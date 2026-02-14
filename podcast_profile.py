import os
import requests
import numpy as np
import json
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from openai import OpenAI
from pydub import AudioSegment
from pydub.effects import compress_dynamic_range, high_pass_filter, low_pass_filter

# =========================
# CONFIG
# =========================

AUTHOR_IDS = [
    "https://openalex.org/A5018917714",
    "https://openalex.org/A5001051737",
    "https://openalex.org/A5000977163"
]

ALLOWED_FIELDS = {
    "Chemistry",
    "Biochemistry",
    "Pharmacology",
    "Materials Science",
    "Biomedical Engineering"
}

DAYS_BACK = 30
TARGET_DURATION_MINUTES = 30
BASE_MINUTES_PER_PAPER = 3
MAX_FEED_ITEMS = 10

BASE_URL = "https://juppifluppi.github.io/researchpodcast"
EPISODES_DIR = "episodes"
FEED_PATH = "feed.xml"
COVER_URL = BASE_URL + "/cover.png"

PODCAST_TITLE = "Research Updates"
PODCAST_DESCRIPTION = "AI-generated analysis of recent scientific publications."
PODCAST_LANGUAGE = "en-us"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# =========================
# AUDIO UTILITIES
# =========================

def normalize(audio):
    if len(audio) == 0:
        return audio
    return audio.apply_gain(-20.0 - audio.dBFS)

def speed_adjust(audio, speed=1.05):
    return audio._spawn(
        audio.raw_data,
        overrides={"frame_rate": int(audio.frame_rate * speed)}
    ).set_frame_rate(audio.frame_rate)

def apply_eq_moderator(audio):
    return low_pass_filter(high_pass_filter(audio, 120), 9000) + 1

def apply_eq_author(audio):
    return low_pass_filter(high_pass_filter(audio, 80), 7000) - 1

# =========================
# EMBEDDING
# =========================

def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# =========================
# OPENALEX HELPERS
# =========================

def openalex_abstract_to_text(inv_index):
    words = []
    for word, positions in inv_index.items():
        for pos in positions:
            words.append((pos, word))
    words.sort()
    return " ".join([w for _, w in words])

# =========================
# AUTHOR PROFILE
# =========================

def build_author_profile():

    author_refs = set()
    author_journal_ids = set()
    author_abstracts = []

    for author_id in AUTHOR_IDS:
        works = requests.get(
            f"https://api.openalex.org/works?filter=author.id:{author_id}&sort=publication_date:desc&per_page=20"
        ).json()

        for work in works.get("results", []):

            for ref in work.get("referenced_works", []):
                author_refs.add(ref)

            if work.get("primary_location") and work["primary_location"].get("source"):
                source = work["primary_location"]["source"]
                if source.get("id"):
                    author_journal_ids.add(source["id"])

            if work.get("abstract_inverted_index"):
                author_abstracts.append(
                    openalex_abstract_to_text(work["abstract_inverted_index"])[:4000]
                )

    if not author_abstracts:
        return None, None, None

    centroid = np.mean([get_embedding(a) for a in author_abstracts], axis=0)

    return centroid, author_refs, author_journal_ids

# =========================
# FIELD FILTER
# =========================

def field_allowed(work):
    if not work.get("primary_topic"):
        return False
    field = work["primary_topic"].get("field", {}).get("display_name")
    return field in ALLOWED_FIELDS

# =========================
# FETCH & RANK PAPERS
# =========================

def fetch_recent_papers():

    centroid, author_refs, author_journal_ids = build_author_profile()

    if centroid is None:
        print("Author profile empty.")
        return []

    # -------------------------------
    # Step 1: Retrieve larger pool
    # -------------------------------

    pool_days = max(DAYS_BACK, 30)
    start_date = (datetime.utcnow() - timedelta(days=pool_days)).strftime("%Y-%m-%d")

    params = {
        "filter": f"publication_date:>{start_date}",
        "sort": "publication_date:desc",
        "per_page": 300
    }

    response = requests.get("https://api.openalex.org/works", params=params)

    if response.status_code != 200:
        print("OpenAlex error:", response.status_code)
        print(response.text)
        return []

    data = response.json()
    results = data.get("results", [])

    if not results:
        print("OpenAlex returned no recent works.")
        return []

    print("Retrieved", len(results), "recent works.")

    # -------------------------------
    # Step 2: Build 2-hop graph
    # -------------------------------

    two_hop_refs = set(author_refs)

    for ref_id in list(author_refs)[:150]:
        try:
            ref_data = requests.get(f"https://api.openalex.org/works/{ref_id}")
            if ref_data.status_code != 200:
                continue
            ref_json = ref_data.json()
            for second_ref in ref_json.get("referenced_works", []):
                two_hop_refs.add(second_ref)
        except:
            continue

    print("1-hop refs:", len(author_refs))
    print("2-hop refs:", len(two_hop_refs))

    # -------------------------------
    # Step 3: Score candidates
    # -------------------------------

    candidates = []

    for work in results:

        refs = set(work.get("referenced_works", []))

        overlap_1 = len(refs & author_refs)
        overlap_2 = len(refs & two_hop_refs)

        graph_score = (0.7 * overlap_1) + (0.3 * overlap_2)

        citations = work.get("cited_by_count", 0)
        citation_multiplier = 1 + np.log1p(citations)

        journal_id = None
        journal_name = "Unknown Journal"

        if work.get("primary_location") and work["primary_location"].get("source"):
            source = work["primary_location"]["source"]
            journal_id = source.get("id")
            journal_name = source.get("display_name", "Unknown Journal")

        journal_score = 1.3 if journal_id in author_journal_ids else 1.0

        pub_date = work.get("publication_date")
        days_old = 0
        if pub_date:
            try:
                days_old = (datetime.utcnow() - datetime.strptime(pub_date, "%Y-%m-%d")).days
            except:
                days_old = 0

        # recency weighting
        recency_score = 2 / (1 + days_old)

        base_score = (
            0.5 * graph_score +
            0.2 * journal_score +
            0.2 * recency_score
        ) * citation_multiplier

        abstract = ""
        if work.get("abstract_inverted_index"):
            abstract = openalex_abstract_to_text(work["abstract_inverted_index"])

        doi = work.get("doi", "")
        if doi:
            doi = doi.replace("https://doi.org/", "")

        candidates.append({
            "title": work.get("title", "Untitled"),
            "journal": journal_name,
            "doi": doi,
            "abstract": abstract,
            "base_score": base_score
        })

    if not candidates:
        print("No candidates after scoring.")
        return []

    print("Scored", len(candidates), "candidates.")

    # -------------------------------
    # Step 4: Preselect before embedding
    # -------------------------------

    candidates = sorted(candidates, key=lambda x: x["base_score"], reverse=True)[:80]

    # -------------------------------
    # Step 5: Embedding refinement
    # -------------------------------

    refined = []

    for c in candidates:

        if c["abstract"]:
            try:
                emb = get_embedding(c["abstract"][:4000])
                sim = cosine_similarity(emb, centroid)
                c["final_score"] = c["base_score"] + 0.2 * sim
            except:
                c["final_score"] = c["base_score"]
        else:
            c["final_score"] = c["base_score"]

        refined.append(c)

    refined = sorted(refined, key=lambda x: x["final_score"], reverse=True)

    print("Returning top", len(refined[:50]), "ranked papers.")

    return refined[:50]

# =========================
# SCRIPT GENERATION
# =========================

def generate_script(selected_papers):

    desired_minutes = max(
        TARGET_DURATION_MINUTES,
        len(selected_papers) * BASE_MINUTES_PER_PAPER
    )

    section = ""
    for p in selected_papers:
        section += f"\nPaper title: {p['title']}\n"

    prompt = (
        f"Create a structured academic podcast dialogue.\n"
        f"Target length approximately {desired_minutes} minutes.\n\n"
        "Discuss ONLY the listed papers.\n\n"
        "Mark each paper start EXACTLY as:\n"
        "=== PAPER: <Title> ===\n\n"
        "Format strictly:\n"
        "MODERATOR:\nText\n\n"
        "AUTHOR:\nText\n"
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Generate precise academic dialogue."},
            {"role": "user", "content": prompt + section}
        ],
        temperature=0.6,
        max_tokens=9000
    )

    return response.choices[0].message.content

# =========================
# AUDIO + CHAPTERS
# =========================

def process_block(speaker, text):

    if not text.strip():
        return None

    voice = "alloy" if speaker == "moderator" else "verse"
    eq = apply_eq_moderator if speaker == "moderator" else apply_eq_author

    speech = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice=voice,
        input=text
    )

    temp = os.path.join(EPISODES_DIR, "temp.mp3")
    with open(temp, "wb") as f:
        f.write(speech.content)

    return eq(AudioSegment.from_mp3(temp))

def generate_audio(script):

    os.makedirs(EPISODES_DIR, exist_ok=True)
    filename = "episode_" + datetime.utcnow().strftime("%Y%m%d") + ".mp3"
    path = os.path.join(EPISODES_DIR, filename)

    segments = []
    speaker = None
    buffer = ""
    chapter_titles = []

    for line in script.split("\n"):
        stripped = line.strip()

        if stripped.startswith("=== PAPER:"):
            title = stripped.replace("=== PAPER:", "").replace("===", "").strip()
            chapter_titles.append(title)
            continue

        if stripped.startswith("MODERATOR"):
            if buffer and speaker:
                seg = process_block(speaker, buffer)
                if seg:
                    segments.append(seg)
            buffer = ""
            speaker = "moderator"
            continue

        if stripped.startswith("AUTHOR"):
            if buffer and speaker:
                seg = process_block(speaker, buffer)
                if seg:
                    segments.append(seg)
            buffer = ""
            speaker = "author"
            continue

        if speaker:
            buffer += " " + stripped

    if buffer and speaker:
        seg = process_block(speaker, buffer)
        if seg:
            segments.append(seg)

    if not segments:
        raise ValueError("No speech segments generated.")

    spoken = segments[0]
    for seg in segments[1:]:
        spoken += seg

    spoken = speed_adjust(normalize(spoken))
    spoken = compress_dynamic_range(spoken, threshold=-20, ratio=2.0)

    intro = normalize(AudioSegment.from_mp3("intro_music.mp3")).fade_out(2000)
    outro = normalize(AudioSegment.from_mp3("intro_music.mp3")).fade_in(2000)

    final = normalize(intro + spoken + outro)
    final.export(path, format="mp3")

    duration_seconds = int(len(final) / 1000)

    chapters = []
    if chapter_titles:
        step = duration_seconds // len(chapter_titles)
        for i, title in enumerate(chapter_titles):
            chapters.append({
                "startTime": i * step,
                "title": title
            })

    chapter_file = filename.replace(".mp3", ".json")
    with open(os.path.join(EPISODES_DIR, chapter_file), "w") as f:
        json.dump({"version": "1.2.0", "chapters": chapters}, f, indent=2)

    return filename, duration_seconds, chapter_file

# =========================
# RSS
# =========================

def update_rss(filename, duration, selected_papers, chapter_file):

    episode_url = f"{BASE_URL}/episodes/{filename}"
    chapter_url = f"{BASE_URL}/episodes/{chapter_file}"
    pub_date = datetime.utcnow().strftime("%a, %d %b %Y %H:%M:%S GMT")

    if os.path.exists(FEED_PATH):
        tree = ET.parse(FEED_PATH)
        rss = tree.getroot()
        channel = rss.find("channel")
    else:
        rss = ET.Element(
            "rss",
            version="2.0",
            attrib={
                "xmlns:itunes": "http://www.itunes.com/dtds/podcast-1.0.dtd",
                "xmlns:podcast": "https://podcastindex.org/namespace/1.0"
            }
        )
        channel = ET.SubElement(rss, "channel")
        ET.SubElement(channel, "title").text = PODCAST_TITLE
        ET.SubElement(channel, "link").text = BASE_URL
        ET.SubElement(channel, "description").text = PODCAST_DESCRIPTION
        ET.SubElement(channel, "language").text = PODCAST_LANGUAGE
        image = ET.SubElement(channel, "itunes:image")
        image.set("href", COVER_URL)

    episode_number = len(channel.findall("item")) + 1
    title = f"Ep. {str(episode_number).zfill(2)}"

    description = "<h3>Discussed Papers</h3><ul>"
    for p in selected_papers:
        doi_link = f"https://doi.org/{p['doi']}"
        description += (
            f"<li><strong>{p['title']}</strong><br>"
            f"<em>{p['journal']}</em><br>"
            f"<a href='{doi_link}'>{doi_link}</a></li>"
        )
    description += "</ul>"

    item = ET.Element("item")
    ET.SubElement(item, "title").text = title
    ET.SubElement(item, "description").text = description
    ET.SubElement(item, "pubDate").text = pub_date
    ET.SubElement(item, "guid").text = episode_url
    ET.SubElement(item, "itunes:episode").text = str(episode_number)
    ET.SubElement(item, "itunes:duration").text = str(duration)

    enclosure = ET.SubElement(item, "enclosure")
    enclosure.set("url", episode_url)
    enclosure.set("type", "audio/mpeg")

    ET.SubElement(item, "podcast:chapters").set("url", chapter_url)

    channel.insert(0, item)
    ET.ElementTree(rss).write(FEED_PATH, encoding="utf-8", xml_declaration=True)

# =========================
# MAIN
# =========================

def main():

    ranked = fetch_recent_papers()

    if not ranked:
        print("No aligned papers found.")
        return

    approx_papers = TARGET_DURATION_MINUTES // BASE_MINUTES_PER_PAPER
    selected = ranked[:max(3, min(10, approx_papers))]

    script = generate_script(selected)
    filename, duration, chapter_file = generate_audio(script)
    update_rss(filename, duration, selected, chapter_file)

    print("Episode generated:", filename)

if __name__ == "__main__":
    main()
