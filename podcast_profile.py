import os
import requests
import numpy as np
import json
import re
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

DAYS_BACK = 7
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
# AUTHOR PROFILING
# =========================

def build_author_centroid():
    author_abstracts = []

    for author_id in AUTHOR_IDS:
        works = requests.get(
            f"https://api.openalex.org/works?filter=author.id:{author_id}&sort=publication_date:desc&per_page=10"
        ).json()

        for work in works.get("results", []):
            if work.get("abstract_inverted_index"):
                text = openalex_abstract_to_text(work["abstract_inverted_index"])
                author_abstracts.append(text[:4000])

    if not author_abstracts:
        return None

    embeddings = [get_embedding(a) for a in author_abstracts]
    return np.mean(embeddings, axis=0)

# =========================
# FETCH PAPERS
# =========================

def fetch_recent_papers():

    start_date = (datetime.utcnow() - timedelta(days=DAYS_BACK)).strftime("%Y-%m-%d")
    centroid = build_author_centroid()

    if centroid is None:
        print("No author profile available.")
        return []

    query = (
        "https://api.openalex.org/works?"
        f"filter=from_publication_date:{start_date}"
        "&per_page=200"
    )

    recent = requests.get(query).json()

    candidates = []

    for work in recent.get("results", []):
        if not work.get("abstract_inverted_index"):
            continue

        abstract = openalex_abstract_to_text(work["abstract_inverted_index"])
        embedding = get_embedding(abstract[:4000])

        similarity = cosine_similarity(embedding, centroid)
        citations = work.get("cited_by_count", 0)
        score = similarity * (1 + np.log1p(citations))

        journal = "Unknown Journal"
        if work.get("primary_location") and work["primary_location"].get("source"):
            journal = work["primary_location"]["source"]["display_name"]

        doi = work.get("doi", "")
        if doi:
            doi = doi.replace("https://doi.org/", "")

        candidates.append({
            "title": work["title"],
            "journal": journal,
            "doi": doi,
            "score": score
        })

    return sorted(candidates, key=lambda x: x["score"], reverse=True)

# =========================
# SCRIPT GENERATION
# =========================

def generate_script(selected_papers):

    desired_papers = len(selected_papers)
    approx_minutes = max(TARGET_DURATION_MINUTES, desired_papers * BASE_MINUTES_PER_PAPER)

    section = ""
    for p in selected_papers:
        section += f"\nPaper title: {p['title']}\n"

    prompt = (
        f"Create a structured academic podcast dialogue.\n"
        f"Target length: approximately {approx_minutes} minutes.\n\n"
        "Discuss ONLY the listed papers.\n"
        "No cross-paper comparisons.\n"
        "No general commentary.\n"
        "No dramatic tone.\n\n"
        "For each paper discuss:\n"
        "- Research problem\n"
        "- Methodology\n"
        "- Main findings\n"
        "Mark each paper start EXACTLY as:\n"
        "=== PAPER: <Title> ===\n\n"
        "Dialogue format strictly:\n"
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

    script = response.choices[0].message.content
    script = script.replace("#", "")
    return script

# =========================
# AUDIO + CHAPTER TRACKING
# =========================

def process_block(speaker, text):
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
    chapter_positions = []

    for line in script.split("\n"):
        stripped = line.strip()

        if stripped.startswith("=== PAPER:"):
            title = stripped.replace("=== PAPER:", "").replace("===", "").strip()
            chapter_positions.append({"title": title, "start": len(segments)})
            continue

        if stripped.startswith("MODERATOR"):
            if buffer and speaker:
                segments.append(process_block(speaker, buffer))
            buffer = ""
            speaker = "moderator"
            continue

        if stripped.startswith("AUTHOR"):
            if buffer and speaker:
                segments.append(process_block(speaker, buffer))
            buffer = ""
            speaker = "author"
            continue

        if speaker:
            buffer += " " + stripped

    if buffer and speaker:
        segments.append(process_block(speaker, buffer))

    spoken = sum(segments)
    spoken = speed_adjust(normalize(spoken))
    spoken = compress_dynamic_range(spoken, threshold=-20, ratio=2.0)

    intro = normalize(AudioSegment.from_mp3("intro_music.mp3")).fade_out(2000)
    outro = normalize(AudioSegment.from_mp3("intro_music.mp3")).fade_in(2000)

    final = normalize(intro + spoken + outro)
    final.export(path, format="mp3")

    duration_seconds = int(len(final) / 1000)

    # Generate chapter JSON
    words_per_minute = 160
    chapters = []
    total_words = len(script.split())
    total_seconds = (total_words / words_per_minute) * 60

    for idx, ch in enumerate(chapter_positions):
        start_word = int((idx / len(chapter_positions)) * total_words)
        start_time = int((start_word / total_words) * total_seconds)
        chapters.append({
            "startTime": start_time,
            "title": ch["title"]
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
        rss = ET.Element("rss", version="2.0",
                         attrib={
                             "xmlns:itunes": "http://www.itunes.com/dtds/podcast-1.0.dtd",
                             "xmlns:podcast": "https://podcastindex.org/namespace/1.0"
                         })
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

    # Dynamically adapt number of papers
    approx_papers = TARGET_DURATION_MINUTES // BASE_MINUTES_PER_PAPER
    selected = ranked[:max(3, min(10, approx_papers))]

    script = generate_script(selected)
    filename, duration, chapter_file = generate_audio(script)
    update_rss(filename, duration, selected, chapter_file)

    print("Episode generated:", filename)

if __name__ == "__main__":
    main()
