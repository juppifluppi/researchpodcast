import os
import requests
import re
import html
import random
import numpy as np
import xml.etree.ElementTree as ET
from openai import OpenAI
from pydub import AudioSegment
from pydub.effects import compress_dynamic_range, high_pass_filter, low_pass_filter
from datetime import datetime, timedelta

# =========================
# CONFIG
# =========================

TARGET_AUTHORS = [
    "Lorenz Meinel",
    "Tessa Lühmann"
]

DAYS_BACK = 7
TOP_SELECTION_TOTAL = 10
MAX_FEED_ITEMS = 20

TARGET_DURATION_MINUTES = 30
BASE_MINUTES_PER_PAPER = 5
CITATION_WEIGHT_FACTOR = 0.04

BASE_URL = "https://juppifluppi.github.io/researchpodcast"
EPISODES_DIR = "episodes"

PODCAST_TITLE = "Research Updates"
PODCAST_DESCRIPTION = "AI-generated deep dive into recent scientific publications."
PODCAST_LANGUAGE = "en-us"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# =========================
# UTILITIES
# =========================

def strip_html(text):
    if not text:
        return ""
    return html.unescape(re.sub("<.*?>", "", text))

def normalize(audio):
    return audio.apply_gain(-20.0 - audio.dBFS)

def speed_adjust(audio, speed=1.10):
    return audio._spawn(
        audio.raw_data,
        overrides={"frame_rate": int(audio.frame_rate * speed)}
    ).set_frame_rate(audio.frame_rate)

def random_pause():
    return AudioSegment.silent(duration=random.randint(250, 500))

def random_long_pause():
    return AudioSegment.silent(duration=random.randint(2000, 3000))

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
# PAPER SELECTION
# =========================

def fetch_crossref():

    start_date = (datetime.utcnow() - timedelta(days=DAYS_BACK)).strftime("%Y-%m-%d")

    # ---- Build author embedding centroid ----
    author_abstracts = []

    for name in TARGET_AUTHORS:
        author_search = requests.get(
            "https://api.openalex.org/authors?search=" + name
        ).json()

        if not author_search.get("results"):
            continue

        author_id = author_search["results"][0]["id"]

        works = requests.get(
            "https://api.openalex.org/works?filter=author.id:" + author_id + "&per_page=40"
        ).json()

        for work in works.get("results", []):
            if work.get("abstract_inverted_index"):
                abstract = openalex_abstract_to_text(work["abstract_inverted_index"])
                if abstract:
                    author_abstracts.append(abstract)

    if not author_abstracts:
        return []

    embeddings = [get_embedding(a[:4000]) for a in author_abstracts[:30]]
    centroid = np.mean(embeddings, axis=0)

    # ---- Fetch recent works (domain-filtered) ----
    domain_query = (
        "https://api.openalex.org/works?"
        f"filter=from_publication_date:{start_date},"
        "concepts.display_name.search:drug|polymer|nanoparticle|biomaterial|lipid"
        "&per_page=100"
    )

    recent = requests.get(domain_query).json()

    candidates = []

    for work in recent.get("results", []):
        if not work.get("abstract_inverted_index"):
            continue

        abstract = openalex_abstract_to_text(work["abstract_inverted_index"])
        if not abstract:
            continue

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
            "summary": abstract,
            "doi": doi,
            "journal": journal,
            "citations": citations,
            "score": score
        })

    ranked = sorted(candidates, key=lambda x: x["score"], reverse=True)

    return ranked[:TOP_SELECTION_TOTAL]

# =========================
# SCRIPT GENERATION (UNCHANGED)
# =========================

def generate_script(selected_papers):

    number_of_papers = len(selected_papers)
    base_duration = number_of_papers * BASE_MINUTES_PER_PAPER
    desired_minutes = max(TARGET_DURATION_MINUTES, base_duration)

    approx_tokens = int(desired_minutes * 180)
    max_tokens = min(9000, approx_tokens)

    moderator_styles = [
        "analytical but slightly contrarian",
        "skeptical and probing",
        "cautiously optimistic yet demanding",
        "methodical and intellectually strict"
    ]

    author_styles = [
        "measured and reflective",
        "technically confident but grounded",
        "carefully enthusiastic",
        "precise and calmly defensive when challenged"
    ]

    moderator_style = random.choice(moderator_styles)
    author_style = random.choice(author_styles)

    citation_info = ""
    for p in selected_papers:
        depth_multiplier = 1 + (p["citations"] * CITATION_WEIGHT_FACTOR)
        citation_info += (
            "\nPaper: " + p["title"] +
            "\nCitations: " + str(p["citations"]) +
            "\nDepth weight: " + str(round(depth_multiplier, 2)) + "x\n"
        )

    section = ""
    for p in selected_papers:
        section += "\n### PAPER_START: " + p["title"] + "\n"

    prompt = (
        "Create a highly dynamic scientific podcast dialogue.\n\n"
        "PERSONALITY:\n"
        "- Moderator is " + moderator_style + ".\n"
        "- Author is " + author_style + ".\n\n"
        "STRICT RULES:\n"
        "- Discuss ONLY the listed papers.\n"
        "- NO cross-paper comparisons.\n"
        "- NO general field commentary.\n"
        "- NO statistical deep dives.\n"
        "- Do not mention specific test names or p-values.\n\n"
        "CONVERSATIONAL DYNAMICS:\n"
        "- Moderator occasionally challenges assumptions.\n"
        "- Include controlled disagreement moments.\n"
        "- Allow subtle emotional nuance.\n"
        "- Include occasional sophisticated analogies.\n\n"
        "CONTENT FOCUS:\n"
        "- Core innovation\n"
        "- Mechanism\n"
        "- Conceptual advance\n"
        "- Why it matters\n"
        "- Real-world implications\n"
        "- High-level limitations\n\n"
        "Target length: approx " + str(desired_minutes) + " minutes.\n\n"
        "Depth weighting:\n" + citation_info + "\n\n"
        "Format strictly:\n\n"
        "### PAPER_START: <Title>\n\n"
        "MODERATOR:\n<Text>\n\n"
        "AUTHOR:\n<Text>\n"
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Generate natural, nuanced dialogue."},
            {"role": "user", "content": prompt + section}
        ],
        temperature=0.9,
        max_tokens=max_tokens
    )

    return response.choices[0].message.content

# =========================
# AUDIO
# =========================

def process_block(speaker, text):
    segments = []
    voice = "alloy" if "moderator" in speaker else "verse"
    pan = -0.08 if "moderator" in speaker else 0.08
    eq_func = apply_eq_moderator if "moderator" in speaker else apply_eq_author

    words = text.split()
    chunks = [" ".join(words[i:i+650]) for i in range(0, len(words), 650)]

    for chunk in chunks:
        speech = client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice=voice,
            input=chunk
        )

        temp_path = os.path.join(EPISODES_DIR, "temp.mp3")
        with open(temp_path, "wb") as f:
            f.write(speech.content)

        segment = AudioSegment.from_mp3(temp_path)
        segment = eq_func(segment).pan(pan)

        segments.append(segment)
        segments.append(random_pause())

    return segments


def generate_audio(script):

    os.makedirs(EPISODES_DIR, exist_ok=True)
    filename = "episode_" + datetime.utcnow().strftime("%Y%m%d") + ".mp3"
    final_path = os.path.join(EPISODES_DIR, filename)

    segments = []
    current_speaker = None
    buffer = ""

    for line in script.split("\n"):
        line = line.strip()

        if line.startswith("### PAPER_START"):
            if buffer:
                segments.extend(process_block(current_speaker, buffer))
                buffer = ""
            segments.append(random_long_pause())
            continue

        speaker_match = re.match(r"^(MODERATOR|AUTHOR)\s*[:\-]?$", line)

        if speaker_match:
            if buffer:
                segments.extend(process_block(current_speaker, buffer))
                buffer = ""
            current_speaker = speaker_match.group(1).lower()
            continue

        if current_speaker:
            buffer += " " + line

    if buffer:
        segments.extend(process_block(current_speaker, buffer))

    spoken = sum(segments)
    spoken = speed_adjust(normalize(spoken), speed=1.10)
    spoken = compress_dynamic_range(spoken, threshold=-20.0, ratio=2.0)

    intro = normalize(AudioSegment.from_mp3("intro_music.mp3")).fade_out(2000)
    outro = normalize(AudioSegment.from_mp3("intro_music.mp3")).fade_in(2000)

    final_audio = normalize(intro + spoken + outro)
    final_audio.export(final_path, format="mp3")

    return filename, int(len(final_audio) / 1000)

# =========================
# MAIN
# =========================

def main():
    papers = fetch_crossref()
    if not papers:
        print("No aligned papers found.")
        return

    script = generate_script(papers)
    filename, duration = generate_audio(script)

    print("Episode generated:", filename, "Duration:", duration)

if __name__ == "__main__":
    main()
