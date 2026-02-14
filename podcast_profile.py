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
TARGET_DURATION_MINUTES = 30
BASE_MINUTES_PER_PAPER = 5
CITATION_WEIGHT_FACTOR = 0.04
MAX_FEED_ITEMS = 20

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

def normalize(audio):
    return audio.apply_gain(-20.0 - audio.dBFS)

def speed_adjust(audio, speed=1.08):
    return audio._spawn(
        audio.raw_data,
        overrides={"frame_rate": int(audio.frame_rate * speed)}
    ).set_frame_rate(audio.frame_rate)

def apply_eq_moderator(audio):
    return low_pass_filter(high_pass_filter(audio, 120), 9000) + 1

def apply_eq_author(audio):
    return low_pass_filter(high_pass_filter(audio, 80), 7000) - 1

def openalex_abstract_to_text(inv_index):
    words = []
    for word, positions in inv_index.items():
        for pos in positions:
            words.append((pos, word))
    words.sort()
    return " ".join([w for _, w in words])

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
# AUTHOR CONCEPT EXTRACTION
# =========================

def extract_author_concepts():

    concept_counter = {}

    for name in TARGET_AUTHORS:

        author_search = requests.get(
            "https://api.openalex.org/authors?search=" + name
        ).json()

        if not author_search.get("results"):
            continue

        author_id = author_search["results"][0]["id"]

        works = requests.get(
            "https://api.openalex.org/works?filter=author.id:" + author_id + "&per_page=50"
        ).json()

        for work in works.get("results", []):
            for concept in work.get("concepts", []):
                if concept.get("level", 0) < 2:
                    continue
                cid = concept["id"]
                score = concept.get("score", 0)
                concept_counter[cid] = concept_counter.get(cid, 0) + score

    sorted_concepts = sorted(concept_counter.items(), key=lambda x: x[1], reverse=True)

    return [cid for cid, _ in sorted_concepts[:3]]

# =========================
# PAPER SELECTION
# =========================

def fetch_papers():

    start_date = (datetime.utcnow() - timedelta(days=DAYS_BACK)).strftime("%Y-%m-%d")
    concept_ids = extract_author_concepts()

    if not concept_ids:
        return []

    concept_filter = "|".join([f"concepts.id:{cid}" for cid in concept_ids])

    query = (
        "https://api.openalex.org/works?"
        f"filter=from_publication_date:{start_date},{concept_filter}"
        "&per_page=150"
    )

    recent = requests.get(query).json()

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
                author_abstracts.append(
                    openalex_abstract_to_text(work["abstract_inverted_index"])
                )

    if not author_abstracts:
        return []

    centroid = np.mean(
        [get_embedding(a[:4000]) for a in author_abstracts[:30]],
        axis=0
    )

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
            "summary": abstract,
            "journal": journal,
            "doi": doi,
            "citations": citations,
            "score": score
        })

    ranked = sorted(candidates, key=lambda x: x["score"], reverse=True)

    return ranked

# =========================
# SCRIPT GENERATION
# =========================

def generate_script(selected_papers):

    number_of_papers = len(selected_papers)
    desired_minutes = max(TARGET_DURATION_MINUTES,
                          number_of_papers * BASE_MINUTES_PER_PAPER)

    approx_tokens = int(desired_minutes * 180)
    max_tokens = min(9000, approx_tokens)

    section = ""
    for p in selected_papers:
        section += f"\n### PAPER_START: {p['title']}\n"

    prompt = (
        "Create an intelligent scientific podcast dialogue.\n\n"
        "Moderator slightly contrarian.\n"
        "Include disagreement moments and emotional nuance.\n"
        "Discuss only listed papers.\n"
        "No statistical deep dives.\n\n"
        "Format:\n\n"
        "### PAPER_START: <Title>\n\n"
        "MODERATOR:\nText\n\n"
        "AUTHOR:\nText\n"
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Generate natural dialogue."},
            {"role": "user", "content": prompt + section}
        ],
        temperature=0.85,
        max_tokens=max_tokens
    )

    return response.choices[0].message.content

# =========================
# AUDIO
# =========================

def process_block(speaker, text):
    voice = "alloy" if "moderator" in speaker else "verse"
    eq = apply_eq_moderator if "moderator" in speaker else apply_eq_author

    speech = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice=voice,
        input=text
    )

    temp = os.path.join(EPISODES_DIR, "temp.mp3")
    with open(temp, "wb") as f:
        f.write(speech.content)

    segment = AudioSegment.from_mp3(temp)
    return eq(segment)

def generate_audio(script):

    os.makedirs(EPISODES_DIR, exist_ok=True)

    filename = "episode_" + datetime.utcnow().strftime("%Y%m%d") + ".mp3"
    path = os.path.join(EPISODES_DIR, filename)

    segments = []
    speaker = None
    buffer = ""

    for line in script.split("\n"):
        if line.startswith("MODERATOR"):
            if buffer:
                segments.append(process_block(speaker, buffer))
                buffer = ""
            speaker = "moderator"
        elif line.startswith("AUTHOR"):
            if buffer:
                segments.append(process_block(speaker, buffer))
                buffer = ""
            speaker = "author"
        else:
            buffer += " " + line

    if buffer:
        segments.append(process_block(speaker, buffer))

    spoken = sum(segments)
    spoken = speed_adjust(normalize(spoken))
    spoken = compress_dynamic_range(spoken, threshold=-20, ratio=2.0)

    intro = normalize(AudioSegment.from_mp3("intro_music.mp3")).fade_out(2000)
    outro = normalize(AudioSegment.from_mp3("intro_music.mp3")).fade_in(2000)

    final = normalize(intro + spoken + outro)
    final.export(path, format="mp3")

    return filename, int(len(final) / 1000)

# =========================
# MAIN
# =========================

def main():

    ranked = fetch_papers()

    if not ranked:
        print("No aligned papers found.")
        return

    # Dynamic number of papers based on target duration
    num_papers = max(3, min(8, TARGET_DURATION_MINUTES // BASE_MINUTES_PER_PAPER))
    selected = ranked[:num_papers]

    script = generate_script(selected)
    filename, duration = generate_audio(script)

    print("Episode generated:", filename, "Duration:", duration)

if __name__ == "__main__":
    main()
