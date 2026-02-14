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
                cid = concept["id"]
                score = concept.get("score", 0)
                concept_counter[cid] = concept_counter.get(cid, 0) + score

    sorted_concepts = sorted(concept_counter.items(), key=lambda x: x[1], reverse=True)

    top_concepts = [cid for cid, _ in sorted_concepts[:5]]

    return top_concepts

# =========================
# PAPER SELECTION
# =========================

def fetch_crossref():

    start_date = (datetime.utcnow() - timedelta(days=DAYS_BACK)).strftime("%Y-%m-%d")

    # Step 1: Get dominant concept IDs
    concept_ids = extract_author_concepts()

    if not concept_ids:
        return []

    # Step 2: Build centroid from author abstracts
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
                author_abstracts.append(abstract)

    if not author_abstracts:
        return []

    embeddings = [get_embedding(a[:4000]) for a in author_abstracts[:30]]
    centroid = np.mean(embeddings, axis=0)

    # Step 3: Fetch recent works constrained to dominant concepts
    concept_filter = "|".join(concept_ids)

    recent_query = (
        "https://api.openalex.org/works?"
        f"filter=from_publication_date:{start_date},concepts.id:{concept_filter}"
        "&per_page=150"
    )

    recent = requests.get(recent_query).json()

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

    section = ""
    for p in selected_papers:
        section += "\n### PAPER_START: " + p["title"] + "\n"

    prompt = (
        "Create a highly dynamic scientific podcast dialogue.\n\n"
        "Discuss ONLY the listed papers.\n"
        "No cross-paper comparisons.\n"
        "No statistical deep dives.\n\n"
        "Format strictly:\n\n"
        "### PAPER_START: <Title>\n\n"
        "MODERATOR:\nText\n\n"
        "AUTHOR:\nText\n"
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Generate natural, nuanced dialogue."},
            {"role": "user", "content": prompt + section}
        ],
        temperature=0.8,
        max_tokens=max_tokens
    )

    return response.choices[0].message.content

# =========================
# MAIN
# =========================

def main():
    papers = fetch_crossref()
    if not papers:
        print("No aligned papers found.")
        return

    script = generate_script(papers)
    print("Selected papers:")
    for p in papers:
        print("-", p["title"])

if __name__ == "__main__":
    main()
