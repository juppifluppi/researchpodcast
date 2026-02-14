import os
import requests
import re
import html
import random
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
    "Tessa Lühmann",
    "Josef Kehrein",
    "Marcus Gutmann",
]

DAYS_BACK = 7
TOP_SELECTION_TOTAL = 10
MAX_FEED_ITEMS = 20

TARGET_DURATION_MINUTES = 30
BASE_MINUTES_PER_PAPER = 5
CITATION_WEIGHT_FACTOR = 0.08

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
# AUTHOR PROFILE BUILDING
# =========================

def build_author_profile():

    abstracts = []

    for author in TARGET_AUTHORS:
        url = (
            "https://api.crossref.org/works?"
            f"query.author={author}"
            "&rows=40"
        )

        response = requests.get(url)
        data = response.json()

        for item in data.get("message", {}).get("items", []):
            abstract = strip_html(item.get("abstract", ""))
            if abstract:
                abstracts.append(abstract)

    combined_text = "\n".join(abstracts[:30])

    prompt = (
        "Analyze the following abstracts and summarize the core research themes, "
        "technological focus, applications, and scientific philosophy of these authors. "
        "Be concise but precise.\n\n" + combined_text
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You extract research identity profiles."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=800
    )

    return response.choices[0].message.content

# =========================
# FETCH RECENT PAPERS
# =========================

def fetch_recent_papers():

    start_date = (datetime.utcnow() - timedelta(days=DAYS_BACK)).strftime("%Y-%m-%d")

    url = (
        "https://api.crossref.org/works?"
        f"filter=from-pub-date:{start_date}"
        "&rows=120"
    )

    response = requests.get(url)
    data = response.json()

    papers = []

    for item in data.get("message", {}).get("items", []):
        abstract = strip_html(item.get("abstract", ""))
        if not abstract:
            continue

        papers.append({
            "title": item.get("title", ["No title"])[0],
            "summary": abstract,
            "doi": item.get("DOI", ""),
            "journal": item.get("container-title", ["Unknown Journal"])[0],
            "citations": item.get("is-referenced-by-count", 0),
        })

    return papers

# =========================
# ALIGNMENT RANKING
# =========================

def rank_by_author_alignment(papers, author_profile):

    text = ""
    for i, p in enumerate(papers):
        text += (
            f"\nPaper {i+1}\n"
            f"Title: {p['title']}\n"
            f"Abstract: {p['summary'][:1200]}\n"
        )

    prompt = (
        "Given this research profile:\n\n"
        + author_profile +
        "\n\nSelect the "
        + str(TOP_SELECTION_TOTAL) +
        " papers most aligned with this profile.\n"
        "Return ONLY numbers separated by commas.\n\n"
        + text
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert research evaluator."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=300
    )

    numbers = re.findall(r"\d+", response.choices[0].message.content)
    indices = [int(n) - 1 for n in numbers]

    selected = []
    for i in indices:
        if 0 <= i < len(papers):
            selected.append(papers[i])

    return selected[:TOP_SELECTION_TOTAL]

# =========================
# SCRIPT GENERATION
# =========================

def generate_script(selected_papers):

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

    section = ""
    citation_info = ""

    for p in selected_papers:
        depth_multiplier = 1 + (p["citations"] * CITATION_WEIGHT_FACTOR)
        citation_info += (
            "\nPaper: " + p["title"] +
            "\nCitations: " + str(p["citations"]) +
            "\nDepth weight: " + str(round(depth_multiplier, 2)) + "x\n"
        )
        section += "\n### PAPER_START: " + p["title"] + "\n"

    prompt = (
        "Create a highly dynamic scientific podcast dialogue.\n\n"
        "PERSONALITY:\n"
        "- Moderator is " + moderator_style + ".\n"
        "- Author is " + author_style + ".\n\n"
        "RULES:\n"
        "- Discuss ONLY the listed papers.\n"
        "- NO cross-paper commentary.\n"
        "- NO statistical deep dives.\n"
        "- Avoid mentioning specific tests or p-values.\n\n"
        "DYNAMICS:\n"
        "- Moderator occasionally challenges assumptions.\n"
        "- Include controlled disagreement moments.\n"
        "- Add subtle emotional nuance.\n"
        "- Occasionally use sophisticated analogies.\n\n"
        "Focus on:\n"
        "- Core innovation\n"
        "- Mechanism\n"
        "- Conceptual advance\n"
        "- Real-world implications\n"
        "- High-level limitations\n\n"
        "Depth weighting:\n" + citation_info + "\n\n"
        "Format strictly:\n\n"
        "### PAPER_START: <Title>\n\n"
        "MODERATOR:\n<Text>\n\n"
        "AUTHOR:\n<Text>\n"
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Generate natural, nuanced scientific dialogue."},
            {"role": "user", "content": prompt + section}
        ],
        temperature=0.9,
        max_tokens=9000
    )

    return response.choices[0].message.content

# =========================
# METADATA
# =========================

def generate_episode_metadata(papers, episode_number):

    paper_list = ""
    for p in papers:
        paper_list += "- " + p["title"] + " (" + p["journal"] + ") DOI: https://doi.org/" + p["doi"] + "\n"

    prompt = (
        "Generate a short episode title (max 6 words) and a 3-4 sentence summary.\n\n"
        "Papers:\n" + paper_list
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Generate concise podcast metadata."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.6,
        max_tokens=300
    )

    text = response.choices[0].message.content.strip()
    lines = text.split("\n", 1)

    raw_title = lines[0].strip()
    raw_title = re.sub(r"^\s*\d+[\).\s-]*", "", raw_title)
    raw_title = re.sub(r"(?i)^episode\s*title\s*:\s*", "", raw_title)
    raw_title = raw_title.replace("*", "").strip()

    summary = lines[1] if len(lines) > 1 else ""

    show_notes = "<h3>Discussed Papers</h3><ul>"
    for p in papers:
