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

TOPICS = [
    "mRNA lipid nanoparticle",
    "bioconjugation",
    "drug mucin bile interactions",
    "polyoxazoline"
]

DAYS_BACK = 7
MAX_PAPERS_PER_TOPIC = 12
TOP_SELECTION_TOTAL = 6
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
# FETCH PAPERS
# =========================

def extract_paper(item):
    return {
        "title": item.get("title", ["No title"])[0],
        "summary": strip_html(item.get("abstract", "")),
        "doi": item.get("DOI", ""),
        "journal": item.get("container-title", ["Unknown Journal"])[0],
        "citations": item.get("is-referenced-by-count", 0),
    }

def fetch_crossref():
    start_date = (datetime.utcnow() - timedelta(days=DAYS_BACK)).strftime("%Y-%m-%d")
    papers = []

    for topic in TOPICS:
        url = (
            "https://api.crossref.org/works?"
            f"query={topic}"
            f"&filter=from-pub-date:{start_date}"
            f"&rows={MAX_PAPERS_PER_TOPIC}"
        )

        response = requests.get(url)
        data = response.json()

        for item in data.get("message", {}).get("items", []):
            papers.append(extract_paper(item))

    unique = {}
    for p in papers:
        if p["doi"]:
            unique[p["doi"]] = p

    return list(unique.values())

# =========================
# SCRIPT GENERATION (PERSONALITY + DYNAMICS)
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
        "- Allow subtle emotional nuance (skepticism, curiosity, surprise).\n"
        "- Include occasional sophisticated analogies.\n"
        "- Keep tone intelligent and realistic.\n\n"
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
            {"role": "system", "content": "Generate natural, nuanced, intellectually rigorous dialogue."},
            {"role": "user", "content": prompt + section}
        ],
        temperature=0.9,
        max_tokens=max_tokens
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
        "Generate:\n"
        "1) A VERY SHORT episode title (max 6 words).\n"
        "2) A concise 3–4 sentence summary.\n\n"
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
        doi_link = "https://doi.org/" + p["doi"]
        show_notes += (
            "<li><strong>" + p["title"] + "</strong><br>"
            "<em>" + p["journal"] + "</em><br>"
            "DOI: <a href=\"" + doi_link + "\">" + doi_link + "</a>"
            "</li>"
        )
    show_notes += "</ul>"

    full_description = "<p>" + summary + "</p>" + show_notes
    title = "Ep. " + str(episode_number).zfill(2) + " – " + raw_title

    return title, full_description

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
            input=chunk,
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
# RSS
# =========================

def update_rss(filename, duration_seconds, title, description, episode_number):

    feed_path = "feed.xml"
    episode_url = BASE_URL + "/episodes/" + filename
    cover_url = BASE_URL + "/cover.png"
    pub_date = datetime.utcnow().strftime("%a, %d %b %Y %H:%M:%S GMT")

    if not os.path.exists(feed_path):
        rss = ET.Element(
            "rss",
            version="2.0",
            attrib={"xmlns:itunes": "http://www.itunes.com/dtds/podcast-1.0.dtd"}
        )
        channel = ET.SubElement(rss, "channel")

        ET.SubElement(channel, "title").text = PODCAST_TITLE
        ET.SubElement(channel, "link").text = BASE_URL
        ET.SubElement(channel, "description").text = PODCAST_DESCRIPTION
        ET.SubElement(channel, "language").text = PODCAST_LANGUAGE

        image = ET.SubElement(channel, "itunes:image")
        image.set("href", cover_url)
    else:
        tree = ET.parse(feed_path)
        rss = tree.getroot()
        channel = rss.find("channel")

    item = ET.Element("item")
    ET.SubElement(item, "title").text = title
    ET.SubElement(item, "description").text = description
    ET.SubElement(item, "pubDate").text = pub_date
    ET.SubElement(item, "guid").text = episode_url
    ET.SubElement(item, "itunes:episode").text = str(episode_number)
    ET.SubElement(item, "itunes:duration").text = str(duration_seconds)

    enclosure = ET.SubElement(item, "enclosure")
    enclosure.set("url", episode_url)
    enclosure.set("type", "audio/mpeg")

    channel.insert(0, item)

    items = channel.findall("item")
    for old in items[MAX_FEED_ITEMS:]:
        channel.remove(old)

    ET.ElementTree(rss).write(feed_path, encoding="utf-8", xml_declaration=True)

# =========================
# MAIN
# =========================

def main():
    papers = fetch_crossref()[:TOP_SELECTION_TOTAL]
    script = generate_script(papers)
    filename, duration = generate_audio(script)

    if os.path.exists("feed.xml"):
        tree = ET.parse("feed.xml")
        channel = tree.getroot().find("channel")
        episode_number = len(channel.findall("item")) + 1
    else:
        episode_number = 1

    title, description = generate_episode_metadata(papers, episode_number)
    update_rss(filename, duration, title, description, episode_number)

    print("Episode generated:", title)

if __name__ == "__main__":
    main()
