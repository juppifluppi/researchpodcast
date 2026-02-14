import os
import requests
import re
import html
import json
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

TRACK_AUTHORS = []

DAYS_BACK = 7
MAX_PAPERS_PER_TOPIC = 12
TOP_SELECTION_TOTAL = 6
MAX_FEED_ITEMS = 20

BASE_URL = "https://juppifluppi.github.io/researchpodcast"
EPISODES_DIR = "episodes"

PODCAST_TITLE = "Research Updates"
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
    audio = high_pass_filter(audio, 120)
    audio = low_pass_filter(audio, 9000)
    return audio + 1

def apply_eq_author(audio):
    audio = high_pass_filter(audio, 80)
    audio = low_pass_filter(audio, 7000)
    return audio - 1

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
# SCRIPT GENERATION
# =========================

def generate_script(selected_papers):

    section = ""
    for p in selected_papers:
        section += f"\n### PAPER_START: {p['title']}\n"

    prompt = """
Create a long-form, in-depth scientific podcast conversation in dialogue format.

Language: English.
Moderator analytical and critical.
Author technically detailed.

Discuss:
- theory
- methodology
- limitations
- implications
- cross-paper trends

Format:

MODERATOR:
<Text>

AUTHOR:
<Text>

Each paper must begin exactly with:
### PAPER_START: <Title>
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You generate a long-form scientific podcast dialogue."},
            {"role": "user", "content": prompt + section}
        ],
        temperature=0.8,
        max_tokens=14000
    )

    return response.choices[0].message.content

# =========================
# EPISODE TITLE + SHOW NOTES
# =========================

def generate_episode_metadata(script, papers):

    paper_list = "\n".join([
        f"- {p['title']} ({p['journal']}) DOI: {p['doi']}"
        for p in papers
    ])

    prompt = f"""
Generate:

1) A compelling but professional podcast episode title (max 15 words).
2) A short episode summary (4–6 sentences).
3) Structured show notes including:
   - Bullet list of discussed papers
   - For each paper: one-sentence technical highlight

Papers:
{paper_list}
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You generate professional podcast metadata."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=800
    )

    text = response.choices[0].message.content

    parts = text.split("\n", 1)
    title = parts[0].strip()
    description = parts[1].strip() if len(parts) > 1 else text

    return title, description

# =========================
# AUDIO PROCESSING
# =========================

def process_block(speaker, text):
    segments = []

    if not text.strip():
        return segments

    if "moderator" in speaker:
        voice = "alloy"
        pan = -0.08
        eq_func = apply_eq_moderator
    else:
        voice = "verse"
        pan = 0.08
        eq_func = apply_eq_author

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
        segment = eq_func(segment)
        segment = segment.pan(pan)

        segments.append(segment)
        segments.append(random_pause())

    return segments

def generate_audio(script):

    os.makedirs(EPISODES_DIR, exist_ok=True)
    filename = f"episode_{datetime.utcnow().strftime('%Y%m%d')}.mp3"
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

        speaker_match = re.match(r"^(MODERATOR|Moderator|AUTHOR|Author)\s*[:\-]?$", line)

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

    spoken = compress_dynamic_range(
        spoken,
        threshold=-20.0,
        ratio=2.0,
        attack=5,
        release=50
    )

    intro = AudioSegment.from_mp3("intro_music.mp3")
    intro = normalize(intro).fade_out(2000)

    outro = AudioSegment.from_mp3("intro_music.mp3")
    outro = normalize(outro).fade_in(2000)

    final_audio = intro + spoken + outro
    final_audio = normalize(final_audio)

    final_audio.export(final_path, format="mp3")

    duration_seconds = int(len(final_audio) / 1000)
    return filename, duration_seconds

# =========================
# RSS GENERATION
# =========================

def update_rss(filename, duration_seconds, episode_title, description):

    feed_path = "feed.xml"
    episode_url = f"{BASE_URL}/episodes/{filename}"
    pub_date = datetime.utcnow().strftime('%a, %d %b %Y %H:%M:%S GMT')

    if not os.path.exists(feed_path):
        rss = ET.Element("rss", version="2.0")
        channel = ET.SubElement(rss, "channel")

        ET.SubElement(channel, "title").text = PODCAST_TITLE
        ET.SubElement(channel, "link").text = BASE_URL
        ET.SubElement(channel, "language").text = PODCAST_LANGUAGE
    else:
        tree = ET.parse(feed_path)
        rss = tree.getroot()
        channel = rss.find("channel")

    item = ET.Element("item")
    ET.SubElement(item, "title").text = episode_title
    ET.SubElement(item, "description").text = description
    ET.SubElement(item, "pubDate").text = pub_date
    ET.SubElement(item, "guid").text = episode_url

    enclosure = ET.SubElement(item, "enclosure")
    enclosure.set("url", episode_url)
    enclosure.set("type", "audio/mpeg")

    ET.SubElement(item, "itunes:duration").text = str(duration_seconds)

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
    episode_title, description = generate_episode_metadata(script, papers)
    update_rss(filename, duration, episode_title, description)
    print("Episode generated:", episode_title)

if __name__ == "__main__":
    main()
