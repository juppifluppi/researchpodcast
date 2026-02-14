import os
import requests
import re
import json
import html
from openai import OpenAI
from pydub import AudioSegment
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET

# =========================
# CONFIG
# =========================

TOPICS = ["lipid nanoparticle", "bioconjugation"]  # Customize
DAYS_BACK = 14
MAX_PAPERS_PER_TOPIC = 12
TOP_SELECTION_TOTAL = 6
MAX_FEED_ITEMS = 20

SITE_URL = "https://juppifluppi.github.io"
EPISODES_DIR = "episodes"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# =========================
# UTILITIES
# =========================

def strip_html(text):
    if not text:
        return ""
    clean = re.sub("<.*?>", "", text)
    return html.unescape(clean)

def normalize(audio):
    return audio.apply_gain(-20.0 - audio.dBFS)

def speed_up(audio, speed=1.18):
    return audio._spawn(
        audio.raw_data,
        overrides={"frame_rate": int(audio.frame_rate * speed)}
    ).set_frame_rate(audio.frame_rate)

# =========================
# FETCH FROM CROSSREF
# =========================

def fetch_crossref():
    start_date = (datetime.utcnow() - timedelta(days=DAYS_BACK)).strftime("%Y-%m-%d")
    papers = []

    for topic in TOPICS:
        url = (
            "https://api.crossref.org/works?"
            f"query={topic}"
            f"&filter=from-pub-date:{start_date}"
            f"&rows={MAX_PAPERS_PER_TOPIC}"
            "&sort=published"
            "&order=desc"
        )

        response = requests.get(url)
        data = response.json()

        for item in data.get("message", {}).get("items", []):
            title = item.get("title", ["No title"])[0]
            abstract = strip_html(item.get("abstract", ""))
            doi = item.get("DOI", "")
            journal = item.get("container-title", ["Unknown Journal"])[0]
            citations = item.get("is-referenced-by-count", 0)
            link = f"https://doi.org/{doi}" if doi else ""

            papers.append({
                "topic": topic,
                "title": title,
                "summary": abstract,
                "doi": doi,
                "journal": journal,
                "citations": citations,
                "link": link
            })

    return papers

# =========================
# RANK PAPERS
# =========================

def rank_papers(papers):
    text = ""
    for i, p in enumerate(papers):
        text += f"""
Paper {i+1}
Topic: {p['topic']}
Title: {p['title']}
Journal: {p['journal']}
Citations: {p['citations']}
Abstract: {p['summary']}
"""

    prompt = f"""
Select the {TOP_SELECTION_TOTAL} most important papers.
Return ONLY numbers separated by commas.

Prioritize:
- theoretical contribution
- methodological innovation
- conceptual novelty
- forward-looking implications
- citation count as signal
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a strict academic evaluator."},
            {"role": "user", "content": prompt + text}
        ],
        temperature=0.2,
    )

    numbers = re.findall(r"\d+", response.choices[0].message.content)
    indices = [int(n) - 1 for n in numbers]

    selected = []
    for i in indices:
        if 0 <= i < len(papers):
            selected.append(papers[i])

    return selected[:TOP_SELECTION_TOTAL]

# =========================
# GENERATE SCRIPT (~6000 WORDS)
# =========================

def generate_script(selected_papers):
    section = ""
    for p in selected_papers:
        section += f"""
### PAPER_START: {p['title']}

Topic: {p['topic']}
Journal: {p['journal']}
Citations: {p['citations']}
Abstract: {p['summary']}
DOI: {p['doi']}
"""

    prompt = f"""
Erstelle ein ca. 6000 Wörter langes technisches Forschungspodcast-Skript.

Sprache: Deutsch.
Zielgruppe: Forschende.
Keine Einleitungen.
Keine Musik-Erwähnung.
Keine Vereinfachungen.

Jedes Paper MUSS mit der Marker-Zeile beginnen:
### PAPER_START: <Original Title>

Diskutiere:
- Theoretischer Rahmen
- Relevante Methodik
- Zentrale Ergebnisse
- Limitationen
- Forschungsimplikationen
- Emergende Trends übergreifend
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Du bist ein präziser wissenschaftlicher Analyst."},
            {"role": "user", "content": prompt + section}
        ],
        temperature=0.4,
        max_tokens=8500
    )

    return response.choices[0].message.content

# =========================
# CHUNKED TTS (FIXES 2000 TOKEN LIMIT)
# =========================

def generate_audio(script):
    os.makedirs(EPISODES_DIR, exist_ok=True)
    filename = f"episode_{datetime.utcnow().strftime('%Y%m%d')}.mp3"
    final_path = os.path.join(EPISODES_DIR, filename)

    # Split into safe chunks
    max_words_per_chunk = 900
    words = script.split()
    chunks = [
        " ".join(words[i:i + max_words_per_chunk])
        for i in range(0, len(words), max_words_per_chunk)
    ]

    audio_segments = []

    for chunk in chunks:
        speech = client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice="alloy",
            input=chunk,
        )

        temp_path = os.path.join(EPISODES_DIR, "temp_chunk.mp3")
        with open(temp_path, "wb") as f:
            f.write(speech.content)

        segment = AudioSegment.from_mp3(temp_path)
        audio_segments.append(segment)

    spoken = sum(audio_segments)

    intro = AudioSegment.from_mp3("intro_music.mp3")

    spoken = speed_up(normalize(spoken))
    intro = normalize(intro - 5)

    combined = intro.fade_out(1500) + spoken + intro.fade_in(1500)
    combined = normalize(combined)

    combined.export(final_path, format="mp3")

    return filename

# =========================
# RSS UPDATE (SAFE XML)
# =========================

def update_rss(filename):
    feed_path = "feed.xml"
    episode_url = f"{SITE_URL}/episodes/{filename}"
    pub_date = datetime.utcnow().strftime('%a, %d %b %Y %H:%M:%S GMT')

    rss = ET.Element("rss", version="2.0")
    channel = ET.SubElement(rss, "channel")

    ET.SubElement(channel, "title").text = "Private Research Podcast"
    ET.SubElement(channel, "link").text = SITE_URL
    ET.SubElement(channel, "description").text = "Automated Research Monitoring"

    old_items = []
    if os.path.exists(feed_path):
        tree = ET.parse(feed_path)
        old_channel = tree.getroot().find("channel")
        old_items = old_channel.findall("item")

    item = ET.SubElement(channel, "item")
    ET.SubElement(item, "title").text = f"Research Update {filename}"
    ET.SubElement(item, "pubDate").text = pub_date

    enclosure = ET.SubElement(item, "enclosure")
    enclosure.set("url", episode_url)
    enclosure.set("type", "audio/mpeg")

    for old in old_items[:MAX_FEED_ITEMS - 1]:
        channel.append(old)

    tree = ET.ElementTree(rss)
    tree.write(feed_path, encoding="utf-8", xml_declaration=True)

# =========================
# MAIN
# =========================

def main():
    papers = fetch_crossref()
    if not papers:
        print("No papers found.")
        return

    ranked = rank_papers(papers)
    script = generate_script(ranked)
    filename = generate_audio(script)
    update_rss(filename)

if __name__ == "__main__":
    main()
