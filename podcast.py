import os
import requests
import re
import html
import json
from openai import OpenAI
from pydub import AudioSegment
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET

# =========================
# CONFIG
# =========================

TOPICS = ["lipid nanoparticle", "bioconjugation"]
DAYS_BACK = 14
MAX_PAPERS_PER_TOPIC = 12
TOP_SELECTION_TOTAL = 6
MAX_FEED_ITEMS = 20

BASE_URL = "https://juppifluppi.github.io/researchpodcast"
EPISODES_DIR = "episodes"

PODCAST_TITLE = "Research Updates"
PODCAST_AUTHOR = "Jupp"
PODCAST_DESCRIPTION = "AI deep-dive into recent publications."
PODCAST_LANGUAGE = "de-de"
PODCAST_CATEGORY = "Science"

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

def speed_up(audio, speed=1.18):
    return audio._spawn(
        audio.raw_data,
        overrides={"frame_rate": int(audio.frame_rate * speed)}
    ).set_frame_rate(audio.frame_rate)

# =========================
# FETCH PAPERS
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
            papers.append({
                "topic": topic,
                "title": item.get("title", ["No title"])[0],
                "summary": strip_html(item.get("abstract", "")),
                "doi": item.get("DOI", ""),
                "journal": item.get("container-title", ["Unknown Journal"])[0],
                "citations": item.get("is-referenced-by-count", 0),
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
Title: {p['title']}
Journal: {p['journal']}
Citations: {p['citations']}
Abstract: {p['summary']}
"""

    prompt = f"""
Select the {TOP_SELECTION_TOTAL} most important papers.
Return ONLY numbers separated by commas.
Consider theoretical impact, novelty, forward implications, and citations.
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

    return [papers[i] for i in indices if 0 <= i < len(papers)][:TOP_SELECTION_TOTAL]

# =========================
# SCRIPT GENERATION (~6000 WORDS)
# =========================

def generate_script(selected_papers):
    section = ""
    for p in selected_papers:
        section += f"\n### PAPER_START: {p['title']}\n"

    prompt = f"""
Erstelle ein ca. 6000 Wörter langes technisches Forschungspodcast-Skript.

Sprache: Deutsch.
Keine Einleitungen.
Keine Musik-Erwähnung.

Jedes Paper beginnt exakt mit:
### PAPER_START: <Title>

Diskutiere:
- Theorie
- Methodik
- Ergebnisse
- Limitationen
- Implikationen
- Übergreifende Trends
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Du bist ein wissenschaftlicher Analyst."},
            {"role": "user", "content": prompt + section}
        ],
        temperature=0.4,
        max_tokens=8500
    )

    return response.choices[0].message.content

# =========================
# SUMMARY
# =========================

def generate_summary(script):
    prompt = "Erstelle eine prägnante wissenschaftliche Zusammenfassung (5–7 Sätze)."

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Du bist wissenschaftlicher Redakteur."},
            {"role": "user", "content": prompt + script[:3000]}
        ],
        temperature=0.3,
        max_tokens=400
    )

    return response.choices[0].message.content

# =========================
# AUDIO GENERATION (CHUNKED)
# =========================

def generate_audio(script):
    os.makedirs(EPISODES_DIR, exist_ok=True)
    filename = f"episode_{datetime.utcnow().strftime('%Y%m%d')}.mp3"
    final_path = os.path.join(EPISODES_DIR, filename)

    words = script.split()
    chunks = [" ".join(words[i:i+900]) for i in range(0, len(words), 900)]

    segments = []
    for chunk in chunks:
        speech = client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice="alloy",
            input=chunk,
        )
        temp = os.path.join(EPISODES_DIR, "temp.mp3")
        with open(temp, "wb") as f:
            f.write(speech.content)
        segments.append(AudioSegment.from_mp3(temp))

    spoken = sum(segments)
    intro = AudioSegment.from_mp3("intro_music.mp3")

    spoken = speed_up(normalize(spoken))
    intro = normalize(intro - 5)

    combined = intro.fade_out(1500) + spoken + intro.fade_in(1500)
    combined = normalize(combined)
    combined.export(final_path, format="mp3")

    duration_seconds = int(len(combined) / 1000)
    return filename, duration_seconds

# =========================
# CHAPTER MARKERS
# =========================

def generate_chapters(script, filename):
    matches = list(re.finditer(r"### PAPER_START: (.+)", script))
    words = script.split()
    total_words = len(words)
    words_per_minute = 177
    total_seconds = (total_words / words_per_minute) * 60

    chapters = []
    for match in matches:
        title = match.group(1)
        start_words = len(script[:match.start()].split())
        start_seconds = int((start_words / total_words) * total_seconds)
        chapters.append({"startTime": start_seconds, "title": title})

    chapter_file = filename.replace(".mp3", ".json")
    chapter_path = os.path.join(EPISODES_DIR, chapter_file)

    with open(chapter_path, "w") as f:
        json.dump({"version": "1.2.0", "chapters": chapters}, f, indent=2)

    return chapter_file

# =========================
# RSS UPDATE
# =========================

def update_rss(filename, duration_seconds, summary, chapter_file):
    feed_path = "feed.xml"
    episode_url = f"{BASE_URL}/episodes/{filename}"
    chapter_url = f"{BASE_URL}/episodes/{chapter_file}"
    cover_url = f"{BASE_URL}/cover.png"

    pub_date = datetime.utcnow().strftime('%a, %d %b %Y %H:%M:%S GMT')

    rss = ET.Element("rss", version="2.0")
    rss.set("xmlns:itunes", "http://www.itunes.com/dtds/podcast-1.0.dtd")
    rss.set("xmlns:podcast", "https://podcastindex.org/namespace/1.0")

    channel = ET.SubElement(rss, "channel")
    ET.SubElement(channel, "title").text = PODCAST_TITLE
    ET.SubElement(channel, "link").text = BASE_URL
    ET.SubElement(channel, "description").text = PODCAST_DESCRIPTION
    ET.SubElement(channel, "language").text = PODCAST_LANGUAGE
    ET.SubElement(channel, "itunes:author").text = PODCAST_AUTHOR
    ET.SubElement(channel, "itunes:explicit").text = "no"

    image = ET.SubElement(channel, "itunes:image")
    image.set("href", cover_url)
    ET.SubElement(channel, "itunes:category").set("text", PODCAST_CATEGORY)

    episode_number = 1
    if os.path.exists(feed_path):
        tree = ET.parse(feed_path)
        old_channel = tree.getroot().find("channel")
        items = old_channel.findall("item")
        episode_number = len(items) + 1
        for old in items[:MAX_FEED_ITEMS - 1]:
            channel.append(old)

    item = ET.SubElement(channel, "item")
    ET.SubElement(item, "title").text = f"Episode {episode_number}"
    ET.SubElement(item, "description").text = summary
    ET.SubElement(item, "itunes:duration").text = str(duration_seconds)
    ET.SubElement(item, "itunes:episode").text = str(episode_number)
    ET.SubElement(item, "pubDate").text = pub_date
    ET.SubElement(item, "guid").text = episode_url

    enclosure = ET.SubElement(item, "enclosure")
    enclosure.set("url", episode_url)
    enclosure.set("type", "audio/mpeg")

    ET.SubElement(item, "podcast:chapters").set("url", chapter_url)

    ET.ElementTree(rss).write(feed_path, encoding="utf-8", xml_declaration=True)

# =========================
# MAIN
# =========================

def main():
    papers = fetch_crossref()
    ranked = rank_papers(papers)
    script = generate_script(ranked)
    summary = generate_summary(script)
    filename, duration = generate_audio(script)
    chapter_file = generate_chapters(script, filename)
    update_rss(filename, duration, summary, chapter_file)

if __name__ == "__main__":
    main()
