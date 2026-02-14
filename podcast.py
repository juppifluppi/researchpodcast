import os
import requests
import re
import html
import json
import random
from openai import OpenAI
from pydub import AudioSegment
from pydub.effects import compress_dynamic_range, high_pass_filter, low_pass_filter
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

def speed_adjust(audio, speed=1.02):
    return audio._spawn(
        audio.raw_data,
        overrides={"frame_rate": int(audio.frame_rate * speed)}
    ).set_frame_rate(audio.frame_rate)

def random_pause():
    return AudioSegment.silent(duration=random.randint(300, 700))

def random_long_pause():
    return AudioSegment.silent(duration=random.randint(2000, 3200))

def apply_eq_moderator(audio):
    audio = high_pass_filter(audio, 120)
    audio = low_pass_filter(audio, 8500)
    return audio + 1

def apply_eq_author(audio):
    audio = high_pass_filter(audio, 80)
    audio = low_pass_filter(audio, 6500)
    return audio - 1

def add_subtle_reverb(audio, delay_ms=70):
    echo = audio - 12
    echo = AudioSegment.silent(delay_ms) + echo
    return audio.overlay(echo)

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
Consider impact, novelty, forward implications, and citations.
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
# SCRIPT GENERATION
# =========================

def generate_script(selected_papers):

    personality_modes = [
        "leicht skeptisch und analytisch",
        "neugierig und forschend",
        "methodologisch kritisch",
        "strategisch denkend mit Blick auf Trends",
        "konzeptionell theorieorientiert"
    ]

    moderator_personality = random.choice(personality_modes)

    section = ""
    for p in selected_papers:
        section += f"\n### PAPER_START: {p['title']}\n"

    prompt = f"""
Erstelle ein ca. 6000 Wörter langes wissenschaftliches Podcastgespräch im Dialogformat.

Sprache: Deutsch.
Natürlich klingend.
Moderator-Persönlichkeit: {moderator_personality}

Format:

MODERATOR:
<Text>

AUTOR:
<Text>

Erlaubt:
- Mikro-Hesitationen
- gelegentliche Unterbrechungen
- leichte Spannung
- am Ende jedes Papers kritische Zusammenfassung

Jedes Paper beginnt exakt mit:
### PAPER_START: <Title>
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Du erzeugst ein lebendiges wissenschaftliches Gespräch."},
            {"role": "user", "content": prompt + section}
        ],
        temperature=0.85,
        max_tokens=8500
    )

    return response.choices[0].message.content

# =========================
# AUDIO BLOCK PROCESSING
# =========================

def process_block(speaker, text):
    segments = []

    if not text.strip():
        return segments

    if "moderator" in speaker:
        voice = "alloy"
        pan = -0.12
        eq_func = apply_eq_moderator
    else:
        voice = "verse"
        pan = 0.12
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
        segment = add_subtle_reverb(segment)

        segments.append(segment)
        segments.append(random_pause())

    return segments

# =========================
# AUDIO GENERATION
# =========================

def generate_audio(script):
    os.makedirs(EPISODES_DIR, exist_ok=True)
    filename = f"episode_{datetime.utcnow().strftime('%Y%m%d')}.mp3"
    final_path = os.path.join(EPISODES_DIR, filename)

    segments = []
    current_speaker = None
    buffer = ""

    lines = script.split("\n")

    for line in lines:
        line = line.strip()

        if line.startswith("### PAPER_START"):
            if buffer:
                segments.extend(process_block(current_speaker, buffer))
                buffer = ""
            segments.append(random_long_pause())
            continue

        speaker_match = re.match(r"^(MODERATOR|Moderator|AUTOR|Autor)\s*[:\-]?$", line)

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

    if len(spoken) < 5000:
        raise ValueError("No speech generated. Dialogue parsing failed.")

    spoken = speed_adjust(normalize(spoken), speed=1.02)

    if os.path.exists("room_tone.mp3"):
        room = AudioSegment.from_mp3("room_tone.mp3") - 32
        if len(room) < len(spoken):
            loops = len(spoken) // len(room) + 1
            room = room * loops
        spoken = spoken.overlay(room[:len(spoken)])

    spoken = compress_dynamic_range(
        spoken,
        threshold=-22.0,
        ratio=2.3,
        attack=5,
        release=60
    )

    intro = AudioSegment.from_mp3("intro_music.mp3")
    intro = normalize(intro).fade_out(2500)

    final_audio = intro + spoken
    final_audio = normalize(final_audio)

    final_audio.export(final_path, format="mp3")

    duration_seconds = int(len(final_audio) / 1000)
    return filename, duration_seconds

# =========================
# MAIN
# =========================

def main():
    papers = fetch_crossref()
    ranked = rank_papers(papers)
    script = generate_script(ranked)
    filename, duration = generate_audio(script)
    print("Episode generated:", filename, "Duration:", duration)

if __name__ == "__main__":
    main()
