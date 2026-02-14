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

# =========================
# CONFIG
# =========================

TOPICS = ["lipid nanoparticle", "bioconjugation", "drug mucin bile interactions"]
TRACK_AUTHORS = ["Lorenz Meinel", "Tessa Lühmann", "Robert Luxenhofer", "Christel Bergström"]

DAYS_BACK = 7
MAX_PAPERS_PER_TOPIC = 12
MAX_PAPERS_PER_AUTHOR = 6
TOP_SELECTION_TOTAL = 6

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

def speed_adjust(audio, speed=1.06):  # faster again
    return audio._spawn(
        audio.raw_data,
        overrides={"frame_rate": int(audio.frame_rate * speed)}
    ).set_frame_rate(audio.frame_rate)

def random_pause():
    return AudioSegment.silent(duration=random.randint(250, 500))

def random_long_pause():
    return AudioSegment.silent(duration=random.randint(1800, 2600))

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

def fetch_crossref():

    start_date = (datetime.utcnow() - timedelta(days=DAYS_BACK)).strftime("%Y-%m-%d")
    papers = []

    # --- Topic-based search ---
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

    # --- Author-based search ---
    for author in TRACK_AUTHORS:
        url = (
            "https://api.crossref.org/works?"
            f"query.author={author}"
            f"&filter=from-pub-date:{start_date}"
            f"&rows={MAX_PAPERS_PER_AUTHOR}"
        )

        response = requests.get(url)
        data = response.json()

        for item in data.get("message", {}).get("items", []):
            papers.append(extract_paper(item))

    # remove duplicates by DOI
    unique = {}
    for p in papers:
        if p["doi"]:
            unique[p["doi"]] = p
    return list(unique.values())

def extract_paper(item):
    return {
        "title": item.get("title", ["No title"])[0],
        "summary": strip_html(item.get("abstract", "")),
        "doi": item.get("DOI", ""),
        "journal": item.get("container-title", ["Unknown Journal"])[0],
        "citations": item.get("is-referenced-by-count", 0),
    }

# =========================
# SCRIPT GENERATION
# =========================

def generate_script(selected_papers):

    section = ""
    for p in selected_papers:
        section += f"\n### PAPER_START: {p['title']}\n"

    prompt = f"""
Erstelle ein natürlich klingendes wissenschaftliches Podcastgespräch im Dialogformat.

Sprache: Deutsch.
Moderator kritisch, analytisch.
Autor reflektiert.

Format:

MODERATOR:
<Text>

AUTOR:
<Text>

Erlaubt:
- kurze Unterbrechungen
- Mikro-Hesitationen
- kritische Rückfragen
- am Ende jedes Papers kurze Zusammenfassung

Jedes Paper beginnt exakt mit:
### PAPER_START: <Title>
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Du erzeugst ein lebendiges wissenschaftliches Gespräch."},
            {"role": "user", "content": prompt + section}
        ],
        temperature=0.8,
        max_tokens=8000
    )

    return response.choices[0].message.content

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

    spoken = speed_adjust(normalize(spoken), speed=1.06)

    spoken = compress_dynamic_range(
        spoken,
        threshold=-20.0,
        ratio=2.0,
        attack=5,
        release=50
    )

    intro = AudioSegment.from_mp3("intro_music.mp3")
    intro = normalize(intro).fade_out(2000)

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
    script = generate_script(papers[:TOP_SELECTION_TOTAL])
    filename, duration = generate_audio(script)
    print("Episode generated:", filename, "Duration:", duration)

if __name__ == "__main__":
    main()
