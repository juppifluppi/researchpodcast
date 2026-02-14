import os
import requests
import re
from openai import OpenAI
import smtplib
from email.message import EmailMessage
from pydub import AudioSegment
from datetime import datetime, timedelta

# =========================
# CONFIG
# =========================

TOPICS = ["TOPIC_ONE", "TOPIC_TWO"]  # Add or remove topics
DAYS_BACK = 14
MAX_PAPERS_PER_TOPIC = 12
TOP_SELECTION_TOTAL = 6  # total papers discussed

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
POSTEO_PASS = os.getenv("POSTEO_PASS")
POSTEO_EMAIL = os.getenv("POSTEO_EMAIL")

client = OpenAI(api_key=OPENAI_API_KEY)

# =========================
# FETCH FROM CROSSREF (14 DAYS)
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
            abstract = item.get("abstract", "")
            doi = item.get("DOI", "")
            link = f"https://doi.org/{doi}" if doi else ""

            papers.append({
                "topic": topic,
                "title": title,
                "summary": abstract,
                "doi": doi,
                "link": link
            })

    return papers


# =========================
# RANK PAPERS ROBUSTLY
# =========================

def rank_papers(papers):
    text = ""
    for i, p in enumerate(papers):
        text += f"""
Paper {i+1}
Topic: {p['topic']}
Title: {p['title']}
Abstract: {p['summary']}
"""

    prompt = f"""
Select the {TOP_SELECTION_TOTAL} most important papers.

Return ONLY numbers separated by commas.
Example: 2,5,7,9

Prioritize:
- theoretical contribution
- methodological innovation
- conceptual novelty
- strong conclusions
- forward-looking implications
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a strict academic evaluator."},
            {"role": "user", "content": prompt + text}
        ],
        temperature=0.2,
    )

    content = response.choices[0].message.content
    numbers = re.findall(r"\d+", content)
    indices = [int(n) - 1 for n in numbers]

    selected = []
    for i in indices:
        if 0 <= i < len(papers):
            selected.append(papers[i])

    return selected[:TOP_SELECTION_TOTAL]


# =========================
# GENERATE 10K WORD SCRIPT
# =========================

def generate_script(selected_papers):
    section = ""
    for p in selected_papers:
        section += f"""
Topic: {p['topic']}
Title (Original English): {p['title']}
Abstract: {p['summary']}
DOI: {p['doi']}
Link: {p['link']}
"""

    prompt = f"""
Erstelle ein ca. 10000 Wörter langes, hochgradig technisches Forschungspodcast-Skript.

Sprache: Deutsch.
Zielgruppe: Forschende mit Fachkenntnis.
Keine Vereinfachungen.
Keine Einleitung wie „Willkommen“.
Keine Meta-Kommentare.
Keine Erwähnung von Musik.

Anforderungen:

- Direkter Einstieg in inhaltliche Analyse.
- Jedes Paper bekommt eine klar abgegrenzte Sektion.
- Detaillierte Diskussion:
    • Theoretischer Rahmen
    • Methodik (nur wenn konzeptionell relevant)
    • Zentrale Ergebnisse
    • Kritische Bewertung
    • Limitationen
    • Konkrete Forschungsimplikationen
- Präzise Terminologie.
- Keine populärwissenschaftliche Sprache.
- Abschließende Synthese übergreifender Trends.

Diskutiere explizit jedes einzelne Paper.

Papers:
{section}
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Du bist ein präziser wissenschaftlicher Analyst."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.4,
        max_tokens=14000
    )

    return response.choices[0].message.content


# =========================
# AUDIO PROCESSING
# =========================

def normalize(audio):
    return audio.apply_gain(-20.0 - audio.dBFS)


def speed_up(audio, speed=1.18):
    return audio._spawn(
        audio.raw_data,
        overrides={"frame_rate": int(audio.frame_rate * speed)}
    ).set_frame_rate(audio.frame_rate)


def generate_audio(script):
    speech = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=script,
    )

    with open("spoken.mp3", "wb") as f:
        f.write(speech.content)

    intro = AudioSegment.from_mp3("intro_music.mp3")
    spoken = AudioSegment.from_mp3("spoken.mp3")

    spoken = normalize(spoken)
    spoken = speed_up(spoken)

    intro = normalize(intro - 5)
    outro = intro

    combined = intro.fade_out(1500) + spoken + outro.fade_in(1500)
    combined = normalize(combined)

    combined.export("podcast.mp3", format="mp3")
    return "podcast.mp3"


# =========================
# EMAIL
# =========================

def send_email(script, audio_path, selected_papers):
    references = "\n\n==============================\n"
    references += "REFERENZEN DER BESPROCHENEN PAPER\n"
    references += "==============================\n\n"

    for i, p in enumerate(selected_papers, 1):
        references += f"{i}. {p['title']}\n"
        if p.get("doi"):
            references += f"   DOI: {p['doi']}\n"
        if p.get("link"):
            references += f"   Link: {p['link']}\n"
        references += "\n"

    full_text = script + references

    msg = EmailMessage()
    msg["Subject"] = "Weekly Research Podcast"
    msg["From"] = POSTEO_EMAIL
    msg["To"] = POSTEO_EMAIL
    msg.set_content("Der neue Forschungspodcast ist angehängt.")

    msg.add_attachment(full_text, subtype="plain", filename="podcast.txt")

    with open(audio_path, "rb") as f:
        msg.add_attachment(
            f.read(),
            maintype="audio",
            subtype="mpeg",
            filename="podcast.mp3"
        )

    with smtplib.SMTP_SSL("posteo.de", 465) as smtp:
        smtp.login(POSTEO_EMAIL, POSTEO_PASS)
        smtp.send_message(msg)


# =========================
# MAIN
# =========================

def main():
    papers = fetch_crossref()
    if not papers:
        print("No papers found.")
        return

    ranked = rank_papers(papers)
    if not ranked:
        print("Ranking failed.")
        return

    script = generate_script(ranked)
    audio = generate_audio(script)
    send_email(script, audio, ranked)


if __name__ == "__main__":
    main()
