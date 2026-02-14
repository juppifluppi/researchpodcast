import os
import requests
from openai import OpenAI
import smtplib
from email.message import EmailMessage
from pydub import AudioSegment
from datetime import datetime, timedelta

# =========================
# CONFIG
# =========================

TOPIC = "YOUR_TOPIC_HERE"
MAX_PAPERS = 15
TOP_SELECTION = 4

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
POSTEO_PASS = os.getenv("POSTEO_PASS")
POSTEO_EMAIL = os.getenv("POSTEO_EMAIL")

client = OpenAI(api_key=OPENAI_API_KEY)

# =========================
# FETCH PAPERS (CROSSREF)
# =========================

def fetch_crossref():
    one_week_ago = (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d")

    url = (
        "https://api.crossref.org/works?"
        f"query={TOPIC}"
        f"&filter=from-pub-date:{one_week_ago}"
        f"&rows={MAX_PAPERS}"
        "&sort=published"
        "&order=desc"
    )

    response = requests.get(url)
    data = response.json()

    papers = []

    for item in data["message"]["items"]:
        title = item.get("title", ["No title"])[0]
        abstract = item.get("abstract", "No abstract available.")
        doi = item.get("DOI", "")
        link = f"https://doi.org/{doi}" if doi else ""

        papers.append({
            "title": title,
            "summary": abstract,
            "link": link
        })

    return papers


# =========================
# RANK PAPERS
# =========================

def rank_papers(papers):
    paper_text = ""
    for i, p in enumerate(papers):
        paper_text += f"\nPaper {i+1}:\nTitle: {p['title']}\nAbstract: {p['summary']}\n"

    prompt = f"""
Select the {TOP_SELECTION} most conceptually important papers in {TOPIC}.

Prioritize:
- strong conclusions
- forward-looking implications
- conceptual impact
- broad relevance

Return only the numbers.
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a strict academic evaluator."},
            {"role": "user", "content": prompt + paper_text}
        ],
        temperature=0.3,
    )

    ranked_numbers = response.choices[0].message.content
    indices = [int(x) - 1 for x in ranked_numbers.split() if x.isdigit()]

    return [papers[i] for i in indices[:TOP_SELECTION]]


# =========================
# GENERATE GERMAN SCRIPT (~20 min)
# =========================

def generate_script(selected_papers):
    paper_section = ""
    for p in selected_papers:
        paper_section += f"\nTitel (Original): {p['title']}\nAbstract: {p['summary']}\nLink: {p['link']}\n"

    prompt = f"""
Erstelle ein ca. 20-minütiges Forschungspodcast-Skript (~2800 Wörter).

Sprache: Deutsch
Titel bleiben im englischen Original.

Fokus:
- Zentrale Ergebnisse
- Schlussfolgerungen
- Bedeutung für das Feld
- Zukunftsausblick

Methodische Details nur sehr knapp.

Struktur:
1. Einstieg nach Musik
2. Wochenüberblick
3. Analyse der Papers
4. Gemeinsame Trends
5. Abschließender Ausblick

Thema: {TOPIC}

Papers:
{paper_section}
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Du bist ein analytischer Wissenschaftsjournalist."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
    )

    return response.choices[0].message.content


# =========================
# AUDIO MASTERING
# =========================

def normalize_audio(audio):
    return audio.apply_gain(-20.0 - audio.dBFS)


def generate_audio(script):
    # Generate speech
    speech = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=script,
    )

    spoken_path = "spoken.mp3"
    with open(spoken_path, "wb") as f:
        f.write(speech.content)

    intro = AudioSegment.from_mp3("intro_music.mp3")
    spoken = AudioSegment.from_mp3(spoken_path)

    # Normalize spoken voice
    spoken = normalize_audio(spoken)

    # Normalize intro music slightly quieter
    intro = normalize_audio(intro - 5)

    intro = intro.fade_out(2000)
    outro = intro.fade_in(2000)

    combined = intro + spoken + outro

    # Final normalization
    combined = normalize_audio(combined)

    final_path = "podcast.mp3"
    combined.export(final_path, format="mp3")

    return final_path


# =========================
# SEND EMAIL
# =========================

def send_email(script, audio_path):
    msg = EmailMessage()
    msg["Subject"] = f"Weekly Research Podcast – {TOPIC}"
    msg["From"] = POSTEO_EMAIL
    msg["To"] = POSTEO_EMAIL
    msg.set_content("Dein wöchentlicher Forschungspodcast ist angehängt.")

    msg.add_attachment(script, subtype="plain", filename="podcast.txt")

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
    ranked = rank_papers(papers)
    script = generate_script(ranked)
    audio = generate_audio(script)
    send_email(script, audio)


if __name__ == "__main__":
    main()
