import os
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from openai import OpenAI
import smtplib
from email.message import EmailMessage

# =========================
# CONFIG
# =========================

TOPIC = "Lipid nanoparticles"
MAX_PAPERS = 15
TOP_SELECTION = 5

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
POSTEO_PASS = os.getenv("POSTEO_PASS")
POSTEO_EMAIL = os.getenv("POSTEO_EMAIL")

client = OpenAI(api_key=OPENAI_API_KEY)


# =========================
# FETCH ARXIV PAPERS
# =========================

def fetch_arxiv():
    query = f"http://export.arxiv.org/api/query?search_query=all:{TOPIC}&start=0&max_results={MAX_PAPERS}&sortBy=submittedDate&sortOrder=descending"
    response = requests.get(query)
    root = ET.fromstring(response.content)

    namespace = {"atom": "http://www.w3.org/2005/Atom"}

    papers = []

    for entry in root.findall("atom:entry", namespace):
        title = entry.find("atom:title", namespace).text.strip()
        summary = entry.find("atom:summary", namespace).text.strip()
        link = entry.find("atom:id", namespace).text.strip()

        papers.append({
            "title": title,
            "summary": summary,
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
You are a research analyst.

Given the following new papers in {TOPIC}, rank the top {TOP_SELECTION}
based on:
- novelty
- likely impact
- methodological rigor
- broader implications

Return only the numbers of the top papers in order.
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a precise academic evaluator."},
            {"role": "user", "content": prompt + paper_text}
        ],
        temperature=0.3,
    )

    ranked_numbers = response.choices[0].message.content
    indices = [int(x) - 1 for x in ranked_numbers.split() if x.isdigit()]

    return [papers[i] for i in indices[:TOP_SELECTION]]


# =========================
# GENERATE PODCAST SCRIPT
# =========================

def generate_script(selected_papers):
    paper_section = ""
    for p in selected_papers:
        paper_section += f"\nTitle: {p['title']}\nAbstract: {p['summary']}\nLink: {p['link']}\n"

    prompt = f"""
Create a 30-minute research podcast script (~4500 words).

Topic: {TOPIC}

Structure:
- Intro overview of this week's research landscape
- Deep dive into each paper
- Connect themes across papers
- Critical evaluation
- Future outlook

Audience: expert researchers
Style: analytical, precise, intellectually engaging (like The Economist tech briefing)

Papers:
{paper_section}
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a high-level research podcast writer."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
    )

    return response.choices[0].message.content


# =========================
# TEXT TO SPEECH
# =========================

def generate_audio(script):
    speech = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=script,
    )

    audio_path = "podcast.mp3"
    with open(audio_path, "wb") as f:
        f.write(speech.content)

    return audio_path


# =========================
# SEND EMAIL (POSTEO)
# =========================

def send_email(script, audio_path):
    msg = EmailMessage()
    msg["Subject"] = f"Weekly Research Podcast – {TOPIC}"
    msg["From"] = POSTEO_EMAIL
    msg["To"] = POSTEO_EMAIL
    msg.set_content("Your weekly research podcast is attached.\n\nEnjoy.")

    # Attach script
    msg.add_attachment(
        script,
        subtype="plain",
        filename="podcast.txt"
    )

    # Attach audio
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
    papers = fetch_arxiv()
    ranked = rank_papers(papers)
    script = generate_script(ranked)
    audio = generate_audio(script)
    send_email(script, audio)


if __name__ == "__main__":
    main()
