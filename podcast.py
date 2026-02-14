import os
import requests
import re
from openai import OpenAI
from pydub import AudioSegment
from datetime import datetime, timedelta

# =========================
# CONFIG
# =========================

TOPICS = ["TOPIC_ONE", "TOPIC_TWO"]  # customize
DAYS_BACK = 14
MAX_PAPERS_PER_TOPIC = 12
TOP_SELECTION_TOTAL = 6

SITE_URL = "https://juppifluppi.github.io"
EPISODES_DIR = "episodes"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

# =========================
# FETCH CROSSREF
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
- citation count (as signal, not sole criterion)
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
Topic: {p['topic']}
Title (Original English): {p['title']}
Journal: {p['journal']}
Citations: {p['citations']}
Abstract: {p['summary']}
DOI: {p['doi']}
"""

    prompt = f"""
Erstelle ein ca. 6000 Wörter langes technisches Forschungspodcast-Skript.

Sprache: Deutsch.
Zielgruppe: Forschende mit Fachkenntnis.
Keine Einleitungen.
Keine Musik-Erwähnung.
Keine Vereinfachungen.

Struktur:

- Direkter Einstieg in analytische Diskussion.
- Jedes Paper erhält eine klar abgegrenzte Sektion.
- Pro Paper:
    • Theoretischer Rahmen
    • Konzeptionell relevante Methodik
    • Zentrale Ergebnisse
    • Kritische Einordnung
    • Limitationen
    • Konkrete Forschungsimplikationen
- Abschließend:
    • Synthese übergreifender Trends
    • Identifikation emergenter Forschungscluster
    • Bewertung möglicher Paradigmenverschiebungen

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
        max_tokens=9000
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

    os.makedirs(EPISODES_DIR, exist_ok=True)

    filename = f"episode_{datetime.utcnow().strftime('%Y%m%d')}.mp3"
    spoken_path = os.path.join(EPISODES_DIR, "spoken.mp3")

    with open(spoken_path, "wb") as f:
        f.write(speech.content)

    intro = AudioSegment.from_mp3("intro_music.mp3")
    spoken = AudioSegment.from_mp3(spoken_path)

    spoken = speed_up(normalize(spoken))
    intro = normalize(intro - 5)
    outro = intro

    combined = intro.fade_out(1500) + spoken + outro.fade_in(1500)
    combined = normalize(combined)

    final_path = os.path.join(EPISODES_DIR, filename)
    combined.export(final_path, format="mp3")

    return filename

# =========================
# RSS UPDATE
# =========================

def update_rss(filename):
    feed_path = "feed.xml"
    episode_url = f"{SITE_URL}/episodes/{filename}"
    pub_date = datetime.utcnow().strftime('%a, %d %b %Y %H:%M:%S GMT')

    item = f"""
    <item>
        <title>Research Update {filename}</title>
        <enclosure url="{episode_url}" type="audio/mpeg"/>
        <pubDate>{pub_date}</pubDate>
    </item>
    """

    if os.path.exists(feed_path):
        with open(feed_path, "r") as f:
            content = f.read()
        content = content.replace("</channel>", item + "\n</channel>")
    else:
        content = f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
<channel>
<title>Private Research Podcast</title>
<link>{SITE_URL}</link>
<description>Automated Research Monitoring</description>
{item}
</channel>
</rss>
"""

    with open(feed_path, "w") as f:
        f.write(content)

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
