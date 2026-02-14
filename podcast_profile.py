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

TARGET_AUTHORS = [
    "Lorenz Meinel",
    "Tessa Lühmann",
    "Josef Kehrein",
    "Robert Luxenhofer"
]

DAYS_BACK = 7
TOP_SELECTION_TOTAL = 10
MAX_FEED_ITEMS = 20

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
# AUTHOR PROFILE
# =========================

def build_author_profile():

    abstracts = []

    for author in TARGET_AUTHORS:
        url = "https://api.crossref.org/works?query.author=" + author + "&rows=40"
        response = requests.get(url)
        data = response.json()

        for item in data.get("message", {}).get("items", []):
            abstract = strip_html(item.get("abstract", ""))
            if abstract:
                abstracts.append(abstract)

    combined = "\n".join(abstracts[:30])

    prompt = (
        "Analyze these abstracts and summarize the core research themes, "
        "technologies, applications, and scientific philosophy of the authors:\n\n"
        + combined
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You extract research identity profiles."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=800
    )

    return response.choices[0].message.content

# =========================
# FETCH RECENT PAPERS
# =========================

def fetch_recent_papers():

    start_date = (datetime.utcnow() - timedelta(days=DAYS_BACK)).strftime("%Y-%m-%d")
    url = "https://api.crossref.org/works?filter=from-pub-date:" + start_date + "&rows=120"

    response = requests.get(url)
    data = response.json()

    papers = []

    for item in data.get("message", {}).get("items", []):
        abstract = strip_html(item.get("abstract", ""))
        if not abstract:
            continue

        papers.append({
            "title": item.get("title", ["No title"])[0],
            "summary": abstract,
            "doi": item.get("DOI", ""),
            "journal": item.get("container-title", ["Unknown Journal"])[0],
            "citations": item.get("is-referenced-by-count", 0),
        })

    return papers

# =========================
# ALIGNMENT RANKING
# =========================

def rank_by_author_alignment(papers, author_profile):

    text = ""
    for i, p in enumerate(papers):
        text += (
            "\nPaper " + str(i+1) + "\n"
            "Title: " + p["title"] + "\n"
            "Abstract: " + p["summary"][:1200] + "\n"
        )

    prompt = (
        "Given this research profile:\n\n"
        + author_profile +
        "\n\nSelect the "
        + str(TOP_SELECTION_TOTAL) +
        " papers most aligned with this profile. "
        "Return ONLY numbers separated by commas.\n\n"
        + text
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert research evaluator."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=300
    )

    numbers = re.findall(r"\d+", response.choices[0].message.content)
    indices = [int(n) - 1 for n in numbers]

    selected = []
    for i in indices:
        if 0 <= i < len(papers):
            selected.append(papers[i])

    return selected[:TOP_SELECTION_TOTAL]

# =========================
# SCRIPT GENERATION
# =========================

def generate_script(selected_papers):

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

    section = ""
    for p in selected_papers:
        section += "\n### PAPER_START: " + p["title"] + "\n"

    prompt = (
        "Create a dynamic scientific podcast dialogue.\n\n"
        "Moderator is " + moderator_style + ".\n"
        "Author is " + author_style + ".\n\n"
        "Rules:\n"
        "- Discuss ONLY the listed papers.\n"
        "- No cross-paper commentary.\n"
        "- No statistical deep dives.\n"
        "- Avoid mentioning specific tests or p-values.\n\n"
        "Include:\n"
        "- Controlled disagreement moments\n"
        "- Occasional analogies\n"
        "- Subtle emotional nuance\n"
        "- Probing questions from moderator\n\n"
        "Format strictly:\n\n"
        "### PAPER_START: <Title>\n\n"
        "MODERATOR:\nText\n\n"
        "AUTHOR:\nText\n"
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Generate natural, nuanced scientific dialogue."},
            {"role": "user", "content": prompt + section}
        ],
        temperature=0.9,
        max_tokens=9000
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
        "Generate a short episode title (max 6 words) and a 3-4 sentence summary.\n\n"
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

    raw_title = lines[0]
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

    title = "Ep. " + str(episode_number).zfill(2) + " – " + raw_title
    description = "<p>" + summary + "</p>" + show_notes

    return title, description

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
            input=chunk
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

    print("Building author research profile...")
    author_profile = build_author_profile()

    print("Fetching recent papers...")
    recent_papers = fetch_recent_papers()

    print("Ranking papers by alignment...")
    selected_papers = rank_by_author_alignment(recent_papers, author_profile)

    if not selected_papers:
        print("No aligned papers found.")
        return

    script = generate_script(selected_papers)
    filename, duration = generate_audio(script)

    if os.path.exists("feed.xml"):
        tree = ET.parse("feed.xml")
        channel = tree.getroot().find("channel")
        episode_number = len(channel.findall("item")) + 1
    else:
        episode_number = 1

    title, description = generate_episode_metadata(selected_papers, episode_number)
    update_rss(filename, duration, title, description, episode_number)

    print("Episode generated:", title)


if __name__ == "__main__":
    main()
