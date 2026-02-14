def generate_audio(script):
    os.makedirs(EPISODES_DIR, exist_ok=True)
    filename = f"episode_{datetime.utcnow().strftime('%Y%m%d')}.mp3"
    final_path = os.path.join(EPISODES_DIR, filename)

    lines = script.split("\n")
    segments = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith("### PAPER_START"):
            segments.append(random_long_pause())
            continue

        if line.startswith("MODERATOR:"):
            text = line.replace("MODERATOR:", "").strip()
            voice = "alloy"
            pan_value = -0.12
            eq = lambda a: low_pass_filter(high_pass_filter(a, 120), 8500) + 1
        elif line.startswith("AUTOR:"):
            text = line.replace("AUTOR:", "").strip()
            voice = "verse"
            pan_value = 0.12
            eq = lambda a: low_pass_filter(high_pass_filter(a, 80), 6500) - 1
        else:
            continue

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

            # EQ difference
            segment = eq(segment)

            # Stereo separation
            segment = segment.pan(pan_value)

            # Slight reverb difference
            segment = add_subtle_reverb(segment, delay_ms=70 if voice=="alloy" else 90)

            segments.append(segment)
            segments.append(random_pause())

    spoken = sum(segments)

    # Slightly slower pace
    spoken = speed_adjust(normalize(spoken), speed=1.02)

    # Room tone
    room = AudioSegment.from_mp3("room_tone.mp3") - 32
    if len(room) < len(spoken):
        loops = len(spoken) // len(room) + 1
        room = room * loops
    room = room[:len(spoken)]
    spoken = spoken.overlay(room)

    # Compression
    spoken = compress_dynamic_range(
        spoken,
        threshold=-22.0,
        ratio=2.3,
        attack=5,
        release=60
    )

    # Adaptive intro ducking
    intro = AudioSegment.from_mp3("intro_music.mp3")
    intro = normalize(intro)

    ducked_intro = intro.fade_out(2500)
    combined = ducked_intro.overlay(spoken[:3000], position=len(intro)-3000)

    full_audio = intro + spoken

    full_audio = normalize(full_audio)
    full_audio.export(final_path, format="mp3")

    duration_seconds = int(len(full_audio) / 1000)
    return filename, duration_seconds
