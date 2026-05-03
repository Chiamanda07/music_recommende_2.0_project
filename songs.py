"""
Song database for the RAG music recommender.

Each entry is a dictionary with:
  - title:            song name
  - artist:           performing artist or band
  - genre:            broad genre category
  - mood_description: natural-language description of the emotional vibe,
                      used to compute embeddings for retrieval
"""

SONGS = [
    # ── Happy / Joyful ────────────────────────────────────────────────────────
    {
        "title": "Happy",
        "artist": "Pharrell Williams",
        "genre": "Pop",
        "mood_description": "Pure feel-good energy and infectious happiness that makes you want to get up and dance.",
    },
    {
        "title": "Can't Stop the Feeling!",
        "artist": "Justin Timberlake",
        "genre": "Pop",
        "mood_description": "Uplifting and carefree, overflowing with joy and the urge to move your body.",
    },
    {
        "title": "Good as Hell",
        "artist": "Lizzo",
        "genre": "Pop/R&B",
        "mood_description": "Empowering self-love and joy, reminding you to feel amazing about who you are.",
    },
    {
        "title": "Walking on Sunshine",
        "artist": "Katrina and the Waves",
        "genre": "Pop/Rock",
        "mood_description": "Bright, beaming happiness — a classic feel-good anthem full of radiant energy.",
    },
    {
        "title": "Levitating",
        "artist": "Dua Lipa",
        "genre": "Pop/Dance",
        "mood_description": "Fun and carefree, like floating on air with excitement and playful confidence.",
    },
    {
        "title": "Here Comes the Sun",
        "artist": "The Beatles",
        "genre": "Classic Rock",
        "mood_description": "Gentle optimism and relief after a hard time, warmth slowly returning to your life.",
    },
    {
        "title": "Dog Days Are Over",
        "artist": "Florence + The Machine",
        "genre": "Indie Rock",
        "mood_description": "Euphoric release and the rush of running toward happiness after leaving darkness behind.",
    },
    {
        "title": "Beautiful Day",
        "artist": "U2",
        "genre": "Rock",
        "mood_description": "Gratitude and wonder at the world around you, bursting with hopeful, open-hearted optimism.",
    },

    # ── Sad / Melancholy ──────────────────────────────────────────────────────
    {
        "title": "Someone Like You",
        "artist": "Adele",
        "genre": "Pop/Soul",
        "mood_description": "Deep heartbreak and longing, mourning the end of a relationship and wishing someone well from afar.",
    },
    {
        "title": "The Night We Met",
        "artist": "Lord Huron",
        "genre": "Indie Folk",
        "mood_description": "Hauntingly sad and wistful, desperately wishing you could return to a single perfect moment.",
    },
    {
        "title": "Driver's License",
        "artist": "Olivia Rodrigo",
        "genre": "Pop",
        "mood_description": "Raw teenage heartbreak and grief, crying alone while memories of someone flood your mind.",
    },
    {
        "title": "Skinny Love",
        "artist": "Bon Iver",
        "genre": "Indie Folk",
        "mood_description": "Fragile sadness and emotional exhaustion, watching a love quietly fall apart.",
    },
    {
        "title": "Fix You",
        "artist": "Coldplay",
        "genre": "Alternative Rock",
        "mood_description": "Tender sadness paired with a glimmer of hope, the ache of wanting to heal someone you love.",
    },
    {
        "title": "Mad World",
        "artist": "Gary Jules",
        "genre": "Indie/Alternative",
        "mood_description": "Quiet despair and feeling completely disconnected from the world around you.",
    },
    {
        "title": "Liability",
        "artist": "Lorde",
        "genre": "Indie Pop",
        "mood_description": "Feeling like too much for others to handle, sitting alone with the sting of rejection.",
    },

    # ── Heartbroken ───────────────────────────────────────────────────────────
    {
        "title": "Happier",
        "artist": "Olivia Rodrigo",
        "genre": "Pop",
        "mood_description": "Painful acceptance that the person you love is better off without you, bittersweet and heavy.",
    },
    {
        "title": "All I Want",
        "artist": "Kodaline",
        "genre": "Indie/Alternative",
        "mood_description": "Deep longing and quiet grief over a love that is lost and irreplaceable.",
    },
    {
        "title": "I Can't Make You Love Me",
        "artist": "Bonnie Raitt",
        "genre": "Pop/R&B",
        "mood_description": "Graceful heartbreak and the aching acceptance that love cannot be forced.",
    },
    {
        "title": "Slow Dancing in a Burning Room",
        "artist": "John Mayer",
        "genre": "Pop/Blues",
        "mood_description": "Watching a relationship collapse in slow motion, full of regret and sorrow.",
    },
    {
        "title": "Breakeven",
        "artist": "The Script",
        "genre": "Pop/Rock",
        "mood_description": "The unfair pain of a breakup where one person is fine and the other is completely shattered.",
    },

    # ── Nostalgic / Reflective ────────────────────────────────────────────────
    {
        "title": "1979",
        "artist": "The Smashing Pumpkins",
        "genre": "Alternative Rock",
        "mood_description": "Wistful haze of youth and teenage memories, longing for a carefree time that is gone.",
    },
    {
        "title": "Summer of '69",
        "artist": "Bryan Adams",
        "genre": "Rock",
        "mood_description": "Nostalgic longing for the greatest days of your youth and the people you shared them with.",
    },
    {
        "title": "Yesterday",
        "artist": "The Beatles",
        "genre": "Classic Rock",
        "mood_description": "Gentle nostalgia and regret, wishing you could undo the past and return to better days.",
    },
    {
        "title": "Vienna",
        "artist": "Billy Joel",
        "genre": "Pop/Rock",
        "mood_description": "Wise and reflective, urging you to slow down and appreciate where you are in life.",
    },
    {
        "title": "The Scientist",
        "artist": "Coldplay",
        "genre": "Alternative Rock",
        "mood_description": "Bittersweet reflection on a relationship, wishing you could go back and do things differently.",
    },

    # ── Calm / Relaxed ────────────────────────────────────────────────────────
    {
        "title": "Weightless",
        "artist": "Marconi Union",
        "genre": "Ambient",
        "mood_description": "Deeply calming and meditative, scientifically designed to reduce anxiety and still the mind.",
    },
    {
        "title": "Clair de Lune",
        "artist": "Claude Debussy",
        "genre": "Classical",
        "mood_description": "Peaceful and dreamy, like moonlight on still water — serene and beautifully quiet.",
    },
    {
        "title": "Sunday Morning",
        "artist": "Maroon 5",
        "genre": "Pop/Soul",
        "mood_description": "Lazy, comfortable, and warm — the perfect slow morning with nowhere to be.",
    },
    {
        "title": "Banana Pancakes",
        "artist": "Jack Johnson",
        "genre": "Acoustic/Folk",
        "mood_description": "Carefree and cozy, wanting to stay in bed all day and pretend the world can wait.",
    },
    {
        "title": "Midnight City",
        "artist": "M83",
        "genre": "Indie Electronic",
        "mood_description": "Dreamy and cinematic, drifting through a quiet city at night lost in your own thoughts.",
    },

    # ── Romantic ──────────────────────────────────────────────────────────────
    {
        "title": "Perfect",
        "artist": "Ed Sheeran",
        "genre": "Pop",
        "mood_description": "Pure romantic love and devotion, slow dancing with the person who is your entire world.",
    },
    {
        "title": "At Last",
        "artist": "Etta James",
        "genre": "Soul/R&B",
        "mood_description": "Deep romantic fulfillment and relief, finally finding the love you have always dreamed of.",
    },
    {
        "title": "Thinking Out Loud",
        "artist": "Ed Sheeran",
        "genre": "Pop/Soul",
        "mood_description": "Timeless and tender love, the kind that grows deeper as you grow old together.",
    },
    {
        "title": "Bloom",
        "artist": "The Paper Kites",
        "genre": "Indie Folk",
        "mood_description": "Soft and intimate new love, gentle and vulnerable like something fragile and beautiful.",
    },
    {
        "title": "Can't Help Falling in Love",
        "artist": "Elvis Presley",
        "genre": "Pop/Classic",
        "mood_description": "Helpless, overwhelming love — surrendering completely to a feeling you never expected.",
    },

    # ── Energetic / Pump-up ───────────────────────────────────────────────────
    {
        "title": "HUMBLE.",
        "artist": "Kendrick Lamar",
        "genre": "Hip-Hop",
        "mood_description": "Intense confidence and unstoppable drive, asserting dominance with sharp, high-energy focus.",
    },
    {
        "title": "Lose Yourself",
        "artist": "Eminem",
        "genre": "Hip-Hop",
        "mood_description": "Raw determination and laser focus, seizing your one opportunity before it slips away.",
    },
    {
        "title": "Eye of the Tiger",
        "artist": "Survivor",
        "genre": "Rock",
        "mood_description": "Classic workout and motivation anthem, pushing hard through every challenge with fierce drive.",
    },
    {
        "title": "Run the World (Girls)",
        "artist": "Beyoncé",
        "genre": "Pop/R&B",
        "mood_description": "Powerful, fierce, and unstoppable — full of bold energy and commanding confidence.",
    },
    {
        "title": "Power",
        "artist": "Kanye West",
        "genre": "Hip-Hop",
        "mood_description": "Feeling invincible and in total control, ready to take on the world with supreme confidence.",
    },

    # ── Angry / Frustrated ────────────────────────────────────────────────────
    {
        "title": "You Oughta Know",
        "artist": "Alanis Morissette",
        "genre": "Alternative Rock",
        "mood_description": "Explosive anger and raw betrayal after being hurt by someone who moved on without a second thought.",
    },
    {
        "title": "Killing in the Name",
        "artist": "Rage Against the Machine",
        "genre": "Rock/Metal",
        "mood_description": "Intense, righteous anger at authority and injustice, refusing to follow orders you don't believe in.",
    },
    {
        "title": "I Will Survive",
        "artist": "Gloria Gaynor",
        "genre": "Disco/Pop",
        "mood_description": "Turning pain and anger into defiant strength — you have been hurt but you are not done.",
    },
    {
        "title": "Fighter",
        "artist": "Christina Aguilera",
        "genre": "Pop/Rock",
        "mood_description": "Channeling hurt and betrayal into fierce determination — grateful for the pain that made you stronger.",
    },
    {
        "title": "Roar",
        "artist": "Katy Perry",
        "genre": "Pop",
        "mood_description": "Finding your voice and refusing to be silenced anymore, standing up for yourself with force.",
    },

    # ── Anxious / Overwhelmed ─────────────────────────────────────────────────
    {
        "title": "Breathin",
        "artist": "Ariana Grande",
        "genre": "Pop",
        "mood_description": "Struggling with anxiety and the desperate effort to just keep breathing and stay above water.",
    },
    {
        "title": "Stressed Out",
        "artist": "Twenty One Pilots",
        "genre": "Alternative/Pop",
        "mood_description": "The crushing anxiety of growing up — wishing you could go back to a simpler, safer time.",
    },
    {
        "title": "Under Pressure",
        "artist": "Queen & David Bowie",
        "genre": "Rock",
        "mood_description": "The heavy weight of external pressure building and threatening to crack everything around you.",
    },
    {
        "title": "1-800-273-8255",
        "artist": "Logic",
        "genre": "Hip-Hop",
        "mood_description": "Confronting the darkest feelings head-on with a message that you are worth fighting for.",
    },
    {
        "title": "Numb",
        "artist": "Linkin Park",
        "genre": "Rock/Alternative",
        "mood_description": "Emotional exhaustion from trying to meet everyone's expectations and losing yourself in the process.",
    },

    # ── Hopeful / Inspired ────────────────────────────────────────────────────
    {
        "title": "Rise Up",
        "artist": "Andra Day",
        "genre": "Soul/Gospel",
        "mood_description": "Powerful, soaring hope and resilience — rising through pain and hardship with unwavering belief.",
    },
    {
        "title": "Shake It Out",
        "artist": "Florence + The Machine",
        "genre": "Indie Rock",
        "mood_description": "Choosing to let go of darkness and move forward with renewed energy and self-belief.",
    },
    {
        "title": "Unwritten",
        "artist": "Natasha Bedingfield",
        "genre": "Pop",
        "mood_description": "Open-ended optimism and the thrill of a blank page — your story is yours to write.",
    },
    {
        "title": "Hall of Fame",
        "artist": "The Script ft. will.i.am",
        "genre": "Pop/Rock",
        "mood_description": "Inspired belief that dedication and heart can take you all the way to greatness.",
    },

    # ── Chill / Lo-fi ─────────────────────────────────────────────────────────
    {
        "title": "Redbone",
        "artist": "Childish Gambino",
        "genre": "R&B/Funk",
        "mood_description": "Smooth, hazy, and hypnotic — a late-night groove that wraps around you like warm air.",
    },
    {
        "title": "Electric Feel",
        "artist": "MGMT",
        "genre": "Indie Pop/Electronic",
        "mood_description": "Effortlessly cool and dreamy, a spacey groove that makes everything feel a little magical.",
    },
    {
        "title": "The Less I Know the Better",
        "artist": "Tame Impala",
        "genre": "Psychedelic Pop",
        "mood_description": "Groovy and bittersweet, dancing through feelings you would rather not think too hard about.",
    },
    {
        "title": " Motion Picture Soundtrack",
        "artist": "Radiohead",
        "genre": "Alternative",
        "mood_description": "Dreamy and deeply introspective, floating quietly through your own thoughts late at night.",
    },
    {
        "title": "Nights",
        "artist": "Frank Ocean",
        "genre": "R&B",
        "mood_description": "Late-night reflection and nostalgia, shifting between hushed introspection and smooth groove.",
    },

    # ── Focused / Productive ──────────────────────────────────────────────────
    {
        "title": "Pursuit of Happiness",
        "artist": "Kid Cudi",
        "genre": "Hip-Hop",
        "mood_description": "Relentless drive and determination on the road to purpose, even when the journey feels hard.",
    },
    {
        "title": "Stronger",
        "artist": "Kanye West",
        "genre": "Hip-Hop",
        "mood_description": "Channeling setbacks into fuel — every obstacle makes you harder, better, faster, stronger.",
    },
    {
        "title": "Work Hard, Play Hard",
        "artist": "Wiz Khalifa",
        "genre": "Hip-Hop",
        "mood_description": "High-energy hustle mindset, grinding through the work so you can enjoy the reward.",
    },
    {
        "title": "Titanium",
        "artist": "David Guetta ft. Sia",
        "genre": "Electronic/Pop",
        "mood_description": "Bulletproof resilience and unshakeable focus — nothing can break you when you are in the zone.",
    },
]
