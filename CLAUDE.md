# Music Recommender Simulation — Project Context

> This file captures domain knowledge, design decisions, and attribute guidance for this project.
> Reference it when implementing `recommender.py`, writing tests, or filling out `model_card.md`.

---

## How Real Platforms Predict What You'll Love

### Spotify
Uses a **hybrid ensemble** of three systems running in parallel:
1. **Collaborative filtering** via matrix factorization on 30-day implicit feedback (streams, saves, skips, playlist adds)
2. **NLP** — crawls music blogs and Reddit to extract cultural context and mood descriptors around songs
3. **Raw audio CNNs** — convolutional neural networks on audio spectrograms to handle new releases with zero play history

Discover Weekly generates a taste profile per user via collaborative filtering, then uses audio models to fill cold-start gaps.

### YouTube / Google
Two-stage deep neural network architecture (documented in their 2016 RecSys paper):
- **Candidate generation network** — narrows billions of videos to hundreds of candidates
- **Ranking network** — scores candidates using fine-grained signals (watch time weighted more than clicks)
- Watch time is the primary implicit signal — a 30-second view of a 3-minute video is a stronger skip signal than no click

### Apple Music
- **Human curation as a feature signal** — editor-curated playlists create ground-truth quality labels
- **Collaborative filtering** on Apple ecosystem behavior (purchases, library adds, play counts)
- **Shazam acquisition** — audio fingerprinting and attribute data at scale

---

## Collaborative vs. Content-Based Filtering

| Dimension | Collaborative Filtering | Content-Based Filtering |
|---|---|---|
| Data needed | User behavior history | Audio/metadata features |
| New user problem | Fails (no history) | Works immediately |
| New song problem | Fails (no plays) | Works immediately |
| Discovery potential | High — finds surprising connections | Low — stays within taste bubble |
| Explainability | Hard ("people like you...") | Easy ("similar tempo, energy...") |
| Compute cost | High (massive matrix ops) | Moderate (feature extraction) |

**Collaborative filtering** ignores what a song *is* — it focuses on what people *do*. If users A and B both loved songs X and Y, and A also loves Z, B probably will too.

**Content-based filtering** analyzes the *attributes of the music itself* — tempo, energy, mood, genre — and matches users to songs with similar feature profiles.

**This simulator uses content-based filtering** — you have song attributes but no cross-user behavior data.

---

## Core Data Types in Recommendation Systems

### User Data
```
UserProfile
├── favorite_genre    (str — categorical preference)
├── favorite_mood     (str — categorical preference)
├── target_energy     (float — continuous preference)
└── likes_acoustic    (bool — binary threshold signal)
```

### Song / Item Data
```
Song
├── id, title, artist   (identifiers)
├── genre, mood         (categorical — str)
├── energy, tempo_bpm   (continuous — float)
├── valence             (continuous — float, emotional positiveness)
├── danceability        (continuous — float)
└── acousticness        (continuous — float)
```

### Interaction Matrix (collaborative filtering — not used here)
A sparse matrix: rows = users, columns = songs, values = interaction strength.
Not applicable to this simulator (no multi-user data), but important to understand conceptually.

### Embeddings (modern systems)
Both users and songs get compressed into a shared float vector space.
Similarity = dot product or cosine distance between vectors.
This simulator approximates this with a weighted scoring formula instead.

---

## songs.csv Attribute Guide

Full dataset: 10 songs, 10 attributes each.

| Attribute | Type | Range in Dataset | What It Captures |
|---|---|---|---|
| `genre` | categorical str | pop, lofi, rock, ambient, jazz, synthwave, indie pop | Style classification |
| `mood` | categorical str | happy, chill, intense, moody, relaxed, focused | Emotional tone |
| `energy` | float | 0.28 – 0.93 | High-intensity vs. laid-back feel |
| `tempo_bpm` | float | 60 – 152 | Pace / rhythm |
| `valence` | float | 0.48 – 0.84 | Musical positiveness (sad → joyful) |
| `danceability` | float | 0.41 – 0.88 | Groove / rhythmic stability |
| `acousticness` | float | 0.05 – 0.92 | Acoustic vs. electronic |

### Attribute Notes

**`genre` + `mood`** — Strongest signals for categorical matching. Binary: either matches or doesn't.
Genre is how users consciously describe their taste. Mood is often more predictive at the session level
(a lofi fan might want "focused" lofi during study, not "chill" lofi).

**`energy`** — Widest spread (0.28–0.93), making it your best continuous differentiator.
Spacewalk Thoughts (0.28) and Gym Hero (0.93) are practically opposite listening experiences.

**`tempo_bpm`** — Strongly correlated with energy. Chill songs cluster 60–90 BPM; intense songs
cluster 110–152. Provides a redundant signal that reinforces energy matching.

**`valence`** — Quietly carries mood information numerically. Intense/moody songs cluster below 0.50;
happy songs cluster above 0.76. Useful for cross-genre mood matching.

**`danceability`** — Ambient and lofi cluster low (0.41–0.62); pop and synthwave cluster high (0.73–0.88).

**`acousticness`** — Most polarizing attribute. Cleanly separates electronic/produced tracks from organic
ones. Maps directly to `UserProfile.likes_acoustic`.

---

## Scoring Attribute Weights (Recommended Approach)

```
score = (genre_match × 0.35)
      + (mood_match  × 0.25)
      + (1 - |song.energy - user.target_energy|) × 0.25
      + (acousticness_match × 0.15)
```

Where:
- `genre_match` = 1.0 if exact match, else 0.0
- `mood_match` = 1.0 if exact match, else 0.0
- `acousticness_match` = song.acousticness if `likes_acoustic`, else (1 - song.acousticness)

Fine-grained attributes (valence, danceability, tempo) can be added to the energy term for deeper matching.

---

## The Discovery Problem (Exploration vs. Exploitation)

The most important unsolved tension in recommender design.

**Without discovery** — a "pop + happy + high energy" user always gets:
- Sunrise City, Gym Hero, Rooftop Lights (all obvious matches)

They never discover Night Drive Loop (synthwave, 0.75 energy, high danceability) which might also resonate.

**With feature-vector proximity** — look at the full attribute profile of liked songs, not just the genre label:

```
User loves: Sunrise City
  → energy: 0.82, danceability: 0.79, valence: 0.84

Nearest neighbor by feature distance:
  → Rooftop Lights:   energy 0.76, dance 0.82, valence 0.81  ← obvious
  → Night Drive Loop: energy 0.75, dance 0.73, valence 0.49  ← surprising but plausible
```

The valence gap (0.84 vs. 0.49) makes Night Drive Loop a *discovery* rather than an obvious match.

**To implement discovery:** Add a small bonus score for songs outside the user's primary genre that
still match on 2+ continuous attributes (energy, danceability, valence).

---

## Model Card Guidance

When filling out `model_card.md`, focus on these areas:

**Strengths**
- Genre + mood matching is fast, transparent, and explainable
- Every recommendation can be justified in plain language ("We recommended this because it matches your preferred genre and energy level")
- Works immediately — no training data or user history required

**Limitations**
- With only 10 songs, genre + mood alone creates a very small candidate pool (1–2 results for common preferences)
- No collaborative signal — the system cannot surface songs that "people like you" love
- Cold-start advantage comes with a discovery ceiling — purely categorical matching has no mechanism for serendipity

**Bias Risks**
- Acousticness and energy implicitly encode production style, which can correlate with cultural and economic factors (acoustic songs may skew toward certain demographics or income levels of artists)
- Genre labels are culturally constructed and inconsistently applied across regions and eras
- A "pop" bias: pop songs are overrepresented in training datasets everywhere, which can cause non-pop genres to score lower even when they match user intent

**Evaluation**
- Run `pytest tests/` to verify recommend() returns songs sorted by score
- Manually test edge cases: user preferences that match zero songs, user preferences that match all songs, `k` larger than catalog size
