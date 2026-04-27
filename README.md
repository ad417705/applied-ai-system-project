# VibeFinder 1.0 — Music Recommender Simulation

## Original Project (Modules 1–3)

**Project 3: Music Recommender Simulation**

You are working for a startup music platform that wants to understand how big-name apps like Spotify or TikTok predict what users will love next. The original mission was to simulate and explain how a basic music recommendation system works — designing a modular Python architecture that transforms song data and listener "taste profiles" into personalized suggestions. The system was built around a content-based filtering approach: no listening history or user behavior data required, just audio attributes and a weighted scoring formula that scores every song in the catalog against the user's expressed preferences and returns the top-k ranked results.

---

## Title and Summary

**VibeFinder 1.0** is a content-based music recommender simulation built in Python. It takes a listener's preferences — genre, mood, energy level, acoustic feel, and emotional tone — scores every song in a 20-track catalog against those preferences using a weighted multi-signal formula, and returns a ranked list of recommendations with plain-language explanations for each result.

The project was built and extended across three phases:

**Phase 1 — Interactive CLI**
Added an `interactive_mode()` that lets any user enter their own preferences at the command line and receive live, personalized recommendations in real time. Every field is optional — pressing Enter skips to a sensible default so users can get results immediately without filling in every preference.

**Phase 2 — Automated Testing (22 tests)**
Expanded the test suite to cover edge cases (k larger than catalog size, genres missing from the catalog, empty preferences), scoring bounds (score always in [0.0, 1.0], a perfect match always equals 1.0), explanation accuracy (reasons only fire when the song genuinely satisfies the signal), and weighted variant ordering (changing weights reshuffles results in a mathematically verifiable way). All 22 tests pass.

**Phase 3 — Model Evaluation (the most important addition)**
Built a standalone evaluation script (`tests/evaluate.py`) that runs all six demo profiles through the recommender and produces a structured metrics report. This is where the project moves from "does it run?" to "is it actually good?" — the difference between a demo and a real system.

The evaluator measures:

- **Precision@k** — what fraction of the top-k results actually match what the user asked for (genre or mood)
- **Genre diversity** — how many unique genres appear in each result set (a proxy for discovery vs. exploitation)
- **Discovery rate** — what fraction of results fall _outside_ the user's primary genre
- **Catalog coverage** — what percentage of the full 20-song catalog gets surfaced across all profiles combined (a low number means certain songs are systematically buried)
- **Score distribution** — mean, standard deviation, min, and max of scores per profile; near-zero std-dev means the ranking is nearly arbitrary
- **Weight sensitivity** — a head-to-head comparison of default vs. experimental weights showing how a single hyperparameter change reshuffles the top-5

Why this matters: without evaluation, you are guessing. The metrics exposed three structural problems that were invisible from simply running the demo — the Ultra-Neutral profile had near-zero score spread (rankings were essentially random), the K-Pop profile is permanently capped at 74.4% of the maximum possible score due to a catalog gap, and three songs in the catalog are never recommended under any of the six demo profiles. These are exactly the kinds of silent failures that get shipped to real users when teams skip the evaluation step.

---

## Architecture Overview

```
data/songs.csv (20 songs)          User Input (CLI or PROFILES dict)
        │                                        │
        │  load_songs()                          │  interactive_mode()
        ▼                                        ▼
                         score_song()
               [genre × 2.0] + [mood × 1.5] + [energy Gaussian × 1.5]
               + [valence Gaussian × 1.0] + [acoustic × 1.0] + [instrumental × 0.8]
               ÷ 7.8  →  score ∈ [0.0, 1.0]
                                │
                                ▼
                       recommend_songs()
                       sort all 20 scores → return top-k
                                │
                      ┌─────────┴──────────┐
                      ▼                    ▼
              CLI Output             tests/evaluate.py
         rank, score bar,          precision@k, diversity,
           explanation             coverage, weight sensitivity
```

**`data/songs.csv`** — 20-song catalog. Each row has 11 attributes: `id`, `title`, `artist`, `genre`, `mood`, `energy`, `tempo_bpm`, `valence`, `danceability`, `acousticness`, `instrumentalness`.

**`src/recommender.py`** — The core engine. Contains the `Song` and `UserProfile` dataclasses, `score_song()` (the weighted formula with Gaussian proximity kernels for continuous signals), `recommend_songs()` (the main functional entry point), `recommend_songs_weighted()` (for A/B weight experiments), and the `Recommender` class (OOP wrapper used by the test suite).

**`src/main.py`** — CLI runner with two modes: option 1 runs 6 hardcoded demo profiles plus a weight-shift experiment; option 2 launches `interactive_mode()` for custom preferences.

**`tests/test_recommender.py`** — 22 unit tests covering edge cases, scoring correctness, explanation accuracy, and weighted variant behavior.

**`tests/evaluate.py`** — Standalone Phase 3 evaluation script. Prints a full metrics report including precision@k, diversity, coverage, score distribution, weight sensitivity, and automated insights.

---

## Setup Instructions

**Prerequisites:** Python 3.10+ and `pytest` installed.

```bash
# 1. Navigate to the project directory
cd applied-ai-system-project

# 2. (Optional) Install pytest if not already installed
pip install pytest

# 3. Run the recommender
python src/main.py
```

When the program starts you will see:

```
============================================================
         VibeFinder 1.0 — Music Recommender
============================================================
  1. Run demo profiles
  2. Interactive mode — enter your own preferences
============================================================
  Choose an option [1]:
```

- Press **1** (or Enter) to run all 6 demo profiles automatically, plus the weight-shift experiment.
- Press **2** to enter your own preferences and get a personalized result.

**To run the evaluation report:**

```bash
python tests/evaluate.py
```

**To run the full test suite:**

```bash
pytest tests/ -v
```

---

## Sample Interactions

### Example 1 — High-Energy Pop (Demo mode, Profile 1)

**Preferences:** `genre=pop | mood=happy | energy=0.85 | valence=0.82 | acoustic=no`

```
============================================================
                    High-Energy Pop
============================================================
  Genre   : pop
  Mood    : happy
  Energy  : 0.85
  Valence : 0.82
------------------------------------------------------------
  #1  Sunrise City  —  Neon Echo
       [███████████████████░]  97%
       • matches your preferred genre (pop)
       • matches your preferred mood (happy)
       • energy (0.82) is close to your target (0.85)
       • emotional tone (0.84) matches your bright/joyful preference
       • has a produced/electronic sound (0.18)
       • features strong vocals (0.04)
------------------------------------------------------------
  #2  Gym Hero  —  Max Pulse
       [████████████████░░░░]  79%
       • matches your preferred genre (pop)
       • energy (0.93) is close to your target (0.85)
       • has a produced/electronic sound (0.05)
       • features strong vocals (0.03)
------------------------------------------------------------
  #3  Rooftop Lights  —  Indigo Parade
       [██████████████░░░░░░]  68%
       • matches your preferred mood (happy)
       • energy (0.76) is close to your target (0.85)
       • emotional tone (0.81) matches your bright/joyful preference
------------------------------------------------------------
```

Sunrise City scores 97% because it satisfies all 6 signals simultaneously. Gym Hero hits on genre and energy but misses mood, landing at 79%. Rooftop Lights earns points on mood and energy but is in a different genre (indie pop), dropping it to 68%.

---

### Example 2 — Chill Lofi (Demo mode, Profile 2)

**Preferences:** `genre=lofi | mood=chill | energy=0.38 | valence=0.60 | acoustic=yes | instrumental=yes`

```
============================================================
                        Chill Lofi
============================================================
  Genre   : lofi
  Mood    : chill
  Energy  : 0.38
  Valence : 0.60
------------------------------------------------------------
  #1  Library Rain  —  Paper Lanterns
       [███████████████████░]  97%
       • matches your preferred genre (lofi)
       • matches your preferred mood (chill)
       • energy (0.35) is close to your target (0.38)
       • emotional tone (0.60) matches your dark/introspective preference
       • has a strong acoustic feel (0.86)
       • is primarily instrumental (0.88)
------------------------------------------------------------
  #2  Midnight Coding  —  LoRoom
       [███████████████████░]  94%
       • matches your preferred genre (lofi)
       • matches your preferred mood (chill)
       • energy (0.42) is close to your target (0.38)
       • has a strong acoustic feel (0.71)
       • is primarily instrumental (0.78)
------------------------------------------------------------
  #3  Focus Flow  —  LoRoom
       [███████████████░░░░░]  77%
       • matches your preferred genre (lofi)
       • energy (0.40) is close to your target (0.38)
       • has a strong acoustic feel (0.78)
       • is primarily instrumental (0.93)
------------------------------------------------------------
```

The Chill Lofi profile benefits from a well-stocked catalog — 3 lofi songs and several chill-mood tracks exist, so the model has strong candidates. Library Rain scores 97% because every attribute including acousticness (0.86) and instrumentalness (0.88) aligns with what the user asked for.

---

### Example 3 — Interactive mode (custom preferences)

```
  Available genres: pop, lofi, rock, ambient, jazz, synthwave, indie pop
  Your preferred genre [none]: jazz

  Available moods: happy, chill, intense, moody, relaxed, focused
  Your preferred mood [none]: relaxed

  Target energy  (0.0 calm → 1.0 intense) [0.5]: 0.35
  Target valence (0.0 dark  → 1.0 joyful)  [0.5]: 0.70
  Prefer acoustic / organic sound? (y/n) [n]: y
  Prefer instrumental (no vocals)?  (y/n) [n]: n
  How many recommendations? [5]: 3
```

```
============================================================
                   Your Custom Profile
============================================================
  #1  Coffee Shop Stories  —  Slow Stereo          94%
       • matches your preferred genre (jazz)
       • matches your preferred mood (relaxed)
       • energy (0.37) is close to your target (0.35)
       • has a strong acoustic feel (0.89)
------------------------------------------------------------
  #2  Library Rain  —  Paper Lanterns               52%
       • energy (0.35) is close to your target (0.35)
       • has a strong acoustic feel (0.86)
------------------------------------------------------------
  #3  Desert Wind  —  The Hollow Pines              50%
       • energy (0.31) is close to your target (0.35)
       • has a strong acoustic feel (0.88)
------------------------------------------------------------
```

There is only one jazz song in the catalog, so Coffee Shop Stories wins convincingly. The #2 and #3 results fall back to continuous signals (energy proximity and acousticness) because no other song matches on genre or mood — which is exactly the graceful degradation the Gaussian kernel is designed to produce.

---

### Example 4 — Evaluation report

```bash
python tests/evaluate.py
```

```
──────────────────────────────────────────────────────────────────────
  METRICS SUMMARY  (k=5 per profile, catalog=20 songs)
──────────────────────────────────────────────────────────────────────
  Profile                            Prec@5  Genres   Discov  AvgScore
  --------------------------------------------------------------------
  High-Energy Pop                      60%       4     60%     0.695
  Chill Lofi                           80%       3     40%     0.774
  Deep Intense Rock                    40%       5     80%     0.619
  Contradictory Energy+Acoustic        80%       4     60%     0.466
  Missing Genre (K-Pop)                60%       5    100%     0.590
  Ultra-Neutral Listener              N/A        5    N/A      0.410
  --------------------------------------------------------------------
  Macro-average (excl. N/A)            64%     4.3     —        0.592

──────────────────────────────────────────────────────────────────────
  CATALOG COVERAGE
──────────────────────────────────────────────────────────────────────
  Songs surfaced: 17 / 20  (85%)
  Never recommended: Night Drive Loop, Piano at 3AM, Desert Wind
```

---

## Design Decisions and Trade-offs

### Content-Based Filtering Over Collaborative Filtering

The system uses content-based filtering — matching audio attributes of songs to user preferences — instead of collaborative filtering (learning from what similar users listened to). This was a deliberate choice: it requires no user history, works immediately for any new listener who can express a preference, and produces easily explainable results. The trade-off is that it cannot surface surprising cross-genre discoveries because it never learns what "people like you" enjoy. Real platforms like Spotify use a hybrid of both approaches precisely because neither alone is sufficient.

### Weighted Multi-Signal Formula

Each signal is assigned a maximum point value based on how strongly it differentiates listener intent:

| Signal               | Max Points | Rationale                                                            |
| -------------------- | ---------- | -------------------------------------------------------------------- |
| Genre match          | 2.0        | Strongest conscious preference — users know what genre they want     |
| Mood match           | 1.5        | Session-level intent, often more predictive than genre at the moment |
| Energy proximity     | 1.5        | Widest spread in catalog (0.19–0.96), best continuous differentiator |
| Valence proximity    | 1.0        | Carries mood information numerically across genre boundaries         |
| Acousticness fit     | 1.0        | Cleanly separates electronic from organic production                 |
| Instrumentalness fit | 0.8        | Lowest weight — vocal preference is secondary to all others          |

Categorical signals (genre, mood) use binary all-or-nothing scoring because there is no such thing as "almost pop." Continuous signals use a Gaussian proximity kernel so small mismatches are forgiven while large gaps are penalized steeply. This combination means a song with the right genre but slightly wrong energy still scores reasonably — which matches how real listeners think.

### Gaussian Kernel for Continuous Signals

Rather than a simple linear distance (`1 - |song_val - target|`), the formula uses a Gaussian bell curve: `exp(-(diff²) / (2σ²))` with σ=0.25. A difference of 0.25 in energy still earns ~61% of the maximum energy points; a difference of 0.50 drops to ~14%. This forgiveness zone is intentional — a song at energy 0.70 is still a reasonable recommendation for someone who asked for 0.85. The σ=0.25 value was chosen manually and is the primary hyperparameter to tune based on evaluation results.

### No Persistence, No Training

The system re-scores all 20 songs on every run. There is no caching, no trained model, and no stored embeddings. This keeps the architecture fully transparent and traceable — you can follow exactly why any song received any score. The trade-off is that it does not scale beyond a small catalog. At 20 million songs, you would need approximate nearest neighbor search and pre-computed embeddings, not a full re-score on every query.

---

## Testing Summary

### What Worked

The 22-test suite (`pytest tests/ -v`) passes completely and confirmed:

- A perfect-match song reliably scores exactly 1.0 — the formula is arithmetically correct.
- Requesting k=999 songs from a 2-song catalog returns 2 results without raising an error.
- The weighted variant correctly reorders results when energy weight is doubled and genre weight is halved — a change that is mathematically predictable and verified in the test.
- The explanation engine fires genre, mood, energy, valence, acousticness, and instrumental reasons only when the song actually satisfies the threshold, and falls back to the generic message only when nothing fires.

### What the Evaluation Revealed (Evaluation Insights)

Running `tests/evaluate.py` uncovered three problems that unit tests could not catch:

**Problem 1: Ultra-Neutral Listener has near-random rankings (std=0.029)**
When a user expresses no genre or mood preference, the five returned songs score within a 0.07-point range of each other. Swapping any two would make essentially no difference to the user. The underlying cause is that without categorical bonuses, the formula relies entirely on continuous signals around a 0.5 midpoint — and most songs land near 0.5 on those dimensions. The fix: add a tiebreaker signal (danceability or tempo proximity) that activates when categorical matches are absent.

**Problem 2: Missing genre permanently caps the maximum score at 74.4%**
The catalog contains no kpop songs. The 2.0 genre bonus can never fire, so the theoretical ceiling for any K-Pop profile recommendation is 5.8 / 7.8 = 74.4%. The system silently returns lower-quality results without telling the user why. The fix: detect when the requested genre has no catalog match and surface an explicit warning message.

**Problem 3: Three songs never get recommended under any profile**
Night Drive Loop (synthwave/moody), Piano at 3AM (classical/melancholic), and Desert Wind (folk/relaxed) do not appear in any top-5 across all six profiles. This reveals a structural bias in the demo profile set — it does not represent the full range of listener types these songs would serve. It also shows that niche genres with low energy and mid-range valence are systematically ranked below louder or more genre-aligned alternatives.

### What This Taught About Testing

Unit tests verify that code behaves correctly for known inputs. Evaluation metrics reveal whether the model is actually useful — these are not the same question. A system can pass every unit test and still silently underserve users. The evaluation step is the bridge between "technically correct" and "actually good."

---

## Reflection: What This Project Taught About AI and Problem-Solving

**There is no perfect model.**

The most important insight from this project is that every design decision in a recommendation system involves a trade-off, and optimizing for one thing almost always degrades something else. Making genre matching stronger improves precision for users with clear preferences but increases "genre lock" for users who want variety. A tight Gaussian kernel makes energy matching more precise but punishes songs that are slightly outside the target. Prioritizing catalog coverage means sometimes recommending a song the user does not like just to ensure it gets surfaced.

Real systems at companies like Spotify manage this by running thousands of A/B experiments continuously, letting user behavior data tell them which trade-off is better for which user segment. Without that feedback loop, a small simulation like VibeFinder has to make explicit bets about what matters most — and then measure the consequences.

The weight sensitivity section in `evaluate.py` is a small version of that A/B process. Doubling the energy weight and halving the genre weight changed the #2 and #3 positions for the High-Energy Pop profile without changing the top-5 set at all — meaning the genre signal was already doing more work than the energy signal needed. That kind of discovery only becomes visible through evaluation, not through reading the code.

The broader lesson: AI systems are not correct or incorrect in isolation. They are better or worse at specific tasks, for specific users, measured by specific metrics. Knowing which metrics matter — and why — is as important as knowing how to write the model. You can build a system that "works" by every mechanical definition and still fails the people it is meant to serve, unless you define success in terms of outcomes, not just outputs.

---

## Responsible AI Reflection

### Limitations and Biases in the System

The most significant bias is **genre dominance**. At 2.0 points — more than any other signal — genre match controls the outcome of most recommendations. A technically excellent song in the "wrong" genre can never outscore a mediocre song in the "right" genre, no matter how closely it matches on every other dimension. This creates a filter bubble by design: users who say "pop" will almost always get pop, and the system has no mechanism for expanding their taste over time.

The second structural bias is in the **mood labels themselves**. The scoring treats "chill" and "relaxed" as completely different (0 points for a mismatch) even though they describe almost the same emotional state. This is a consequence of binary categorical matching — the system has no sense of semantic similarity between labels. A user who types "relaxed" gets zero mood credit for a song tagged "chill," even though a human listener would consider both perfectly reasonable.

**Acousticness** is a subtler bias. Songs with high acousticness (organic, natural sound) tend to come from artists working with smaller production budgets, while highly produced electronic tracks require significant studio resources. The acousticness signal is neutral by design (it rewards either end based on user preference), but the underlying attribute correlates with economics of music production. In a larger system, this could cause certain demographics of artists to be systematically surfaced or buried based on signals that appear technical but carry cultural weight.

Finally, the 20-song catalog itself is a bias. Every genre not in the catalog is permanently unreachable — users who want kpop, classical, folk, or blues as their primary genre start at a ceiling of 74.4% max score, regardless of how well other signals match. A small catalog does not just limit variety; it structurally disadvantages some users from the start.

### Could This System Be Misused?

A music recommender feels low-stakes, and in isolation it largely is. But the pattern underneath — weighted scoring on user-expressed attributes — is identical to the logic used in hiring tools, loan approval systems, and content moderation. The risk with VibeFinder is not that it harms anyone directly; it is that it normalizes the idea that a simple weighted formula, applied without evaluation, produces fair and useful results.

The specific misuse risk here is **gaming the system**. If an artist or label knew the exact weights (genre=2.0, mood=1.5, energy=1.5), they could optimize their song metadata to score highly against common user profiles — not by making better music, but by labeling it more strategically. This is the same problem search engine optimization created for Google, and it degrades the quality of results for everyone.

Prevention starts with **not publishing the weights openly in a production system**, running **periodic audits** to check whether certain artists or genres are being over- or under-surfaced relative to catalog share, and most importantly, **not treating the current weights as permanent**. They were chosen manually with reasoning, but they were not validated against actual user satisfaction data. Any system that gets deployed as-is, without a feedback loop, is a system that cannot correct its own biases.

### What Surprised Me While Testing

The biggest surprise was how invisible some failures were until evaluation was added. Before `tests/evaluate.py` existed, running the demo felt like it was working — results looked plausible, the scores were reasonable numbers, and no errors were thrown. The unit tests all passed. Everything appeared functional.

Then the evaluation script revealed that the Ultra-Neutral Listener profile was producing results with a standard deviation of 0.029 — meaning the five songs it returned were essentially indistinguishable in score. The difference between #1 and #5 was 0.069 points. A small change in any catalog attribute would reorder those results completely. The ranking was not wrong — it was arbitrary. And that only became visible once a metric was defined to catch it.

The second surprise was the weight sensitivity result. The expectation, based on how the experiment was framed, was that doubling the energy weight would significantly shake up the High-Energy Pop ranking. Instead, all 5 songs in the top-5 remained identical — only positions #2 and #3 swapped. The formula was more stable than anticipated, which is reassuring from a robustness standpoint, but it also means the "genre dominates everything" narrative was somewhat overstated. The genre signal matters, but the other signals are doing real work too.

### Collaboration With AI During This Project

AI (Claude) was used throughout this project as a co-developer — writing code, designing the evaluation framework, and structuring the test suite. The collaboration worked best when the task was well-defined with clear success criteria (for example, "write 22 unit tests that cover these specific edge cases"). The AI produced thorough, correct tests on the first pass and caught an important edge case — that `recommend_songs_weighted(weights=None)` should fall back to the default behavior and produce identical results — that had not been considered.

**One instance where the AI suggestion was genuinely helpful:** When designing the evaluation script, the AI suggested measuring score distribution standard deviation per profile, specifically to detect near-arbitrary rankings. This was not in the original architecture spec, which only called for precision@k, diversity, and coverage. Adding std-dev as a metric is what ultimately surfaced the Ultra-Neutral Listener problem — a real flaw that would have shipped undetected without that suggestion. It reframed evaluation from a pass/fail check into a diagnostic tool.

**One instance where the AI's suggestion was flawed:** In the weight sensitivity experiment section of `src/main.py`, the pre-written observation text states: _"songs with a strong energy match but wrong genre rise in ranking [B] compared to [A], confirming that the genre weight dominates results in the default configuration."_ This conclusion was written before the experiment was actually run end-to-end through the evaluator. When `evaluate.py` ran the experiment, the top-5 set showed 5/5 overlap between default and experimental rankings — not a single song entered or left the top-5. The conclusion overstated what the data showed. The framing was confident and plausible-sounding, but it was a claim about dominance that the actual numbers did not fully support. This is a good example of why AI-generated analysis should always be verified against real output rather than taken at face value — the AI reasoned correctly about the direction of the change, but overstated its magnitude.

---

## Project Structure

```
applied-ai-system-project/
├── data/
│   └── songs.csv              20-song catalog with 11 attributes per track
├── src/
│   ├── main.py                CLI runner — demo mode and interactive mode
│   └── recommender.py         Core scoring engine, dataclasses, functional API
├── tests/
│   ├── test_recommender.py    22 unit tests (run with: pytest tests/ -v)
│   └── evaluate.py            Phase 3 evaluation metrics report
├── architecture.md            System diagram and implementation phase notes
└── README.md                  This file
```
