# VibeFinder 1.0 — Architecture

## System Overview

Content-based music recommender. Takes a user's taste preferences, scores every song in the catalog against those preferences using a weighted multi-signal formula, and returns the top-k ranked results with plain-language explanations.

No training data or listening history required. Works immediately for any user who can express a preference.

---

## Data Flow

```
songs.csv (20 songs)          User Input (CLI)
       │                             │
       │  load_songs()               │  interactive_mode() or PROFILES dict
       ▼                             ▼
                    score_song()
              [genre × 2.0] + [mood × 1.5] + [energy gaussian × 1.5]
              + [valence gaussian × 1.0] + [acoustic × 1.0] + [instrumental × 0.8]
              ÷ 7.8  →  score ∈ [0.0, 1.0]
                             │
                             ▼
                    recommend_songs()
                    sort all 20 scores → return top-k
                             │
                             ▼
                    CLI Output (main.py)
                    rank, title, score bar, explanation
```

---

## Components

**`data/songs.csv`** — 20-song catalog with 11 attributes per song (genre, mood, energy, valence, danceability, acousticness, instrumentalness, tempo_bpm, title, artist, id).

**`src/recommender.py`** — Core scoring engine. Contains the `Song` and `UserProfile` dataclasses, `score_song()` (the weighted formula), `recommend_songs()` (the main entry point), `recommend_songs_weighted()` (for A/B weight experiments), and the `Recommender` class (OOP wrapper).

**`src/main.py`** — CLI runner. Offers a menu: option 1 runs 6 hardcoded demo profiles, option 2 launches `interactive_mode()` so any user can enter their own preferences and get live results.

**`tests/test_recommender.py`** — Unit tests covering edge cases, scoring correctness, explanation accuracy, and weighted variant behavior.

**`tests/evaluate.py`** — Standalone metrics script. Runs all 6 demo profiles through `recommend_songs()` and reports precision@k, genre diversity in results, and catalog coverage.

---

## Where Humans and Testing Intervene

**Human — input:** User enters genre, mood, energy, and other preferences through the interactive CLI. All fields are optional; pressing Enter skips to a neutral default.

**Human — output:** User reads the ranked recommendations and explanation bullets, then decides whether to refine preferences and try again.

**Human — evaluation:** After running `python tests/evaluate.py`, a person reads the precision@k, diversity, and coverage numbers and decides whether the scoring weights need tuning.

**Automated testing:** `pytest tests/ -v` runs the full unit test suite after any code change. Tests catch regressions in edge cases (k > catalog size, missing genre, empty preferences), scoring bounds (score always in [0, 1], perfect match = 1.0), and weighted variant ordering.

---

## Implementation Phases

| Phase | What Changes | How to Verify |
|-------|-------------|---------------|
| 1 — Interactive CLI | Add `interactive_mode()` to `src/main.py` | `python -m src.main` → choose option 2, enter preferences |
| 2 — Core Tests | Expand `tests/test_recommender.py` from 2 → ~17 tests | `pytest tests/ -v` — all pass |
| 3 — Eval Metrics | Add `tests/evaluate.py` with precision@k, diversity, coverage | `python tests/evaluate.py` — metrics table prints |

Phases 2 and 3 are independent of Phase 1.
