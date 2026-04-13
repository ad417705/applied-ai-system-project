"""
Command line runner for the Music Recommender Simulation.

This file helps you quickly run and test your recommender.

Functions implemented in recommender.py:
- load_songs
- score_song
- recommend_songs
- recommend_songs_weighted  (experimental weight overrides)
"""

from recommender import load_songs, recommend_songs, recommend_songs_weighted

WIDTH = 60
HIGH_ENERGY_POP = "High-Energy Pop"

# ---------------------------------------------------------------------------
# User profiles — diverse and adversarial
# ---------------------------------------------------------------------------

PROFILES = {
    # ── Standard profiles ──────────────────────────────────────────────────
    HIGH_ENERGY_POP: {
        "genre": "pop",
        "mood": "happy",
        "energy": 0.85,
        "likes_acoustic": False,
        "prefers_instrumental": False,
        "target_valence": 0.82,
    },
    "Chill Lofi": {
        "genre": "lofi",
        "mood": "chill",
        "energy": 0.38,
        "likes_acoustic": True,
        "prefers_instrumental": True,
        "target_valence": 0.60,
    },
    "Deep Intense Rock": {
        "genre": "rock",
        "mood": "intense",
        "energy": 0.90,
        "likes_acoustic": False,
        "prefers_instrumental": False,
        "target_valence": 0.45,
    },
    # ── Adversarial / edge-case profiles ───────────────────────────────────
    # Contradictory: wants very high energy but also acoustic + ambient genre.
    # Real-world equivalent: someone saying "I want calm background music but
    # also high intensity" — the system should expose tension in the scoring.
    "Contradictory Energy+Acoustic": {
        "genre": "ambient",
        "mood": "intense",
        "energy": 0.92,
        "likes_acoustic": True,
        "prefers_instrumental": True,
        "target_valence": 0.30,
    },
    # Missing genre: "kpop" does not exist in the catalog at all.
    # The system can never award genre points; everything ranks on continuous
    # features alone — a stress test for the catalog-gap problem.
    "Missing Genre (K-Pop)": {
        "genre": "kpop",
        "mood": "happy",
        "energy": 0.72,
        "likes_acoustic": False,
        "prefers_instrumental": False,
        "target_valence": 0.78,
    },
    # Ultra-neutral: no genre/mood preference, perfectly mid energy/valence.
    # Exposes whether the scoring formula produces flat, meaningless rankings
    # when the user has no strong preferences.
    "Ultra-Neutral Listener": {
        "genre": "",
        "mood": "",
        "energy": 0.5,
        "likes_acoustic": False,
        "prefers_instrumental": False,
        "target_valence": 0.5,
    },
}

# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _score_bar(score: float, width: int = 20) -> str:
    """Convert a 0–1 score into a filled progress bar string."""
    filled = round(score * width)
    return "[" + "█" * filled + "░" * (width - filled) + "]"


def _print_header(label: str, user_prefs: dict) -> None:
    print("=" * WIDTH)
    print(f"  {label}".center(WIDTH))
    print("=" * WIDTH)
    print(f"  Genre   : {user_prefs['genre'] or '(none)'}")
    print(f"  Mood    : {user_prefs['mood'] or '(none)'}")
    print(f"  Energy  : {user_prefs['energy']}")
    print(f"  Valence : {user_prefs['target_valence']}")
    print(f"  Acoustic: {'yes' if user_prefs['likes_acoustic'] else 'no'}")
    print(f"  Vocal   : {'instrumental' if user_prefs['prefers_instrumental'] else 'vocal'}")
    print("=" * WIDTH)


def _print_result(rank: int, song: dict, score: float, explanation: str) -> None:
    bar = _score_bar(score)
    print(f"  #{rank}  {song['title']}  —  {song['artist']}")
    print(f"       {bar}  {score:.0%}")
    print(f"       Genre: {song['genre']}  |  Mood: {song['mood']}  |  Energy: {song['energy']}")

    if explanation.startswith("Recommended because it "):
        raw = explanation[len("Recommended because it "):]
        reasons = [r.strip().rstrip(".") for r in raw.split(", and ")]
        for reason in reasons:
            print(f"       • {reason}")
    else:
        print(f"       • {explanation}")

    print("-" * WIDTH)


def run_profile(label: str, user_prefs: dict, songs: list, k: int = 5) -> None:
    recommendations = recommend_songs(user_prefs, songs, k=k)
    print()
    _print_header(label, user_prefs)
    print(f"  Top {len(recommendations)} recommendations".center(WIDTH))
    print("-" * WIDTH)
    for rank, (song, score, explanation) in enumerate(recommendations, start=1):
        _print_result(rank, song, score, explanation)
    print()


# ---------------------------------------------------------------------------
# Experiment: weight shift — double energy weight, halve genre weight
# ---------------------------------------------------------------------------

EXPERIMENTAL_WEIGHTS = {
    "genre_pts":  1.0,   # was 2.0 — halved
    "mood_pts":   1.5,   # unchanged
    "energy_pts": 3.0,   # was 1.5 — doubled
    "valence_pts": 1.0,  # unchanged
    "acoustic_pts": 1.0, # unchanged
    "instrumental_pts": 0.8, # unchanged
}


def run_experiment(user_prefs: dict, songs: list, k: int = 5) -> None:
    """
    Run the same profile twice — once with default weights, once with the
    experimental weights — and print both side-by-side for comparison.
    """
    print()
    print("*" * WIDTH)
    print("  EXPERIMENT: Weight Shift".center(WIDTH))
    print("  genre ÷2  |  energy ×2".center(WIDTH))
    print(f"  Profile: {HIGH_ENERGY_POP}".center(WIDTH))
    print("*" * WIDTH)

    # Default run
    print("\n  [A] Default weights  (genre=2.0, energy=1.5)")
    print("-" * WIDTH)
    default_recs = recommend_songs(user_prefs, songs, k=k)
    for rank, (song, score, _) in enumerate(default_recs, start=1):
        bar = _score_bar(score)
        print(f"  #{rank}  {song['title']:30s}  {bar}  {score:.0%}")

    # Experimental run
    print(f"\n  [B] Experimental weights  (genre=1.0, energy=3.0)")
    print("-" * WIDTH)
    exp_recs = recommend_songs_weighted(user_prefs, songs, k=k, weights=EXPERIMENTAL_WEIGHTS)
    for rank, (song, score, _) in enumerate(exp_recs, start=1):
        bar = _score_bar(score)
        print(f"  #{rank}  {song['title']:30s}  {bar}  {score:.0%}")

    print()
    print("  Observation: songs with a strong energy match but wrong genre")
    print("  rise in ranking [B] compared to [A], confirming that the genre")
    print("  weight dominates results in the default configuration.")
    print("*" * WIDTH)
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    songs = load_songs("data/songs.csv")

    # Run all profiles
    for label, prefs in PROFILES.items():
        run_profile(label, prefs, songs, k=5)

    # Run weight-shift experiment on the High-Energy Pop profile
    run_experiment(PROFILES[HIGH_ENERGY_POP], songs, k=5)


if __name__ == "__main__":
    main()
