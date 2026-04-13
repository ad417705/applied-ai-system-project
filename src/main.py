"""
Command line runner for the Music Recommender Simulation.

This file helps you quickly run and test your recommender.

Functions implemented in recommender.py:
- load_songs
- score_song
- recommend_songs
"""

from recommender import load_songs, recommend_songs

WIDTH = 60


def _score_bar(score: float, width: int = 20) -> str:
    """Convert a 0–1 score into a filled progress bar string."""
    filled = round(score * width)
    return "[" + "█" * filled + "░" * (width - filled) + "]"


def _print_header(user_prefs: dict) -> None:
    print("=" * WIDTH)
    print("  MUSIC RECOMMENDER SIMULATION".center(WIDTH))
    print("=" * WIDTH)
    print(f"  Genre   : {user_prefs['genre']}")
    print(f"  Mood    : {user_prefs['mood']}")
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

    # Split explanation into individual reason lines
    if explanation.startswith("Recommended because it "):
        raw = explanation[len("Recommended because it "):]
        reasons = [r.strip().rstrip(".") for r in raw.split(", and ")]
        for reason in reasons:
            print(f"       • {reason}")
    else:
        print(f"       • {explanation}")

    print("-" * WIDTH)


def main() -> None:
    songs = load_songs("data/songs.csv")

    user_prefs = {
        "genre": "pop",
        "mood": "happy",
        "energy": 0.8,
        "likes_acoustic": False,
        "prefers_instrumental": False,
        "target_valence": 0.8,
    }

    recommendations = recommend_songs(user_prefs, songs, k=5)

    print()
    _print_header(user_prefs)
    print(f"  Top {len(recommendations)} recommendations".center(WIDTH))
    print("-" * WIDTH)

    for rank, (song, score, explanation) in enumerate(recommendations, start=1):
        _print_result(rank, song, score, explanation)

    print()


if __name__ == "__main__":
    main()
