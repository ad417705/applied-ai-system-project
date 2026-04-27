"""
Phase 2 test suite for VibeFinder 1.0.

Coverage targets (per architecture.md):
  - Edge cases: k > catalog, k=0, missing genre, empty preferences
  - Scoring bounds: score always in [0, 1], perfect match = 1.0
  - Scoring correctness: genre > mood weight, Gaussian proximity, directional signals
  - Explanation accuracy: matches trigger reasons, fallback fires when nothing matches
  - Weighted variant ordering: weight shift reorders results
"""

import math
import pytest
from src.recommender import (
    Song,
    UserProfile,
    Recommender,
    score_song,
    recommend_songs,
    recommend_songs_weighted,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def make_song(
    id=1,
    title="Test Song",
    artist="Test Artist",
    genre="pop",
    mood="happy",
    energy=0.8,
    tempo_bpm=120,
    valence=0.9,
    danceability=0.8,
    acousticness=0.2,
    instrumentalness=0.05,
) -> Song:
    return Song(
        id=id, title=title, artist=artist, genre=genre, mood=mood,
        energy=energy, tempo_bpm=tempo_bpm, valence=valence,
        danceability=danceability, acousticness=acousticness,
        instrumentalness=instrumentalness,
    )


def make_user(
    genre="pop",
    mood="happy",
    energy=0.8,
    likes_acoustic=False,
    prefers_instrumental=False,
    target_valence=0.5,
) -> UserProfile:
    return UserProfile(
        favorite_genre=genre,
        favorite_mood=mood,
        target_energy=energy,
        likes_acoustic=likes_acoustic,
        prefers_instrumental=prefers_instrumental,
        target_valence=target_valence,
    )


def make_small_catalog() -> list:
    return [
        make_song(id=1, genre="pop", mood="happy", energy=0.8, valence=0.84,
                  acousticness=0.18, instrumentalness=0.04),
        make_song(id=2, title="Chill Lofi Loop", genre="lofi", mood="chill",
                  energy=0.4, valence=0.6, acousticness=0.9, instrumentalness=0.78),
    ]


def make_small_recommender() -> Recommender:
    return Recommender(make_small_catalog())


# ---------------------------------------------------------------------------
# 1. Scoring bounds — score is always in [0, 1]
# ---------------------------------------------------------------------------

def test_score_song_bounds_perfect_match_is_1():
    """A song that exactly satisfies every signal should score 1.0."""
    song = make_song(
        genre="pop", mood="happy",
        energy=0.80, valence=0.80,
        acousticness=0.0, instrumentalness=0.0,
    )
    user = make_user(
        genre="pop", mood="happy",
        energy=0.80, likes_acoustic=False,
        prefers_instrumental=False, target_valence=0.80,
    )
    score = score_song(song, user)
    assert math.isclose(score, 1.0, abs_tol=1e-9)


def test_score_song_bounds_never_exceeds_1():
    """Score must never exceed 1.0 regardless of inputs."""
    song = make_song(energy=0.5, valence=0.5, acousticness=1.0, instrumentalness=1.0)
    user = make_user(energy=0.5, target_valence=0.5, likes_acoustic=True, prefers_instrumental=True)
    assert score_song(song, user) <= 1.0


def test_score_song_bounds_never_below_0():
    """Score must never go below 0.0 for any combination of inputs."""
    song = make_song(genre="rock", mood="intense", energy=0.0, valence=0.0,
                     acousticness=1.0, instrumentalness=1.0)
    user = make_user(genre="pop", mood="happy", energy=1.0, likes_acoustic=False,
                     prefers_instrumental=False, target_valence=1.0)
    assert score_song(song, user) >= 0.0


def test_score_song_bounds_all_catalog_songs():
    """Every song in the 20-song catalog scores in [0, 1] for a fixed user profile."""
    import csv, os
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "songs.csv")
    user = make_user(genre="pop", mood="happy", energy=0.75, likes_acoustic=False,
                     prefers_instrumental=False, target_valence=0.75)
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            song = Song(
                id=int(row["id"]), title=row["title"], artist=row["artist"],
                genre=row["genre"], mood=row["mood"],
                energy=float(row["energy"]), tempo_bpm=float(row["tempo_bpm"]),
                valence=float(row["valence"]), danceability=float(row["danceability"]),
                acousticness=float(row["acousticness"]),
                instrumentalness=float(row["instrumentalness"]),
            )
            s = score_song(song, user)
            assert 0.0 <= s <= 1.0, f"Song '{song.title}' scored {s}"


# ---------------------------------------------------------------------------
# 2. Scoring correctness — signal weights and proximity
# ---------------------------------------------------------------------------

def test_score_song_genre_match_outweighs_mood_only():
    """A genre match (2.0 pts) should produce a higher score than a mood-only match (1.5 pts)."""
    base = {"energy": 0.5, "valence": 0.5, "acousticness": 0.5, "instrumentalness": 0.5,
            "tempo_bpm": 100, "danceability": 0.5}

    song_genre_match = make_song(genre="pop", mood="intense", **base)
    song_mood_match  = make_song(genre="rock", mood="happy", **base)

    user = make_user(genre="pop", mood="happy", energy=0.5, target_valence=0.5)

    assert score_song(song_genre_match, user) > score_song(song_mood_match, user)


def test_score_song_energy_gaussian_peak_at_target():
    """Energy score peaks exactly at target and decays symmetrically."""
    user = make_user(genre="", mood="", energy=0.6, target_valence=0.5)

    at_target  = make_song(genre="x", mood="x", energy=0.6, valence=0.5,
                           acousticness=0.5, instrumentalness=0.5)
    off_target = make_song(genre="x", mood="x", energy=0.9, valence=0.5,
                           acousticness=0.5, instrumentalness=0.5)

    assert score_song(at_target, user) > score_song(off_target, user)


def test_score_song_acoustic_rewards_high_when_user_likes_acoustic():
    """likes_acoustic=True should reward a high-acousticness song over a low one."""
    high_acoustic = make_song(acousticness=0.95, instrumentalness=0.5,
                              genre="x", mood="x", energy=0.5, valence=0.5)
    low_acoustic  = make_song(acousticness=0.05, instrumentalness=0.5,
                              genre="x", mood="x", energy=0.5, valence=0.5)
    user = make_user(genre="", mood="", energy=0.5, likes_acoustic=True, target_valence=0.5)

    assert score_song(high_acoustic, user) > score_song(low_acoustic, user)


def test_score_song_acoustic_rewards_low_when_user_dislikes_acoustic():
    """likes_acoustic=False should reward a low-acousticness (electronic) song."""
    high_acoustic = make_song(acousticness=0.95, instrumentalness=0.5,
                              genre="x", mood="x", energy=0.5, valence=0.5)
    low_acoustic  = make_song(acousticness=0.05, instrumentalness=0.5,
                              genre="x", mood="x", energy=0.5, valence=0.5)
    user = make_user(genre="", mood="", energy=0.5, likes_acoustic=False, target_valence=0.5)

    assert score_song(low_acoustic, user) > score_song(high_acoustic, user)


def test_score_song_instrumental_directional():
    """prefers_instrumental=True rewards high instrumentalness; False rewards low."""
    vocal   = make_song(instrumentalness=0.02, acousticness=0.5,
                        genre="x", mood="x", energy=0.5, valence=0.5)
    instru  = make_song(instrumentalness=0.95, acousticness=0.5,
                        genre="x", mood="x", energy=0.5, valence=0.5)

    user_wants_inst  = make_user(genre="", mood="", energy=0.5,
                                 prefers_instrumental=True, target_valence=0.5)
    user_wants_vocal = make_user(genre="", mood="", energy=0.5,
                                 prefers_instrumental=False, target_valence=0.5)

    assert score_song(instru, user_wants_inst) > score_song(vocal, user_wants_inst)
    assert score_song(vocal, user_wants_vocal) > score_song(instru, user_wants_vocal)


# ---------------------------------------------------------------------------
# 3. Recommender class — edge cases
# ---------------------------------------------------------------------------

def test_recommend_returns_songs_sorted_by_score():
    """Top result should be the song that matches genre + mood preferences."""
    user = make_user(genre="pop", mood="happy", energy=0.8, likes_acoustic=False)
    rec = make_small_recommender()
    results = rec.recommend(user, k=2)

    assert len(results) == 2
    assert results[0].genre == "pop"
    assert results[0].mood == "happy"


def test_recommend_k_larger_than_catalog_returns_all():
    """Requesting more results than catalog size should return the whole catalog silently."""
    rec = make_small_recommender()
    user = make_user()
    results = rec.recommend(user, k=999)
    assert len(results) == len(rec.songs)


def test_recommend_k_zero_returns_empty_list():
    """k=0 should return an empty list, not an error."""
    rec = make_small_recommender()
    user = make_user()
    results = rec.recommend(user, k=0)
    assert results == []


def test_recommend_does_not_mutate_catalog():
    """recommend() must never reorder the internal songs list."""
    rec = make_small_recommender()
    original_ids = [s.id for s in rec.songs]

    user = make_user(genre="lofi", mood="chill", energy=0.4)
    rec.recommend(user, k=2)

    assert [s.id for s in rec.songs] == original_ids


# ---------------------------------------------------------------------------
# 4. Explanation accuracy
# ---------------------------------------------------------------------------

def test_explain_recommendation_returns_non_empty_string():
    user = make_user(genre="pop", mood="happy", energy=0.8, likes_acoustic=False)
    rec = make_small_recommender()
    explanation = rec.explain_recommendation(user, rec.songs[0])
    assert isinstance(explanation, str)
    assert explanation.strip() != ""


def test_explain_mentions_genre_when_matched():
    """Explanation should reference the genre when it exactly matches."""
    user = make_user(genre="pop", mood="happy", energy=0.8)
    rec = make_small_recommender()
    pop_song = next(s for s in rec.songs if s.genre == "pop")
    explanation = rec.explain_recommendation(user, pop_song)
    assert "pop" in explanation.lower()


def test_explain_mentions_mood_when_matched():
    """Explanation should reference the mood when it exactly matches."""
    user = make_user(genre="lofi", mood="chill", energy=0.4)
    rec = make_small_recommender()
    chill_song = next(s for s in rec.songs if s.mood == "chill")
    explanation = rec.explain_recommendation(user, chill_song)
    assert "chill" in explanation.lower()


def test_explain_fallback_when_nothing_matches_strongly():
    """When no signal fires, explanation should fall back to the generic message.

    Conditions to suppress every reason branch:
      - genre mismatch → no genre reason
      - mood mismatch  → no mood reason
      - energy 0.9 vs target 0.5 → diff 0.4 ≥ 0.15 threshold → no energy reason
      - valence 0.1 vs target 0.5 → diff 0.4 ≥ 0.15 threshold → no valence reason
      - acousticness 0.5, likes_acoustic=False → 0.5 ≥ 0.4 → no acoustic reason
      - instrumentalness 0.5, prefers_instrumental=False → 0.5 ≥ 0.3 → no instrumental reason
    """
    song = make_song(
        genre="metal", mood="angry",
        energy=0.9, valence=0.1,
        acousticness=0.5, instrumentalness=0.5,
    )
    user = make_user(genre="jazz", mood="relaxed", energy=0.5, target_valence=0.5,
                     likes_acoustic=False, prefers_instrumental=False)
    rec = Recommender([song])
    explanation = rec.explain_recommendation(user, song)
    assert explanation == "Closest match available in the catalog."


# ---------------------------------------------------------------------------
# 5. Functional API — recommend_songs and weighted variant
# ---------------------------------------------------------------------------

def test_recommend_songs_returns_correct_tuple_structure():
    """recommend_songs must return (dict, float, str) tuples sorted by score desc."""
    prefs = {"genre": "pop", "mood": "happy", "energy": 0.8,
             "likes_acoustic": False, "prefers_instrumental": False, "target_valence": 0.8}
    songs = [s.__dict__ for s in make_small_catalog()]

    results = recommend_songs(prefs, songs, k=2)

    assert len(results) == 2
    for song_dict, score, explanation in results:
        assert isinstance(song_dict, dict)
        assert 0.0 <= score <= 1.0
        assert isinstance(explanation, str) and explanation.strip() != ""

    scores = [r[1] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_recommend_songs_missing_genre_still_returns_k_results():
    """A genre not in the catalog should never cause fewer than k results."""
    prefs = {"genre": "kpop", "mood": "happy", "energy": 0.72,
             "likes_acoustic": False, "prefers_instrumental": False, "target_valence": 0.78}
    songs = [s.__dict__ for s in make_small_catalog()]

    results = recommend_songs(prefs, songs, k=2)
    assert len(results) == 2


def test_recommend_songs_empty_preferences_all_scores_uniform_range():
    """Ultra-neutral preferences produce scores that are all in a narrow mid-range band."""
    prefs = {"genre": "", "mood": "", "energy": 0.5,
             "likes_acoustic": False, "prefers_instrumental": False, "target_valence": 0.5}
    songs = [s.__dict__ for s in make_small_catalog()]

    results = recommend_songs(prefs, songs, k=2)
    scores = [r[1] for r in results]
    # No categorical bonus fires; scores should all be well below 0.7
    assert all(s < 0.7 for s in scores)


def test_recommend_songs_weighted_reorders_results():
    """Doubling energy weight and halving genre weight should change the ranking.

    Song A (pop/happy) wins by default because genre+mood give 3.5 categorical pts,
    but its energy=0.10 is far from target=0.85 (gaussian ≈ 0.011).
    Song B (rock/intense) gets zero categorical pts but matches energy perfectly.

    Verified math:
      DEFAULT  → A=0.694, B=0.436  → A wins
      WEIGHTED → A=0.534, B=0.590  → B wins (energy×2 outweighs genre÷2)
    """
    prefs = {"genre": "pop", "mood": "happy", "energy": 0.85,
             "likes_acoustic": False, "prefers_instrumental": False, "target_valence": 0.5}

    catalog = [
        make_song(id=1, genre="pop", mood="happy", energy=0.10, valence=0.5,
                  acousticness=0.5, instrumentalness=0.5),
        make_song(id=2, title="High Energy Non-Pop", genre="rock", mood="intense",
                  energy=0.85, valence=0.5, acousticness=0.5, instrumentalness=0.5),
    ]
    song_dicts = [s.__dict__ for s in catalog]

    default_top = recommend_songs(prefs, song_dicts, k=1)[0][0]["id"]

    weights = {"genre_pts": 1.0, "mood_pts": 1.5, "energy_pts": 3.0,
               "valence_pts": 1.0, "acoustic_pts": 1.0, "instrumental_pts": 0.8}
    weighted_top = recommend_songs_weighted(prefs, song_dicts, k=1, weights=weights)[0][0]["id"]

    assert default_top == 1, "Default should pick the pop/genre-match song"
    assert weighted_top == 2, "With energy×2, perfect-energy rock song should win"
    assert default_top != weighted_top


def test_recommend_songs_weighted_none_behaves_like_default():
    """Passing weights=None to recommend_songs_weighted must match recommend_songs exactly."""
    prefs = {"genre": "pop", "mood": "happy", "energy": 0.8,
             "likes_acoustic": False, "prefers_instrumental": False, "target_valence": 0.8}
    songs = [s.__dict__ for s in make_small_catalog()]

    default_scores  = [r[1] for r in recommend_songs(prefs, songs, k=2)]
    weighted_scores = [r[1] for r in recommend_songs_weighted(prefs, songs, k=2, weights=None)]

    assert default_scores == weighted_scores
