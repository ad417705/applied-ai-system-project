"""
recommender.py — Music Recommender Simulation
==============================================
Core module for the content-based music recommender.

This module contains:
  - Song          dataclass representing a track and its audio attributes
  - UserProfile   dataclass representing a listener's taste preferences
  - score_song    scores one Song against one UserProfile (0–1 scale)
  - Recommender   OOP class that ranks a catalog and explains results
  - load_songs    reads data/songs.csv and returns a list of dicts
  - recommend_songs  functional entry point used by src/main.py

Scoring approach (Algorithm Recipe)
------------------------------------
Each song receives up to 7.8 raw points across six signals:

    genre match    2.0 pts  — exact string match (highest weight)
    mood match     1.5 pts  — exact string match
    energy fit     1.5 pts  — Gaussian proximity kernel (σ=0.25)
    valence fit    1.0 pts  — Gaussian proximity kernel (σ=0.25)
    acoustic fit   1.0 pts  — directional: song value or 1−value
    instrumental   0.8 pts  — directional: song value or 1−value

    final score = raw_points / 7.8   → [0.0, 1.0]

Categorical signals (genre, mood) use binary on/off points because label
matches are all-or-nothing — there is no "almost pop."  Continuous signals
(energy, valence) use a Gaussian kernel so small mismatches are forgiven
gently while large gaps are penalized steeply.
"""

import csv
import math
from typing import List, Dict, Tuple
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Song:
    """
    Represents a single track and all of its audio/metadata attributes.

    Identifiers
    -----------
    id          : unique integer row id from songs.csv
    title       : song title string
    artist      : artist or project name

    Categorical attributes (used for exact matching)
    ------------------------------------------------
    genre       : style label, e.g. "pop", "lofi", "rock", "R&B"
    mood        : emotional label, e.g. "happy", "chill", "intense"

    Continuous attributes (used for proximity scoring, all in [0, 1] unless noted)
    -------------------------------------------------------------------------------
    energy          : overall intensity — 0.0 (calm) to 1.0 (high intensity)
    tempo_bpm       : beats per minute; not used in scoring but available for filtering
    valence         : emotional positiveness — 0.0 (dark/sad) to 1.0 (bright/joyful)
    danceability    : rhythmic groove — 0.0 (non-danceable) to 1.0 (very danceable)
    acousticness    : organic vs. produced — 0.0 (electronic) to 1.0 (fully acoustic)
    instrumentalness: vocal presence — 0.0 (vocal-heavy) to 1.0 (fully instrumental)
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float
    instrumentalness: float = 0.5


@dataclass
class UserProfile:
    """
    Represents a listener's taste preferences used to score and rank songs.

    Categorical preferences (matched exactly against Song fields)
    -------------------------------------------------------------
    favorite_genre      : the genre the user most wants to hear, e.g. "lofi"
    favorite_mood       : the mood the user is in, e.g. "chill"

    Continuous preferences (compared to Song attributes via proximity)
    ------------------------------------------------------------------
    target_energy       : desired energy level, 0.0 (calm) to 1.0 (intense)
    target_valence      : desired emotional tone, 0.0 (dark/sad) to 1.0 (bright/joyful)

    Binary preferences (flip a directional weight in the score)
    -----------------------------------------------------------
    likes_acoustic      : True → reward high acousticness; False → reward low acousticness
    prefers_instrumental: True → reward high instrumentalness; False → reward vocal-heavy tracks

    Default values allow existing tests to construct UserProfile with four
    positional arguments without breaking when the new fields are added.
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool
    prefers_instrumental: bool = False
    target_valence: float = 0.5


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

# The maximum possible raw score — the sum of every signal's point ceiling.
# Dividing by this value normalizes any raw total into the [0, 1] range.
_MAX_RAW = 2.0 + 1.5 + 1.5 + 1.0 + 1.0 + 0.8  # 7.8


def _gaussian(song_val: float, target: float, sigma: float = 0.25) -> float:
    """
    Gaussian (bell-curve) proximity kernel for continuous attribute matching.

    Returns 1.0 when song_val exactly equals target, and decays smoothly
    toward 0.0 as the gap widens.  Sigma (σ) controls the width of the
    forgiveness zone — with σ=0.25, a difference of 0.25 still earns ~0.61,
    while a difference of 0.50 only earns ~0.14.

    Used for energy and valence where nearby values should still score well,
    but large mismatches deserve a steep penalty.

    Parameters
    ----------
    song_val : the song's value for a given attribute (e.g. song.energy)
    target   : the user's preferred value for that attribute
    sigma    : standard deviation of the kernel; lower = tighter tolerance
    """
    return math.exp(-((song_val - target) ** 2) / (2 * sigma ** 2))


# ---------------------------------------------------------------------------
# Core scoring function — Algorithm Recipe
# ---------------------------------------------------------------------------

def score_song(song: Song, user: UserProfile) -> float:
    """
    Score a single Song against a UserProfile and return a value in [0, 1].

    This is the "judge" function — every song in the catalog is passed
    through here so that recommend_songs can rank them numerically.

    Algorithm Recipe
    ----------------
    Signal              Max pts   Method
    ──────────────────  ───────   ──────────────────────────────────────────
    genre match         2.0       2.0 if exact label match, else 0.0
    mood match          1.5       1.5 if exact label match, else 0.0
    energy fit          1.5       1.5 × Gaussian(song.energy, target_energy)
    valence fit         1.0       1.0 × Gaussian(song.valence, target_valence)
    acoustic fit        1.0       song.acousticness        if likes_acoustic
                                  (1 − song.acousticness)  otherwise
    instrumental fit    0.8       song.instrumentalness        if prefers_instrumental
                                  (1 − song.instrumentalness)  otherwise
    ──────────────────  ───────
    Total max raw       7.8

    final score = sum of all signal points / 7.8

    Parameters
    ----------
    song : Song dataclass instance to be evaluated
    user : UserProfile dataclass instance representing listener preferences

    Returns
    -------
    float between 0.0 (no match at all) and 1.0 (perfect match on every signal)
    """
    # Categorical signals — binary: full points or zero
    genre_pts = 2.0 if song.genre == user.favorite_genre else 0.0
    mood_pts  = 1.5 if song.mood  == user.favorite_mood  else 0.0

    # Continuous signals — Gaussian kernel rewards proximity, penalizes distance
    energy_pts  = 1.5 * _gaussian(song.energy,  user.target_energy)
    valence_pts = 1.0 * _gaussian(song.valence, user.target_valence)

    # Directional binary signals — reward the "right end" of the 0–1 scale
    acoustic_pts = (
        1.0 * song.acousticness if user.likes_acoustic
        else 1.0 * (1.0 - song.acousticness)
    )
    instrumental_pts = (
        0.8 * song.instrumentalness if user.prefers_instrumental
        else 0.8 * (1.0 - song.instrumentalness)
    )

    raw = genre_pts + mood_pts + energy_pts + valence_pts + acoustic_pts + instrumental_pts
    return raw / _MAX_RAW


# ---------------------------------------------------------------------------
# OOP interface — Recommender
# ---------------------------------------------------------------------------

class Recommender:
    """
    Object-oriented wrapper around score_song for ranking and explanation.

    Holds the full song catalog and exposes two methods:
      - recommend()              → sorted list of top-k Song objects
      - explain_recommendation() → human-readable string for one song

    Required by tests/test_recommender.py
    """

    def __init__(self, songs: List[Song]):
        """
        Parameters
        ----------
        songs : full catalog of Song dataclass instances to rank against
        """
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        """
        Score every song in the catalog and return the top-k matches.

        Uses sorted() (not .sort()) so self.songs is never mutated —
        the catalog stays in its original order for future calls.

        Parameters
        ----------
        user : UserProfile describing the listener's current preferences
        k    : maximum number of songs to return (default 5)

        Returns
        -------
        List of up to k Song objects, ranked highest score first
        """
        scored = sorted(self.songs, key=lambda s: score_song(s, user), reverse=True)
        return scored[:k]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        """
        Build a plain-language explanation for why a song was recommended.

        Checks each scoring signal individually and appends a reason phrase
        when the song performs well on that signal for this user.  If no
        signal fires strongly enough, returns a generic fallback message.

        Parameters
        ----------
        user : UserProfile the song was scored against
        song : the specific Song to explain

        Returns
        -------
        A human-readable string starting with "Recommended because it ..."
        or "Closest match available in the catalog." as a fallback.
        """
        reasons = []

        # Categorical matches — only mention when they actually match
        if song.genre == user.favorite_genre:
            reasons.append(f"matches your preferred genre ({song.genre})")
        if song.mood == user.favorite_mood:
            reasons.append(f"matches your preferred mood ({song.mood})")

        # Continuous proximity — mention when the gap is small (< 0.15)
        if abs(song.energy - user.target_energy) < 0.15:
            reasons.append(
                f"energy ({song.energy:.2f}) is close to your target ({user.target_energy:.2f})"
            )
        if abs(song.valence - user.target_valence) < 0.15:
            mood_word = "bright/joyful" if user.target_valence > 0.6 else "dark/introspective"
            reasons.append(
                f"emotional tone ({song.valence:.2f}) matches your {mood_word} preference"
            )

        # Directional binary signals — mention when the song is clearly on the right end
        if user.likes_acoustic and song.acousticness > 0.6:
            reasons.append(f"has a strong acoustic feel ({song.acousticness:.2f})")
        elif not user.likes_acoustic and song.acousticness < 0.4:
            reasons.append(f"has a produced/electronic sound ({song.acousticness:.2f})")

        if user.prefers_instrumental and song.instrumentalness > 0.6:
            reasons.append(f"is primarily instrumental ({song.instrumentalness:.2f})")
        elif not user.prefers_instrumental and song.instrumentalness < 0.3:
            reasons.append(f"features strong vocals ({song.instrumentalness:.2f})")

        if reasons:
            return "Recommended because it " + ", and ".join(reasons) + "."
        return "Closest match available in the catalog."


# ---------------------------------------------------------------------------
# Functional API — used by src/main.py
# ---------------------------------------------------------------------------

def load_songs(csv_path: str) -> List[Dict]:
    """
    Read songs.csv and return every row as a typed dictionary.

    Uses csv.DictReader so the header row becomes the dict keys
    automatically.  All numeric fields are cast from str to their
    correct Python types so that score_song can do math on them.

    Parameters
    ----------
    csv_path : path to the CSV file, relative to the project root
               (e.g. "data/songs.csv")

    Returns
    -------
    List of dicts, one per song row, with keys:
        id (int), title (str), artist (str), genre (str), mood (str),
        energy (float), tempo_bpm (float), valence (float),
        danceability (float), acousticness (float), instrumentalness (float)
    """
    songs = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            songs.append({
                "id":               int(row["id"]),
                "title":            row["title"],
                "artist":           row["artist"],
                "genre":            row["genre"],
                "mood":             row["mood"],
                "energy":           float(row["energy"]),
                "tempo_bpm":        float(row["tempo_bpm"]),
                "valence":          float(row["valence"]),
                "danceability":     float(row["danceability"]),
                "acousticness":     float(row["acousticness"]),
                "instrumentalness": float(row["instrumentalness"]),
            })
    print(f"Loaded {len(songs)} songs from {csv_path}")
    return songs


def _song_dict_to_obj(song: Dict) -> Song:
    """
    Convert a song dictionary (as returned by load_songs) into a Song dataclass.

    This adapter lets the functional API (which works with plain dicts) reuse
    the same score_song and explain_recommendation logic that expects Song objects.
    """
    return Song(
        id=song["id"],
        title=song["title"],
        artist=song["artist"],
        genre=song["genre"],
        mood=song["mood"],
        energy=song["energy"],
        tempo_bpm=song["tempo_bpm"],
        valence=song["valence"],
        danceability=song["danceability"],
        acousticness=song["acousticness"],
        instrumentalness=song["instrumentalness"],
    )


def _prefs_dict_to_profile(user_prefs: Dict) -> UserProfile:
    """
    Convert a user-preferences dictionary (as used in main.py) into a UserProfile.

    .get() with defaults ensures that callers do not need to supply every key —
    omitted keys fall back to neutral mid-range values so scoring still works.

    Expected keys: genre, mood, energy, likes_acoustic,
                   prefers_instrumental, target_valence
    """
    return UserProfile(
        favorite_genre=user_prefs.get("genre", ""),
        favorite_mood=user_prefs.get("mood", ""),
        target_energy=user_prefs.get("energy", 0.5),
        likes_acoustic=user_prefs.get("likes_acoustic", False),
        prefers_instrumental=user_prefs.get("prefers_instrumental", False),
        target_valence=user_prefs.get("target_valence", 0.5),
    )


def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """
    Score every song in the catalog and return the top-k recommendations.

    This is the main functional entry point called by src/main.py.
    It bridges the dict-based API used in main.py with the dataclass-based
    scoring and explanation logic defined above.

    How it works
    ------------
    1. Convert user_prefs dict → UserProfile dataclass (_prefs_dict_to_profile)
    2. For every song dict in the catalog:
         a. Convert it to a Song dataclass    (_song_dict_to_obj)
         b. Score it against the user profile (score_song)
         c. Generate a plain-language reason  (explain_recommendation)
         d. Bundle all three into a tuple:    (song_dict, score, explanation)
    3. sorted() ranks the full list by score, highest first.
       sorted() is used instead of .sort() so the original songs list
       is never mutated — it returns a brand-new sorted list.
    4. [:k] slices off the top results.

    Parameters
    ----------
    user_prefs : dict of listener preferences (see _prefs_dict_to_profile for keys)
    songs      : list of song dicts as returned by load_songs
    k          : number of top results to return (default 5)

    Returns
    -------
    List of up to k tuples: (song_dict, score_float, explanation_str),
    sorted from highest score to lowest.
    """
    user = _prefs_dict_to_profile(user_prefs)
    rec = Recommender([])

    # List comprehension: score and explain every song in one pass.
    # The walrus operator (:=) assigns song_obj once per iteration so it can
    # be passed to both score_song and explain_recommendation without a
    # second conversion call.
    scored = [
        (song_dict, score_song(song_obj := _song_dict_to_obj(song_dict), user),
         rec.explain_recommendation(user, song_obj))
        for song_dict in songs
    ]

    # sorted() returns a new ranked list; the original catalog is untouched.
    return sorted(scored, key=lambda entry: entry[1], reverse=True)[:k]
