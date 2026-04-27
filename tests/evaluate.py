"""
tests/evaluate.py — Phase 3 evaluation metrics for VibeFinder 1.0

Usage (run from project root):
    python tests/evaluate.py
    python -m tests.evaluate

Metrics reported
----------------
precision@k     Fraction of top-k results that match the user's preferred
                genre OR preferred mood (proxy for relevance without ground
                truth labels).  N/A when neither preference is set.

genre diversity Unique genres represented in the top-k result set.
                Higher = more eclectic recommendations.

discovery rate  Fraction of top-k results whose genre does NOT match the
                user's primary genre.  Surfaces the exploitation/exploration
                tension: a score of 0% means the model only ever confirms
                what the user already likes.

catalog coverage
                Fraction of the 20-song catalog surfaced at least once
                across all profiles combined.  Low coverage indicates that
                certain songs are systematically buried by the scoring formula.

score distribution
                Per-profile mean, std-dev, min, and max of the raw scores
                returned by score_song().  A very narrow spread (low std)
                means the ranking is nearly random; a very wide spread
                means a small set of songs dominate every run.

weight sensitivity
                Head-to-head comparison of default weights vs. the
                experimental weight shift (genre÷2, energy×2) on the
                High-Energy Pop profile.  Concretely shows how changing a
                single hyperparameter reshuffles the top-5.
"""

import os
import sys
import math
import statistics
from typing import Dict, List, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.recommender import load_songs, recommend_songs, recommend_songs_weighted  # noqa: E402

# ---------------------------------------------------------------------------
# Profiles — identical to PROFILES in src/main.py
# ---------------------------------------------------------------------------

PROFILES: Dict[str, dict] = {
    "High-Energy Pop": {
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
    "Contradictory Energy+Acoustic": {
        "genre": "ambient",
        "mood": "intense",
        "energy": 0.92,
        "likes_acoustic": True,
        "prefers_instrumental": True,
        "target_valence": 0.30,
    },
    "Missing Genre (K-Pop)": {
        "genre": "kpop",
        "mood": "happy",
        "energy": 0.72,
        "likes_acoustic": False,
        "prefers_instrumental": False,
        "target_valence": 0.78,
    },
    "Ultra-Neutral Listener": {
        "genre": "",
        "mood": "",
        "energy": 0.5,
        "likes_acoustic": False,
        "prefers_instrumental": False,
        "target_valence": 0.5,
    },
}

EXPERIMENTAL_WEIGHTS = {
    "genre_pts":        1.0,
    "mood_pts":         1.5,
    "energy_pts":       3.0,
    "valence_pts":      1.0,
    "acoustic_pts":     1.0,
    "instrumental_pts": 0.8,
}

W = 70  # output width


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _is_relevant(song: dict, prefs: dict) -> bool | None:
    """
    Return True if the song satisfies the user's genre OR mood preference.
    Return None when the user expressed no genre/mood preference at all
    (evaluation is not meaningful for that profile).
    """
    genre = prefs.get("genre", "")
    mood  = prefs.get("mood", "")
    if not genre and not mood:
        return None
    return (bool(genre) and song["genre"] == genre) or \
           (bool(mood)  and song["mood"]  == mood)


def precision_at_k(
    results: List[Tuple[dict, float, str]], prefs: dict
) -> float | None:
    """Fraction of top-k results that are relevant.  None → N/A."""
    relevance = [_is_relevant(song, prefs) for song, _, _ in results]
    if all(r is None for r in relevance):
        return None
    hits = sum(1 for r in relevance if r is True)
    return hits / len(results)


def genre_diversity(results: List[Tuple[dict, float, str]]) -> int:
    """Number of unique genres in the result list."""
    return len({song["genre"] for song, _, _ in results})


def discovery_rate(
    results: List[Tuple[dict, float, str]], prefs: dict
) -> float | None:
    """Fraction of results whose genre does NOT match the user's preferred genre."""
    genre = prefs.get("genre", "")
    if not genre:
        return None
    mismatches = sum(1 for song, _, _ in results if song["genre"] != genre)
    return mismatches / len(results)


def score_stats(results: List[Tuple[dict, float, str]]) -> dict:
    scores = [s for _, s, _ in results]
    return {
        "mean": statistics.mean(scores),
        "std":  statistics.pstdev(scores),
        "min":  min(scores),
        "max":  max(scores),
    }


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

def _fmt_pct(value: float | None, precision: int = 0) -> str:
    if value is None:
        return "  N/A "
    return f"{value * 100:{5}.{precision}f}%"


def _fmt_score(value: float) -> str:
    return f"{value:.3f}"


def _score_bar(score: float, width: int = 16) -> str:
    filled = round(score * width)
    return "[" + "█" * filled + "░" * (width - filled) + "]"


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------

def section(title: str) -> None:
    print()
    print("─" * W)
    print(f"  {title}")
    print("─" * W)


def print_summary_table(
    all_results: Dict[str, List[Tuple[dict, float, str]]]
) -> None:
    section("METRICS SUMMARY  (k=5 per profile, catalog=20 songs)")
    header = f"  {'Profile':<32}  {'Prec@5':>7}  {'Genres':>6}  {'Discov':>7}  {'AvgScore':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    precision_vals = []
    for label, results in all_results.items():
        prefs = PROFILES[label]
        p     = precision_at_k(results, prefs)
        gd    = genre_diversity(results)
        dr    = discovery_rate(results, prefs)
        stats = score_stats(results)

        p_str  = _fmt_pct(p)
        dr_str = _fmt_pct(dr)

        print(f"  {label:<32}  {p_str}  {gd:>6}  {dr_str}  {stats['mean']:>8.3f}")
        if p is not None:
            precision_vals.append(p)

    print("  " + "-" * (len(header) - 2))
    macro_p  = statistics.mean(precision_vals) if precision_vals else None
    macro_gd = statistics.mean(
        [genre_diversity(r) for r in all_results.values()]
    )
    macro_avg_score = statistics.mean(
        [score_stats(r)["mean"] for r in all_results.values()]
    )
    print(f"  {'Macro-average (excl. N/A)':<32}  {_fmt_pct(macro_p)}  {macro_gd:>6.1f}  {'   —   ':>7}  {macro_avg_score:>8.3f}")


def print_catalog_coverage(
    all_results: Dict[str, List[Tuple[dict, float, str]]],
    all_songs: List[dict],
) -> None:
    section("CATALOG COVERAGE")
    surfaced_ids = {song["id"] for results in all_results.values()
                    for song, _, _ in results}
    coverage = len(surfaced_ids) / len(all_songs)
    never_surfaced = [s for s in all_songs if s["id"] not in surfaced_ids]

    print(f"  Songs surfaced: {len(surfaced_ids)} / {len(all_songs)}  ({coverage:.0%})")
    print()
    if never_surfaced:
        print(f"  Never recommended ({len(never_surfaced)} songs):")
        for s in never_surfaced:
            print(f"    • {s['title']:30s}  [{s['genre']:12s}  {s['mood']:12s}  energy={s['energy']:.2f}]")
    else:
        print("  All catalog songs were recommended at least once.")

    print()
    print(f"  Insight: {100 - coverage*100:.0f}% of the catalog is unreachable under the current")
    print("  profile set.  These songs may be structurally penalised by the")
    print("  weighted formula or simply have no matching profile in the demo set.")


def print_score_distribution(
    all_results: Dict[str, List[Tuple[dict, float, str]]]
) -> None:
    section("SCORE DISTRIBUTION")
    col = f"  {'Profile':<32}  {'Mean':>6}  {'Std':>6}  {'Min':>6}  {'Max':>6}  {'Spread':>6}"
    print(col)
    print("  " + "-" * (len(col) - 2))

    for label, results in all_results.items():
        st = score_stats(results)
        spread = st["max"] - st["min"]
        print(
            f"  {label:<32}  {st['mean']:>6.3f}  {st['std']:>6.3f}"
            f"  {st['min']:>6.3f}  {st['max']:>6.3f}  {spread:>6.3f}"
        )

    print()
    print("  Std-dev < 0.05 → ranking is nearly uniform (weights undifferentiated).")
    print("  Std-dev > 0.15 → strong separation; top results are clearly dominant.")


def print_per_profile_results(
    all_results: Dict[str, List[Tuple[dict, float, str]]]
) -> None:
    section("TOP-5 RECOMMENDATIONS PER PROFILE")
    for label, results in all_results.items():
        prefs = PROFILES[label]
        print(f"\n  [{label}]")
        print(f"  genre={prefs['genre'] or '(none)'}  mood={prefs['mood'] or '(none)'}"
              f"  energy={prefs['energy']}  valence={prefs['target_valence']}")
        for rank, (song, score, _) in enumerate(results, 1):
            bar      = _score_bar(score)
            relevant = _is_relevant(song, prefs)
            flag     = "✓" if relevant else ("·" if relevant is None else "✗")
            print(f"    #{rank} {flag}  {bar} {score:.2%}  "
                  f"{song['title']:28s}  [{song['genre']}, {song['mood']}]")


def print_weight_sensitivity(
    songs: List[dict],
    label: str = "High-Energy Pop",
    k: int = 5,
) -> None:
    section(f"WEIGHT SENSITIVITY  (profile: {label})")
    prefs = PROFILES[label]

    default_recs  = recommend_songs(prefs, songs, k=k)
    weighted_recs = recommend_songs_weighted(prefs, songs, k=k, weights=EXPERIMENTAL_WEIGHTS)

    print(f"  {'Rank':<5}  {'[A] Default  genre=2.0 energy=1.5':^38}  {'[B] Experimental  genre=1.0 energy=3.0':^38}")
    print("  " + "-" * 82)
    for i, ((sd, ss, _), (sw, sw_score, _)) in enumerate(
        zip(default_recs, weighted_recs), 1
    ):
        moved = "↑" if sw_score > ss else ("↓" if sw_score < ss else "=")
        print(
            f"  #{i:<4}  {sd['title']:28s} {ss:.2%}"
            f"    {moved}    {sw['title']:28s} {sw_score:.2%}"
        )

    # Rank-order overlap
    default_ids  = [sd["id"] for sd, _, _ in default_recs]
    weighted_ids = [sw["id"] for sw, _, _ in weighted_recs]
    overlap = len(set(default_ids) & set(weighted_ids))
    print()
    print(f"  Top-{k} set overlap: {overlap}/{k} songs appear in both rankings.")
    print(f"  Weight shift moved {k - overlap} song(s) into / out of the top-{k}.")
    print()
    print("  Interpretation: doubling the energy weight brings songs with a strong")
    print("  continuous energy match to the surface, even if they fail the genre")
    print("  check.  This demonstrates that categorical signals dominate the default")
    print("  configuration — a design choice worth revisiting if users report that")
    print("  recommendations feel 'genre-locked'.")


def print_insights(
    all_results: Dict[str, List[Tuple[dict, float, str]]], songs: List[dict]
) -> None:
    section("ACTIONABLE INSIGHTS")

    # 1. Genre lock detection
    locked_profiles = [
        label for label, results in all_results.items()
        if PROFILES[label]["genre"]
        and all(song["genre"] == PROFILES[label]["genre"]
                for song, _, _ in results)
    ]
    if locked_profiles:
        print("  [!] Genre lock detected — the following profiles received zero")
        print("      cross-genre discoveries in their top-5:")
        for lp in locked_profiles:
            print(f"        • {lp}")
        print("      Consider adding a diversity bonus: +0.05 to songs outside the")
        print("      primary genre that score ≥ 0.7 on energy + valence combined.")
    else:
        print("  [✓] No complete genre lock detected across demo profiles.")

    # 2. Coverage gap
    surfaced_ids   = {song["id"] for results in all_results.values()
                      for song, _, _ in results}
    unsurfaced_pct = (1 - len(surfaced_ids) / len(songs)) * 100
    print()
    if unsurfaced_pct > 30:
        print(f"  [!] {unsurfaced_pct:.0f}% of the catalog is never surfaced.")
        print("      Add a 'catalog exploration' profile (random genre, mid energy)")
        print("      or implement a popularity-floor that guarantees every song")
        print("      appears at least once per N recommendation sessions.")
    else:
        print(f"  [✓] Catalog coverage is reasonable ({100 - unsurfaced_pct:.0f}% surfaced).")

    # 3. Score compression
    low_spread_profiles = [
        label for label, results in all_results.items()
        if score_stats(results)["std"] < 0.05
    ]
    print()
    if low_spread_profiles:
        print("  [!] Low score spread (std < 0.05) in the following profiles:")
        for lsp in low_spread_profiles:
            st = score_stats(all_results[lsp])
            print(f"        • {lsp}  (std={st['std']:.3f})")
        print("      Low spread means rankings are nearly arbitrary — a small noise")
        print("      perturbation would reorder results.  Consider adding a")
        print("      tiebreaker signal (danceability or tempo proximity).")
    else:
        print("  [✓] All profiles show meaningful score differentiation (std ≥ 0.05).")

    # 4. Missing genre penalty summary
    missing_genre_profiles = [
        label for label in PROFILES
        if PROFILES[label]["genre"]
        and not any(s["genre"] == PROFILES[label]["genre"] for s in songs)
    ]
    print()
    if missing_genre_profiles:
        print("  [!] The following profiles request a genre absent from the catalog:")
        for mgp in missing_genre_profiles:
            print(f"        • {mgp}  (genre='{PROFILES[mgp]['genre']}')")
        print("      These users can never receive the max 2.0 genre bonus, capping")
        print("      their theoretical best score at  5.8 / 7.8 = 74.4%.")
        print("      Expanding the catalog or mapping near-equivalent genres would fix this.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    csv_path = os.path.join(PROJECT_ROOT, "data", "songs.csv")
    songs = load_songs(csv_path)

    print()
    print("=" * W)
    print("  VibeFinder 1.0 — Evaluation Report".center(W))
    print("=" * W)

    # Run all profiles
    all_results: Dict[str, List[Tuple[dict, float, str]]] = {}
    for label, prefs in PROFILES.items():
        all_results[label] = recommend_songs(prefs, songs, k=5)

    print_summary_table(all_results)
    print_catalog_coverage(all_results, songs)
    print_score_distribution(all_results)
    print_per_profile_results(all_results)
    print_weight_sensitivity(songs)
    print_insights(all_results, songs)

    print()
    print("=" * W)
    print("  End of evaluation report".center(W))
    print("=" * W)
    print()


if __name__ == "__main__":
    main()
