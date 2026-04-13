# Model Card: Music Recommender Simulation

## 1. Model Name

**VibeFinder 1.0** — Content-Based Music Recommender (Classroom Simulation)

---

## 2. Intended Use

VibeFinder suggests songs from a 20-song catalog based on a user's stated genre, mood, energy, and acoustic preferences. It is built for classroom exploration — not production use — and requires no listening history or training data.

---

## 3. How the Model Works

Each song is scored against the user profile across six signals: genre match (strongest), mood match, energy proximity, valence proximity, acoustic fit, and vocal fit. The six scores are summed and divided by the maximum possible total to produce a 0–1 ranking score. Songs are sorted highest-first and the top results are returned with a plain-language explanation.

---

## 4. Data

The catalog is `data/songs.csv` with **20 songs** across 16 genres and 9 moods. Most genres have only one representative song, so after a genre match the system must rely on energy and valence to differentiate. No lyrics, cultural context, or listening history is included, and many global genres (K-pop, Afrobeats, Bossa Nova) are absent.

---

## 5. Strengths

- **Explainable.** Every result includes a plain-language reason tied to specific signals.
- **Cold-start friendly.** Works immediately for any user who can state a preference — no history needed.
- **Coherent for clear profiles.** High-Energy Pop, Chill Lofi, and Deep Intense Rock all returned 97% top matches that matched intuition immediately.

---

## 6. Limitations and Bias

**Genre weight dominates.** At 2.0 points, a genre match outweighs a perfect mood + energy combination, which can suppress great songs from adjacent genres. The weight-shift experiment confirmed this: halving genre and doubling energy moved Rooftop Lights (indie pop, better energy fit) above Gym Hero (pop, weaker energy fit).

**Binary mood matching is too coarse.** "Chill" and "relaxed" score identically to a complete mismatch — a song tagged "relaxed" earns zero mood points for a "chill" user. Semantically close moods are treated as completely unrelated.

**Missing genres silently cap confidence.** The K-Pop profile could never earn genre points, capping its best possible score at ~70%. The system returns plausible-looking results with no warning that the user's core preference is unrepresented.

**Contradictory preferences produce quiet failures.** A user who wants high energy + acoustic + ambient gets results at only 52% with no indication their preferences conflict.

**Acousticness encodes production style bias.** Rewarding acoustic preference systematically buries highly-produced global genres (reggaeton, electronic) even when they match on every other signal.

---

## 7. Evaluation

| Profile | Top Result | Score | Surprise? |
|---------|-----------|-------|-----------|
| High-Energy Pop | Sunrise City | 97% | No |
| Chill Lofi | Library Rain | 97% | No |
| Deep Intense Rock | Storm Runner | 97% | No — only 1 rock song exists |
| Contradictory Energy+Acoustic | Spacewalk Thoughts | 52% | Yes — chill/ambient ranked #1 despite energy mismatch |
| Missing Genre (K-Pop) | Sunrise City | 70% | Mild — mood/valence carried it |
| Ultra-Neutral | Velvet Skyline | 45% | Yes — essentially arbitrary within a flat score band |

The weight-shift experiment (genre ÷2, energy ×2) showed Rooftop Lights climbing from #3 to #2, confirming that the default config is closer to a "genre filter with an energy tiebreaker" than a true multi-signal ranker.

---

## 8. Future Work

- **Soft mood/genre matching** — partial credit for semantically close labels ("chill" ≈ "relaxed", "indie pop" ≈ "pop").
- **Discovery bonus** — small score boost for out-of-genre songs that match well on 2+ continuous features.
- **Conflict detection** — warn the user when their preferences are internally contradictory before returning results.
- **Larger catalog** — at least 5–10 songs per genre to make rankings meaningful rather than single-song genre matches.

---

## 9. Personal Reflection

My biggest learning moment was realizing how much work goes into building even a simple model — decisions about what to weight, how to normalize, and what counts as a "match" compound quickly. I used Claude throughout the project, which was a huge help for the mathematics and implementation details, but I made a point of reading and understanding every piece of generated code before moving on. That habit of double-checking mattered: a few times the logic looked right on the surface but didn't behave as expected until I traced through it manually. What surprised me most was how a straightforward weighted formula — no neural networks, no training data — can still produce recommendations that *feel* personalized and intelligent for users with clear preferences. The algorithm is simple, but the output reads like it knows you. If I were to extend this project, I'd try implementing YouTube's two-stage deep neural network approach: a candidate generation network to narrow the catalog, followed by a ranking network that weights signals like watch time — a much more realistic model of how production recommenders actually work.
