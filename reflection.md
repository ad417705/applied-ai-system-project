# Reflection: Profile Comparisons

## High-Energy Pop vs. Chill Lofi

Both profiles hit 97% on their top result, but for completely opposite reasons. High-Energy Pop
rewarded low acousticness, high energy (0.82–0.93), and bright valence — landing Sunrise City and
Gym Hero. Chill Lofi rewarded high acousticness, low energy (0.35–0.42), and instrumental tracks
— landing Library Rain and Midnight Coding. The same scoring formula produced two totally different
worlds, which makes sense: the user preferences are almost mirror images of each other on every
continuous axis.

## Deep Intense Rock vs. Contradictory Energy+Acoustic

Rock got Storm Runner at 97% because there is exactly one rock song in the catalog — a perfect
genre+mood hit. Contradictory Energy+Acoustic topped out at 52% because the user asked for high
energy (0.92) AND acoustic AND ambient, but ambient songs in the catalog are all low energy (0.19–
0.28). The system quietly returned Spacewalk Thoughts — a chill, acoustic ambient song — as #1
despite a massive energy mismatch. This is the system being "tricked": it found the best available
compromise but presented it with the same confidence bar format as a genuine match.

## Missing Genre (K-Pop) vs. Ultra-Neutral Listener

K-Pop topped out at 70% — never earning a genre point, but still producing reasonable mood/energy
matches (Sunrise City, Rooftop Lights). The Ultra-Neutral Listener topped out at 45%, and the
top-ranked song (Velvet Skyline, R&B) was essentially random within a narrow band — it won only
because its energy (0.55) was closest to the neutral target (0.5) and it happened to score well on
the vocal preference. The difference tells you something important: a missing genre is a recoverable
failure (mood and energy still guide the ranking), but having *no* preferences at all breaks the
scoring model's ability to differentiate — every song ends up roughly tied on the signals that
actually matter.

## Experiment: Default Weights vs. Energy-Boosted Weights (High-Energy Pop)

With default weights (genre=2.0, energy=1.5), Gym Hero ranked #2 and Rooftop Lights ranked #3.
After doubling energy and halving genre (genre=1.0, energy=3.0), Rooftop Lights climbed to #2 and
Gym Hero dropped to #3. Rooftop Lights (indie pop, energy 0.76) is actually a closer energy match
than Gym Hero (pop, energy 0.93), but in the default config the genre label "pop" on Gym Hero
outweighed the energy gap. This shows that the default system is more of a "genre filter with an
energy tiebreaker" than a true multi-signal ranker — a real recommender would want those two signals
to compete more evenly.
