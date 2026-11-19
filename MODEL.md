# How the forecaster works

`main.py` is the whole entry: one file, Python standard library only, no
network, deterministic. It reads every `*.in` case in the input directory and
writes one `*.out` forecast per case.

## The quantity being forecast

Each trophy's Exophase earn rate is a share of the players the tracker
follows. That share moves for two reasons that fight each other:

- **Progression pushes rates up.** Players work through the game and earn
  trophies; the longer the game has been out, the larger the share that has
  reached each milestone.
- **Dilution pushes rates down.** New owners flood in faster than anyone can
  progress, so the denominator grows faster than the numerator. The house
  saw this on launch night: players tracked went 638 → 3,868 at the first
  refresh and every rate but the 100% one fell.

So a rate's path is a rise that decelerates toward the share of owners who
will ever earn that trophy. A straight line fitted to the whole history and
extended overshoots, because the early, steep part of the rise dominates the
fit. The forecaster corrects for that three ways.

## 1. Recent trend

For each trophy, the logit of its earn rate is regressed on
`x = ln(1 + hours since release)` over the last 24 hours of *distinct* reads
only. Log-time is the right clock because launch dynamics are fastest at
first and slow down. Distinct reads matter because Exophase refreshes in
jumps: the long flat stretches between refreshes are one observation, not
many (a read that only lacks a trophy counts as the same page, not a new
state). Reads are weighted by recency (24 h half-life). The slope is shrunk
toward its tier's mean slope (platinum / gold / silver / bronze,
pseudo-count 4), so a trophy with a noisy history borrows strength from its
tier. The median is anchored at the last read:

```
median_logit = logit(last rate) + slope * (x_target - x_last)
```

## 2. Damping

The extrapolated move is multiplied by 0.7. A rate's logit growth must
decelerate as it approaches its ceiling, so a linear extrapolation of the
recent slope overshoots on a 4.5-day horizon. The factor 0.7 is the
correction that minimised the pinball loss on synthetic launch weeks built
to match the two real anchors available (launch night: 638 players tracked,
Chapter 1 at 1.25%; 13:11 UTC on Nov 19, from Exophase's live page:
Chapter 1 at 0.73%, Chapter 2 at 0.14%, platinum at 0.03%, the
near-universal trophy at 99.99%). The same correction also won on a second,
differently-shaped synthetic world, so it is not tuned to one guessed
future. On a 90-minute horizon (the practice cases) the damping barely
matters, because the extrapolated move is tiny.

## 3. Honest spread

The spread is the regression's prediction standard error (residual variance
plus slope uncertainty, which grows with extrapolation distance), scaled by
0.9, floored at 0.25 and capped at 1.2 logit units. Quantiles are
`median + z_tau * sigma` on the logit scale, `z = (-1.645, -0.674, 0, 0.674,
1.645)`, mapped back to percent. On the synthetic launch weeks this gives
about 90% coverage of the 5%–95% band, as a calibrated forecast should.

## Special cases

- **Stale page.** When the history holds only one distinct state, the page
  has not refreshed in hours while the game is live, so the next refresh is
  a dilution jump down. The median is shifted −0.8 logit and the spread
  widened to at least 0.9. This is what the practice cases look like; the
  final's history is rich, so this branch does not fire there.
- **Near-universal trophy.** A trophy at 99.5% or above (the one every
  player earns in the first minutes) cannot rise and barely falls: tiny
  spread, median pinned near its last rate.

## Validation

Backtested on synthetic launch weeks (two independently-built worlds: one
where rates rise monotonically toward saturation, one where earners follow a
convolution of the owner inflow with a time-to-earn distribution, giving the
real dilution dip then recovery; three player-growth shapes and three
progression speeds each, several seeds). Mean pinball loss on the logit
scale, lower is better:

| world | this forecaster | house curve baseline | house baseline |
|---|---|---|---|
| convolution (dilution) | **0.29** | 0.40 | 0.84 |
| monotone saturation | **0.09** | 0.27 | 0.33 |

The house baselines are the ones the kit documents (flat median with a
random-walk spread, and a logit-linear trend on the last 24 h). The
forecaster beats both in every world tested, with a worst case of 0.39
against the curve baseline's worst of 0.60.

## What it does not do

- No network, no hard-coded rates, no trophy names used for anything but
  documentation. Every number comes from the case file it is given.
- No model calls, no randomness, no installed packages. Runs in ~0.05 s on
  the practice cases and ~0.04 s on a 359-read final-shaped case, far inside
  the 8 s limit, and writes byte-identical output on repeat runs.
