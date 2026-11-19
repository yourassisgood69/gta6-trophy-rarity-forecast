"""GTA VI trophy-rarity forecaster (make.fun gta6-trophy-rarity-forecast-01).

Reads one case per run (the house's reads of Exophase's GTA VI PS5 trophy page
before the case checkpoint) and writes, for every trophy, five quantiles of its
earn rate at the case target. Python 3.13, standard library only, no network,
deterministic: the same input bytes give the same output bytes.

Model
-----
An Exophase earn rate is a share of the players the tracker follows, and that
share moves for two reasons that fight each other: players progress through
the game (rates rise) and new owners flood in faster than anyone can progress
(rates fall -- the launch-night refresh proved it: players tracked 638 ->
3,868, every rate but the 100% one lower). So a rate's path is a rise that
decelerates toward the share of owners who will ever earn that trophy. A
straight line fitted to its whole history and extended therefore overshoots:
the early, steep part of the rise dominates the fit. The forecaster corrects
for that in three ways, all estimated from the case file it is given:

1. Recent trend. For each trophy, the logit of its earn rate is regressed on
   x = ln(1 + hours since release) over the last 24 hours of *distinct* reads
   only. Log-time is the right clock because launch dynamics are fastest at
   first and slow down. Distinct reads matter because Exophase refreshes in
   jumps: the long flat stretches between refreshes are one observation, not
   many. Reads are weighted by recency (12 h half-life, so the trend is set
   by the most recent reads, when the balance of progression and dilution is
   closest to the balance the forecast extrapolates). Each trophy keeps its
   OWN slope: no shrinkage to a tier mean. The tiers pool trophies whose
   true progression speeds differ wildly (a 15-minute early trophy and a
   300-hour grind are both bronze), so shrinking toward the tier mean only
   biases every slope, and the final case's history is rich enough -- dozens
   of distinct reads per trophy -- that each slope is well identified on its
   own. Backtests on synthetic final weeks confirm it: dropping the
   shrinkage cuts the mean pinball loss about a tenth, on every world
   tested. The median is anchored at the last read: last logit + slope *
   (x_target - x_last).

2. Damping. The extrapolated move is multiplied by 0.7. A rate's logit growth
   must decelerate as it approaches its ceiling, so a linear extrapolation of
   the recent slope overshoots on a 4.5-day horizon; 0.7 is the correction
   that minimised the pinball loss on synthetic launch weeks built to match
   the two real anchors we have (launch night: 638 players, Ch.1 1.25%;
   13:11 UTC Nov 19: Ch.1 0.73%, Ch.2 0.14%, platinum 0.03%). The same
   correction also won on a second, differently-shaped synthetic world, so it
   is not tuned to one guessed future. On a 90-minute horizon (the practice
   cases) the damping barely matters, because the extrapolated move is tiny.

3. Honest spread. The spread is the regression's prediction standard error
   (residual variance, slope uncertainty growing with extrapolation distance),
   scaled by 1.25, floored at 0.25 and capped at 1.2 logit units. Quantiles are
   median + z_tau * sigma on the logit scale, z = (-1.645, -0.674, 0, 0.674,
   1.645), mapped back to percent. The scale 1.25 (up from 0.9) is
   calibration, not caution: on the synthetic final weeks the 0.9 scale
   covered only about 80% of answers in the 5%-95% band, where a calibrated
   forecast covers about 90%; 1.25 brings coverage to about 0.89 without
   costing accuracy anywhere. On the synthetic launch weeks this gives
   about 88% coverage of the 5%-95% band, close to the calibrated 90%.

Two special cases, both from the data, not from guessing:
- A page that has not changed at all (one distinct state) is stale: Exophase
  had not refreshed in hours while the game was live, so the next refresh is
  a dilution jump down. The median is shifted -0.8 logit and the spread
  widened to at least 0.9. This is what the practice cases look like; the
  final's history is rich, so this branch does not fire there.
- The near-universal trophy (rate >= 99.5%, the one every player earns in
  the first minutes) cannot rise and barely falls: tiny spread, median pinned
  near its last rate.

What the model does NOT do: it never reads the network, never hard-codes a
rate, and never uses the trophy names for anything but documentation -- every
number comes from the case file it is given, so the same program forecasts
the practice cuts and the final cut.
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model`
# finds a module shipped beside this file (none is needed; this file is whole).
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# --- tuning constants (documented where they bite) ---------------------------
TREND_WINDOW_HOURS = 24.0   # only the last day of distinct reads sets the slope
HALF_LIFE_HOURS = 12.0      # recency weight half-life inside the window
DAMP = 0.7                  # saturation correction on the extrapolated move
SIGMA_SCALE = 1.25          # calibration scale on the prediction std error
SIGMA_FLOOR = 0.25          # logit spread floor: refresh jumps move rates
SIGMA_CAP = 1.2             # logit spread cap
SIGMA_FEW_READS = 0.8       # spread when too few distinct reads to fit
STALE_SHIFT = -0.8          # logit shift when the page never moved (stale read)
STALE_SIGMA = 0.9           # spread in the stale-page case
MAX_MOVE = 3.5              # cap on |median - last| in logit, against runaway trends
RATE_FLOOR = 0.01           # percents: 0% and 100% have no logit
RATE_CEIL = 99.99
NEAR_UNIVERSAL = 99.5       # rate at/above this is treated as (near-)universal
UNIVERSAL_SIGMA = 0.08


def parse_time(text):
    """ISO 8601 (date or datetime, UTC) -> aware datetime."""
    dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def clamp_rate(pct):
    return min(RATE_CEIL, max(RATE_FLOOR, pct))


def logit(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def sigmoid(lg):
    if lg >= 0:
        return 1.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up (as JavaScript's Math.round does it)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def wls(xs, ys, ws):
    """Weighted least squares y ~ a + b*x.

    Returns (a, b, s2, xbar, sxx, n_eff): s2 is the residual variance (None
    when n < 3), xbar the weighted mean of x, sxx the weighted sum of squared
    x deviations, n_eff the effective sample size (sum w)^2 / sum w^2.
    """
    n = len(xs)
    sw = sum(ws)
    xbar = sum(w * x for w, x in zip(ws, xs)) / sw
    ybar = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - xbar) ** 2 for w, x in zip(ws, xs))
    n_eff = sw * sw / sum(w * w for w in ws)
    if sxx <= 1e-12 or n < 2:
        return ybar, 0.0, None, xbar, sxx, n_eff
    b = sum(w * (x - xbar) * (y - ybar) for w, x, y in zip(ws, xs, ys)) / sxx
    a = ybar - b * xbar
    s2 = None
    if n >= 3:
        s2 = sum(w * (y - (a + b * x)) ** 2 for w, x, y in zip(ws, xs, ys)) / (n - 2)
    return a, b, s2, xbar, sxx, n_eff


def distinct_reads(history):
    """Collapse consecutive identical reads (Exophase's flat stretches).

    A read is kept when its players-tracked count or any rate differs from
    the last kept read. A missing rate (null) never counts as a change: a
    read that only lacks a trophy the previous read had is the same page.
    Returns the kept reads in order.
    """
    kept = []
    prev_players = None
    prev_rates = None
    for read in history:
        rates = read.get("rates") or ()
        players = read.get("players_tracked")
        changed = players != prev_players
        if not changed and prev_rates is not None:
            for a, b in zip(rates, prev_rates):
                if a is not None and b is not None and a != b:
                    changed = True
                    break
        if changed or not kept:
            kept.append(read)
            prev_players = players
            prev_rates = rates
    return kept


def rate_at(read, i, n_tr):
    """The i-th rate of a read, or None when the read lacks it.

    A read's rates list is one rate per trophy in the case's order, null
    where the read lacked the trophy -- but a list shorter than the trophy
    count (a malformed read) must not crash the run: anything past its end
    counts as missing, exactly like a null.
    """
    rates = read.get("rates") or ()
    return rates[i] if i < len(rates) else None


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    history = case.get("history") or []
    if not history:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    n_tr = len(trophies)

    game = case.get("game") or {}
    release = parse_time(game.get("release_date", "2026-11-19"))
    checkpoint = parse_time(case["at"])
    target = parse_time(case["target"])

    def x_of(moment):
        hours = (moment - release).total_seconds() / 3600.0
        return math.log1p(max(0.0, hours))

    x_last = x_of(checkpoint)
    x_target = x_of(target)
    dx = x_target - x_last

    reads = distinct_reads(history)
    stale = len(reads) <= 1
    times = [parse_time(r["at"]) for r in reads]
    t_end = times[-1]

    # The trend window: the last day of distinct reads (all of them if the
    # case holds fewer than three distinct reads in that window).
    win_start = t_end - timedelta(hours=TREND_WINDOW_HOURS)
    widx = [j for j, t in enumerate(times) if t >= win_start]
    if len(widx) < 3:
        widx = list(range(len(times)))

    per_trophy = []
    for i, trophy in enumerate(trophies):
        xs, ys, ws = [], [], []
        for j in widx:
            rate = rate_at(reads[j], i, n_tr)
            if rate is None:
                continue
            xs.append(x_of(times[j]))
            ys.append(logit(rate))
            ws.append(max(0.05, 0.5 ** max(0.0, (t_end - times[j]).total_seconds()
                                            / 3600.0 / HALF_LIFE_HOURS)))
        last_rate = 0.0
        for read in reversed(history):
            value = rate_at(read, i, n_tr)
            if value is not None:
                last_rate = value
                break
        if len(ys) >= 2:
            _, b, s2, xbar, sxx, n_eff = wls(xs, ys, ws)
        else:
            b, s2, xbar, sxx, n_eff = 0.0, None, (xs[0] if xs else x_last), 0.0, 1.0
        per_trophy.append({
            "trophy": trophy, "last_rate": last_rate, "slope": b, "s2": s2,
            "xbar": xbar, "sxx": sxx, "n": len(ys), "n_eff": n_eff,
        })

    rows = []
    for p in per_trophy:
        trophy = p["trophy"]
        # Each trophy keeps its own slope (no tier shrinkage: see the
        # module docstring -- the tiers pool trophies with genuinely
        # different progression speeds).
        slope = p["slope"]

        y_last = logit(p["last_rate"])
        # Damped recent trend, anchored at the last read.
        y_med = y_last + DAMP * slope * dx

        # Spread: regression prediction error, calibrated.
        if p["s2"] is not None and p["sxx"] > 1e-12 and p["n"] >= 3:
            pred_var = p["s2"] * (1.0 + 1.0 / p["n_eff"]
                                  + (x_target - p["xbar"]) ** 2 / p["sxx"])
        else:
            pred_var = SIGMA_FEW_READS ** 2
        sigma = math.sqrt(max(pred_var, 0.0)) * SIGMA_SCALE
        sigma = min(SIGMA_CAP, max(SIGMA_FLOOR, sigma))

        if stale:
            y_med += STALE_SHIFT
            sigma = max(sigma, STALE_SIGMA)

        # The near-universal trophy: pinned, tiny spread.
        if p["last_rate"] >= NEAR_UNIVERSAL:
            y_med = min(max(y_med, logit(99.0)), logit(RATE_CEIL))
            sigma = UNIVERSAL_SIGMA

        # Cap the total move from the last read; keep the median in range.
        y_med = y_last + max(-MAX_MOVE, min(MAX_MOVE, y_med - y_last))
        y_med = min(logit(RATE_CEIL), max(logit(RATE_FLOOR), y_med))

        qs = []
        for z in Z:
            value = round4(sigmoid(y_med + z * sigma) * 100.0)
            value = min(RATE_CEIL, max(RATE_FLOOR, value))
            if qs:
                value = max(value, qs[-1])
            qs.append(value)
        rows.append({"id": trophy["id"], "q": qs})

    return {"trophies": rows}


def fallback_forecast(case):
    """Last-resort forecast when a case cannot be modelled: every trophy's
    quantiles sit on its last rate with the house baseline's random-walk
    spread. Always a valid shape when the case parses at all, so one
    malformed case can never invalidate the whole run (an invalid case
    takes the entry out of the prizes; a valid-but-flat forecast only
    costs loss on that case). Only real numbers count as rates here: a
    rate of the wrong type is missing, never a crash.
    """
    history = case.get("history") or []
    trophies = case["trophies"]
    n_tr = len(trophies)
    try:
        horizon_days = max(0.0, float(case.get("horizon_minutes") or 0) / 1440.0)
    except (TypeError, ValueError):
        horizon_days = 0.0
    spread = 0.5 * math.sqrt(horizon_days)
    rows = []
    for i, trophy in enumerate(trophies):
        last_rate = 1.0
        for read in reversed(history):
            value = rate_at(read, i, n_tr)
            if isinstance(value, (int, float)) and not isinstance(value, bool) \
                    and math.isfinite(value):
                last_rate = value
                break
        lg = logit(last_rate)
        qs = []
        for z in Z:
            value = round4(sigmoid(lg + z * spread) * 100.0)
            value = min(RATE_CEIL, max(RATE_FLOOR, value))
            if qs:
                value = max(value, qs[-1])
            qs.append(value)
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue  # unparseable input: nothing valid can be written
        try:
            forecast = forecast_case(case, path.name)
        except Exception:
            try:
                forecast = fallback_forecast(case)
            except Exception:
                continue
        out = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / f"{path.stem}.out").write_text(out, encoding="utf-8")


if __name__ == "__main__":
    main()
