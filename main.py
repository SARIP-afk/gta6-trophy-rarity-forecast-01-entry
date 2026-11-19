"""gta6-trophy-rarity-forecast-01: forecast each GTA VI trophy's Exophase earn rate.

Model (own design, standard library only, deterministic, offline):
  - Work on the logit scale. For each trophy, fit by least squares a line of
    logit(rate) against ln(1 + hours since 2026-11-19T00:00:00Z) over ALL reads
    in the case history. The log-time axis is the dilution axis: Exophase's
    tracked-player count explodes after launch, so rates mostly fall as the
    log of elapsed time grows, and a straight line in log-time extrapolates
    that dilution instead of repeating the last read.
  - Shrink each trophy's slope halfway toward the median slope of its tier
    (platinum / gold / silver / bronze), so a noisy single trophy cannot run
    away on its own. Anchor the shrunk line at the series mean and cap the
    median's move from the last read's logit at 3 logit units.
  - Spread (logit std) combines, in quadrature: (a) a random walk
    0.5 * sqrt(horizon in days), (b) the residual std of the trophy's fit,
    (c) a flat-history term of 0.9 when the trophy's rate took at most two
    distinct values in the history: a frozen page means an Exophase refresh
    the case has not seen yet, and the launch-night refresh moved the median
    rate down ~1.6 logit, so a flat history must carry a wide band.
  - A trophy at exactly 100% in every read stays 100% at every quantile:
    it is earned by every tracked player, and the first refresh left it there.
  - Pending-refresh prior: a (nearly) flat history means no Exophase refresh
    has been seen yet, but one is due before the target, so the median is
    shifted down by half the first refresh's observed move (~1.6 logit).
  - Rates are clamped to 0.01..99.99 before any logit, so 0% and 100% reads
    never break the transform.

Usage: python3 main.py, reading $MAKEFUN_INPUT_DIR (default /input) and
writing $MAKEFUN_OUTPUT_DIR (default /output), one <id>.out per <id>.in.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist, median

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# Launch night, the origin of the log-time axis (2026-11-19T00:00:00Z).
ORIGIN = datetime(2026, 11, 19, tzinfo=timezone.utc)

DEFAULT_LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
NORMAL = NormalDist()

# Random-walk spread, logit units per sqrt(day), as in the house baseline.
RW_SIGMA = 0.5
# Extra spread when a trophy's history is (nearly) flat: an unseen refresh.
FLAT_SPREAD = 0.9
# How far the median may move from the last read, in logit units.
MEDIAN_MOVE_CAP = 3.0
# Pending-refresh prior: when a trophy's history is (nearly) flat, no refresh
# has been seen yet, but Exophase refreshes on its own schedule (the page was
# frozen from launch night until the 04:30 UTC refresh, which moved the median
# rate down ~1.6 logit). A flat history therefore expects a refresh before the
# target, so the median is shifted down by half that observed move: a hedge
# that costs about the same whether the refresh lands or not, instead of
# betting everything on "no refresh". Halved again when exactly two distinct
# rates were seen (one refresh already visible in the trend).
FLAT_MEDIAN_SHIFT = -0.8
FLAT_MEDIAN_SHIFT_TWO_STATES = -0.4
FLAT_SHIFT_MIN_HORIZON_MIN = 60.0
# Shrinkage weight toward the tier-median slope (0.5 = halfway).
SHRINK = 0.5
# Spread floor / cap, logit units.
SPREAD_MIN = 0.10
SPREAD_MAX = 2.50


def parse_ts(text):
    """Parse an ISO-8601 UTC timestamp (with 'Z' or an offset)."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def clamp_rate(pct):
    return min(99.99, max(0.01, pct))


def logit_pct(pct):
    p = clamp_rate(pct)
    return math.log(p / (100.0 - p))


def inv_logit_pct(lg):
    return 100.0 / (1.0 + math.exp(-lg))


def round4(x):
    """Round to 4 decimals, half up (as the house baseline rounds)."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value.is_integer() else value


def log_time(read_at):
    """ln(1 + hours from launch-night origin to the read)."""
    hours = (read_at - ORIGIN).total_seconds() / 3600.0
    return math.log1p(max(hours, -0.999999))


def ols(points):
    """Least-squares line y = a + b*x over [(x, y)]; returns (a, b, resid_std).

    With fewer than two distinct x values there is no slope to fit.
    """
    n = len(points)
    if n == 0:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    xbar = sum(xs) / n
    ybar = sum(ys) / n
    sxx = sum((x - xbar) ** 2 for x in xs)
    if n < 2 or sxx <= 0.0:
        return (ybar, 0.0, 0.0)
    sxy = sum((x - xbar) * (y - ybar) for x, y in points)
    slope = sxy / sxx
    intercept = ybar - slope * xbar
    if n <= 2:
        resid = 0.0
    else:
        resid = math.sqrt(
            sum((y - (intercept + slope * x)) ** 2 for x, y in points) / (n - 2)
        )
    return (intercept, slope, resid)


def forecast_case(case, name):
    levels = case.get("levels") or DEFAULT_LEVELS
    zs = [NORMAL.inv_cdf(tau) for tau in levels]
    history = case.get("history") or []
    trophies = case["trophies"]

    horizon_minutes = case.get("horizon_minutes")
    if horizon_minutes is None:
        horizon_minutes = (
            parse_ts(case["target"]) - parse_ts(case["at"])
        ).total_seconds() / 60.0
    horizon_days = max(horizon_minutes, 0.0) / 1440.0
    target_x = log_time(parse_ts(case["target"]))

    # First pass: one log-time fit per trophy, over all its non-null reads.
    fits = []
    for i, trophy in enumerate(trophies):
        points = []
        distinct = set()
        all_hundred = True
        for read in history:
            rates = read.get("rates") or []
            if i >= len(rates):
                continue
            rate = rates[i]
            if rate is None:
                continue
            distinct.add(rate)
            if rate != 100:
                all_hundred = False
            points.append((log_time(parse_ts(read["at"])), logit_pct(rate)))
        fits.append(
            {
                "points": points,
                "distinct": len(distinct),
                "all_hundred": all_hundred and bool(points),
                "fit": ols(points),
            }
        )

    # Tier-median slopes for shrinkage (trophies with a real fitted slope).
    tier_slopes = {}
    for i, trophy in enumerate(trophies):
        fit = fits[i]["fit"]
        if fit is None or fit[1] == 0.0:
            continue
        tier_slopes.setdefault(trophy.get("tier", "bronze"), []).append(fit[1])
    tier_median = {t: median(v) for t, v in tier_slopes.items()}

    rows = []
    for i, trophy in enumerate(trophies):
        info = fits[i]
        fit = info["fit"]
        points = info["points"]

        if info["all_hundred"]:
            rows.append({"id": trophy["id"], "q": [100, 100, 100, 100, 100]})
            continue
        if fit is None or not points:
            # No usable history for this trophy: wide band around 1%.
            center = logit_pct(1.0)
            spread = SPREAD_MAX
            rows.append(
                {
                    "id": trophy["id"],
                    "q": [round4(inv_logit_pct(center + z * spread)) for z in zs],
                }
            )
            continue

        intercept, slope, resid = fit
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        xbar = sum(xs) / len(xs)
        ybar = sum(ys) / len(ys)
        last_logit = ys[-1]

        # Shrink the slope toward the tier median, anchored at the mean.
        tier = trophy.get("tier", "bronze")
        if tier in tier_median:
            slope = (1.0 - SHRINK) * slope + SHRINK * tier_median[tier]
        med_logit = ybar + slope * (target_x - xbar)
        # Pending-refresh prior on a (nearly) flat history: see the constant
        # above. A flat history carries no trend, so the OLS median would sit
        # on the last read and bet everything on "no refresh before target".
        if horizon_minutes >= FLAT_SHIFT_MIN_HORIZON_MIN:
            if info["distinct"] <= 1:
                med_logit += FLAT_MEDIAN_SHIFT
            elif info["distinct"] == 2:
                med_logit += FLAT_MEDIAN_SHIFT_TWO_STATES
        # Cap how far the median may move from the last read.
        med_logit = min(max(med_logit, last_logit - MEDIAN_MOVE_CAP),
                        last_logit + MEDIAN_MOVE_CAP)

        # Spread: random walk + fit residual + flat-history refresh risk.
        rw = RW_SIGMA * math.sqrt(horizon_days)
        flat = FLAT_SPREAD if info["distinct"] <= 2 else 0.0
        spread = math.sqrt(rw * rw + resid * resid + flat * flat)
        spread = min(max(spread, SPREAD_MIN), SPREAD_MAX)

        rows.append(
            {
                "id": trophy["id"],
                "q": [round4(inv_logit_pct(med_logit + z * spread)) for z in zs],
            }
        )
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
