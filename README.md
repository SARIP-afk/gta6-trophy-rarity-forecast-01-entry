[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/SARIP-afk/gta6-trophy-rarity-forecast-01-entry/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/SARIP-afk/gta6-trophy-rarity-forecast-01-entry/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

# gta6-trophy-rarity-forecast-01 — log-time dilution forecast

Forecast, for every GTA VI trophy, its Exophase earn rate at the case's
target: a least-squares line of logit(rate) against ln(1 + hours since
launch night), fitted over the whole history, with each trophy's slope
shrunk halfway toward its tier median. Exophase's tracked-player count
explodes after launch, so rates mostly fall along that log-time axis, and
the line extrapolates the dilution instead of repeating the last read.
The spread combines a random walk (0.5 logit per sqrt day), the fit's
residual, and a flat-history term for the refresh the case has not seen
yet; a flat history also shifts the median down by half the first
refresh's observed move (~1.6 logit). A trophy at 100% in every read
stays 100%. Python 3.13, standard library only, deterministic, offline,
no model calls.

## Validation

- Practice cases p00/p01: valid output, byte-identical across runs,
  ~0.1 s per run.
- Synthetic case s00 (p00's history, target the live 11:37 UTC page of
  19 Nov): mean pinball loss on the logit scale **0.440** vs **1.384**
  for the house baseline starter, 90% coverage 1.000, typical miss
  0.21 points.
- Synthetic final-like trend case (2 days of reads, logit rates falling
  linearly in log-time): loss **0.240** vs **0.350** for the starter.
- Robustness: single-read history, 0% / 100% / null rates, final
  horizon (6,510 min), and empty history all produce valid output
  without crashing.

## Running it

```sh
mkdir out; MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Each `<id>.in` is read and an `<id>.out` is written holding exactly one
row per trophy: `{"trophies": [{"id": 1, "q": [q05, q25, q50, q75, q95]}, ...]}`.

## Contents

- `main.py` — the forecaster; the only code that matters
- `LICENSE` — MIT, as in the kit's starter

---

Optional: earn referral rewards, a second way to make money:
[make.fun/r/cxyf1jsy](https://make.fun/r/cxyf1jsy)
