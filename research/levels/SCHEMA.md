# research/levels/ — data schema

Logging only. Nothing in this package can place, modify or cancel an order;
`test_research_levels_isolation.py` enforces that two ways (runtime import
graph + static AST scan). It suggests entry/stop/TP prices, which makes the
constraint matter more here than elsewhere, not less — those are numbers a
reader could mistake for instructions.

## Why this package exists

The bot's 48 recorded trades sit **7.2pp below the random-walk baseline**
for their own RR (27.1% actual vs 34.4% needed at RR 1.92), i.e. no
demonstrated entry edge. The user's September book returned **+4.885% per
closing fill** on n=9, with alpha that survives a beta check on two of four
symbols (ETH +1.63% while ETH moved −0.1%; CASHCAT +11.42% while CASHCAT
fell 18%).

The gap between those two is judgement that exists nowhere in the codebase.
This package collects the data needed to find out what it is.

## Three files

| file | one row per | written when |
|---|---|---|
| `episodes.csv` | (coin, level, bar) whose gate fired | at detection |
| `outcomes.csv` | episode | once ≥42 bars (~7d) have elapsed |
| `labels.csv` | episode | when the user records a verdict |

Outcomes and labels answer **different** questions. Outcomes say whether a
gated setup made money. Labels say whether the gate agrees with the
judgement it is trying to reproduce. A gate that is profitable but fires on
setups the user would never take has not captured their strategy; a gate
that misses setups they would take is leaving edge behind. Neither shows up
in outcomes alone.

## Everything is measured in sigma

`sigma_4h` is the stdev of 4h log returns over the trailing 120 bars. Every
width, distance and band is expressed in multiples of it.

This is not stylistic. A first hand-encoding of the user's rule used raw
percent and **fired on 0 of 108 defensible parameter sets**, because the
same setup measured 124% on CASHCAT and 9.8% on BTC. Normalised, those are
9.9σ and 13.2σ — the same event. After normalisation the identical concept
fired on **216 of 216** parameter sets. A raw-percent threshold cannot
serve both a memecoin and a major, and the user trades both.

## episodes.csv

**Identity** — `episode_id` (`COIN_BARTS_LEVELPRICE`; one bar can gate
several levels, so price is part of the key), `coin`, `detected_at_utc`,
`bar_ts_ms`.

**Level** — `level_price`, `level_kind`, `level_touches`, `level_holds`,
`level_hold_rate`, `level_pivot_count`, `level_bars_since_last_touch`.

A touch is a bar entering the ±1σ band. It **holds** unless some bar in the
next 12 *closes* more than 3σ beyond the level. Both details were wrong in
the first cut: judging the break inside the touching bar required a candle
to dip 1σ and close 3σ past, which never happens, so every level scored a
perfect 100% and the metric carried no information. Keying off the close
rather than the wick matters too — CASHCAT printed 0.14211 against a 0.14
level and reversed, and counting wicks as breaks would discard exactly the
tests worth counting.

**Gate features** — `run_sigma`, `pull_sigma`, `run_pct`, `pull_pct`,
`bars_since_high`, `retests`, `bars_since_last_retest`, `near_sigma`,
`dist_to_level_pct`, `cond_run_pullback`, `cond_retest`, `cond_near`,
`gate_fires`.

**Volume — captured, never gated.** `vol_ratio_recent_ref`,
`vol_trend_slope`, `vol_down_up_ratio`. The user recalled volume contracting
into the level; at the actual fill bar every window measured **0.90–1.25×**
its reference — normal to elevated. So the remembered feature is not the one
first computed. Three competing readings (low absolute level, declining
trend, down-vs-up split) are recorded on every episode so the question gets
settled from data rather than argued. `review --stats` prints the separation
between takes and skips for all three; near-zero gaps on all of them settle
it in the negative.

**Suggested trade** — `atr_pct`, `entry_buffer_sigma`, `suggested_entry`,
`suggested_stop`, `suggested_tp`, `suggested_rr`, `opposing_level_price`.

Entry sits **0.35σ above** the level rather than at it. Both levels the user
named would have missed as exact limits: CASHCAT's 20-day low was 0.14211
against a named 0.14, and BTC's was 74,903 against 74,900 — **a $3 miss**.
The stop sits 3σ *below the level*, not at the next support. When the user
noted the next structure under CASHCAT 0.14 was 0.080, that is a −44% move,
which at 2% account risk forces a $32 position on a $700 account. A level
breaking and price reaching the next level are different events, and only
the first invalidates the trade.

## outcomes.csv

`reached_level`, `bars_to_reach`, `fill_price`, `mae_pct`, `mfe_pct`,
`mae_sigma`, `mfe_sigma`, `close_after_{24,72,168}h_pct`,
`hit_suggested_stop`, `hit_suggested_tp`, `bars_to_stop`, `bars_to_tp`,
`outcome_r`, `level_broke`.

MAE is recorded because it is the number that decides whether a stop is
survivable. All four of the user's September winners would have been
stopped out by a 10%-of-margin stop; ZEC went **−151.8% of margin** against
the position before returning +$31.84.

## labels.csv

`episode_id`, `labelled_at_utc`, `verdict` (`take` / `skip` / `maybe`),
`confidence`, `note`.

## All tunables live in `config.py`

Every parameter is defined once, in `research/levels/config.py`, tagged with
what it rests on:

| tag | meaning |
|---|---|
| `STRUCTURAL` | forced by the problem. Changing it breaks correctness, not calibration. |
| `EVIDENCED` | supported by a measurement recorded in `DECISIONS.md`. |
| `GUESSED` | my invention. No evidence. Expected to move once labels exist. |

Thirteen of the parameters are tagged `GUESSED`. That count is the honest
measure of how much of this system is currently assumption.

Before consolidation the system had 31 knobs across four modules, and
`near_sigma` was filtered **twice** — at 5.0 in the gate and 2.0 in the
trader — so a setup could pass one and fail the other with nothing naming
which was authoritative. It is now enforced once, in the gate, at 2.0. That
makes the gate stricter than it was during the backfill, so episodes
collected before 2026-09-18 came through a looser filter than episodes
collected after.

The review shortlist is ordered by hold rate then reward/risk. It previously
used a weighted sum with coefficients 0.45/0.30/0.15/0.10 that I invented;
those encoded a claim about relative importance that nothing supports, and
presenting a guess as a score made it harder to see that it was one.

## Calibration status — read before using any of this

Gate thresholds (`run_sigma ≥ 6`, `pull_sigma ≥ 5`, `retests ≥ 2`) are
**placeholders fitted to two levels the user named after the fact, on charts
used to generate the hypotheses.** That is n=2 and it is circular. The
216/216 sensitivity result measures robustness across *parameters*, not
across *setups*.

A recent pass produced **39 episodes against roughly 9 trades a month by
hand — about 25× too loose.** That is expected and is the thing being
measured. The scanner logs generously and `review` surfaces a ranked
shortlist; do not tighten the gate before labels exist, because fitting
thresholds to the same two charts that produced them is precisely how this
pivot generated its four earlier false positives.

## Usage

```sh
python -m research.levels.scanner --dry-run       # one pass, print, write nothing
python -m research.levels.scanner --backfill 90   # replay history, then run
python -m research.levels.scanner                 # long-lived, 15m poll

python -m research.levels.review                  # ranked shortlist
python -m research.levels.review --label <id> take --note "clean retest"
python -m research.levels.review --stats          # take-vs-skip feature separation
```
