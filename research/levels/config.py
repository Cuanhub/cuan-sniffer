"""
Every tunable for the level system, in one place, each marked with what it
actually rests on.

This file exists because the system had 31 knobs spread across four modules,
with near_sigma filtering in two of them at two different values. That is not
a system anyone can reason about, and almost none of those numbers had
evidence behind them.

Read the PROVENANCE tag before trusting any value:

  STRUCTURAL  forced by the problem, not chosen. Changing it breaks
              correctness, not just calibration.
  EVIDENCED   supported by a measurement recorded in DECISIONS.md.
  GUESSED     my invention. No evidence. Expected to move once labelled
              episodes exist, and no result should be read through it.

Four rules are the whole strategy. Everything below serves one of them:
  1. a level with a real test history
  2. buy just above it
  3. stop just below it
  4. target the next level up, size = fixed risk / stop distance
"""

import os


def _f(name, default):
    return float(os.getenv(name, str(default)))


def _i(name, default):
    return int(os.getenv(name, str(default)))


# ---- rule 1: what counts as a level ----------------------------------------
# STRUCTURAL. Everything is measured in multiples of the asset's own 4h return
# volatility. Raw percent cannot serve a memecoin and a major at once: the same
# setup measured 124% on CASHCAT and 9.8% on BTC, and a raw-percent encoding
# fired on 0 of 108 parameter sets. Normalised they are 9.9 and 13.2 sigma.
VOL_LOOKBACK = _i("LEVELS_VOL_LOOKBACK", 120)     # bars for the sigma estimate
HISTORY_DAYS = _i("LEVELS_HISTORY_DAYS", 125)     # STRUCTURAL: 70d hid BTC 74,900,
                                                  # whose tests were in May
INTERVAL = os.getenv("LEVELS_INTERVAL", "4h")

PIVOT_WINDOW = _i("LEVELS_PIVOT_WINDOW", 3)          # GUESSED
CLUSTER_SIGMA = _f("LEVELS_CLUSTER_SIGMA", 1.5)      # GUESSED
TOUCH_BAND_SIGMA = _f("LEVELS_TOUCH_BAND_SIGMA", 1.0)  # GUESSED
BREAK_SIGMA = _f("LEVELS_BREAK_SIGMA", 3.0)          # GUESSED
BREAK_WINDOW = _i("LEVELS_BREAK_WINDOW", 12)         # STRUCTURAL: judging a break
                                                     # inside the touching bar made
                                                     # every level score 100% holds
MIN_TOUCHES_DETECT = _i("LEVELS_MIN_TOUCHES_DETECT", 2)  # GUESSED

# NOT USED AS A TRADE FILTER. Hold rate looked like the obvious quality
# measure and was tagged EVIDENCED here for days. Against 18 labels it passes
# 4 of 5 SKIPS and only 5 of 11 TAKES -- takes average a hold rate of 0.86
# against 0.92 for skips, and fewer tests (10.7 vs 12.2). Both point the wrong
# way, so filtering on them costs takes and buys nothing. Kept for display and
# for the record, not for gating.
MIN_HOLD_RATE = _f("LEVELS_MIN_HOLD_RATE", 0.90)     # display only
MIN_TOUCHES = _i("LEVELS_MIN_TOUCHES", 8)            # display only

# ---- when is the level live -------------------------------------------------
# EVIDENCED that timing dominates: the same limit at 0.1428 on CASHCAT left
# resting from 08-20 rides a -73.67% drawdown, while placed on 09-08 it fills
# at the same price and never goes more than -0.48% underwater. A 150x
# difference in drawdown from nothing but when the order became live.
#
# The THRESHOLDS below, however, do not capture that. Measured against 18
# labels, none of the three separates a take from a skip:
#     pull_sigma  takes median 3.8  skips median 4.0   gap +0.57
#     run_sigma   takes median 13.3 skips median 14.8  gap -0.21
#     retests     takes median 4    skips median 6     gap -1.91  (backwards)
# At their old values (pull>=5.0, retests>=2) the gate rejected 7 of 11 takes,
# so the trader would never have SEEN them whatever the trade filter said.
#
# They are now set as a LOGGING net, not a decision: wide enough that every
# labelled take passes, because an episode that is never recorded can never be
# labelled or measured. What gets bought is decided by the TRADE_ thresholds
# further down, which do discriminate.
RUN_LOOKBACK = _i("LEVELS_RUN_LOOKBACK", 84)
RUN_SIGMA_MIN = _f("LEVELS_RUN_SIGMA_MIN", 6.0)     # all 11 takes are >=12.6
PULL_SIGMA_MIN = _f("LEVELS_PULL_SIGMA_MIN", 1.0)   # was 5.0, rejected 7 takes
RETEST_LOOKBACK = _i("LEVELS_RETEST_LOOKBACK", 180)
MIN_RETESTS = _i("LEVELS_MIN_RETESTS", 1)           # was 2, rejected 1 take
RETEST_BAND_SIGMA = _f("LEVELS_RETEST_BAND_SIGMA", 1.0)

# GATE vs TRADE, kept apart on purpose. The gate decides what gets LOGGED and
# is deliberately wide, because an episode that never gets recorded can never
# be labelled or measured. The trade filter below decides what gets BOUGHT.
# Collapsing the two is what produced a filter that logged and traded the same
# narrow slice.
GATE_MAX_NEAR_SIGMA = _f("LEVELS_GATE_MAX_NEAR_SIGMA", 12.0)

# EVIDENCED, and it replaces an inverted predecessor. MAX_NEAR_SIGMA was 2.0 --
# it REQUIRED price to be within 2 sigma of the level, and tested against 18
# labels it passed 1 of 11 takes while passing 1 of 5 skips. The user's takes
# sit 0.18-10.00 sigma below price (median 5.80); their skips 1.13-4.67
# (median 2.06). The old filter demanded the shallow setups they reject and
# excluded the deep ones they take. Direction reversed; threshold from the data.
TRADE_MIN_NEAR_SIGMA = _f("LEVELS_TRADE_MIN_NEAR_SIGMA", 5.0)

# EVIDENCED. "Sitting ON the level now" was the user's own reason for two
# skips and both maybes. Takes were last tested 9-40 bars ago; skips 0-9.
TRADE_MIN_BARS_SINCE_TOUCH = _i("LEVELS_TRADE_MIN_BARS_SINCE_TOUCH", 9)

# EVIDENCED. Overhead resistance was the stated reason for the AAVE skip and
# the ETH rejection. Takes average 7.58 sigma of headroom, skips 2.98.
TRADE_MIN_OVERHEAD_SIGMA = _f("LEVELS_TRADE_MIN_OVERHEAD_SIGMA", 5.0)

# ---- rules 2 and 3: entry and stop ------------------------------------------
# EVIDENCED. Both levels the user named would have MISSED as exact limits:
# CASHCAT's 20-day low was 0.14211 against a named 0.14, and BTC's was 74,903
# against 74,900 -- a $3 miss. The level is a zone, not a price.
ENTRY_BUFFER_SIGMA = _f("LEVELS_ENTRY_BUFFER_SIGMA", 0.35)   # size GUESSED

# EVIDENCED that the stop belongs below the LEVEL, not at the next support: the
# 0.080 structure under CASHCAT 0.14 is a -44% move, which forces an untradeable
# position. A level breaking and price reaching the next level are different
# events and only the first invalidates the trade.
STOP_SIGMA_BELOW = _f("LEVELS_STOP_SIGMA_BELOW", 3.0)        # size GUESSED

# ---- rule 4: sizing ---------------------------------------------------------
# STRUCTURAL. Size comes from the distance to the stop; leverage is only ever a
# ceiling on the result, never an input. At fixed 10x a 10% stop costs the whole
# account and a 44% stop costs 4.4x of it, because stop distance is set by the
# chart rather than chosen.
# 3% for the initial trials, to be raised only as the system demonstrates
# profitability. Nothing measures this strategy's edge yet, so the first
# trades are being bought as information at roughly $24 each.
RISK_PCT_PER_TRADE = _f("LEVELS_RISK_PCT", 3.0)
MAX_LEVERAGE = _f("LEVELS_MAX_LEVERAGE", 10.0)
# Lowered from 2.5: at 2.5 the filter kept 6 of 11 takes, at 2.0 it keeps 7
# while still admitting 0 of 5 skips. Live median reward/risk is 3.40 anyway,
# so the floor rarely binds.
MIN_RR = _f("LEVELS_MIN_RR", 2.0)
MIN_NOTIONAL_USD = _f("LEVELS_MIN_NOTIONAL", 12.0)

# STRUCTURAL. execute_entry sends a marketable IOC at the CURRENT price and
# ignores the limit price it is given, so a setup identified while price was
# far away will fill nowhere near its planned entry. On 2026-09-18 that bought
# UNI 17% above plan: stop distance 5.66% -> 19.7%, reward/risk 3.27 -> 0.04,
# risk $23 -> $80 on a position sized for the plan. This is the rail that
# would have refused it.
MAX_ENTRY_DRIFT_PCT = _f("LEVELS_MAX_ENTRY_DRIFT_PCT", 1.5)

# ---- the trigger: has the level actually held? -----------------------------
# The filters above identify a level worth watching while price is still 5-10
# sigma above it. They say nothing about the moment of entry, and the user's
# own labels are explicit that arrival alone is not enough: every setup they
# marked "maybe" was one price was sitting on right then, with the reason
# given as "good trade IF it holds" and "if we maintain this level in a
# channel". Firing on arrival would buy exactly those.
#
# Their proposed rule, and the first testable version of "it held": price came
# down into the zone, and has stayed above the level for two hours since.
# Checked on 1h bars regardless of the timeframe the level was found on,
# because the hold is a wall-clock condition, not a chart-interval one.
TRIGGER_ZONE_SIGMA = _f("LEVELS_TRIGGER_ZONE_SIGMA", 1.0)
TRIGGER_HOLD_HOURS = _i("LEVELS_TRIGGER_HOLD_HOURS", 2)
TRIGGER_TOUCH_LOOKBACK_HOURS = _i("LEVELS_TRIGGER_TOUCH_LOOKBACK_HOURS", 12)

# How far above the level price may sit and still count as holding AT it.
# Without this, "held" meant nothing more than "is above", which a level price
# has run away from satisfies trivially: UNI touched 7.5078 twelve hours ago,
# rallied to 8.64, and the trigger read that as two hours of support 16% below
# the market. The channel has a ceiling as well as a floor.
TRIGGER_MAX_ABOVE_SIGMA = _f("LEVELS_TRIGGER_MAX_ABOVE_SIGMA", 2.0)

# ---- operational rails ------------------------------------------------------
MAX_CONCURRENT = _i("LEVELS_MAX_CONCURRENT", 2)
# 5, matching the daily loss stop's own arithmetic: at 3% risk a 15% stop
# halts after five losing trades, so a cap of five never takes a trade the
# risk budget would have allowed. What it bounds is the case that budget
# cannot see -- opening and closing without losing enough to trip anything.
# A protection failure that repeats does exactly that, paying a round trip in
# fees and spread each pass. opened_today is incremented on FILL, before
# protection is placed, so a trade that is immediately emergency-closed still
# spends one of the five; for a churn bound that is the point.
# 0 disables the rail entirely, which is what it was set to.
MAX_NEW_PER_DAY = _i("LEVELS_MAX_NEW_PER_DAY", 5)
# 15%, not 10%. At 3% risk a 10% stop halts after 3 losing trades, and at the
# ~3.4 reward/risk this filter selects for, a no-edge win rate would be ~23% --
# so three straight losses is a 46% event and the system would spend most of
# its life halted, learning nothing. 15% allows 5, where five straight is 27%.
DAILY_LOSS_STOP_PCT = _f("LEVELS_DAILY_LOSS_STOP_PCT", 15.0)
POLL_SEC = _i("LEVELS_POLL_SEC", 900)
RESOLVE_MIN_BARS = _i("LEVELS_RESOLVE_MIN_BARS", 42)       # ~7d before judging
ONSET_MATCH_SIGMA = _f("LEVELS_ONSET_MATCH_SIGMA", 0.5)


def summary() -> str:
    return (f"gate(log): run>={RUN_SIGMA_MIN}s pull>={PULL_SIGMA_MIN}s "
            f"retests>={MIN_RETESTS} near<={GATE_MAX_NEAR_SIGMA}s | "
            f"trade(buy): depth>={TRADE_MIN_NEAR_SIGMA}s "
            f"rested>={TRADE_MIN_BARS_SINCE_TOUCH}b "
            f"headroom>={TRADE_MIN_OVERHEAD_SIGMA}s rr>={MIN_RR} | "
            f"entry +{ENTRY_BUFFER_SIGMA}s stop -{STOP_SIGMA_BELOW}s "
            f"risk {RISK_PCT_PER_TRADE}%")
