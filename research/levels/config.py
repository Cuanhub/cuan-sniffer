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

# EVIDENCED. Hold rate is what separates a level price respects from one it
# merely crosses: BTC 74,900 held 10 of 10 tests; a naive band around CASHCAT
# 0.14 held 8 of 18. Observed spread across detected levels is 0.43-1.00.
MIN_HOLD_RATE = _f("LEVELS_MIN_HOLD_RATE", 0.90)     # threshold itself GUESSED
MIN_TOUCHES = _i("LEVELS_MIN_TOUCHES", 8)            # GUESSED

# ---- when is the level live -------------------------------------------------
# EVIDENCED that timing dominates: the same limit at 0.1428 on CASHCAT left
# resting from 08-20 rides a -73.67% drawdown, while placed on 09-08 it fills
# at the same price and never goes more than -0.48% underwater. A 150x
# difference in drawdown from nothing but when the order became live.
# The THRESHOLDS are GUESSED -- fitted to two levels named after the fact.
RUN_LOOKBACK = _i("LEVELS_RUN_LOOKBACK", 84)
RUN_SIGMA_MIN = _f("LEVELS_RUN_SIGMA_MIN", 6.0)
PULL_SIGMA_MIN = _f("LEVELS_PULL_SIGMA_MIN", 5.0)
RETEST_LOOKBACK = _i("LEVELS_RETEST_LOOKBACK", 180)
MIN_RETESTS = _i("LEVELS_MIN_RETESTS", 2)
RETEST_BAND_SIGMA = _f("LEVELS_RETEST_BAND_SIGMA", 1.0)

# GUESSED. Single definition -- this was previously applied twice, at 5.0 in
# the gate and 2.0 in the trader, so a setup could pass one and fail the other
# with nothing naming which was authoritative.
MAX_NEAR_SIGMA = _f("LEVELS_MAX_NEAR_SIGMA", 2.0)

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
RISK_PCT_PER_TRADE = _f("LEVELS_RISK_PCT", 5.0)
MAX_LEVERAGE = _f("LEVELS_MAX_LEVERAGE", 10.0)
MIN_RR = _f("LEVELS_MIN_RR", 2.5)                 # GUESSED
MIN_NOTIONAL_USD = _f("LEVELS_MIN_NOTIONAL", 12.0)

# ---- operational rails ------------------------------------------------------
MAX_CONCURRENT = _i("LEVELS_MAX_CONCURRENT", 2)
MAX_NEW_PER_DAY = _i("LEVELS_MAX_NEW_PER_DAY", 0)          # 0 = unlimited
DAILY_LOSS_STOP_PCT = _f("LEVELS_DAILY_LOSS_STOP_PCT", 10.0)
POLL_SEC = _i("LEVELS_POLL_SEC", 900)
RESOLVE_MIN_BARS = _i("LEVELS_RESOLVE_MIN_BARS", 42)       # ~7d before judging
ONSET_MATCH_SIGMA = _f("LEVELS_ONSET_MATCH_SIGMA", 0.5)


def summary() -> str:
    return (f"level: hold>={MIN_HOLD_RATE} touches>={MIN_TOUCHES} | "
            f"timing: run>={RUN_SIGMA_MIN}s pull>={PULL_SIGMA_MIN}s "
            f"retests>={MIN_RETESTS} near<={MAX_NEAR_SIGMA}s | "
            f"trade: entry +{ENTRY_BUFFER_SIGMA}s stop -{STOP_SIGMA_BELOW}s "
            f"rr>={MIN_RR} risk {RISK_PCT_PER_TRADE}%")
