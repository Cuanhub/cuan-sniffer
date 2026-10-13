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

# EVIDENCED at 5.0, lowered to 3.0 on 2026-09-20 because the evidence went
# stale. Overhead resistance was the stated reason for the AAVE skip and the
# ETH rejection, and takes averaged 7.58 sigma of headroom against 2.98 for
# skips -- at 37 labels it was 8.53 against 4.90, so the separation held.
#
# What changed is the stop. Every one of those labels was made when the stop
# sat a fixed 3 sigma below the level, and under that rule low headroom really
# did mean a bad trade: a wide stop needs a distant target to clear the rr
# floor, so the two moved together. The structural stop broke the link. ETH
# 2562.9 now prices at rr 4.09 with 3.40 sigma of headroom, because its stop is
# 0.94% rather than 5%. Headroom and reward/risk have decoupled, and the floor
# is still measuring the old relationship.
#
# 3.0 admits ETH 2562.9 (3.40s, rr 4.09), LIT 4.6 (3.19s, rr 3.11), HYPE 89.597
# (4.04s, rr 4.90) and SOL 107.555 (4.31s, rr 5.15) -- four setups labelled
# take by the user on 2026-09-20, each priced under the new stop rule.
#
# This is a threshold set to get trades rather than one fitted to evidence, and
# it should be treated as provisional until there are labels and outcomes on
# setups priced the new way. MIN_RR stays at 2.0: this loosens where the target
# may sit, not what the trade has to pay.
TRADE_MIN_OVERHEAD_SIGMA = _f("LEVELS_TRADE_MIN_OVERHEAD_SIGMA", 3.0)

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

# ---- the structural stop -----------------------------------------------------
# STOP_SIGMA_BELOW alone puts the stop a fixed distance under the level
# regardless of where the level's own lows sit, and on a tight base that is
# absurd. UNI 8.5254 is a double bottom whose lows are 8.4516 and 8.4554 -- an
# invalidation 0.9% away -- and the 3-sigma rule stopped it at 8.073, 5.9%
# below, which priced the trade at rr 0.91 and had the rr floor reject it. The
# user's own stop was 8.44. Same story on ZEC: 3-sigma gave rr 0.92, their
# 1426 gave 2.97. The stop was manufacturing the rejection.
#
# The structural stop goes just under the lows that actually formed the level,
# because that is where the idea is wrong rather than where a fixed multiple
# lands.
#
# Buffer under the lowest touch. Small: the point is to sit below the wick
# that made the level, not to add another sigma of room on top of it.
STOP_STRUCT_BUFFER_SIGMA = _f("LEVELS_STOP_STRUCT_BUFFER_SIGMA", 0.25)

# Never tighter than this below the level, whatever the lows say. A stop
# placed at the exact low is the price most likely to be wicked, and a base
# whose touches all land on one candle would otherwise produce a stop inches
# away. The user's own stops on ZEC and UNI sat 0.55-0.67 sigma under their
# levels, so 0.5 admits how they actually trade without permitting tighter.
STOP_MIN_SIGMA = _f("LEVELS_STOP_MIN_SIGMA", 0.5)

# How far back to look for the touches that define the level.
STOP_STRUCT_LOOKBACK = _i("LEVELS_STOP_STRUCT_LOOKBACK", 120)

# ---- rule 4: sizing ---------------------------------------------------------
# STRUCTURAL. Size comes from the distance to the stop; leverage is only ever a
# ceiling on the result, never an input. At fixed 10x a 10% stop costs the whole
# account and a 44% stop costs 4.4x of it, because stop distance is set by the
# chart rather than chosen.
# 3% for the initial trials, to be raised only as the system demonstrates
# profitability. Nothing measures this strategy's edge yet, so the first
# trades are being bought as information at roughly $24 each.
# Flat leverage. Every position is the same size: equity x FLAT_LEVERAGE,
# capped by what the venue allows on that coin and by free margin.
#
# Size used to be derived from the stop distance so that risk stayed pinned at
# RISK_PCT_PER_TRADE. That is defensible and it is also why sizing was
# unpredictable: a 0.9% stop asked for 3.3x equity, a 13% stop asked for
# 0.22x. The same idea produced a huge position or a pointless one depending
# on where the stop happened to land.
#
# Flat sizing gives up constant risk to get a system that behaves the same way
# every time. Risk per trade is then whatever the stop distance implies --
# 5x into a 2% stop risks 10% of equity, into a 10% stop risks 50% -- so the
# rails that still bind are MAX_CONCURRENT, MAX_NEW_PER_DAY and the daily loss
# stop, not a per-trade risk budget.
FLAT_LEVERAGE = _f("LEVELS_FLAT_LEVERAGE", 5.0)

# Share of equity committed as MARGIN on each trade. The position is this
# times FLAT_LEVERAGE, so 3% at 5x is a 15%-of-equity position: $23 of margin
# controlling $115 of notional on a $768 account.
#
# Note this is margin allocated, NOT risk taken. What the trade actually
# risks is the notional times the stop distance, which is smaller again -- the
# median structural stop sits 1.76% below the fill, so at 5x it risks 8.8% of
# the margin committed. That distinction is not pedantry: conflating the two is
# what left the daily loss stop fifty-four losses away from firing.
#
# 5%, raised from 3% on 2026-10-04. EVIDENCED by replaying all 39 closed trades
# with the rails simulated rather than rescaled, since a bigger loss trips the
# daily stop and the drawdown halt sooner and a halt removes the trades after
# it:
#
#     size   end equity        net   worst drawdown   drawdown halt
#       3%       795.47    +26.89             4.1%           never
#       5%       813.40    +44.82             6.5%           never
#       8%       840.29    +71.71             9.9%    never, by 0.1pt
#      10%       830.15    +61.57            10.9%      trips 10-02
#
# 5% nearly doubles the net for a drawdown comfortably inside the 10% halt.
# 8% was the best outcome and is NOT taken: it misses halting by a tenth of a
# point on a 39-trade sample in a mostly-rising market, which is luck rather
# than margin. The 10% row is the argument -- it ends up WORSE than 8% because
# the halt fired and cut off the winners behind it.
#
# Execution cost does not bind at this size. Measured across 17 fills since the
# slippage metric was corrected: median 0.6 bps, mean 2.8. Going 3% -> 5% adds
# about four cents per round trip against an average win of $7.44.
# 5% again on 2026-10-10, after 3% -> 5% (10-04), 5% -> 3% (10-08) and back.
#
# Worth keeping the correction visible rather than quietly restoring it: the
# original 5% was chosen by replaying 39 trades whose worst drawdown was 4.1%,
# a sample drawn almost entirely from a rising market, and the drawdown halt
# fired four days later. That analysis argued against 8% precisely because
# sizing up on one regime's book is unsafe -- 5% carried the same error.
#
# So this is restored on a judgement about the tape, not on evidence that 5%
# is right. If the tape is wrong, this is the dial that makes it expensive.
MARGIN_PCT_PER_TRADE = _f("LEVELS_MARGIN_PCT", 5.0)

# Kept because the review flags and the trade log still report what a position
# risks; it no longer decides size.
RISK_PCT_PER_TRADE = _f("LEVELS_RISK_PCT", 3.0)
MAX_LEVERAGE = _f("LEVELS_MAX_LEVERAGE", 10.0)
# Lowered from 2.5: at 2.5 the filter kept 6 of 11 takes, at 2.0 it keeps 7
# while still admitting 0 of 5 skips. Live median reward/risk is 3.40 anyway,
# so the floor rarely binds.
MIN_RR = _f("LEVELS_MIN_RR", 2.0)
MIN_NOTIONAL_USD = _f("LEVELS_MIN_NOTIONAL", 12.0)

# How far the CURRENT hourly bar may have traded below the level before the
# setup is refused, as a fraction of the distance from the level to the stop.
# 1.0 means the bar has already touched the stop. 0 disables the check.
#
# entry_is_actionable already refuses when price is at or below the stop -- but
# it tests the MID, so a dip through the stop that bounced back before the
# pass looked at it sails through. Extending the same rule to the bar's LOW is
# a bug fix rather than a new filter: buying a level after the bar has traded
# past the point where we would admit the trade was wrong is incoherent
# whatever the data says.
#
# And the data is blunt about it. Across 53 live entries, bucketing by how far
# the live bar's low sat from the level toward the stop:
#
#     above the level          15 trades   47% win   +$34.34
#     0-50% toward the stop    18 trades   28% win   +$ 3.82
#     50-100%                   9 trades   33% win   -$ 1.63
#     already past the stop    11 trades    0% win   -$27.37
#
# Eleven entries, not one winner. That is the trade the operator described on
# STRK as "no double bottom had been created yet" -- the level is being
# violated as we buy it, and level_has_held cannot see it because it judges
# closed bars only and the forming bar is excluded by design.
#
# 1.0 is the unarguable setting and needs no evidence. Tightening toward 0.5
# would also have caught STRK itself (which sat at 0.505) and vetoed 20 trades
# of which 3 won, but that IS a fitted threshold on 53 trades -- it keeps
# fewer trades for the same profit, so it buys nothing except the one case.
ENTRY_MAX_BREAK_FRAC = _f("LEVELS_ENTRY_MAX_BREAK_FRAC", 1.0)

# Refuse a long when the perp is trading this far ABOVE the oracle, in bps.
# Negative disables the check; 0.0 means "refuse any premium to spot".
#
# EVIDENCED, and the first feature in this project to reach statistical
# significance. Screened over 248 resolved episodes carrying positioning
# context, scored on expectancy and stratified by reward/risk the same way the
# price features were -- all nineteen of which failed:
#
#     mark-vs-oracle basis        n    mean R   win rate
#     Q1   -52.5 .. -2.0 bps      62    -0.052      27%
#     Q2    -1.9 .. +1.9          62    +0.006      27%
#     Q3    +2.0 .. +7.7          62    -0.485      16%
#     Q4    +7.8 .. +29.3         62    -0.694      11%
#
#     perp ABOVE oracle  n=153  mean R -0.524  p = 0.0035
#     perp AT or BELOW   n= 95  mean R +0.044  p = 0.0024
#
# Mark above oracle means perp buyers are paying a premium over spot --
# leveraged longs crowding in. Buying support while longs already pay up is
# buying a crowded trade, and crowded longs are what get flushed. It is the
# mechanism funding is supposed to measure and cannot, because Hyperliquid
# pins funding at its base rate 77% of the time.
#
# 0.0 rather than a tuned number: it is the natural boundary (perp at parity
# with spot) and the only cut in the sweep where the KEPT set has positive
# expectancy. Wider cuts veto fewer trades but degrade what is kept --
# +0.044 at 0bps, -0.031 at 2bps, -0.083 at 4bps, -0.127 at 6bps.
#
# Held as a veto to be re-measured, not as a sizing input. The cohort is one
# regime and a losing one, 248 is enough for significance and not for comfort,
# and it is the first of roughly thirty features screened to pass -- some of
# that is luck by construction. The sample reaches ~500 by 2026-10-15.
ENTRY_MAX_BASIS_BPS = _f("LEVELS_ENTRY_MAX_BASIS_BPS", 0.0)

# How many resistance rungs above price the target may choose between.
#
# 3, which is what call.py already displayed as its ladder -- this only makes
# the choice from it rather than always taking the first. Aiming at the nearest
# rung tied the target to the admission test: the ceiling above a support
# barely moves, so a support near price has little headroom and poor
# reward/risk however good the support itself is, and every freshly formed
# shelf was refused for it. Measured 2026-09-28 across the live universe, 41
# levels were being refused that way, median 42% return on margin and median 7
# touches on the rung they should have aimed at.
#
# GUESSED at 3. Raising it reaches for bigger targets that are less likely to
# be hit and hold a slot longer; the take-profit study found per-slot-day
# return is U-shaped in target distance, so further is not automatically
# better. Worth revisiting once there are labels on which rung was right.
TARGET_MAX_RUNGS = _i("LEVELS_TARGET_MAX_RUNGS", 3)

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

# The 2h hold is for FLIPPED RESISTANCE only, and that is what it was always
# for. The user's words: it "was more in the notion of a channel forming on a
# recent flip from s/r". Broken resistance becoming support is a claim that
# has to be proven, and watching it hold a channel for two hours is the proof.
#
# Ordinary support is a different animal. It taps and sends -- the user again:
# "support usually taps then sends higher and doesn't sit for 2h". Demanding
# two hours of sitting there asks it to behave like the thing it is not, and
# the cost is not theoretical: by the time the hold completes, price has run,
# and MAX_ENTRY_DRIFT_PCT then refuses the fill. That is the miss measured on
# 2026-09-19, where not one of 139 tradeable levels was inside the 1.5% drift
# guard and the nearest was 3.10% away.
#
# 1 bar for a swing low: the level was reached, and the last closed candle is
# back above it. A tap and a reclaim, which is the event itself rather than a
# wait for it to become something else.
TRIGGER_HOLD_HOURS_SWING = _i("LEVELS_TRIGGER_HOLD_HOURS_SWING", 1)

# When a level BEHAVES like a flipped resistance regardless of how it was
# classified: it capped price for at least FLIP_CAPPED_BARS_MIN of the last 120
# bars, and did so within the last FLIP_CAPPED_RECENT_BARS.
#
# These two numbers already existed, hardcoded inside review_flags, where they
# only produced an advisory line. They now decide which HOLD a setup must
# clear, so they belong here under a name.
#
# The reason is a classification bug, not a fitted filter. level_origin marks
# only what the resistance detector happened to find, so a level can be
# labelled swing_low while plainly still acting as resistance -- and a
# swing_low gets the easy 1h hold instead of the 2h channel hold a flip must
# prove. ARB 0.186655 was bought on 2026-10-13 after capping price for 66 of
# the last 120 bars, most recently 6 bars earlier, on a 1h hold. SOL 101.72 was
# the same shape: scored swing_low while having capped price 95 of 120 bars,
# last 12 bars ago.
#
# Deliberately NOT justified by outcomes. Hold rate, touch count and the
# capping flag itself were all screened against the live book and all came out
# flat -- the capping flag at p = 0.74. This corrects which rule a setup is
# measured by, which is a question of consistency rather than of edge.
FLIP_CAPPED_BARS_MIN = _i("LEVELS_FLIP_CAPPED_BARS_MIN", 25)
FLIP_CAPPED_RECENT_BARS = _i("LEVELS_FLIP_CAPPED_RECENT_BARS", 48)
TRIGGER_TOUCH_LOOKBACK_HOURS = _i("LEVELS_TRIGGER_TOUCH_LOOKBACK_HOURS", 12)

# How far above the level price may sit and still count as holding AT it.
# Without this, "held" meant nothing more than "is above", which a level price
# has run away from satisfies trivially: UNI touched 7.5078 twelve hours ago,
# rallied to 8.64, and the trigger read that as two hours of support 16% below
# the market. The channel has a ceiling as well as a floor.
# 1.0, tightened from 2.0 on 2026-09-20 because 2.0 could not be filled.
#
# The trigger and the entry drift guard were arithmetically incompatible. The
# trigger fired up to 2.0 sigma above the level; entry sits 0.35 sigma above
# it; so at the trigger's own limit price is 1.65 sigma past the entry, and
# MAX_ENTRY_DRIFT_PCT allows 1.5%. That only fits where sigma <= 0.909% --
# seven of thirty coins. Everything volatile enough to trigger often was
# barred from filling by arithmetic, not by judgement. USELESS triggered three
# times on 2026-09-20 at 4.28%, 4.42% drift; at sigma 2.40% it could not have
# done otherwise.
#
# At 1.0 the worst case is 0.65 sigma past entry, which fits 1.5% wherever
# sigma <= 2.31% -- 26 of 30. It also matches TRIGGER_ZONE_SIGMA: price must
# still be inside the band that counts as touching the level, rather than a
# band twice that wide.
#
# The cost is fewer triggers: price has to be caught nearer the level instead
# of anywhere in a 2-sigma window above it. That is the intended trade. A
# trigger that cannot be filled is not a signal, it is a log line.
TRIGGER_MAX_ABOVE_SIGMA = _f("LEVELS_TRIGGER_MAX_ABOVE_SIGMA", 1.0)

# How long a level stays on the watchlist after it clears the quality bar.
#
# STRUCTURAL. The quality filter and the entry trigger describe the SAME level
# at two different moments, and they disagree by construction: depth demands
# price sit >=5 sigma ABOVE the level, the trigger demands it sit <=2 sigma
# above. Evaluated in one pass on one candidate those are an empty set, which
# is why the trader took zero trades between 2026-09-18 15:53 and 2026-09-19
# 00:00 -- 3041 decisions, every one "not_triggered", the trigger never once
# firing. trigger.py always said this out loud: the filters pick levels "worth
# watching, while price is still 5-10 sigma above them", and the trigger
# decides "the moment of entry", which "is a separate question".
#
# Watching is what connects the two moments. A level admitted while price is
# far above stays eligible while price comes back to it, so depth and rest are
# checked once, at admission, rather than again at entry where they can no
# longer be true.
#
# 48h: the user's takes sit 0.18-10.00 sigma below price (median 5.80), and a
# 5-sigma pullback is a multi-day move for a major. Too short silently
# re-creates the deadlock for anything slower than the window.
WATCH_TTL_HOURS = _i("LEVELS_WATCH_TTL_HOURS", 48)

# How close a watched level must be before it is worth telling a human about.
#
# The trigger fires at <=2.0 sigma above the level, so 3.0 means "within about
# one sigma of being tradeable". Alerting on everything eligible would mean
# ~140 levels a pass, which is not a signal, it is a mailing list.
# 5.0, raised from 3.0 on 2026-09-19 because 3.0 sent nothing at all.
#
# Measured against 164 tradeable levels that morning: the closest sat 3.3
# sigma above its level, p10 was 5.5 and the median 9.8. In a market where
# price taps support and runs, almost nothing lingers within one sigma of the
# trigger, so a threshold one sigma out is silent exactly when there is the
# most to watch. At 5.0 about a dozen qualify, and since the digest already
# takes the five closest and sends at most one an hour, a looser threshold
# costs nothing in volume -- it only decides how far out "approaching" starts.
NEAR_ALERT_SIGMA = _f("LEVELS_NEAR_ALERT_SIGMA", 5.0)

# At most one approaching-levels digest per hour. The board is there for
# looking on purpose; this is only so a zone does not get reached unnoticed.
NEAR_ALERT_COOLDOWN_SEC = _i("LEVELS_NEAR_ALERT_COOLDOWN_SEC", 3600)

# How long a resting entry bid stays on the book before it is pulled.
#
# A bid is only useful while the setup it was placed for is still the setup:
# levels break, the channel ceiling moves, and a bid left sitting is an
# opinion nobody is holding any more. 12h, against a 48h watchlist TTL,
# because an order carries money and a watch does not.
LIMIT_ENTRY_TTL_HOURS = _i("LEVELS_LIMIT_ENTRY_TTL_HOURS", 12)

# ---- operational rails ------------------------------------------------------
# 4, raised from 2 on 2026-09-26. EVIDENCED: across the log to that date, 130
# passes declined to open with the message "holding 2 and bidding on 0, max 2"
# -- roughly 16 hours where a qualifying setup could not be taken because the
# cap, not the strategy, said no. One of those two slots was held for about two
# days by a PONS position the venue had already closed, which is fixed
# separately; the rail was still the binding constraint the rest of the time.
#
# What this costs, at MARGIN_PCT_PER_TRADE=5 and FLAT_LEVERAGE=5: four
# positions commit 20% of equity as margin and 100% as notional, and four
# structural stops firing together is about 1.8% of equity -- well inside the
# 5% daily loss stop, so that rail still bites first and this one does not
# quietly become the risk limit.
#
# Watch the interaction with MAX_NEW_PER_DAY below: filling four slots spends
# four of the five daily opens, leaving one replacement. Live fill rate is
# ~1.1/day so it should not bind, but if the daily cap starts refusing trades
# while slots are free, that is this change asking for the other one.
# 4 again on 2026-10-10, having gone 2 -> 4 (09-26), 4 -> 2 (10-08) and back.
# Cut when the drawdown halt fired at 735.43 against an 817.51 peak; restored
# on the operator's read that the market has settled.
#
# A regime dial, not a finding. Nothing in the data says 4 beats 2 -- the one
# measurement that exists is unflattering, 0-for-12 on 10-07 with four slots
# open in a falling tape. What the dial buys is participation when levels are
# being reached; what it costs is four ways to be wrong at once.
MAX_CONCURRENT = _i("LEVELS_MAX_CONCURRENT", 4)
# 5, matching the daily loss stop's own arithmetic: at 3% risk a 15% stop
# halts after five losing trades, so a cap of five never takes a trade the
# risk budget would have allowed. What it bounds is the case that budget
# cannot see -- opening and closing without losing enough to trip anything.
# A protection failure that repeats does exactly that, paying a round trip in
# fees and spread each pass. opened_today is incremented on FILL, before
# protection is placed, so a trade that is immediately emergency-closed still
# spends one of the five; for a churn bound that is the point.
# 0 disables the rail entirely, which is what it was set to.
# 20, raised from 5 on 2026-10-01, and demoted from the primary rail to a
# backstop. Drawdown is what should bound the day -- a count cap refuses good
# setups while the account is flat and perfectly healthy, which it did 433
# times in the log, more than every other reason combined.
#
# It is not removed, because the case it bounds is the one no loss stop can
# see: opening and closing repeatedly while losing too little to trip
# anything. The per-coin cooldown now covers most of that, and 20 sits well
# above the ~18 losses the 5% daily stop allows, so in practice drawdown binds
# first and this only catches a runaway loop.
MAX_NEW_PER_DAY = _i("LEVELS_MAX_NEW_PER_DAY", 20)
# How long a coin is left alone after it stopped us out. 0 disables.
#
# EVIDENCED, and specifically NOT after a winning close. Across the 26 live
# trades to 2026-09-30 there were 12 re-entries:
#
#   after a LOSS, under 2h :  3 trades   0W/3L   -$10.06
#   after a LOSS, 15h plus :  7 trades   2W/5L   +$ 3.96
#   after a WIN            :  2 trades   2W/0L   +$26.63
#
# The harm is concentrated and obvious. LIT was bought three times inside four
# hours on 2026-09-30 -- twice at the same level, 28 and 19 minutes after being
# stopped out of it -- while being the worst performer in the universe at -10%.
# That one coin was $13.16 of a $19.99 losing run. A level that just stopped us
# out is a level that broke; buying it again immediately is buying the break.
#
# A WIN means the level did its job, and both re-entries after one were
# profitable (PUMP +20.35 re-entered 3.4h after a +4.25). So this keys off the
# realised result, not merely off closing.
#
# 6h is the middle of the gap in the evidence: nothing in the sample sits
# between 0.8h and 15.9h, so any value from 1h to 15h blocks the same three
# losers and preserves the same winners. 6h claims no more precision than that.
COIN_REENTRY_COOLDOWN_HOURS = _f("LEVELS_COIN_REENTRY_COOLDOWN_HOURS", 6.0)

# Minimum gap between two entries, in minutes, regardless of coin. 0 disables.
#
# Not a tuned parameter but a correction to a design error. The open loop fills
# every free slot in a single pass, so four triggers in one scan became four
# positions in the SAME MINUTE -- APT, AVAX, HYPE and INJ at 00:06 on
# 2026-10-13, all six of that morning's entries closed inside the hour for
# -$18.99 and the account drawdown halt fired.
#
# In a universe whose median pairwise hourly correlation is 0.47 (0.54 against
# BTC), four simultaneous longs are not four bets. They are one bet sized four
# times: four positions behave like 2.4 independent ones.
#
# EVIDENCED, and it is the largest effect in the live book by a wide margin.
# Across 90 closed trades, split by how many other entries landed within an
# hour:
#
#     alone        33 trades   45% win   +$59.90
#     +1 other     20 trades   15% win   -$32.83
#     +2 others     9 trades    0% win   -$24.40
#     +3 or more   28 trades   14% win   -$37.36
#
# Isolated entries have made money. Clustered entries have lost $94.59. The
# account would be up about $60 without them.
#
# 60 minutes because that is the window the clustering was measured over, not
# because 60 optimises anything. What it buys is that each position gets an
# hour to declare itself before another correlated one is added behind it.
ENTRY_SPACING_MINUTES = _f("LEVELS_ENTRY_SPACING_MINUTES", 60.0)
# 5%, lowered from 15% on 2026-10-01. The old value was calibrated against a
# risk-per-trade that was wrong by about 11x.
#
# Its reasoning -- "at 3% risk a 10% stop halts after 3 losing trades" -- was
# sound, but 3% is MARGIN_PCT_PER_TRADE, the margin committed, not the amount
# at risk. The structural stop sits a median 1.76% below the fill, so at 5x the
# real risk is $2.11 on $797 of equity: 0.26%, not 3%. Measured against the
# observed $2.20 average loss, a 15% stop halts after FIFTY-FOUR consecutive
# losses, which is not a rail. 5% halts after ~18.
#
# Same margin-vs-risk conflation that once sized positions off equity x
# leverage instead of margin x leverage.
#
# EVIDENCED against the live book: the worst day so far was 6 trades for
# -$12.44, or 1.5%, so this leaves roughly 3x headroom over anything actually
# seen while still bounding a blowout.
DAILY_LOSS_STOP_PCT = _f("LEVELS_DAILY_LOSS_STOP_PCT", 5.0)

# Halt when equity falls this far below its own high-water mark. 0 disables.
#
# The rail that did not exist. The daily stop resets at 00:00 UTC, so a slow
# bleed never trips it: three days to 2026-09-30 lost $19.99 across 17 trades
# and no single day came close to halting. Nothing measured the account's
# trajectory at all.
#
# 10% of the peak. Worst drawdown through the first 26 trades was 3.25%, so
# this is about 3x headroom -- loose enough not to fire on normal variance,
# tight enough to stop a strategy that has quietly stopped working.
#
# Mark to market, not realised-only. I wrote the opposite here on 2026-10-01
# and it was wrong: on a unified account the spot USDC balance this reads
# ALREADY carries unrealised PnL on open perps. Measured 2026-10-02 with two
# positions open -- equity 808.95, unrealised +4.16, and the realised-only
# ledger 804.88, which is 808.95 - 4.16 to within a cent of market drift
# between the two calls.
#
# That makes the rail better than the old note claimed, not worse: an open
# drawdown trips it at the time it happens rather than when it is booked. The
# consequence worth knowing is that it can halt while positions are still open.
# That is correct -- a halt blocks NEW opens only, and reconcile runs before
# preflight, so the open book keeps its stops and its protection either way.
ACCOUNT_DD_HALT_PCT = _f("LEVELS_ACCOUNT_DD_HALT_PCT", 10.0)

# How many days of OPENING balances the drawdown is measured against. The
# reference is the highest of the last N day-opens, so 1 means "today's
# opening balance" and a large number approximates the old all-time peak.
#
# Changed from an all-time high-water mark on 2026-10-13. That mark was set at
# 817.51 on 09-27 and never came down, so the rail became a ratchet: the
# longer the account sat below its best-ever day the less room it had, and it
# fired twice in six days while the account was flat-to-recovering. A rail
# that tightens because of something that happened two weeks ago is measuring
# history, not risk.
#
# 3, not 1. Measured against the real equity path since 09-20, the worst
# drawdown by reference was:
#
#     all-time peak            10.25%   -> the 10% limit fires (it did, twice)
#     that day's opening balance 5.07%  -> a 10% limit NEVER fires
#     best of the last 3 opens   7.46%  -> a 10% limit never fires, 2.5pt margin
#
# A one-day reference at 10% is a dead rail -- it has never come within half
# the limit, and the worst single day (-5.07%) is already what the 5% realised
# daily stop catches.
#
# What is GIVEN UP, and it is worth being exact because I first wrote the
# opposite here: a trailing reference cannot catch a slow grind, because the
# reference falls with the account. A steady -6.66/day -- the rate that cost
# $19.99 over the three days to 09-30 -- measures as 0.81% on day one and only
# 3.13% by day thirty against a 3-day window. It never halts. Only an
# all-time mark catches that shape, and an all-time mark is the ratchet we are
# removing.
#
# So this rail now catches a FAST collapse (10% or worse inside three days,
# which 10-11..10-13's -7.46% approached) and no longer catches a slow one.
# The slow case is left to the operator reading the daily report, deliberately,
# because the alternative is a rail that tightens forever.
#
# Set to 1 for a strictly daily reference, but tighten ACCOUNT_DD_HALT_PCT to
# about 6% at the same time or the rail stops doing anything at all.
ACCOUNT_DD_LOOKBACK_DAYS = _i("LEVELS_ACCOUNT_DD_LOOKBACK_DAYS", 3)
# 450s, halved from 900 on 2026-09-19.
#
# The bot sends a marketable order when the trigger fires and refuses it if
# price has drifted more than MAX_ENTRY_DRIFT_PCT from the plan. Support that
# taps and sends is therefore gone before a 15-minute poll looks: measured
# across 139 tradeable watched levels, NOT ONE was inside the 1.5% drift
# guard, the nearest being 3.10%. Polling twice as often halves that blind
# window. It does not close it -- a tap that resolves inside 7.5 minutes is
# still missed, and catching those needs a resting bid at the level rather
# than a faster loop.
#
# Affordable only because the 1h bars the trigger reads are now fetched once
# per coin per pass instead of once per candidate: ~140 fetches became ~30,
# so twice the passes cost less than the old single pass. Watch the 429 rate
# before cutting this further; a rate limit here truncates the pass.
POLL_SEC = _i("LEVELS_POLL_SEC", 450)

# The scanner's own cadence, deliberately NOT the trader's.
#
# scanner.py imported POLL_SEC directly, so halving the trader's loop to
# catch faster entries silently halved the scanner's too -- doubling the
# universe scans of a job that has no reason to hurry. The trader polls
# quickly because it is trying to catch a support tap before price leaves;
# the scanner is accumulating a record, and a record does not get better for
# being sampled twice as often, only more expensive. Two jobs, two knobs.
#
# 900s, which is what the scanner ran at before the trader's change reached
# it by accident.
SCANNER_POLL_SEC = _i("LEVELS_SCANNER_POLL_SEC", 900)
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
