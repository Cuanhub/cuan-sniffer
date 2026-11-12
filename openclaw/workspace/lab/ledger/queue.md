# Research queue

Ordered. Each item says what grounds it. Take from the top unless a result
or `lab.py why` gives a better-grounded idea; then insert it with its
grounding. Remove an item when it is registered (note the experiment id).

1. **Stop multiple, re-measured on the conservative engine.** 1.0x / 1.5x /
   2.0x / 2.5x ATR under the band low. Grounding: stop/ATR is the dominant
   known effect (t = +9.75), but the 1.5x choice came from a scratch harness
   that resolved market entries on their own decision bar and selected
   plans[0] -- both since fixed. Primary: per-trade net R difference vs the
   live 1.5x, paired by setup.

2. **Target distance cap.** Refuse trades whose target is more than X% away.
   Grounding: the scratch harness showed "target <= 5.67%" at +27.2R over
   352 trades vs +17.1R/491 uncapped -- untested properly, and a far target
   is also a slow one (more exposure to the 3-loss stand-down).

3. **Bids vs market entries within a reward:risk bucket.** Descriptive in
   H-RRFLOOR: filled bids in rr 1-2 did worse than random bids (-0.095R,
   iid p 0.014) while market entries did not (+0.046). Untested -- a
   question about adverse selection when a bid fills on a falling price.
   (The floor itself is settled: H-RRFLOOR found no edge in any bucket.)

4. **Re-entry cooldown on the bid path.** The 6h post-loss cooldown only
   applies to market entries; ~90% of entries are bids. Measure whether
   applying it to bids would have helped (band.run(cooldown_on_bids=True)
   vs False, paired). Grounding: 2026-10-27 audit finding. A descriptive
   look on that date found no detectable effect (907 -> 901 trades, -46.6R ->
   -46.5R; the 46 blocked trades averaged +0.228R) -- so this is low value
   unless new data or a different window suggests otherwise.

5. **Band touch count as a veto.** Trades on bands reached into more than N
   times since forming. Grounding: the operator's bands sit at the 8th
   percentile of touches; the live code flags > 20 for review but never
   refuses.

6. **Bid TTL.** 6h / 12h / 24h. Grounding: arbitrary today (12h); a stale
   bid fills in a different market than the one that justified it.

7. **Market vs bid entries.** Are in-band market entries worse than bids?
   Grounding: they are the minority path and have never been measured
   apart; no market band entry has filled live yet.

8. **Liquidity sweep -> reclaim -> order block, as an entry confirmation.**
   The operator asked for it (2026-10-26). Use the trendline H2 design:
   compare against the same wait-and-enter mechanics with the sweep events
   moved to random times (circular shift) -- NOT against the baseline,
   which rewards abstention. Graveyard: sweep confirmation died at n = 15;
   this is the full-universe, controlled version.

9. **Trendline-break confirmation entry, forward re-test.** H-TRENDLINE H2
   showed +0.035R per setup over random-event waiting (p 0.037 uncorrected,
   Holm 0.097), positive in both halves and all 29 variants. Re-run the
   SAME design (research/backtest/experiments/trendline_breakout.py, H2
   only) once the archive holds 60+ days of bars that did not exist on
   2026-10-27 -- that new data is a clean out-of-sample test. Grounding:
   consistent but unestablished; the one trendline framing with signal.

10. **Volatility regime.** ATR percentile at decision (the template's
   example). Grounding: none beyond plausibility -- low priority.

11. **Funding extremes as a veto.** Context data is collected live but no
    funding field has earned evidence. Grounding: the basis veto (perp
    above oracle) is the only context feature that has.

12. **15m confirmation, re-test.** H-LTF15M's design, unchanged, once the 15m
    archive holds ~6 months (the first test had 52 days and could not rule
    out a material effect).

13. **Kronos K1, forward re-test.** H-KRONOS's K1 alone -- same model,
    settings and rule (research/backtest/experiments/kronos_filter.py) -- on
    trades from bars after 2026-10-27, once ~900 exist (~6 months).
    Grounding: +0.193R, positive in both halves, not established (Holm p
    0.41). A long shot: at that size six months detect it about one time in
    four, and its approved trades were break-even (+0.010R). Runs from the
    repo (research/kronos/.venv), not the lab: the package has no PyTorch.

14. **Weekly cross-sectional momentum** (28-day formation, 7-day hold,
    terciles, equal weight). Grounding: the predictability scan (exploratory)
    found +64 bps a week over 195 weeks of Hyperliquid daily candles, z 1.65,
    Holm 0.59, breaking even at a 52 bps one-way fee; Liu, Tsyvinski & Wu
    (2022, JF) on 2014-2018. Power: ~560 weeks for 80% at this size, so it
    needs a longer, broader daily history than this venue's, pre-registered
    there first, then a forward test here. Survivorship (today's top 45) must
    be addressed in the design.

15. **Delta-neutral funding carry -- MEASURED (H-CARRY, 2026-11-05).** What
    remains is not a test but a build and a forward run: spot orders in the
    bot, an hourly rebalancer that tops up each short's margin from spot with
    limit orders, exits on a funding turn or a spot-perp gap beyond a bound.
    Grounding: results/carry (every coin paid on average; ~3-14% a year on
    capital at the last 12 months' rates; the squeeze and the wrapped token
    are the risks). The user's call.

16. **Daily trend-following, forward run -- H-TREND PASSED (2026-11-12).**
    Each coin held in the direction of its 14/28/56-day trend, sized to its
    30-day volatility, long and short, ten majors. Grounding:
    research/backtest/results/trend_daily -- +12.6 bps a day over rotated
    controls (Holm p 0.002), both halves positive, net Sharpe 1.17 (CI 0.40
    to 1.90), survives a day's delay and 20 bps costs; 2022-2026 is weaker
    (Sharpe 0.5 to 1.0) and the test is on Binance spot bars. What remains
    is the forward run: research/trend/paper.py is built (paper only,
    import-isolated, criteria in research/trend/PREREG_forward.md) and
    starts when the operator loads com.cuansniffer.trendpaper.plist.

Do NOT queue without new reasoning: killzone timing, HTF bias, per-coin
selection, lowering the reward:risk floor -- see graveyard.md. Also, from the
predictability scan: hourly price-pattern rules on this universe (nothing at
1h to a week beyond the last hour's bounce), switching on "trending vs
ranging" regimes (trendiness does not persist), and premium as a
cross-sectional signal (nothing in 60 days).
