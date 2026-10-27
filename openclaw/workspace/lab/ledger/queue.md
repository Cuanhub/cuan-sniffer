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

3. **MIN_RR from the operator's boundary labels.** Do refusals at rr
   1.5-2.0 lose? Needs `labels.csv` rows on boundary cases (tools/boundary).
   Grounding: the rr floor was set by inference from a different strategy's
   episodes. This one needs the operator to label first -- ask, don't
   guess.

4. **Re-entry cooldown on the bid path.** The 6h post-loss cooldown only
   applies to market entries; ~90% of entries are bids. Measure whether
   applying it to bids would have helped (band.run(cooldown_on_bids=True)
   vs False, paired). Grounding: 2026-10-27 audit finding.

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

Do NOT queue without new reasoning: killzone timing, HTF bias, per-coin
selection -- see graveyard.md.
