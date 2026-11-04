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

Do NOT queue without new reasoning: killzone timing, HTF bias, per-coin
selection, lowering the reward:risk floor -- see graveyard.md.
