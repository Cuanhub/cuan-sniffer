# NOTES — things noticed while building, deliberately not built

Per the build brief: anything not in SIMPLIFY_PLAN.md goes here instead of into
the code.

## Deferred (not built)

- **Loss-streak stand-down: dropped.** The brief allowed keeping it "unless it's
  a few lines off `user_fills` `closedPnl`". It isn't. `closedPnl` is per *fill*,
  not per trade: closing one position in three fills yields three rows, so
  counting consecutive losing trades needs fills grouped into trades (by coin,
  contiguous time, and `startPosition` crossing zero). That is ~25 lines of
  fiddly grouping with its own failure modes. Dropped. If wanted later, the
  honest input is `startPosition` reaching 0, not a row count.
- **`/take` (forced entry): dropped**, as instructed. `FORCEABLE` and the
  `state["forced"]` plumbing go with it.
- **Websocket (AUDIT B-1).** Still absent. The 60s protection loop narrows the
  blind window from 450s to 60s but does not remove it; the venue-side stop is
  what actually covers a gap.
- **`positionTpsl` auto-sizing.** Unresolved (SIMPLIFY_PLAN §1.5) and no longer
  probeable, since the brief rules out testnet. The 60s loop re-asserts an
  explicit full-size stop every cycle, which is correct either way, so this is
  an optimisation we are simply not taking.
- **Deploy wiring.** `deploy/launchd/com.cuansniffer.reconcile.plist` points at
  the deleted `tools/reconcile_loop.py`. It needs to point at
  `tools/protect_loop.py` with a 60s interval. Not edited — config is yours.
  Note: `launchctl list` shows no `com.cuansniffer.reconcile` job loaded, so the
  reconcile loop was never actually running and REVIEW.md finding 2 was latent
  rather than live.

## Observations worth your attention

- **`hold` does move on this account.** Trader logs show
  `spot_usdc={'total': '698.509836', 'hold': '139.878885'}` while positions were
  open, and `hold: '0.0'` now that it is flat. The old
  `_get_margin_summary_unified` docstring said this could not be established.
  It can: `hold` tracks committed margin. Does not change the plan (equity is
  `total`), but it retires an open question.
- **`com.cuansniffer.trader` last exited with status 1** at the time of this
  work, which is the `halt(...); return 1` path in the old main loop.
- **The telegram control process is live** (pid 41917) and imports the modules
  being replaced. Commits are additive until the last one for this reason.
