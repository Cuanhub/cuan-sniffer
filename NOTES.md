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

## Found while building, noted rather than built

- **`execute.close` cannot close a position whose limit price has no
  representation.** `round_price` refuses rather than substituting (a real fix
  — the old code substituted the smallest increment), but that means a
  degenerate mark price makes a position both unprotectable and uncloseable.
  `flatten_all` reports the failure loudly, which is the honest outcome. A
  market-order fallback would close it, but that is a new path and not in the
  plan. Only reachable at mark prices that cannot occur for a real perp
  (`test_a_stop_that_would_round_across_the_mark_is_refused` uses 1e-9 on BTC).
- **`flatten_all` cancels resting orders before closing** (rule 5 as written).
  That briefly removes the stops. If the venue then goes down for the full
  120s window, the positions are left both open and stopless, where before
  they at least had stops. The 60s protection loop re-places them on its next
  cycle, so the exposure is bounded by that, not by the outage. Worth knowing;
  the alternative ordering has its own problem (a stop firing mid-flatten
  double-reduces).
- **A `TP` is best-effort and never flattens.** A missing take-profit costs
  upside, not safety. Only the stop is a safety invariant.

## Integration status (deliberate stopping point)

The five `core/` modules and `tools/protect_loop.py` are built, tested and
mypy-clean. **The old code is still in place and still running.** Nothing has
been deleted yet, and no caller has been switched over.

That is on purpose. Switching one caller and not the other would leave two live
order-placing paths at once — two sources of truth for what the account holds,
which is the exact problem class this rebuild removes. The deletion phase is
all-or-nothing and is the next piece of work:

1. `core/venue.py` needs three methods the trader still calls and the plan did
   not list, all replacing existing behaviour rather than adding any:
   `set_leverage` / `max_leverage_for` (Hyperliquid enforces leverage per
   asset, so sizing is meaningless without them) and an `available` figure on
   `Equity` from `tokenToAvailableAfterMaintenance`, which sizing caps against.
2. Rewire `tools/live_levels_trader.py`: 29 call sites across 16 backend
   methods. `place_limit_entry` and `get_realised_pnl` have no replacement by
   design (rules 3 and 6).
3. Rewire `tools/telegram_control.py`: `/close`, `/closeall`, `/halt` become
   one call each into `core.safety`. Drop `/take`.
4. Delete `live_execution_backend.py`, `execution_backend.py`,
   `live_data_guard.py`, `tools/reconcile_loop.py`, and the ~1,160 lines of
   execution/safety functions inside `live_levels_trader.py`.
5. Retire the old tests that cover deleted code: `test_flatten.py`,
   `test_protection_verification.py`, `test_reconcile_loop.py`,
   `test_e2e_fake_venue.py`, `test_margin_summary.py`,
   `test_slippage_reporting.py`, and the execution/safety parts of
   `test_live_trader.py`, `test_rolling_drawdown.py` and `test_loss_streak.py`.

Until step 4 lands, the measured "after" line count is a projection, not an
achieved number.

**The live processes matter here.** `com.cuansniffer.telegram` is running
(pid 41917) and imports `live_execution_backend`; `com.cuansniffer.trader` last
exited with status 1. Both pick up working-tree changes on their next restart,
so the deletion phase should be done with those jobs unloaded.
