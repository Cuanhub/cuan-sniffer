# NOTES — things noticed while building, deliberately not built

Per the build brief: anything not in SIMPLIFY_PLAN.md goes here instead of into
the code.

## Deferred (not built)

- **Loss-streak stand-down: dropped, and this one needs your call.** The brief
  allowed keeping it "unless it's a few lines off `user_fills` `closedPnl`".
  It isn't: `closedPnl` is per *fill*, so counting consecutive losing trades
  needs fills grouped into trades (by coin, contiguous time, `startPosition`
  crossing zero) — ~25 lines of fiddly grouping.

  Under the "keep anything with clear evidence" rule it is genuinely
  ambiguous, so I have not reinstated it on my own judgement. The evidence
  points both ways: it was added deliberately (`2862bc3`), but the trader's
  own comment records it firing "an hour before the market turned and then
  locking the account out" on 2026-10-15, which is why `--clear-standdown`
  had to be added the next day (`f9edfc3`). That is evidence of harm, not of
  value. Say the word and it comes back.
- **`/take` (forced entry): dropped**, as instructed. `FORCEABLE` and the
  `state["forced"]` plumbing go with it.
- **Reason-aware exit slippage ladder: dropped.** The five
  `LIVE_EXIT_SLIPPAGE_BPS_*` knobs collapse to one `EXEC_EXIT_IOC_BPS`. No
  measurement sits behind the individual values — they read as estimates —
  so this is not covered by the "clear evidence" rule. Flagging it because it
  is a removal, not a refactor.
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

## Integration status (SUPERSEDED -- see "Integration landed" below)

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

## Settled since the plan

- **Equity marks to market — the loss limits now act.** SIMPLIFY_PLAN 3C left
  rule 6 log-only because it was unknown whether spot USDC `total` moves with
  unrealised perp PnL. It does, and it was answerable from data already on
  disk: across 265 equity readings in the trader's own log, every consecutive
  pair with a position open and no fill between them has
  d(total)/d(hold) = 5.0000 — the flat leverage this system trades at — so
  `total` moves by the whole change in position notional. Equity is
  `usdc_total` alone; adding `unrealizedPnl` would double-count, which is why
  there is no `with_unrealized` property. The series is kept as
  `tests/fixtures/hyperliquid/equity_hold_series.txt` and asserted by a test,
  so if the relationship ever stops holding the suite says so before the
  limits act on a wrong number. No testnet was needed.
- **Resting GTC bids are kept.** Rule 3 would have removed them, and the
  evidence behind them is one-sided: across 139 tradeable watched levels on
  2026-09-19 not one was inside the 1.5% entry drift guard, the nearest 3.10%
  away. A marketable IOC cannot catch a level that taps and leaves. What was
  dangerous was the `pending` dict, not the order — and there is none now. A
  bid is visible in `open_orders`, carries its own venue timestamp for expiry,
  and a partial fill is just a smaller position with a correctly sized stop
  beside a smaller resting order. Findings 6, 9 and 11 were all `pending`,
  not the bid.

## File audit (reported, nothing deleted)

Asked for after the stand-down rewrite. Nothing here was removed — the
project's stated retention stance is "delete nothing" (`e384524`), so this is
a list, not an action.

### No fabricated data reached the live files

Checked `episodes.csv`, `live_trades.csv`, `outcomes.csv` and `po3.csv` for the
fixture symbols that leaked in before (AAA/BBB/CCC/DDD and similar): **zero
rows**. The autouse isolation in `conftest.py` is holding.

The only deliberately fake things in the tree are `tests/fake_venue.py` (the
test double) and the two derived fixtures, both declared in
`tests/fixtures/hyperliquid/PROVENANCE.md`.

### Dead weight

| what | size | note |
|---|---|---|
| `archive/` | **898 MB**, 966 files | 836 MB of it is `legacy_bot_data_2026-09-18` from the retired SMC bot. Never imported; `pytest.ini` already excludes it. Pure disk. |
| `research/levels/data/live_trades.csv` | **26 MB**, 197,098 rows | **98.4% are `skip` rows.** The actual audit trail is 162 `entry`, 110 `protection`, 109 `closed` and 1 `emergency_exit` — 382 rows buried under 193,858. |
| `research/levels/data/_fetch_failures.log` | 4.2 MB | append-only, nothing reads it |
| `research/levels/data/_live_state.json` | 111 KB | 76 KB of it is `watching` with 803 entries. The TTL prune is supposed to bound this; 803 suggests it is not keeping up. |
| `.env.example` | — | **110 of its 121 variables reference nothing in the code.** It documents the retired SMC/V3 bot (`CHART_PATTERN_*`, `DAILY_ZONE_*`, `CHOP_REVERSAL_EXCEPTION`, …). Anyone configuring from it is configuring a system that no longer exists. |

### Dies with the deletion phase

Seven test files, 1,623 lines, exist only to test code that is being removed:

```
test_protection_verification.py  488    test_margin_summary.py      129
test_e2e_fake_venue.py           401    test_slippage_reporting.py  101
test_flatten.py                  237    test_fake_contracts.py       88
test_reconcile_loop.py           179
```

Plus the execution/safety portions of `test_error_rate.py`,
`test_rolling_drawdown.py`, `test_loss_streak.py` and `test_drawdown_rail.py`,
which test functions inside `live_levels_trader.py` that go away.

`deploy/launchd/com.cuansniffer.reconcile.plist` points at
`tools/reconcile_loop.py`, which is being deleted. It is not loaded in
`launchctl` anyway.

### Not dead, just unscheduled

`tools/near_firing.py` and `tools/screen_context.py` are manual CLI tools with
no plist, referenced only from docs and config. Working as intended.

### Current runtime state

`_live_state.json` shows the bot **manually halted from Telegram**
(`halt_kind: "manual"`, reason `"halted from Telegram: DISARMED pending…"`)
and flat — `owned: {}`, `pending: {}`. So the deletion phase can be done
without racing a live position.

## Integration landed

The deletion phase is done. `live_execution_backend.py`,
`execution_backend.py`, `live_data_guard.py` and `tools/reconcile_loop.py`
are gone, along with 1,043 lines of execution and safety functions inside
`live_levels_trader.py`. Both callers go through `core/`.

### Running configuration

| job | state |
|---|---|
| `com.cuansniffer.trader` | running, **DRY RUN** (`LEVELS_LIVE_ARMED=no` in the installed plist) |
| `com.cuansniffer.protect` | running, 60s cycle — **newly installed** |
| `com.cuansniffer.telegram` | running, on the new core |
| `com.cuansniffer.scanner` | running |
| `com.cuansniffer.retention` | periodic |

The halt is still `manual` and was not cleared: arming is the operator's.

### What the protect job may do to the account

It is NOT read-only, and the plist it was derived from said it was. It places
reduce-only stops, cancels them, and calls `flatten_all` — which cancels
every resting order and market-closes every position the venue holds. It does
that when a position cannot be given a valid stop after two attempts.

**Consequence worth knowing: a position opened by hand at the venue will have
a stop placed on it, and will be market-closed if no valid stop price can be
found for it.** There is no arming gate on this, deliberately — protection
must work when nobody is trading, because then nothing else is looking. If
you want to hold a manual position outside the bot's reach, unload this job
first.

### Two things I caused, recorded for honesty

- A dry `--once` run during the integration drained four stale queued halts
  and overwrote the halt reason. The original text, `"DISARMED pending C-1
  and D-1 merge (audit Phase 2)"`, now survives only in git history and in
  this file.
- `trader.err.log` holds ~183KB of tracebacks from 18:06, when launchd
  restarted the trader against a half-edited working tree mid-integration.
  They reference `backend` at line 751, which no longer exists. Nothing has
  been written to it since the restart. Left in place per the "delete
  nothing" stance, but it will mislead anyone reading the tail of that file.

### Line counts, measured

| | total |
|---|---|
| before (execution + safety) | 3,720 |
| after (`core/` + `protect_loop` + glue left in the trader) | 2,199 |

41%, not the 85% the plan projected, and well over the "well under 600"
target. The target was set when rules 3, 6 and 9 removed resting bids, the
stand-down and the acting loss limits; restoring all three on evidence added
roughly 470 code lines. The scope moved, so the target moved with it — that
should have been re-forecast when the instruction changed, not reported at
the end.
