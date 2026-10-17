# SIMPLIFY_PLAN — rebuild the execution + safety core

Branch: `simplify-execution-core` (off `audit-phase2-safety`).
Status: **plan only, awaiting approval. No code written.**

Target from the brief: core execution + safety well under 600 lines.
Measured today: **3,720 lines** (see §5). Proposed: **~550**.

---

## 1. What I verified before planning

The brief rests on assumptions about Hyperliquid and about the installed SDK. I
checked each one against the installed SDK (`hyperliquid-python-sdk==0.22.0`),
the official docs, and a **read-only `/info` probe of your own mainnet account**
(2,000 historical order records; no exchange actions, no orders, nothing armed).

**1.1 `sz` is the remaining size. REVIEW.md finding 1 is confirmed on real data.**
Across your own order history:

| inner `status` | `origSz` | `sz` |
|---|---|---|
| `filled` | `23.0` | `0.0` |
| `canceled` | `22.66` | `22.66` |
| `open` | `23.0` | `23.0` |

`totalSz` and `avgPx` appear on **none** of the 2,000 records. So rule 2 is not a
stylistic preference — the parsing it removes is provably wrong.

**1.2 The inner-status vocabulary is far wider than the code assumes.** Your
account's real history contains nine distinct values:

```
open 865 | filled 580 | reduceOnlyCanceled 286 | triggered 231
perpMarginRejected 31 | siblingFilledCanceled 2 | canceled 3
iocCancelRejected 1 | reduceOnlyRejected 1
```

`_resolve_ambiguous_order` tests only the *envelope* status against
`("unknownOid","unknown","")`, so **all nine are "adopted" as fills.** Three real
zero-fill rejects from your own account, and what today's code would report:

```
perpMarginRejected   origSz=1907.0  sz=1907.0  -> "filled 1907.0 @ 0.16819"
iocCancelRejected    origSz=89.1    sz=89.1    -> "filled 89.1 @ 2.0014"
reduceOnlyRejected   origSz=2217.3  sz=2217.3  -> "filled 2217.3 @ 0.127"
```

32 such records exist in the last 2,000 orders. Any of them arriving via the
timeout path becomes a phantom position. This is the single strongest argument
for the rebuild.

**1.3 The venue auto-cancels reduce-only orders when a position closes.**
`reduceOnlyCanceled` appears 286 times. This **corrects REVIEW.md finding 7**:
the "stale stop outlives its position" risk is mostly handled by the venue, which
is why rule 4's "never store oids" is safe. I was more alarmed than the evidence
warranted there.

**1.4 `scheduleCancel` exists — and rule 9 would make this system less safe.**
`Exchange.schedule_cancel(time)` is real
([exchange.py:341](.venv/lib/python3.13/site-packages/hyperliquid/exchange.py#L341)),
min 5s in the future, max 10 triggers/day. But its docstring and the official
docs both say it cancels **all open orders**, with no carve-out for trigger
orders, and the docs do not say otherwise. Under rule 3 the only resting orders
this bot will ever have **are its protective stops**. So the dead-man's switch
would, on bot death, cancel exactly the protection and leave positions naked.
**Recommendation: drop rule 9.** See §3A — the venue-side stop already *is* the
dead-man's switch.

**1.5 `positionTpsl` is available but its auto-sizing is not free.**
`bulk_orders(..., grouping="positionTpsl")` is supported
([exchange.py:144](.venv/lib/python3.13/site-packages/hyperliquid/exchange.py#L144)).
But the official TP/SL docs say position TP/SL has "a size equal to the entire
position **by default**" and "**if a specific size is configured** on these TP/SL
orders, they will be fixed-sized (i.e. they will not resize with the position)".
The SDK's wire encoder **always** emits a size
([signing.py:509](.venv/lib/python3.13/site-packages/hyperliquid/utils/signing.py#L509)),
with no way to omit it. Whether `sz=0` + `positionTpsl` means "track the
position" cannot be settled from the SDK or the docs — it needs a testnet probe.
Your existing protection orders all carry `isPositionTpsl: false`, so this has
never been exercised here. See §3B.

**1.6 Your account is a `unifiedAccount`, and `marginSummary.accountValue` reads
`'0.0'`.** Probed live:

```
abstraction_state: 'unifiedAccount'
marginSummary: {'accountValue': '0.0', 'totalNtlPos': '0.0', 'totalMarginUsed': '0.0'}
withdrawable: 0.0
```

So rule 6's "record venue accountValue at UTC day start" **cannot read
`user_state`** — it would record 0.0 and the daily loss stop would be dead on
arrival (or fire instantly). Equity must come from the spot layer. See §3C.

**1.7 `user_fills_by_time` is a usable venue-truth source for prices.** 385 real
rows in 30 days, with `px`, `sz`, `dir`, `closedPnl`, `fee`, and `startPosition`
(the position size *before* the fill). This is how exits get a price under rule 2,
where `entryPx` cannot help because the position is gone.

**1.8 The account is currently flat** — 0 positions, 0 open orders,
`LEVELS_LIVE_ARMED` not set. Nothing is at risk while we work.

---

## 2. The two root causes, and the rules that kill them

| Root cause | Rule | Mechanism |
|---|---|---|
| Trusts its own state over the venue | 1, 5, 6 | `owned`/`pending` stop existing as a position record. Every decision reads the venue. State holds metadata only. |
| Parses order responses instead of observing results | 2 | A response is a log line. The only question asked is "what does the venue hold now?" |

Everything in §4 is deleted because one of those two answers makes it unnecessary
— not because it was badly written.

---

## 3. Three changes I propose to the brief

### 3A. Drop rule 9 (`scheduleCancel`)

Per §1.4 it is strictly negative here: it cancels the stops and leaves the
positions. The protection it would remove is the only thing standing between a
dead bot and an unbounded position.

What already covers the intent: `launchd` restarts the process (`KeepAlive`), and
the 60s protection loop re-asserts every stop within one cycle of restart. The
venue-side reduce-only stop survives bot death by design — that is the dead-man's
switch, and rule 4 is what guarantees it is always there and correctly sized.

If you want a true dead-man's switch later, the correct shape is a *second*
process that flattens on loss of heartbeat — not an order-cancelling timer. Out
of scope for this rebuild.

### 3B. Build rule 4 explicit-size first, `positionTpsl` as a verified switch

The 60s loop makes auto-sizing an optimisation, not a requirement: if a stop is
the wrong size, the loop replaces it within 60 seconds either way. So:

1. Ship explicit full-size reduce-only triggers, re-derived from venue position
   size every cycle. Correct regardless of §1.5.
2. Add one testnet probe (`probe_position_tpsl.py`, throwaway) that places a
   `positionTpsl` group, changes the position size, and re-reads whether the
   trigger resized.
3. If it auto-sizes, flip one constant and the loop's "wrong size" branch becomes
   dead code we then delete. If it does not, we lose nothing.

This is the only way to answer your question honestly — the docs don't, and the
SDK can't.

### 3C. Rule 6 needs a verified equity source, not `accountValue`

Per §1.6, equity for this account is the spot USDC balance. Two sub-questions,
one answerable now and one not:

- **Answerable:** equity source is `spot_user_state(addr)` → USDC `total`. This is
  what the current code already does, and it is the only path that returns a real
  number on a unified account.
- **Not answerable while flat:** does spot USDC `total` move with *unrealised*
  perp PnL? Rule 6 says "including unrealised", so the limit's correctness turns
  on this. If `total` already includes it, use it directly; if not, equity must be
  `total + Σ unrealizedPnl` from `assetPositions`. Using the wrong one either
  double-counts or ignores open-position losses entirely.

Plan: one testnet probe with a single open position, moved in both directions.
**Until that probe lands, rule 6's limits log their verdict but do not act** — a
loss limit computed from an unverified equity number is worse than none, because
it can flatten a healthy account. This is the one place I'd rather be slow.

Smaller note: an equity-delta loss measure reads a deposit or withdrawal as PnL.
I'll record transfers via `user_non_funding_ledger_updates` and skip the day's
limit if the ledger moved, rather than flatten on a deposit.

---

## 4. What gets deleted, and what replaces it

### 4.1 Deleted outright

| File / function | Lines | Why it stops existing |
|---|---|---|
| `tools/reconcile_loop.py` (whole file) | 191 | Rule 8. The 60s protection loop reads the venue directly and acts; it never compares two beliefs, so it cannot produce finding 2's false divergences. |
| `_fill_response_from_order`, `_resolve_ambiguous_order` | 55 | Rule 2. Provably wrong (§1.1, §1.2). |
| `_parse_fill` | 147 | Rule 2. Fill = venue position delta. |
| `_normalize_open_orders`, `_normalize_open_order_item` | 161 | Rule 1/4. The normaliser existed to make a thin payload answer "is this a stop?". We now ask `frontend_open_orders` for one coin and compare trigger price + size directly. |
| `_verify_protection_orders`, `_verify_single_protection_order`, `_append_verify_notes`, `_validate_protection_price_relations` | 218 | Rule 4. Hard/soft verification existed because placement was one-shot. A loop that re-checks every 60s and replaces what's wrong needs no verdict taxonomy. |
| `place_native_protection`, `place_stop_only` | 260 | Rule 4. One `ensure_protection(coin)` replaces both. |
| `place_limit_entry`, `place_resting_bid`, `reconcile_pending` | 278 | Rule 3. Resting bids and `pending` are gone. |
| `reconcile`, `venue_sanity`, `adopt_or_flatten_orphans`, `has_protection` ×2, `verify_open_protection`, `cancel_protection` | 299 | Rules 1/4/5. All are "compare state to venue" machinery. With no state to compare, the sync cycle is the only reconciler. |
| `accrue_close`, `assumed_loss`, `note_closed_outcome`, `record_equity_sample`, `drawdown_now`, `prune_equity_samples`, `roll_day` | 170 | Rule 6. Limits read equity, so none of the PnL-accrual bookkeeping is load-bearing. |
| `release_halt_if_clear`, `clear_standdown`, `streak_cooldown_remaining` | 90 | Rule 7. Two halt states, one of which clears on date change. |
| `_is_unified_account`, `_get_margin_summary_unified`, `_get_margin_summary_standard`, `_available_after_maintenance` | 160 | Rule 6. One `equity()`, one verified source (§3C). |
| `live_data_guard.py` backoff machinery | 121 | Rule 8's error rail replaces it with a counter. **Caveat below.** |
| `get_realised_pnl` | 46 | Rule 6. Nothing reads realised PnL to decide anything. |
| `flatten`, `execute_exit` reason-aware slippage ladder | 120 | Rule 5. One `flatten_all`; exits are market. |
| `execution_backend.py` (ABC + `FillResult`) | 86 | Single backend since `08b5f9e` removed paper. The ABC is a one-implementation interface, and `FillResult`'s 10 fields collapse to an observed delta. |

**Caveat on `live_data_guard.py`:** deleting the backoff removes the thing that
currently stops a 429 storm from becoming a crash loop. Rule 8's "5 errors in
10 minutes" is an *alert*, not a brake. I'll keep one ~15-line sleep-on-429 in
`venue.py` so the rail that exists today is not silently traded away — and note
that it fixes finding 10 as a side effect (the brake no longer starves `flatten`,
because `flatten_all` reads positions rather than mid-prices).

### 4.2 What replaces it

Five files, no inheritance, no interfaces.

| New module | Budget | Contents |
|---|---|---|
| `core/venue.py` | ~210 | The only Hyperliquid wrapper. `positions()`, `open_orders(coin)`, `equity()`, `mid(coin)`, `send_ioc()`, `send_trigger()`, `cancel(coin, oid)`, `cancel_all(coin)`, `market_close(coin)`, size/price rounding, meta load. Returns venue data; interprets nothing. Keeps `cloid` on every order, for logs only. |
| `core/execute.py` | ~90 | `enter(coin, usd)` / `exit(coin)`: read position → send → sleep → re-read → return the observed delta and `entryPx`. Exit prices from `user_fills_by_time` (§1.7). No response branching. |
| `core/protect.py` | ~110 | The 60s loop. For each venue position: find its reduce-only triggers; if absent or mis-sized, `cancel_all(coin)` then place fresh. Two failures → `flatten_all` that coin + alert. Positions with no metadata get a stop from `MAX_RISK` off current price, or are flattened (rule 1). |
| `core/safety.py` | ~140 | `flatten_all()` (rule 5 — the only kill switch), halt state AUTO/MANUAL (rule 7), equity-based limits (rule 6, gated per §3C), error counter + the four alerts (rule 8). |
| `core/meta.py` | ~60 | Trade metadata only: `{coin: {reason, planned_stop, planned_tp, opened_at}}`. Atomic write. **Cannot** express "what we hold". Losing this file degrades us to rule 1's default stop, not to a forgotten position. |
| | **~610** | trimmed to **~550** target, 600 hard cap |

Callers change shape, not size: `tools/live_levels_trader.py`'s strategy half
keeps its 1,458 lines; `tools/telegram_control.py`'s `/close`, `/closeall`,
`/halt` all become one-line calls into `safety.flatten_all()`.

---

## 5. Line counts

**Before** (execution + safety only; strategy and notifier excluded):

```
  2162  live_execution_backend.py
  1160  execution+safety functions inside tools/live_levels_trader.py (32 functions)
   191  tools/reconcile_loop.py
   121  live_data_guard.py
    86  execution_backend.py
  ----
  3720  TOTAL
```

Out of scope and unchanged: `tools/live_levels_trader.py` strategy remainder
(1,458), `notify.py` (215). `tools/telegram_control.py` (494) shrinks as a
consequence but is not the target.

**After (projected):** ~550, a **~85% reduction**. I'll report measured actuals
against this table when the code lands.

---

## 6. Which REVIEW.md findings each change eliminates

| # | REVIEW.md finding | Eliminated by | How |
|---|---|---|---|
| 1 | Cloid adoption inverted | Rule 2 | The parsing is deleted. Position delta cannot be inverted. |
| 2 | Reconcile loop halts on normal operation | Rule 8 + 1 | File deleted. Nothing compares two beliefs. |
| 3 | `has_protection` returns `True` for non-protective order | Rule 4 | Both copies deleted. The loop asks "is there a reduce-only trigger at the right price and size", with no boolean to invert. |
| 4 | Queued halts verify nothing (`backend=None`) | Rule 5 | No `backend=None` paths. `flatten_all` needs a venue or it raises. |
| 5 | Kill switch blind to orphans | Rules 1 + 5 | `flatten_all` iterates **venue** positions. `owned` no longer exists to be blind with. |
| 6 | Partial resting bid orphans remainder; no "grew" branch | Rule 3 | Resting bids deleted. A position larger than expected is just the current position; the loop sizes the stop to it. |
| 7 | Protection oids never persisted | Rule 4 | Oids re-read each cycle, never stored. (And §1.3: the venue auto-cancels anyway.) |
| 8 | Restart mid-trade leaves no `size_coin` | Rules 1 + 2 | Size is never stored, so it cannot be missing. The venue has it. |
| 9 | Startup orphan sweep ignores `pending` | Rules 1 + 3 | No `pending`; startup is just the first sync cycle. |
| 10 | `flatten` can't outlast the backoff | Rules 5 + the §4.1 caveat | `flatten_all` loops to 120s and uses `market_close` (which reads `szi` itself), so it needs no mid-price and cannot be starved by a mid-price brake. |
| 11 | Post-entry `venue_sanity` fatal-halts on a bid fill | Rules 1 + 3 | Both the function and the trigger condition are gone. |
| 12 | `HALT_ACTION=flatten` bypasses `accrue_close` | Rules 5 + 6 | One flatten path; limits read equity, so there is no ledger to bypass. |
| 13 | `flatten` misreports `attempts_used` | Rule 5 | `flatten_all` reports the venue's final state, not an attempt count. |
| 14 | `cancel_protection` duplicated inline | Rules 4 + 5 | One call site each. |
| 15 | `or` chain hides `0.0` equity | Rule 6 / §3C | One equity function, one source, explicit `None` handling — and §1.6 shows `0.0` is the *real* reading here, which makes this finding urgent rather than cosmetic. |

All 15. Findings 1, 3 and 15 additionally gain real-payload regression tests (§7).

---

## 7. Test plan — real payloads only

**Fixture corpus.** I already have 2,000 real order records, a real `user_state`,
a real `marginSummary`, a real `spot_user_state` and 385 real fills from your
account, read-only. I'll commit these as sanitised JSON under
`tests/fixtures/hyperliquid/` (address and `cloid` scrubbed; sizes, prices,
statuses and field shapes untouched), plus a `record_fixtures.py` that regenerates
them. The fake venue is built **only** from these files — it has no hand-written
payload shapes, and a test asserting a shape the corpus doesn't contain fails.

What the corpus already covers: `filled`, `canceled`, `open`, `triggered`,
`perpMarginRejected`, `iocCancelRejected`, `reduceOnlyRejected`,
`reduceOnlyCanceled`, `siblingFilledCanceled`; all five real `orderType` strings
(`Limit`, `Market`, `Stop Market`, `Take Profit Market`, `Take Profit Limit`);
unified-account zero `accountValue`.

What it cannot cover without testnet: **live order-response envelopes**, success
and error. Those need orders sent. So:

| Scenario (from the brief) | Source |
|---|---|
| fill, partial fill, zero-fill IOC | testnet responses + corpus records |
| timeout-after-fill, timeout-no-fill | corpus records + injected transport exception |
| restart mid-trade | kill between send and re-read; assert sync recovers from venue |
| stop missing / stop wrong size | corpus trigger records, resized |
| position larger than expected | corpus `user_state` with inflated `szi` |
| unknown position at startup | corpus `user_state`, empty metadata |
| flatten under API errors / backoff | injected 429s on `/info` and `/exchange` |
| daily loss breach, drawdown breach | synthetic equity series over the real `spot_user_state` shape |

Three regression tests pinned directly to §1: a `perpMarginRejected` record must
never produce a fill; a `filled` record (`sz: "0.0"`) must never produce a reject;
a coin whose only resting order is non-protective must read as unprotected.

**Testnet credentials.** `.env` currently has `HL_TESTNET=false` and a mainnet
key, and I will not touch config. The 24h run and the two probes in §3B/§3C need
a funded testnet account and key from you. I'll read them from a separate
`.env.testnet` you provide, or from env vars you export — your call, and nothing
in the repo changes either way.

**24h testnet run.** Once tests pass: run armed on testnet ≥24h, then report
every alert and halt with timestamps and causes, plus a diff of what the bot
believed vs what the venue held at each sync. Mainnet stays unarmed; you arm it.

---

## 8. Decisions I need from you

1. **Drop rule 9 (`scheduleCancel`)?** I recommend yes, per §1.4 — as specified it
   would strip the stops off live positions when the bot dies.
2. **Rule 6 gated until the equity probe lands** (§3C)? I recommend yes. The
   alternative is a loss limit that may read 0.0 equity on your unified account.
3. **Testnet credentials** — how do you want to supply them (§7)?
4. **Keep a minimal 429 brake** in `venue.py` (~15 lines) rather than deleting
   `live_data_guard.py` outright (§4.1 caveat)? I recommend yes.
5. **`/take` (forced entry) and the loss-streak stand-down** are strategy, not
   safety, and sit in the deleted region. Keep them (ported as-is), or drop them?
   The brief doesn't say. Default if you don't answer: port as-is.

---

## 9. What this plan does not fix

- **No websocket.** The brief's 60s protection loop narrows the gap from 450s to
  60s, but a gap through a stop is still invisible for up to 60s. The venue-side
  stop is what actually covers this; the loop only guarantees it exists.
- **No edge.** `EDGE_PLAN.md` and the `-0.0413R vs random` note are untouched.
  This work makes the system safe to run, not profitable.
- **Strategy bugs are out of scope** — the 1,458 strategy lines keep whatever is
  wrong with them.
- **`REVIEW.md` finding 7 was overstated** by me; §1.3 is the correction.

---

**Awaiting approval. Nothing will be coded, and nothing armed, until you say go.**
