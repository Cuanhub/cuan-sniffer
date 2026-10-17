# Adversarial review — `audit-phase2-safety` vs `main`

Scope as requested: `live_execution_backend.py`, the kill-switch / halt path
(`tools/live_levels_trader.py`, `tools/telegram_control.py`), and reconciliation
(`tools/reconcile_loop.py`, `reconcile`, `reconcile_pending`). Read-only review:
no code, orders, or config were touched.

The branch's safety work is real and the commit messages are unusually honest
about past failures. The findings below are the places where a fix does not do
what its own docstring claims, and — in two cases — where it inverts the
behaviour it was written to guarantee. Each one is stated as a concrete failure
scenario; the first three were reproduced against the code in the tree.

Test suite state: `test_e2e_fake_venue.py test_reconcile_loop.py test_flatten.py
test_protection_verification.py` → **97 passed**. Findings 1, 2 and 3 are live
while those tests are green, because the fakes and the assertions encode the same
misunderstandings as the code. See "Where the tests agree with the bugs" at the
end.

---

## Severity ranking

| # | Severity | Finding |
|---|----------|---------|
| 1 | Critical | Cloid adoption after a timeout is inverted: a filled order reads as a reject, a zero-fill cancel reads as a full fill |
| 2 | Critical | The 45s reconcile loop queues a halt on every normal exit and every resting-bid fill, and that halt never auto-releases |
| 3 | High | `has_protection` returns `True` for a non-protective resting order, defeating the D-1 unprotected-position check |
| 4 | High | Every queued/Telegram halt runs the D-1 protection check with `backend=None`, so it verifies nothing |
| 5 | High | `/closeall` closes only what the state file lists, so the human kill switch cannot see the positions C-1 exists to find |
| 6 | High | A partially filled resting bid leaves the remainder resting with its oid discarded; nothing handles a position that grew |
| 7 | Medium | Protection oids are never persisted, so `cancel_protection` is a permanent no-op at its only live call site |
| 8 | Medium | A restart between claim and fill leaves `owned[coin]` with no `size_coin`; the loss stop then charges $0 and both reconcilers skip the size check |
| 9 | Medium | The startup orphan sweep ignores `pending`, so `ORPHAN_ACTION=auto` flattens a legitimately filled bid after a restart |
| 10 | Medium | `flatten`'s retry budget (3 × 1s) cannot outlast the 30s API backoff its own call volume can trip |
| 11 | Medium | Post-entry `venue_sanity` reads a mid-pass resting-bid fill as an unknown position and halts fatally |
| 12 | Medium | `HALT_ACTION=flatten` closes positions but leaves `owned` populated and bypasses `accrue_close` |
| 13 | Low | `flatten` always reports `attempts_used == attempts`; unreadable reads silently consume the budget |
| 14 | Low | `execute_one` re-implements `cancel_protection` inline, so its failure alert cannot fire |
| 15 | Low | `_get_margin_summary_standard`'s `or` chain treats a genuine `0.0` equity as missing |

---

## 1. Critical — cloid adoption after a timeout is inverted

**Evidence:** [live_execution_backend.py:204-225](live_execution_backend.py#L204-L225)
(`_fill_response_from_order`), [live_execution_backend.py:248-258](live_execution_backend.py#L248-L258)
(`_resolve_ambiguous_order`), call sites [live_execution_backend.py:307-312](live_execution_backend.py#L307-L312)
and [live_execution_backend.py:426-431](live_execution_backend.py#L426-L431).

This is the C-1 fix. The SDK's `query_order_by_cloid` posts
`{"type": "orderStatus"}` ([.venv/.../hyperliquid/info.py:614](.venv/lib/python3.13/site-packages/hyperliquid/info.py#L614)).
That payload is:

```json
{"status": "order",
 "order": {"order": {"coin","side","limitPx","sz","origSz","oid","orderType","tif",...},
           "status": "open"|"filled"|"canceled"|"rejected"|...,
           "statusTimestamp": ...}}
```

Two problems compound:

**(a) The fields it reads do not exist on this endpoint.** `_fill_response_from_order`
asks for `totalSz` and `avgPx`. Neither appears in an `orderStatus` order object,
so the fallbacks at lines 220-221 are what always execute: `sz`, which is the
*remaining unfilled* size, and `limitPx`, which is not the fill price.

**(b) The inner order status is never read.** Line 250 checks only the
*top-level* `status`, which is `"order"` whenever the venue recognises the cloid
— including for `canceled`, `rejected` and `marginCanceled`. An IOC that was
accepted and immediately cancelled with zero fill is therefore "adopted" by line
256-258.

Reproduced against the code in the tree:

```
requested sz=10.0, limit_px=100.0

A. IOC FULLY FILLED (venue holds 10)      -> totalSz=0.0  avgPx=100.0
B. IOC accepted then CANCELED, zero fill  -> totalSz=10.0 avgPx=100.0
C. IOC partially filled 4 of 10           -> totalSz=6.0  avgPx=100.0
```

Fed through `_parse_fill`:

```
venue FILLED 10 HYPE, client timed out -> filled=False reject='ioc_no_fill'
venue CANCELED, zero fill              -> filled=True  fill_size_usd=$1000.00
```

**Failure scenario (entry).** `exchange.order` times out on a `LONG HYPE`,
$1,000 notional. The venue filled it.
`execute_entry` → `_resolve_ambiguous_order` finds the order → adopts it →
`sz` (remaining) is `0` → `_parse_fill` returns
`filled=False, reject_reason="ioc_no_fill"`. `execute_one` records a reject and
releases the slot. The account holds 10 HYPE, unprotected and untracked, for up
to `POLL_SEC` (450s) — **exactly the C-1 failure this commit says it fixed.**

**Failure scenario (phantom position).** Same timeout, but the venue cancelled
the IOC unfilled (thin book — the normal no-fill outcome for a marketable IOC).
The lookup adopts it, `sz` is the full unfilled size, and the backend reports a
**full fill at the limit price**. `execute_one` then marks the position
`confirmed: True` with `size_coin=10`, and calls `place_native_protection` for a
position that does not exist. On Hyperliquid a reduce-only trigger against a flat
account is rejected, so protection comes back `missing` → the emergency path
fires `flatten` → `flatten` reads the venue, sees flat, reports `flat: True` →
`accrue_close` charges a loss that never happened to `realised_today`. The daily
loss stop and the loss-streak stand-down are both driven off a fabricated trade.
`fill_price` is `limitPx`, so the recorded entry is wrong by the full IOC
slippage band (`LIVE_ENTRY_IOC_BPS`, 25bps).

**Failure scenario (partial).** 4 of 10 fill, timeout, then cancel. The backend
reports 6 — the complement. Protection is placed for 6 coins against a 4-coin
position (reduce-only, so the venue clamps or rejects), and the recorded
`size_coin` is wrong in the direction that makes `assumed_loss` overstate.

`execute_exit` has the identical defect at line 426-431, where "adopted" means
the exit is reported filled when it was cancelled — the slot is released and the
position stays open.

Note that the direction of (a) depends on `sz` being the remaining size. If it
were the original size, (b) alone still turns every cancelled-unfilled IOC into a
reported full fill, and the price is still `limitPx` rather than the realised
average. The bug is critical under either reading.

---

## 2. Critical — the reconcile loop halts the trader on normal operation

**Evidence:** [tools/reconcile_loop.py:104-122](tools/reconcile_loop.py#L104-L122)
(`check`), [tools/reconcile_loop.py:139-145](tools/reconcile_loop.py#L139-L145)
(queues `cmd: halt`), [tools/live_levels_trader.py:2506-2509](tools/live_levels_trader.py#L2506-L2509)
(applies it with `kind="manual"`), [tools/live_levels_trader.py:2361-2376](tools/live_levels_trader.py#L2361-L2376)
(`release_halt_if_clear` recognises only `daily_loss` and `drawdown`).

`check` compares the venue against `state["owned"]` **only** (line 101) and has
no notion of how stale the state file is. The trader polls every 450s; this loop
polls every 45s. Every divergence class it reports has a benign cause inside that
window. Reproduced:

```
Scenario 1 -- a take-profit fires; the trader's 450s poll has not run yet
   -> ['HYPE is gone at the venue but the trader still holds the slot']

Scenario 2 -- a resting GTC bid fills (it lives in pending[], not owned[])
   -> ['HYPE is untracked at the venue (2)']

Scenario 3 -- first run / state file absent -> read_state() == {}
   -> ['HYPE is untracked at the venue (2)']
```

**Failure scenario.** A take-profit fires at T+0. At T+45s the loop sees the
venue flat and `owned` still listing the coin, and queues
`{"cmd": "halt", "arg": "reconcile loop found a venue divergence: HYPE is gone
at the venue but the trader still holds the slot"}`. At the trader's next pass,
`drain_control_queue` runs at [tools/live_levels_trader.py:775](tools/live_levels_trader.py#L775)
— *before* `reconcile` at line 793, which is the thing that would have resolved
it. The halt is applied with `kind="manual"`, which
`release_halt_if_clear` falls through to `return False` on (line 2375-2376). So
the trader stops opening anything until a human runs `--resume`.

That is **every winning and losing trade**: each normal exit halts the account
roughly ten passes before the trader would have reconciled it. Scenario 2 does
the same for the resting-bid entry path, and scenario 3 does it on first run or
any `read_state()` failure, where an empty dict is read as "the trader owns
nothing" — the maximally alarming interpretation of a missing file.

The two fixes are independent: `check` needs to consider `state["pending"]` and
a grace period at least as long as the trader's poll interval before calling
either direction a divergence; and a divergence halt needs a `kind` that
`release_halt_if_clear` can re-check.

---

## 3. High — `has_protection` reports `True` for a non-protective order

**Evidence:** [tools/live_levels_trader.py:1726-1727](tools/live_levels_trader.py#L1726-L1727).

```python
    # An order list that came back and held nothing protective for this coin.
    return any(str(o.get("coin", "")).upper() == coin.upper() for o in orders)
```

The comment describes returning `False`. The code returns `True` whenever *any*
order exists for the coin. `orders` is already coin-filtered by
`get_open_orders(coin)` ([live_execution_backend.py:1358-1361](live_execution_backend.py#L1358-L1361)),
and the loop above has already returned `True` for anything reduce-only or
trigger-flagged — so everything reaching this line is by construction a
**non-protective** order. Reproduced, with the unfilled remainder of a GTC entry
bid as the only resting order:

```
trader.has_protection(HYPE)        -> True      (comment says False)
reconcile_loop.has_protection(same) -> False     <- the two disagree
verify_open_protection             -> []         (D-1 reports nothing bare)
```

`tools/reconcile_loop.py:78-93` is a second copy of the same function that gets
it right. The two have already drifted.

**Failure scenario.** A resting bid partially fills (see finding 6), so a plain
GTC bid for that coin is still resting at the venue. `place_native_protection`
then fails — rate limit, price-relation rejection, anything. The drawdown rail
fires. `halt()` → `verify_open_protection` →
`has_protection` sees the resting entry bid → `True` → `bare == []` → the
"🚨 Halted, and N position(s) are unprotected" alert at
[tools/live_levels_trader.py:1834-1837](tools/live_levels_trader.py#L1834-L1837)
never sends. The entire D-1 guarantee is silently void for that position, and the
operator is told the halt is clean.

`adopt_or_flatten_orphans` reads the same function at line 1781-1786 and will
leave an unprotected orphan alone, logged as "protected", on the same input.

---

## 4. High — queued halts verify nothing

**Evidence:** [tools/live_levels_trader.py:2506-2509](tools/live_levels_trader.py#L2506-L2509)
→ [tools/live_levels_trader.py:1830](tools/live_levels_trader.py#L1830)
→ [tools/live_levels_trader.py:1711-1712](tools/live_levels_trader.py#L1711-L1712).

`drain_control_queue` calls `halt(state, ..., kind="manual")` with **no
`backend`**. `halt` then calls `verify_open_protection(None, state)`, and
`has_protection(None, coin)` returns `None` at line 1711-1712, so nothing is
ever `is False`:

```
has_protection(None, 'HYPE')        -> None
verify_open_protection(None, state) -> []        <- D-1 check is a no-op
```

`HALT_ACTION == "flatten"` is also skipped, since line 1838 requires
`backend is not None`.

**Failure scenario.** The reconcile loop detects the one condition it is built to
detect — `"HYPE is open and unprotected -- no stop is resting at the venue"`
([tools/reconcile_loop.py:120-122](tools/reconcile_loop.py#L120-L122)) — and
queues a halt. The trader applies it through the path that cannot check
protection and cannot flatten. The position stays open and unstopped; the
operator gets the generic "Level trader halted" message. Every Telegram `/halt`
has the same hole.

`test_a_halt_without_a_backend_still_halts`
([test_e2e_fake_venue.py:389-397](test_e2e_fake_venue.py#L389-L397)) asserts only
that `halted` becomes `True`, which institutionalises this.

`drain_control_queue` has `backend` available at its call site
([tools/live_levels_trader.py:775](tools/live_levels_trader.py#L775), inside
`if backend is not None` two lines later) and does not pass it.

---

## 5. High — the kill switch is blind to the positions C-1 is about

**Evidence:** [tools/telegram_control.py:264-266](tools/telegram_control.py#L264-L266),
[tools/telegram_control.py:351-356](tools/telegram_control.py#L351-L356),
[tools/telegram_control.py:391-395](tools/telegram_control.py#L391-L395).

`/closeall` iterates `state["owned"]`. `/close COIN` returns
`"No open position in {coin}."` for anything not in `owned`. `describe` says
`"Flat already. Confirming will do nothing."` when `owned` is empty.

So the operator's emergency button cannot close: a timeout-after-fill orphan, a
filled resting bid still in `pending`, a position the state file lost, or
anything opened by hand at the venue. These are precisely the positions C-1 and
D-1 exist to find, and the one documented incident
([live_execution_backend.py:462-466](live_execution_backend.py#L462-L466), the
2026-10-17 hand-flatten that left ZRO at $178.70) is this shape.

**Failure scenario.** Finding 1's phantom/orphan path leaves 10 HYPE live and
untracked. The operator sees the reconcile-loop alert and sends `/closeall`.
Reply: "Flat already." There is no Telegram command that can close it.

The fix is to union `owned`, `pending`, and the venue's own `assetPositions`
before flattening — `backend.info.user_state` is already available on the
control process, and `render_positions` at line 171 takes a backend.

Secondary: on success, `close_one` (line 274-280) does not cancel the position's
stop and TP, and tells the operator "Its stop and take-profit are still in
place" on failure — which is false for an orphan that never had any.

---

## 6. High — a partially filled resting bid orphans its remainder, and nothing handles a position that grew

**Evidence:** [tools/live_levels_trader.py:2021-2035](tools/live_levels_trader.py#L2021-L2035)
(`reconcile_pending` adopts on any non-zero size and `pending.pop`s without
cancelling), [tools/live_levels_trader.py:2153-2154](tools/live_levels_trader.py#L2153-L2154)
and [tools/live_levels_trader.py:2236](tools/live_levels_trader.py#L2236)
(`reconcile`'s only two branches).

`reconcile_pending` treats any venue size as "the bid filled", sets
`size_coin = abs(szi)`, protects that, then drops the `pending` record — and with
it `p["oid"]`, the only handle on the order. If the bid filled *partially*, the
remainder is still resting at the venue and is now untracked by anything.

`reconcile` has a branch for `abs(sz) < believed * 0.99` (partial close) and one
for `abs(sz) < 1e-12` (flat). There is **no branch for `abs(sz) > believed`**.
`venue_sanity` does not help either: it compares coin *sets*, never sizes
([tools/live_levels_trader.py:1546-1552](tools/live_levels_trader.py#L1546-L1552)).

**Failure scenario.** A $200 GTC bid on HYPE fills 60% at the level. The next
pass adopts 1.2 coins, protects 1.2 coins, drops the pending record. Price
re-tests the level an hour later and the remaining 0.8 fills. The account now
holds 2.0 coins against a stop and a TP covering 1.2. When the stop triggers it
closes 1.2 and leaves 0.8 **live and unstopped**. `reconcile` sees
`abs(sz)=0.8 < believed=1.2 * 0.99` and routes it to the partial-close branch,
which — because `residual_usd` is below `MIN_NOTIONAL_USD` or not, depending on
size — either flattens it or re-protects it, having never recorded that the
position was ever 2.0. The trade's realised PnL is wrong by the 0.8 leg either
way, and `verify_open_protection` reports the whole thing protected throughout
(finding 3, since the partially-filled bid is a resting non-protective order).

Minimum fix: cancel the remainder before popping `pending`, and give `reconcile`
an explicit "venue holds more than we believe" branch that halts.

---

## 7. Medium — protection oids are never persisted

**Evidence:** written nowhere. The three `owned[coin]` writes are
[tools/live_levels_trader.py:1340-1342](tools/live_levels_trader.py#L1340-L1342),
[tools/live_levels_trader.py:1385-1386](tools/live_levels_trader.py#L1385-L1386)
and [tools/live_levels_trader.py:2027-2030](tools/live_levels_trader.py#L2027-L2030);
none records `stop_order_id` / `tp_order_id`. The only read is
[tools/live_levels_trader.py:2189](tools/live_levels_trader.py#L2189):

```python
                cancel_protection(
                    backend, coin,
                    (owned[coin].get("stop_oid"), owned[coin].get("tp_oid")))
```

`grep -n "stop_oid\|tp_oid"` across the repo returns that line and nothing else.
Both gets always return `None`, and `cancel_protection` skips falsy oids at
[tools/live_levels_trader.py:1504-1505](tools/live_levels_trader.py#L1504-L1505).
So the one call site that cleans up after a dust close is a permanent no-op, and
its "⚠️ protection may still be resting" alert can never fire.

`reconcile`'s normal flat branch (line 2236-2253) does not call
`cancel_protection` at all, while the dust branch does — an asymmetry that
suggests the intent was to cancel in both.

**Failure scenario.** A take-profit fires and closes the position. The paired
stop remains resting (if the venue does not auto-cancel it — worth confirming
against Hyperliquid's reduce-only semantics for non-position TPSL triggers, which
is how `_place_trigger_reduce_only` submits them). The coin re-enters on a later
day at a different level. A stale trigger at the *old* stop price is now live
against the new position, sized to the old position. Nothing in this branch can
cancel it, because the oid was never stored.

---

## 8. Medium — a restart between claim and fill leaves a position with no recorded size

**Evidence:** [tools/live_levels_trader.py:1340-1344](tools/live_levels_trader.py#L1340-L1344)
writes `owned[coin]` with `confirmed: False` and **no `size_coin`** before the
order is sent (correct claim-before-send); `size_coin` only arrives at
[tools/live_levels_trader.py:1385-1386](tools/live_levels_trader.py#L1385-L1386).

If the process dies in between — SIGKILL from launchd, a deploy, a crash in
`_parse_fill` — the persisted record has `entry` (the *planned* entry), `stop`,
`tp`, and no size.

Three rails then degrade quietly:

- `assumed_loss` ([tools/live_levels_trader.py:1570-1578](tools/live_levels_trader.py#L1570-L1578))
  returns `0.0` when `size_coin <= 0`. So an unmeasured close of this position is
  charged **$0** to `realised_today` — contradicting the function's own docstring
  ("an unmeasured close must never be cheaper than a measured one") and leaving
  the daily loss stop blind to it.
- `reconcile`'s partial-close branch requires `believed > 0`
  ([tools/live_levels_trader.py:2153](tools/live_levels_trader.py#L2153)), so a
  partial close on this position is never detected.
- `reconcile_loop`'s size check requires `want and want > 0`
  ([tools/reconcile_loop.py:115-117](tools/reconcile_loop.py#L115-L117)), so the
  independent watcher does not check it either.

**Failure scenario.** A deploy restarts the trader three seconds after an entry
IOC goes out. The position is live and protected at the venue. Its state record
has no size. It later stops out. `get_realised_pnl` happens to fail (rate limit),
`assumed_loss` returns 0.0, and a full stop-out is booked as a flat day. Four of
those and the daily loss stop has still seen nothing.

A `size_coin` estimated from the planned notional at claim time would close this.

---

## 9. Medium — the startup orphan sweep ignores `pending`

**Evidence:** [tools/live_levels_trader.py:762](tools/live_levels_trader.py#L762)
(runs before the pass loop), [tools/live_levels_trader.py:1762](tools/live_levels_trader.py#L1762)
(`state_coins` from `owned` only), [tools/live_levels_trader.py:1782-1791](tools/live_levels_trader.py#L1782-L1791).
`ORPHAN_ACTION` defaults to `"auto"`
([research/levels/config.py:905](research/levels/config.py#L905)).

**Failure scenario.** A resting bid fills. Before the next pass can run
`reconcile_pending`, launchd restarts the trader (deploy, or the non-zero exit at
[tools/live_levels_trader.py:834](tools/live_levels_trader.py#L834)). Startup
calls `adopt_or_flatten_orphans`, which compares the venue against `owned` only.
The filled bid is in `pending`, so it reads as an orphan. It has no protection
yet — protection is placed by `reconcile_pending`, which has not run — so
`protected is False` and it is **flattened**, at market, immediately. The
operator gets "⚠️ Positions found that this bot did not open".

Worse, `pending[coin]` survives. `reconcile_pending` then sees the venue flat,
falls to its "still resting" branch
([tools/live_levels_trader.py:2090-2105](tools/live_levels_trader.py#L2090-L2105)),
and prints "bid resting" every pass until the TTL expires, at which point it
cancels an oid that filled hours ago.

Any restart while a bid is filled-but-unreconciled costs a round trip in fees and
spread and loses the trade.

---

## 10. Medium — `flatten` cannot outlast the backoff it can trigger

**Evidence:** [live_execution_backend.py:481-535](live_execution_backend.py#L481-L535),
`RETRY_READ_SLEEP_SEC` default `1.0` at
[live_execution_backend.py:124](live_execution_backend.py#L124),
`API_GLOBAL_BACKOFF_SECONDS` default `30.0` and `API_FAILURE_BURST_LIMIT` `5` at
[live_data_guard.py:35-36](live_data_guard.py#L35-L36),
`LIVE_MAX_MIDPRICE_AGE_SECONDS` `10.0` at [live_data_guard.py:34](live_data_guard.py#L34).

`flatten` is the one function every emergency path depends on — `halt`'s flatten,
`/close`, `/closeall`, the protection-failed exits, the orphan sweep. It makes
3 attempts, 1 second apart. Each attempt costs one `user_state`, one `all_mids`,
one `order`, and `execute_exit` makes another `user_state` and `all_mids` of its
own ([live_execution_backend.py:354](live_execution_backend.py#L354),
[live_execution_backend.py:368](live_execution_backend.py#L368)).

**Failure scenario.** `/closeall` on four positions = up to 4 × 3 × ~4 requests
in a few seconds. Five `all_mids` failures inside 30 seconds trip the global
backoff for 30 seconds. `get_mid_price` then returns `None` once the cached mid is
older than 10s ([live_execution_backend.py:932-951](live_execution_backend.py#L932-L951)),
so `flatten` takes its "no price, not sending blind" branch
([live_execution_backend.py:515-518](live_execution_backend.py#L515-L518)),
burns all three attempts in three seconds, and returns `flat: False`. The
operator is told "🚨 STILL OPEN after 3 attempts" with "Its stop and take-profit
are still in place" — and the kill switch has done nothing during the exact
minute it was needed.

Related asymmetry: `_get_venue_signed_size` calls `record_api_success()` on
success ([live_execution_backend.py:1263](live_execution_backend.py#L1263)) but
never `record_api_failure` on the exception path
([live_execution_backend.py:1270-1276](live_execution_backend.py#L1270-L1276)).
The heaviest query in the system cannot feed the backoff that is supposed to
protect it, while it *can* clear the failure counter.

The retry budget for a kill switch should outlast the backoff window, not be a
tenth of it.

---

## 11. Medium — post-entry `venue_sanity` halts on a mid-pass bid fill

**Evidence:** [tools/live_levels_trader.py:928-931](tools/live_levels_trader.py#L928-L931),
[tools/live_levels_trader.py:1546-1552](tools/live_levels_trader.py#L1546-L1552).

`venue_sanity` compares the venue against `state["owned"]` and returns
`fatal=True` for anything it does not recognise. It does not consult
`state["pending"]`. The main loop's mitigation is to run `reconcile_pending`
first (line 790-792), which handles the top of the pass — but `venue_sanity` runs
**again after every order sent**, at line 928.

**Failure scenario.** A bid rests on ZRO. The pass starts, `reconcile_pending`
sees ZRO unfilled. An entry on HYPE is sent. In those seconds ZRO's bid fills.
The post-entry `venue_sanity` reports `"venue holds ['ZRO'] which this module did
not open"`, `fatal=True` → `halt(..., "after entry")` → trading stops on a
perfectly correct fill. The halt carries `kind=""`, so
`release_halt_if_clear` cannot clear it either (line 2375) and `--resume` is
required.

Adding `state["pending"]` to the `known` set at line 1548 would fix this and
finding 9's root cause at the same time.

---

## 12. Medium — `HALT_ACTION=flatten` flattens but does not book

**Evidence:** [tools/live_levels_trader.py:1822](tools/live_levels_trader.py#L1822)
(`save_state` happens *before* the flatten),
[tools/live_levels_trader.py:1838-1843](tools/live_levels_trader.py#L1838-L1843).

The flatten loop neither pops `state["owned"]` nor calls `accrue_close`, and
`halt` never saves again afterwards. `accrue_close`'s docstring says "EVERY path
that closes a position must come through here"
([tools/live_levels_trader.py:1859](tools/live_levels_trader.py#L1859)); this one
does not.

The default is `verify_stops`
([research/levels/config.py:892](research/levels/config.py#L892)), so this is
latent rather than live — but it is the setting an operator reaches for after an
incident. With it on, a drawdown halt closes four positions and books none of
them; `realised_today` and the loss streak miss the whole set until `reconcile`
picks them up on a later pass, and `release_halt_if_clear` may clear the drawdown
halt in the meantime on an equity reading that the unbooked closes have already
changed.

`test_halt_action_flatten_closes_open_positions`
([test_e2e_fake_venue.py:375-387](test_e2e_fake_venue.py#L375-L387)) asserts the
venue position reaches zero and checks neither `owned` nor `realised_today`.

---

## 13–15. Low

**13.** `flatten` returns `"attempts_used": attempts` unconditionally at
[live_execution_backend.py:534](live_execution_backend.py#L534), so the number in
the "STILL OPEN after N attempts" alert is the configured maximum, not what
happened. Iterations that `continue` on an unreadable size or a missing price
(lines 486-493, 515-518) consume the budget without sending anything, so three
unreadable reads report "3 attempts" having placed no order at all.

**14.** `execute_one` re-implements `cancel_protection` inline at
[tools/live_levels_trader.py:1429-1436](tools/live_levels_trader.py#L1429-L1436)
rather than calling it, so the "protection may still be resting" alert at
[tools/live_levels_trader.py:1518-1523](tools/live_levels_trader.py#L1518-L1523)
cannot fire from the emergency-exit path — which is the path whose own comment
(lines 1424-1428) explains why that alert matters. This is the fourth instance of
the sibling-call-site pattern that commit `e027ab4` was about.

**15.** [live_execution_backend.py:1155-1157](live_execution_backend.py#L1155-L1157)
chains `or` across candidate fields, so a genuine `accountValue` of `0.0` reads
as "field not found" and the function returns `None`. Pre-existing, and `_f`
already distinguishes `None` from `0.0` — the chain just discards that.

---

## Where the tests agree with the bugs

The suite is green on all of the above. Three places where a fake or an
assertion encodes the same misunderstanding as the code:

**Finding 1.** The fake venue's cloid lookup
([test_e2e_fake_venue.py:139-148](test_e2e_fake_venue.py#L139-L148)) returns
`"sz": str(r["sz"])` where `r["sz"]` is the size the order **requested**
([test_e2e_fake_venue.py:59-62](test_e2e_fake_venue.py#L59-L62)) — i.e. it models
`sz` as filled size, which is what the backend assumes and the opposite of what
`orderStatus` returns. It also hardcodes the inner status to `"filled"` for every
order it ever saw, so the cancelled-IOC adoption in finding 1 is unreachable in
the fake. `test_a_timeout_is_resolved_by_cloid_not_assumed_rejected`
([test_e2e_fake_venue.py:282-295](test_e2e_fake_venue.py#L282-L295)) therefore
passes against a payload the venue does not produce, and
`test_an_unknown_order_after_timeout_stays_a_reject` only exercises the
`unknownOid` branch. A fixture built from the real `orderStatus` shape — inner
status `canceled`, `sz` as remaining — fails on both.

**Finding 2.** `test_a_position_state_has_and_the_venue_does_not`
([test_reconcile_loop.py:82-86](test_reconcile_loop.py#L82-L86)) asserts that a
position the venue no longer holds **is** reported as a problem. That is the
false positive from scenario 1 asserted as desired behaviour. The `state()`
helper ([test_reconcile_loop.py:58](test_reconcile_loop.py#L58)) hardcodes
`"pending": {}`, so the pending blind spot is never exercised. No test covers
staleness or a grace period.

**Finding 3.** `has_protection` has no direct test anywhere
(`grep -rn "has_protection"` finds only the two definitions and three internal
call sites). `test_halting_with_a_protected_position_does_not_cry_wolf`
([test_e2e_fake_venue.py:357-369](test_e2e_fake_venue.py#L357-L369)) supplies a
resting order that is both `reduce_only` and `is_trigger`, so it returns `True`
from the loop and never reaches the broken fallback. A case with one plain
non-reduce-only order would fail.

Two further coverage gaps worth noting: `test_flatten.py` and
`test_protection_verification.py` never drive `flatten` under an active API
backoff (finding 10), and nothing tests a venue position **larger** than the
believed size (finding 6).

---

## Suggested order of work

1. Finding 1 — rewrite `_fill_response_from_order` against the real
   `orderStatus` payload (use `origSz - sz` for filled size, read the inner
   `order.status`, and treat `canceled`/`rejected` with zero fill as a true
   reject). Until then, every venue timeout is a coin flip between a phantom
   position and a lost one.
2. Finding 2 — give divergence halts a re-checkable `kind`, teach `check` about
   `pending`, and add a grace period longer than `POLL_SEC`. Until then the
   reconcile loop is a scheduled outage on every trade.
3. Findings 3 and 4 — one-line and one-argument fixes respectively, and both sit
   directly on the D-1 guarantee.
4. Finding 5 — union the venue's own positions into `/closeall`.
5. Findings 6, 9, 11 — all three are the same root cause: `pending` is not part
   of "what this module believes it holds".
