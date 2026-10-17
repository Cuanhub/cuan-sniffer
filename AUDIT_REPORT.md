# Cuan Sniffer — Phase 1 Audit

**Date:** 2026-10-17 · **Branch:** `sprint-4-alt-strategy-lab` · **HEAD:** `e027ab4`
**Scope:** full repo, `git log -p --since="3 weeks ago"` (42 commits)
**Account:** Hyperliquid mainnet, live, equity **$720.17**, currently flat and
unhalted (trading).

## Auditor conflict of interest — read this first

**I wrote the large majority of the code committed in the last three weeks,
in the session immediately preceding this audit.** I am auditing my own work.
Where I could verify a claim against something outside my own judgement — the
test suite, the installed SDK, venue state, git history, the trade ledger — I
did, and the evidence is cited. Where a finding rests on my reading of intent
rather than on external evidence, it is labelled UNVERIFIED. A second pair of
eyes on sections A and C would be worth more than anything else in this
document.

---

## 1. Executive summary

### Top 10 risks, ranked

| # | Risk | Sev |
|---|------|-----|
| 1 | **No measured edge.** The bot's own startup banner states the entry gate measured **−0.0413R versus random entry (z=−0.93, n=688)**. Live: 109 closed trades, **net −$46.65**, profit factor **0.81**, 24.8% win against a 28.9% breakeven. | CRITICAL |
| 2 | No client order IDs on any order; a timed-out-but-filled order leaves an untracked, unprotected position for up to 450s | HIGH |
| 3 | Kill switch blocks new entries but never flattens or re-stops open positions | HIGH |
| 4 | 450s poll cadence is the only market-data path — no streaming, so an adverse move is unseen for up to 7.5 minutes | HIGH |
| 5 | Every dependency unpinned (`>=`); no `pip-audit`, no `mypy`, no pre-commit | HIGH |
| 6 | Four uncommitted files for a new subsystem (Po3) sit in the working tree of a live-deployed repo | HIGH |
| 7 | Admission rails are young and partly fitted — `pull_over_run` threshold chosen from a 9-point sweep on 492 simulated episodes in one regime | MEDIUM |
| 8 | Stop/TP prices round to *nearest* tick, so a stop can land a tick against us | MEDIUM |
| 9 | No structured logging, metrics or dashboard; observability is `print` into a file nobody reads except via Telegram alerts | MEDIUM |
| 10 | ~45 features have been screened against ~1.7k episodes with no family-wise correction; survivorship in the feature set is likely | MEDIUM |

### Go / no-go

**NO-GO as an edge-seeking strategy. Conditional GO as a bounded-risk data
collection exercise.**

The engineering is, with the exceptions below, better than typical: dry-run is
the default, arming needs two independent switches, state load fails closed,
venue reconciliation runs twice a pass and halts on disagreement, sizes round
down through `Decimal`, fees are subtracted from realised PnL, and there are
624 passing tests. **None of that is the problem.**

The problem is finding #1, and it is not a bug. The system measures its own
gate at −0.04R against random entry and has lost money over 109 live trades at
a profit factor of 0.81. Every rail added in the last three weeks reduces the
*rate* of loss; none of them creates an edge. Continuing to run this live is
defensible only as paying for data, and only if that is the explicit, stated
intent and the position sizing reflects it.

At $180 notional per trade and 5% margin on a $720 account, the current
configuration risks roughly **$3 per trade and up to 5% of equity per day**.
That is a tuition fee, not a strategy. It is survivable. It is not an
investment.

**My recommendation:** keep running, but (a) halve `MARGIN_PCT_PER_TRADE` to
2.5% while the edge is unproven, (b) fix findings C-1, D-1 and B-1 before the
next size increase, and (c) set a written date and sample size at which the
edge question gets answered yes or no. Without (c) this runs indefinitely on
hope.

---

## 2. Findings

Severity: **CRITICAL** = stop trading / fix now · **HIGH** = fix this week ·
**MEDIUM** = fix this month · **LOW** = backlog.

| ID | Sev | Area | Evidence | Issue | Failure scenario | Proposed fix | Effort |
|----|-----|------|----------|-------|------------------|--------------|--------|
| **E-1** | CRITICAL | E Strategy | banner in `tools/live_levels_trader.py` startup; `research/levels/data/live_trades.csv` (109 closes) | Gate measures −0.0413R vs random (z=−0.93, n=688). Live net −$46.65, PF 0.81, 24.8% win vs 28.9% breakeven | Continues losing at a bounded rate indefinitely; capital erodes while rails are tuned | Set an explicit decision point (sample size + date). Cut size to 2.5% until then. Do not scale on a sub-1.0 profit factor | 1h decision, 0 code |
| **C-1** | HIGH | C Execution | `live_execution_backend.py:212`, `:325` | `exchange.order(...)` passes no `cloid`. No client order ID anywhere in the repo | Request times out after the venue accepts. Code records a reject; venue holds an **unprotected, untracked** position. `venue_sanity` catches it only on the next pass — up to 450s later | Pass a deterministic `cloid` per intent; on ambiguous response, query by `cloid` before concluding anything | M (4–6h) |
| **D-1** | HIGH | D Rails | `tools/live_levels_trader.py:1634`; `tools/telegram_control.py:318` | `halt()` blocks new entries only. No halt path flattens or re-verifies stops on open positions | Account drawdown halt fires with 4 positions open. Bot stops trading but keeps full exposure, relying entirely on venue-side stops being intact | Add `HALT_ACTION` config: `block` (today) / `flatten`. Default `block`, but re-verify protection on every halt | M (4h) |
| **B-1** | HIGH | B Data | `research/levels/config.py:806` (`POLL_SEC=450`), `live_execution_backend.py:146` (`skip_ws=True`) | No websocket anywhere. All market and account data is REST-polled every 450s | A gap through the stop, a partial fill, or a venue-side liquidation is invisible for up to 7.5 minutes. Native stops mitigate price risk but not state divergence | Subscribe to the user-fills websocket for position/fill events; keep REST as the reconciliation path | L (1–2 days) |
| **F-1** | HIGH | F Ops | `requirements.txt` (all `>=`); no `mypy`/`pip-audit`/`pre-commit` present | No dependency pinning or vulnerability scanning; no type checking on 2,400-line order-placing module | A transitive update to `hyperliquid-python-sdk` or `eth-account` silently changes order semantics on the next rebuild | Pin with a lockfile, add `pip-audit` + `mypy --strict` on the backend to CI | M (half day) |
| **A-1** | HIGH | A Hygiene | `git status`: `research/levels/po3.py`, `test_po3.py`, `tools/collect_po3.py`, `tools/screen_po3.py` untracked | A four-file new subsystem is uncommitted in the working tree of a repo that is deployed by `launchctl` from this same directory | A restart picks up whichever files happen to be on disk. Untracked code is unreviewed and unrevertable | Commit behind a flag or stash. Never leave untracked code in a deployed tree | S (15m) |
| **E-2** | MEDIUM | E Strategy | `research/levels/config.py` (`ENTRY_MAX_PULL_OVER_RUN`), `tools/screen_context.py` | `pull_over_run = 0.60` chosen as best-of-9 on 492 **simulated** episodes (no fees/slippage), 10 days, one falling regime. 33 matched real trades were non-monotone | Rail rejects ~68% of candidates on a threshold that may be noise; real-money behaviour unknown | Re-screen against real fills at ≥100 closes carrying the field. Treat as provisional until then | S (already instrumented) |
| **E-3** | MEDIUM | E Strategy | ~45 features screened across sessions; see `DECISIONS.md` | No family-wise error correction across the full feature search. Day-stratified permutation and chronological holdout are applied, which is good, but the *search* is uncorrected | A surviving feature is a false positive at a rate well above its nominal p | Pre-register the remaining hypotheses; apply Benjamini-Hochberg across the family | S (2h) |
| **A-2** | MEDIUM | A Correctness | `live_execution_backend.py:1841` → `round_to_tick(..., direction="nearest")` | Stop and TP prices round to nearest tick, not away from the position | A long's stop rounds *up* by one tick, exiting marginally early; a TP rounds *down*, exiting early. Small but systematic | Round stops down and TPs up for longs (always in the position's favour) | S (1h) |
| **D-2** | MEDIUM | D Rails | no match for rate-limit/throttle in `tools/live_levels_trader.py` | No explicit order-rate cap. Rate is bounded only by `ENTRY_SPACING_MINUTES=60`, `MAX_CONCURRENT=4`, `MAX_NEW_PER_DAY=20` | A bug in the spacing or concurrency check removes the only brake. Nothing counts orders per unit time independently | Add an independent orders-per-hour counter that halts on breach | S (2h) |
| **F-2** | MEDIUM | F Ops | plain `print()` throughout; `~/Library/Logs/cuan-sniffer/trader.log` | No structured logs, no trade-ID correlation, no metrics, no health endpoint. Observability is Telegram alerts plus grepping a text file | A slow degradation (rising slippage, falling fill rate) is invisible until it shows in PnL | Emit JSON lines with `episode_id` on every decision; a 20-line daily digest would cover most of it | M (half day) |
| **A-3** | LOW | A Correctness | `tools/telegram_control.py:162`, `:367` | `except ValueError: pass` on user-typed numbers — the only swallowed exceptions in the codebase | A malformed `/pause` argument is silently ignored rather than answered | Reply with the parse error | S (15m) |
| **F-3** | LOW | F Hygiene | `tools/live_levels_trader.py:2383` | `equity_peak` is written and read only by itself; displayed nowhere, decides nothing | Dead state that reads as a live risk input to a new maintainer | Remove, or surface it in `/status` | S (15m) |
| **A-4** | LOW | A Hygiene | `test_research_levels_isolation.py:35` | One unused import (deliberate, `noqa`'d) | None — it is the subject of the test | None | — |
| **F-4** | MEDIUM | F Ops | `git status`: `research/levels/data/archive/live_trades-2026-10.csv.gz` modified | The retention archive is **tracked in git** and is appended to by the daily job, so a live process dirties the working tree and grows repo history with binary blobs | Repo size grows unboundedly; every `git status` on a deployed tree is dirty, which is how A-1 went unnoticed | Add `research/levels/data/archive/` to `.gitignore`; back it up out of band | S (15m) |

### Verified as sound (counter-evidence, stated for balance)

These were checked and found correct. They matter because they bound the blast
radius of everything above.

- **No hallucinated SDK calls.** All 11 distinct methods invoked on `Info` /
  `Exchange` verified present on the installed `hyperliquid-python-sdk`. The
  one apparent miss (`info.get`) is a local dict, not the SDK object.
- **No bare `except:`, no naive `datetime.now()`** anywhere in production code.
- **Sizes round down through `Decimal`/`ROUND_DOWN`** (`live_execution_backend.py:1834`) — never oversizes.
- **Dry-run is the default.** Real orders require `--arm` **and**
  `LEVELS_LIVE_ARMED=yes` (`tools/live_levels_trader.py:673`). A paper mode
  already exists, so Phase 2's "add a dry-run mode" is satisfied.
- **No retry loop around `exchange.order`** — a timeout becomes a reject, which
  substantially reduces (but does not eliminate, see C-1) double-submit risk.
- **`venue_sanity` runs twice per pass** (`:826`, `:923`) and halts fatally on
  an untracked position, a concurrency breach, or notional > `MAX_LEVERAGE` × equity.
- **`load_state` refuses to start on a corrupt state file** rather than
  returning an empty one (`:127`) — a corrupt file is not a new account.
- **Fees subtracted from realised PnL** (`live_execution_backend.py:1140`).
- **Mid-price staleness enforced** at 10s before pricing an exit (`:807`, `:837`).
- **No secrets committed.** `.env` is gitignored; only `.env.example` is tracked;
  no secret-shaped literals in tracked source. Bot token is a live credential
  that can close positions and halt trading, and cannot move money.
- **624 tests pass; pyflakes clean** on production code.
- **Retention archives rather than deletes**, with the money rows never on a clock.

---

## 3. Safety-rails scorecard (section D)

| Rail | Status | Evidence / gap |
|------|--------|----------------|
| Max position size per asset | **PRESENT** | `MARGIN_PCT_PER_TRADE=5%` × `FLAT_LEVERAGE=5x` → ~$180 notional; `MIN_NOTIONAL_USD` floor |
| Max total exposure | **PRESENT** | `MAX_CONCURRENT=4` (20% margin committed); `venue_sanity` halts if notional > `MAX_LEVERAGE`×equity |
| Max loss per trade | **PARTIAL** | Bounded by the *structural* stop, which varies by setup (median 1.76%). No hard per-trade dollar cap |
| Daily loss limit | **PRESENT** | `DAILY_LOSS_STOP_PCT=5%` on realised PnL, halts |
| Max drawdown circuit breaker | **PRESENT** | `ACCOUNT_DD_HALT_PCT=10%` over a rolling 24h equity high |
| Kill switch — manual | **PRESENT** | `/halt` via Telegram (queued, applies next pass); `--resume` to clear |
| Kill switch — automatic | **PRESENT** | All three rails route through `halt()` |
| Kill switch stops NEW orders | **PRESENT** | `preflight` returns on the persisted flag before any sizing |
| Kill switch manages OPEN positions | **MISSING** → see D-1 | Blocks entries only. Relies on venue-side stops remaining intact |
| Max order rate / frequency cap | **PARTIAL** → D-2 | Entry spacing 60m + concurrency 4 + 20/day. No independent rate counter |
| Absurd size/price rejection | **PRESENT** | `MIN_NOTIONAL_USD`, `MAX_LEVERAGE` ceiling, `_validate_protection_price_relations`, drift veto |
| Crash/restart recovery | **PRESENT** | `load_state` fails closed; `venue_sanity` detects duplicates/untracked; launchd `KeepAlive` with `SuccessfulExit=false` |
| Alerting on halt / limit breach | **PRESENT** | Every `halt()` sends Telegram, keyed to avoid repeat-fatigue |
| Alerting on error spike | **MISSING** | No error-rate monitor. A rising reject or fetch-failure rate is silent |

**Score: 9 present, 3 partial, 2 missing.** The two missing items (halt does not
manage open positions; no error-rate alerting) are both in the fix plan.

---

## 4. Test coverage gaps

624 tests pass, which is genuinely good coverage for the decision logic. The
gaps are concentrated where failure is most expensive.

| Area | Coverage | Gap |
|------|----------|-----|
| `live_execution_backend.py` (2,065 loc, places all orders) | 4 test files | **No test sends a real order through a fake venue end to end.** `execute_entry`'s response parsing, partial-fill handling and reject paths are the least-covered expensive code in the repo |
| `flatten()` | 18 tests | Good, and new. Added after it was found to report a live short as flat |
| `tools/screen_po3.py`, `tools/collect_po3.py` | **0 tests** | Uncommitted (A-1). `research/levels/po3.py` has 34 |
| `tools/near_firing.py`, `research/levels/review.py` | **0 tests** | Read-only reporting tools; low risk |
| `tools/retention.py` | 23 tests | Covers the append race and copy-truncate |
| Order-state machine (partial → reject → unknown) | **thin** | No test for an *unknown* venue state (timeout with unknown fill) — the exact C-1 scenario |
| Websocket/stream handling | N/A | None exists (B-1) |
| Integration / soak | **none** | No test runs the pass loop against a simulated venue for N passes |

**Highest-value test to add:** a fake venue that accepts an order and then
times out the response, asserting the trader does not record a position as
rejected while the venue holds it.

---

## 5. Prioritised fix plan

Each batch is small enough to review in one sitting. Phase 2 work, pending
approval.

**Batch 0 — hygiene, no behaviour change (30 min)**
- A-1: commit or stash the four Po3 files; confirm the deployed tree is clean
- A-4, F-3: remove `equity_peak` or surface it; tidy the last unused import

**Batch 1 — the decision, not code (1h, no commits)**
- E-1: agree a written decision point (sample size + date) for the edge question
- Cut `MARGIN_PCT_PER_TRADE` 5% → 2.5% while unproven (config-only, reversible)

**Batch 2 — order integrity (CRITICAL path, ~6h)**
- C-1: deterministic `cloid` per intent; on ambiguous response, resolve by
  `cloid` before concluding reject
- New test: venue accepts, response times out → trader must not record a reject
- A-2: round stops/TPs in the position's favour

**Batch 3 — halt completeness (~5h)**
- D-1: `HALT_ACTION` config (`block` default, `flatten` opt-in); re-verify
  protection on every halt
- D-2: independent orders-per-hour counter that halts on breach
- Alerting on error-rate spike (the second MISSING rail)

**Batch 4 — supply chain and types (~half day)**
- F-1: lockfile, `pip-audit`, `mypy` on `live_execution_backend.py` + `tools/`
- Pre-commit running pyflakes + the suite

**Batch 5 — data path (1–2 days, largest)**
- B-1: user-fills websocket for position/fill events; REST stays as reconciliation
- Integration test: pass loop against a simulated venue over many passes

**Batch 6 — observability (~half day)**
- F-2: JSON-line decision logs keyed by `episode_id`; daily digest alert

**Deliberately not in the plan:** E-2 and E-3 are already instrumented and
re-measure on their own schedule. Acting on them now would be more fitting, not
less.

---

## Appendix — architecture map

```
INGESTION        Hyperliquid REST only (no websockets, skip_ws=True)
                 research/levels/venue_client.py — candles, mids, perp contexts
                 scanner: 900s cadence · trader: 450s cadence
                     ↓
SIGNAL           research/levels/levels.py   level detection, sigma normalisation
                 research/levels/gate.py     quality gate (run/pull/retest/near)
                 research/levels/trigger.py  level reached + held
                 research/levels/call.py     candidate assembly
                     ↓
RISK CHECKS      tools/live_levels_trader.py
                 preflight()      halt flag · stand-down · concurrency · day cap
                                  · daily loss stop · rolling 24h drawdown
                 entry_is_actionable()  drift · break-through · perp basis
                                        · pull/run
                 venue_sanity()   untracked positions · concurrency · notional
                     ↓
EXECUTION        live_execution_backend.py
                 execute_entry (marketable IOC) → place_native_protection
                 (venue-side stop + TP, verified) → flatten() for exits
                     ↓
TRACKING         _live_state.json (single writer: the trader)
                 live_trades.csv (audit trail) · episodes.csv · outcomes.csv
                 reconcile() per pass against venue size
                     ↓
ALERTING         notify.py → Telegram (send + getUpdates long-poll)
CONTROL          tools/telegram_control.py → command queue → trader drains
SUPERVISION      launchd: scanner · trader · telegram · retention (daily)
```

**External dependencies (complete):**
`api.hyperliquid.xyz/info` (market + account data, REST),
Hyperliquid exchange endpoint via `hyperliquid-python-sdk` (orders),
`api.telegram.org` (alerts + control).
No database, no RPC, no websockets, no other third-party API. Nothing listens
on a port.

**Riskiest diffs in the 3-week window** (by churn in order-placing code):
`tools/live_levels_trader.py` 1,207 lines changed across 42 commits.
Highest-risk individual commits: `2f45891` (rewrote all three close paths),
`2862bc3` (replaced the drawdown rail and added self-clearing halts),
`f80e3a6` (added a second process with order-placing capability).
