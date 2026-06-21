# Cuan Sniffer — Operations Runbook

> Live operations reference. See README.md for architecture overview and `.env.example` for full parameter reference.

---

## Quick start

```bash
cp .env.example .env
# Fill in: RPC_URL, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, HL_ACCOUNT_ADDRESS, HL_SECRET_KEY
pip install -r requirements.txt
python main.py
```

Use `HL_TESTNET=true` for exchange testnet validation. Paper execution is not supported.

---

## Unified Threshold Framework

> Implemented 2026-06-16, Sprint 7 post-candle-fix.

All confidence gates are controlled by a **single master variable**. Do not set individual gate variables unless you have a documented, evidence-based reason to diverge.

### Master variable

```bash
UNIVERSAL_MIN_CONFIDENCE=0.90   # live-test phase default
```

All of the following variables default to `UNIVERSAL_MIN_CONFIDENCE` when unset:
- `SWING_MIN_CONFIDENCE`
- `SMC_4H_MIN_CONFIDENCE`
- `WEAK_TREND_MIN_CONFIDENCE`
- `CHOP_REVERSAL_MIN_CONFIDENCE`
- `MIN_SIGNAL_CONFIDENCE`

**To raise the confidence floor system-wide:** change `UNIVERSAL_MIN_CONFIDENCE` only.  
**To override one gate:** set the specific variable in `.env` with a comment explaining why.

### Score pre-filters

```bash
REGIME_SCORE_THRESHOLD_STRONG=0.64
REGIME_SCORE_THRESHOLD_WEAK=0.64
REGIME_SCORE_THRESHOLD_CHOP=0.64
```

These are **pre-filters**, not quality gates. Raising them reduces trade volume without improving realized-R. Leave at 0.64 unless the sample shows a clear regime-specific score correlation.

### RR gates

```bash
MIN_STOP_REDESIGN_RR=1.60          # engine TP floor
MIN_EXECUTION_EFFECTIVE_RR=1.55    # executor fill-time gate
STOP_REDESIGN_RR_TOLERANCE=0.05    # engine tolerance band
```

The 0.05 spread (1.55–1.60) creates a tolerance band at execution. `MIN_EXECUTION_EFFECTIVE_RR` must always be ≤ `MIN_STOP_REDESIGN_RR`.

### Feature flags

```bash
SMC_ENABLE_4H_LIVE=true          # re-enabled after candle-fix
HARD_BLOCK_CONTINUATION=false    # removed after candle-fix
HARD_BLOCKED_TIMEFRAMES=         # empty — 4h is no longer blocked
```

**Do not re-enable `HARD_BLOCK_CONTINUATION=true`** without re-running the continuation performance analysis on post-fix data (N≥30 continuation trades).

**Do not set `HARD_BLOCKED_TIMEFRAMES=4h`** — the historical 4H underperformance was caused by look-ahead bias in HTF/regime data, which has been fixed.

### Startup validation

Every boot prints a threshold summary. Check for `[THRESHOLD WARNING]` lines:

```
[THRESHOLD SUMMARY] ──────────────────────────────────────────────────────
  UNIVERSAL_MIN_CONFIDENCE  = 0.90  ← master confidence floor
  CONFIDENCE GATES (all should match universal):
    SWING_MIN_CONFIDENCE         = 0.90  ✓
    ...
```

If any `[THRESHOLD WARNING]` appears, a gate is diverged from the unified framework. Either fix the `.env` value or document the intentional override.

---

## When to raise `UNIVERSAL_MIN_CONFIDENCE`

Raise to `0.92` when:
- N≥60 closed trades at post-candle-fix data quality
- Rolling 20-trade WR is below 40% (kill-switch triggered repeatedly)
- Analysis confirms 0.90–0.91 bucket remains net negative

**Evidence:** In the 107-trade pre-fix sample, conf≥0.92 was the first bucket with positive expectancy (+0.317R avg, 61% WR, PF=1.78). 0.90 is used during live-test phase for adequate trade flow.

---

## Kill-switch

```bash
KILL_MIN_TRADES=5       # minimum trades before kill-switch can fire
KILL_WIN_RATE=0.40      # rolling WR floor
KILL_ROLLING_N=10       # rolling window
KILL_PAUSE_HOURS=12     # pause duration when triggered
```

Per-coin kill-switch:
```bash
COIN_MIN_TRADES=5
COIN_KILL_WIN_RATE=0.35
COIN_ROLLING_N=10
COIN_PAUSE_HOURS=12
```

---

## Tests

```bash
# Candle integrity (23 tests)
pytest test_candle_integrity.py -v

# Unified threshold framework (17 tests)
pytest test_unified_thresholds.py -v

# All
pytest -v
```

All tests must pass before deploying any threshold change.

---

## Hardcoded threshold inventory (2026-06-16)

The following non-env-driven numeric values exist in the codebase. They are intentionally left unchanged — see classification.

| File | Line | Value | Classification |
|---|---|---|---|
| signal_engine.py | ~2669 | `max(0.50, ...)` | Confidence clamp floor — never a gate |
| signal_engine.py | ~3205 | `max(0.55, ...)` | Confidence clamp floor — swing path |
| signal_engine.py | ~2669 | `min(0.95, ...)` | Confidence clamp ceiling |
| executor.py | 3405 | `1.75` in `expected_remaining * 1.75` | Position-size tolerance multiplier, NOT RR |
| executor.py | 53 | `"dead_zone": 0.60` | Session weight dict — not a quality gate |
| risk_manager.py | 58–64 | CONFIDENCE_SIZING_TIERS | Position sizing multipliers — intentionally broad |
| tools/research/analyze_smc_live.py | 784–792 | `0.60, 0.70, 0.80, 0.90` | Analysis bucket boundaries — read-only reporting |
| tools/research/backtest_missed.py | 428–432 | `0.85, 0.80, 0.70, 0.90` | Analysis bucket boundaries — read-only reporting |
| tools/research/analyze_winrate.py | 569–570 | `0.70, 0.80, 0.90` | Chart bucket bins — read-only reporting |
| tools/research/param_suggester.py | 44 | `WEAK_TREND_MIN_CONFIDENCE="0.80"` | Analysis tool default — not live code path |

---

## Common ops procedures

### Restart after crash

```bash
python main.py
```

Bot auto-reconciles open positions from exchange state on startup. No manual cleanup required.

### Temporarily stop new signals without closing positions

```bash
# In .env:
PAUSE_NEW_SIGNALS=true
```

Then restart. Existing positions continue to be managed.

### Check threshold alignment without running the bot

```bash
python -c "
import os
from dotenv import load_dotenv
load_dotenv()
from agent import validate_thresholds
validate_thresholds()
"
```

### Disable a specific coin

```bash
# In .env — add the coin to HARD_BLOCKED_COINS:
HARD_BLOCKED_COINS=ARB,JUP,PYTH,NEWCOIN
```

### Check if kill-switch is active

```bash
cat strategy_filter_state.json
```

### Run analysis on live trade log

```bash
python tools/research/analyze_winrate.py
python tools/research/analyze_smc_live.py
```

---

## Do not

- Do not commit `.env` — it is gitignored. Stage only `.env.example`.
- Do not change `HL_SECRET_KEY` while the bot is running.
- Do not set `HARD_BLOCKED_TIMEFRAMES=4h` — historical basis was contaminated data.
- Do not raise score thresholds above 0.70 without a per-regime regression analysis.
- Do not lower `UNIVERSAL_MIN_CONFIDENCE` below 0.85 — below this bucket expectancy is consistently negative in historical data.
