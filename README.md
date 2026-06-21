# 🚀 Cuan Sniffer — Institutional-Grade Crypto Execution Engine

> **Not a signal bot. Not a toy backtester.**
>
> Cuan Sniffer is a live-capable, capital-aware execution engine designed to identify asymmetric opportunities, deploy capital efficiently, and compound profitable edge through disciplined execution.

---

# 🧠 Core Philosophy

Markets do not reward activity.

Markets do not reward prediction.

Markets reward:

* Risk-adjusted decision making
* Capital efficiency
* Consistent execution
* Surviving long enough for edge to compound

Cuan Sniffer is built around four principles:

1. Find asymmetric opportunities
2. Execute with discipline
3. Protect capital during adverse conditions
4. Scale proven edge aggressively

---

# 📍 Current System Status

## Post-Candle Integrity Era

On 2026-06-16 the system underwent a full architecture audit and candle integrity rebuild.

Critical fixes included:

* Open-time vs close-time candle correction
* HTF regime contamination removal
* Macro regime contamination removal
* Daily zone contamination removal
* 4H SMC signal path rewiring
* Unified Threshold Framework implementation
* Cross-layer threshold validation
* Gate telemetry instrumentation

Historical data collected before these fixes is archived and treated as legacy research data.

All future optimization decisions should prioritize post-fix live data.

---

# 🔄 Current Development Phase

## Phase 1 — Integrity (Complete ✅)

Completed:

* Candle timestamp audit
* HTF contamination removal
* Macro contamination removal
* Daily zone contamination removal
* Threshold unification
* 4H SMC path validation

## Phase 2 — Observability (Active 🔄)

The system now records:

* Engine rejections
* Executor rejections
* Score distributions
* Trade outcomes
* Missed opportunities

Primary objective:

Determine which filters improve expectancy versus which filters only reduce trade frequency.

## Phase 3 — Profitability Optimization (Upcoming)

Future optimization decisions will be based on:

* Post-fix live data only
* Gate rejection analytics
* Walk-forward validation
* Risk-adjusted returns
* Execution quality metrics

No parameter changes should be made without supporting telemetry evidence.

---

# ⚙️ System Architecture

Cuan Sniffer is composed of multiple independent layers.

```text
Market Data
    ↓
Candle Integrity Layer
    ↓
Signal Engine
    ↓
Score Filter
    ↓
Confidence Model
    ↓
Executor Validation
    ↓
RR Validation
    ↓
Risk Manager
    ↓
Execution Engine
    ↓
Live Monitoring
```

Each layer has a single responsibility and can reject a trade candidate before capital is deployed.

---

# 🔬 Candle Integrity Layer

The candle integrity layer exists to eliminate look-ahead bias and incomplete-bar contamination.

### Guarantees

* Uses candle open time as canonical timestamp
* Tracks candle close time separately
* Removes forming candles before signal generation
* Removes incomplete HTF bars before regime calculation
* Removes incomplete macro bars before trend analysis
* Removes incomplete daily bars before zone classification

### Validation

The repository includes:

```bash
python3 test_candle_integrity.py
```

Current status:

```text
23 / 23 checks passing
```

No strategy changes should be evaluated until candle integrity remains green.

---

# 🧩 Signal Engine

The signal engine identifies trade opportunities.

Current strategy families:

## Continuation

Trend-following entries aligned with higher timeframe structure.

## Reversal

Liquidity-driven reversals following exhaustion or sweep conditions.

## Multi-Timeframe Confluence

Setups requiring alignment across:

* Structure
* Regime
* Liquidity
* Risk/Reward

## 4H SMC Path

Institutional-style Smart Money Concepts framework.

Requires:

* Order block interaction
* Liquidity sweep or displacement
* HTF alignment
* Macro alignment
* Minimum confluence score

---

# 🎯 Signal Quality Pipeline

Every signal passes through a strict validation process.

```text
Setup Detected
      ↓
Structure Valid
      ↓
Score Threshold
      ↓
Confidence Model
      ↓
RR Validation
      ↓
Executor Validation
      ↓
Risk Validation
      ↓
Trade Opened
```

Signals can be rejected at any stage.

Rejection is treated as valuable information rather than failure.

---

# 📊 Unified Threshold Framework

Implemented after the candle integrity rebuild.

The objective is to eliminate hidden threshold divergence between components.

## Master Confidence Floor

```env
UNIVERSAL_MIN_CONFIDENCE=0.90
```

Single source of truth for all confidence gates.

Inherited by:

```env
MIN_SIGNAL_CONFIDENCE
SWING_MIN_CONFIDENCE
SMC_4H_MIN_CONFIDENCE
WEAK_TREND_MIN_CONFIDENCE
CHOP_REVERSAL_MIN_CONFIDENCE
```

## Score Thresholds

```env
REGIME_SCORE_THRESHOLD_STRONG=0.64
REGIME_SCORE_THRESHOLD_WEAK=0.64
REGIME_SCORE_THRESHOLD_CHOP=0.64
```

Purpose:

Remove structurally poor setups before confidence evaluation.

Score is intentionally permissive.

Confidence is the primary quality gate.

## Risk / Reward Floors

Signal generation:

```env
MIN_STOP_REDESIGN_RR=1.60
```

Execution:

```env
MIN_EXECUTION_EFFECTIVE_RR=1.55
```

This allows minor fill drift while preserving expectancy.

---

# 🏛 4H Institutional SMC Framework

4H SMC remains enabled.

```env
SMC_ENABLE_4H_LIVE=true
```

The system intentionally favors quality over frequency on the 4H path.

Requirements:

* Order Block interaction
* Liquidity sweep
* HTF confirmation
* Macro confirmation
* Confidence threshold
* RR threshold

The previous 0% win-rate observations occurred during the pre-fix candle era and are no longer considered valid evidence.

---

# 💰 Portfolio-Aware Execution

Capital is allocated at the portfolio level.

The system evaluates:

* Open positions
* Directional exposure
* Sector exposure
* Available risk budget

### Controls

* Maximum open positions
* Directional caps
* Bucket exposure controls
* Dynamic replacement logic

When capital is fully allocated:

The weakest position may be replaced by a stronger opportunity.

---

# 📈 Risk Management

Risk is measured in R.

Not dollars.

Not emotions.

### Controls

* Fixed risk per trade
* Daily loss limits
* Drawdown limits
* Confidence-weighted sizing
* Position concentration controls

### Live Capital Sync

Exchange equity is the source of truth.

The system automatically reconciles:

* Position state
* Wallet state
* Open orders
* Available margin

---

# 🎯 Trade Lifecycle

Default trade management:

```text
Entry
 ↓
+1R Partial
 ↓
Stop to Breakeven
 ↓
Runner Management
 ↓
Final Exit
```

Objectives:

* Protect capital
* Lock gains
* Allow outlier winners

---

# 🛡 Protection Layer

Every position is protected at the exchange level.

### Guarantees

* Native stop-loss
* Native take-profit
* Protection verification
* Missing-order repair
* Continuous auditing

The exchange is treated as the final authority.

---

# ⚡ Execution Engine

Designed for real fills.

Not backtest fills.

### Features

* Slippage-aware execution
* Partial-fill handling
* Order lifecycle management
* Retry logic
* Fill verification
* Position reconciliation

---

# 🔄 Live Monitoring

The system continuously validates:

* Position state
* Open orders
* Risk state
* Equity state
* Protection integrity

Supports:

* Restarts
* Exchange disconnects
* State recovery

---

# 📡 Observability Layer

Optimization is evidence-driven.

Not opinion-driven.

The system tracks:

## Trade Data

```text
trades.csv
orders.csv
```

## Signal Data

```text
signals.csv
missed_signals.csv
```

## Engine Telemetry

```text
gate_rejects.csv
score_distribution.csv
```

## Shadow Research Ledger

```text
shadow_research_candidates.csv
shadow_research_executions.csv
shadow_research_outcomes.csv
```

The shadow research ledger is append-only and joinable by `shadow_id`:
candidates preserve engine-time thesis and planned geometry, executions preserve
executor decisions and final geometry, and outcomes label forward candle MFE,
MAE, close R, TP touch, and stop touch across configured horizons.

## Execution Telemetry

```text
executor_rejects.csv
```

These datasets allow the team to answer:

* Why did a trade execute?
* Why did a trade not execute?
* Which gate rejects most opportunities?
* Which filters improve expectancy?
* Which filters only reduce volume?

---

# 🔍 Gate Analytics

The system includes rejection-path telemetry and audit tooling.

Run:

```bash
python tools/research/audit_gate_rejections.py
```

Available modes:

```bash
python tools/research/audit_gate_rejections.py
python tools/research/audit_gate_rejections.py --hours 24
python tools/research/audit_gate_rejections.py --gate-only
python tools/research/audit_gate_rejections.py --executor-only
```

The objective is to identify:

* Dominant rejection paths
* Confidence bottlenecks
* Score bottlenecks
* RR bottlenecks
* Session bottlenecks
* Structural bottlenecks

Future threshold changes should be justified by telemetry rather than intuition.

---

# 🔬 Research Methodology

The system is developed using evidence-first research.

Evaluation sources:

* Live trades
* Missed trades
* Gate rejection analysis
* Walk-forward validation
* Expectancy analysis
* Risk-adjusted returns

The objective is not finding the best historical parameter.

The objective is finding robust parameter zones that survive future market conditions.

---

# 📊 Primary Success Metrics

The system optimizes for:

## Expectancy

Average R per trade.

## Profit Factor

Gross wins divided by gross losses.

## Drawdown

Maximum capital decline.

## Capital Efficiency

Return per unit of deployed risk.

## Execution Quality

How closely fills match modeled assumptions.

---

# 🧪 Operating Mode

## LIVE MODE

The system only supports live-backend execution. Use `HL_TESTNET=true` for
exchange testnet validation; paper execution is not supported.

---

# ⚙️ Configuration

All behavior is controlled via:

```env
.env
```

Key categories:

* Risk controls
* Position limits
* Confidence thresholds
* RR thresholds
* Session logic
* Execution controls
* SMC controls

See:

```text
.env.example
RUNBOOK.md
```

for complete reference.

---

# 🚀 Deployment

Typical startup sequence:

```bash
git pull

pip install -r requirements.txt

python3 test_candle_integrity.py

python3 test_unified_thresholds.py

python main.py
```

Recommended workflow:

1. Pull latest code
2. Run integrity tests
3. Run threshold tests
4. Verify configuration
5. Start engine

---

# 🏷 Current Baseline

Current research baseline:

```text
telemetry-baseline-v1
```

This tag represents:

* Candle integrity fixed
* HTF contamination removed
* Macro contamination removed
* Daily zone contamination removed
* Threshold framework unified
* 4H SMC enabled
* Telemetry instrumentation installed

Future research should compare against this baseline.

---

# ⚠️ Disclaimer

This is a live trading system.

* Losses will occur
* Markets evolve
* Edge decays
* Misconfiguration can be costly

Risk management exists because uncertainty is permanent.

---

# 🧠 Final Note

Cuan Sniffer is not attempting to predict the future.

Its purpose is to identify favorable risk/reward situations, deploy capital efficiently, and compound edge while preserving survivability.

The goal is simple:

> Find edge. Deploy capital. Compound intelligently.

---

# 🪙 Tagline

**Capture asymmetric moves. Scale winning systems. Protect capital while compounding.**
