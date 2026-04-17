# 🚀 Cuan Sniffer — Live Execution Engine for Crypto Trading

> **Not a signal bot. Not a toy backtester.**
> Cuan Sniffer is a **live-capable, capital-aware execution system** designed to convert high-quality signals into disciplined, risk-controlled trades.

---

## 🧠 Philosophy

Markets don’t pay for being early.
They don’t pay for being active.
They pay for **precision and discipline**.

Cuan Sniffer is built on three principles:

* **Only take asymmetric trades** (high RR, strong structure)
* **Protect capital above all else**
* **Execute like a machine, not a trader**

---

## ⚙️ System Overview

Cuan Sniffer is a **modular trading engine** composed of:

* Signal Engine → Generates trade ideas
* Executor → Validates & executes trades
* Risk Manager → Controls capital exposure
* Protection Layer → Ensures stop/TP integrity
* Live Monitor → Reconciles real positions

Everything flows through a **strict validation pipeline** before capital is deployed.

---

## 🧬 Execution Integrity (Core Edge)

Cuan Sniffer enforces **execution-time discipline**, not just signal-time logic.

### 🔒 Hard Guarantees

* **Execution-time RR validation**
  Trades must meet minimum RR *at fill*, not just at signal

* **TP-consumed rejection**
  No entering trades where the move already happened

* **ATR-based staleness filtering**
  Rejects signals that drift too far from original setup

* **Overextension guard**
  Avoids chasing exhausted moves

* **Fail-closed execution**
  No trade executes without valid market data

👉 Result: fewer trades, higher quality, cleaner equity curve

---

## 💰 Portfolio-Aware Execution

Cuan Sniffer doesn’t just take trades — it **allocates capital intelligently**.

### 🧠 Portfolio Logic

* Max open positions enforced
* Bucketed exposure (majors / SOL beta / alts)
* Directional caps (long vs short)
* Intraday vs swing separation

### 🔄 Position Replacement

When full:

* Weakest position can be replaced
* Stronger signals take priority

Protected positions:

* Partial profits locked
* Trades in profit
* Trades near TP

👉 Capital is always deployed to the **highest expected value opportunities**

---

## 📊 Risk Management

Risk is defined in **R (risk units)**, not emotions.

### Controls

* Fixed % risk per trade
* Daily loss limit (R-based)
* Max drawdown halt
* Confidence-weighted sizing
* Track-based risk (intraday vs swing)

### Live Capital Sync

* Balance derived from **exchange equity**
* Wallet = source of truth
* Automatic recovery after restart

👉 No drift between model and reality

---

## 🎯 Trade Lifecycle

Cuan Sniffer uses a **structured profit-taking model**.

### Default Model

* Partial close at **+1R**
* Stop moves to **breakeven**
* Remaining position runs to TP

### Optional Modes

* Full TP mode (no partials)
* Runner-based exits

👉 Wins are protected, runners capture upside

---

## 🛡️ Protection Layer

Every position is protected at the venue level.

* Native stop-loss placement
* Native take-profit placement
* Auto-repair on missing protection
* Continuous protection auditing

> The exchange is treated as the **final authority** for all positions.

---

## 🔄 Live Position Reconciliation

Cuan Sniffer is built for real-world conditions:

* Detects positions closed outside the bot
* Handles restarts without losing state
* Re-syncs with wallet automatically

👉 No phantom positions. No desync risk.

---

## ⚡ Execution Engine

Designed for **real fills, not theory**.

* Slippage-aware execution
* Partial fill handling
* Order lifecycle tracking
* Retry logic for edge cases

---

## 🧪 Modes

### PAPER MODE

* Simulated execution
* Full logic testing

### LIVE MODE

* Real capital
* Venue-integrated execution
* Wallet-authoritative state

---

## 🌐 Web3 Native Considerations

This system is built with **crypto-native realities** in mind:

* Volatility is a feature, not a bug
* Liquidity can disappear instantly
* Execution matters more than signals
* Wallet state > local state

Or simply:

> **Don’t trust your bot. Trust the chain.**

---

## 🧩 Strategy Layer (Pluggable)

Cuan Sniffer is **strategy-agnostic**.

Current signal types:

* Continuation setups
* Reversal setups
* Multi-timeframe confluence

Future-ready for:

* On-chain signals
* Liquidity flows
* AI-driven strategies

---

## 📈 Metrics That Matter

The system optimizes for:

* Expectancy (R-based)
* Drawdown control
* Capital efficiency
* Execution quality

Not vanity metrics.

---

## 🛠️ Configuration

Fully environment-driven via `.env`

Key controls:

* Risk per trade
* Position limits
* Slippage thresholds
* Session filters
* Execution tuning

👉 No hardcoded behavior. Everything is adjustable.

---

## 🚀 Deployment

Typical flow:

```bash
git pull
pip install -r requirements.txt
python main.py
```

Runs continuously as a live trading agent.

---

## ⚠️ Disclaimer

This is a **live trading system**.

* Losses will occur
* Misconfiguration can be costly
* Always test before deploying capital

---

## 🧠 Final Note

Cuan Sniffer is not trying to predict the market.

It is designed to:

> **Execute only when the odds are already in your favor — and survive when they’re not.**

---

## 🪙 Tagline

**Trade less. Execute better. Survive longer.**
