<<<<<<< HEAD
# 🐶💰 Cuan Sniffer

**Sniffing out high-probability trades before the market moves.**

Cuan Sniffer is a real-time crypto signal engine that combines Solana on-chain whale tracking with Hyperliquid perpetual market data to identify asymmetric trade setups. Signals are delivered via Telegram with entry, stop, TP, R/R, regime, and flow context.

---

## Overview

Cuan Sniffer combines four data layers into a single scored signal:

- 🧠 **Smart Money Concepts** — BOS, CHoCH, order blocks, FVGs, liquidity sweeps
- 🐋 **On-chain whale flow** — tracked Solana wallets, inflow/outflow pressure
- 📊 **Perp sentiment** — Hyperliquid funding rates, open interest, long/short bias
- ⚡ **Multi-timeframe regime** — 1h and 4h EMA trend alignment as context filters

Signals only fire when all layers align. No confluence → no signal.

---

## Live Performance

> 154 signals evaluated against real 1-minute Hyperliquid candles — not backtested simulations.  
> Data collected: March 20–23, 2026.

| Metric | Value |
|---|---|
| Win rate | 46.7% |
| Total R | +79.06R |
| Mean R per signal | +0.513R |
| Avg planned R/R | 2.18R |
| Signal Sharpe | 0.35 |
| Signal Sortino | 2.05 |
| Max drawdown | -7.10R |

**By coin:**

| Coin | Win rate | Mean R | Signals |
|---|---|---|---|
| PYTH | 85.7% | +1.567R | 24 |
| ARB | 81.8% | +1.201R | 15 |
| JUP | 61.5% | +0.882R | 17 |
| BTC | 50.0% | +0.657R | 13 |
| JTO | 44.0% | +0.429R | 29 |

The system is designed for positive expectancy at sub-50% win rates through consistent 2R+ setups. Thresholds and regime filters are adjusted from live performance data, not assumptions — e.g. continuation signals in HTF=chop were removed after data showed they comprised 55% of volume with negative R.

---

## Example Signal

```
🟢 JUP LONG SIGNAL 📈

🎯 Setup: Breakout
🧭 Regime: continuation | htf up | macro down
⏱ TF: 15m

💰 Entry: 0.5842
🛑 SL: 0.5710
🎯 TP: 0.6108
📊 R/R: 2.02R

⚡ Strength: 90%
📈 Funding: Shorts crowded
📦 OI: 48291033

🧠 Breakout detected with confluence
```

---

## Signal Logic

`AdaptiveSignalEngine` scores signals across two setup families:

**Continuation** — BOS/OB/FVG triggers with aligned HTF and macro trend  
**Reversal** — sweep/CHoCH below VWAP with on-chain flow confluence

Scoring components:
- Base trigger (BOS, CHoCH, order block, FVG, liquidity sweep)
- HTF regime alignment (1h EMA10/EMA30 slope)
- Macro regime alignment (4h EMA10/EMA30 slope)
- On-chain whale pressure and 30m net flow imbalance
- Funding rate direction bias

Data-driven filters block signals in unfavourable conditions:
- Continuation in HTF=chop — removed after data showed it was 55% of signals with negative R
- Reversals in HTF=chop + macro=down — blocked after 0% win rate across 12 live trades
- Score threshold of 0.80 — raised from 0.55 after data showed 96% of signals survived at 0.70

Only signals above threshold proceed to ATR-based level construction. Minimum 2R enforced.

---

## Architecture

Two processes share a SQLite database:

```
main.py     — Solana wallet tracker (Alchemy RPC → SQLite)
agent.py    — Signal engine + Telegram dispatcher (Hyperliquid API)
```

SOL signals receive full on-chain context. All other coins use perp sentiment only.

```
agent.py              — Multi-coin signal loop, per-coin state, Telegram dispatch
signal_engine.py      — AdaptiveSignalEngine (SMC scoring, regime filtering)
main.py               — Wallet flow tracker entry point
engine.py             — Per-wallet scanning, throttle, DB writes
db.py                 — SQLAlchemy models (WalletBalance, FlowEvent)
config.py             — Env var loading, live SOL price fetch
notifier.py           — Telegram send with Markdown/plain fallback
alerts.py             — Large flow + funding extreme context alerts
signal_log.py         — CSV signal logger
analyze_winrate.py    — Signal evaluator: R-multiples, Sharpe, regime charts
daily_recap.py        — Telegram daily performance summary
perp_data.py          — Threaded Hyperliquid OHLCV feed
perp_sentiment.py     — Threaded Hyperliquid funding/OI feed
flow_context.py       — On-chain flow snapshot from SQLite
sol_client.py         — Solana RPC wrapper (signatures, balance)
features.py           — Technical indicators (ATR, VWAP, EMAs)
smc_structure.py      — Swing detection, BOS/CHoCH labeling
smc_zones.py          — Order blocks, FVGs
smc_sweeps.py         — Liquidity sweep detection
```

---

## Setup

**Requirements:** Python 3.13+, uv

```bash
git clone https://github.com/YOUR_USERNAME/cuan-sniffer.git
cd cuan-sniffer

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Create venv and install dependencies
uv venv --seed
source .venv/bin/activate
uv pip install -r requirements.txt

# Configure
cp .env.template .env
# Edit .env — add Alchemy RPC key, Telegram token, wallet addresses, coins
```

## Run

```bash
# Terminal 1 — on-chain wallet tracker
python main.py

# Terminal 2 — signal engine + Telegram alerts
python agent.py
```

## Analysis

```bash
# Evaluate signals against real candle data + generate dashboard chart
python analyze_winrate.py

# Filter by regime
python analyze_winrate.py --htf down --macro down
python analyze_winrate.py --coin PYTH --side SHORT
python analyze_winrate.py --no-fetch        # analyze from cache only

# Send daily recap to Telegram
python daily_recap.py
python daily_recap.py --hours 12
python daily_recap.py --dry-run             # preview without sending
```

---

## Configuration

Copy `.env.template` to `.env`:

| Variable | Description |
|---|---|
| `RPC_URL` | Alchemy Solana mainnet RPC endpoint |
| `TELEGRAM_BOT_TOKEN` | From @BotFather |
| `TELEGRAM_CHAT_ID` | From @userinfobot |
| `TRACKED_COINS` | Coins to scan — adjust to your preference |
| `TRACKED_WALLETS` | Comma-separated Solana addresses to monitor |
| `MIN_SOL_ALERT` | Minimum SOL movement to log as whale flow event |
| `POLL_INTERVAL_SECONDS` | Scan frequency in seconds (default: 15) |

---

## Requirements

```
requests
python-dotenv
SQLAlchemy
pandas
numpy
matplotlib
```

---

## Roadmap

- [x] Real-time Telegram signal alerts
- [x] Multi-coin scanning
- [x] On-chain whale flow integration (SOL)
- [x] Live performance attribution by regime, coin, session
- [x] Data-driven threshold and filter tuning from live signals
- [ ] Automated execution via Hyperliquid SDK
- [ ] Per-coin adaptive thresholds based on rolling win rate

---

> ⚠️ Built for high-risk leveraged markets. Not financial advice.
=======
# cuan-sniffer
>>>>>>> 19f52c9bb6432d101c636cb961ccf0c901f42f26
