🐶💰 Cuan Sniffer
A data-driven crypto trading system designed for asymmetric returns
Overview

Cuan Sniffer is a real-time, multi-asset trading system that identifies and executes high-probability crypto trades using a combination of:

Smart Money Concepts (SMC)
On-chain whale flow (Solana)
Perpetual market sentiment (Hyperliquid)
Multi-timeframe regime filters

Unlike typical signal bots, Cuan Sniffer is not just predictive — it is executable.

It includes a full paper trading engine with persistent balance, risk management, and performance tracking, allowing real-world validation of strategy edge.

Core Philosophy

Low win rate. High expectancy. Strict selectivity.

Signals only fire when multi-layer confluence aligns
Minimum ~2R risk/reward enforced
Strategy is designed to be profitable below 50% win rate
Filters and thresholds are derived from live data, not assumptions
System Architecture

Cuan Sniffer operates as a modular, real-time pipeline:

Market Data → Feature Engine → Signal Engine → Execution → Performance Tracking
Components
Signal Layer
signal_engine.py — Core scoring engine (continuation + reversal setups)
features.py — TA indicators (EMA, ATR, VWAP, RSI)
smc_structure.py — Market structure (BOS, CHoCH)
smc_zones.py — Order blocks, fair value gaps
smc_sweeps.py — Liquidity sweep detection
Data Layer
perp_data.py — Hyperliquid OHLCV feed (15m execution)
perp_sentiment.py — Funding rates, open interest
flow_context.py — On-chain whale tracking (SOL only)
Execution Layer (NEW)
executor.py — Signal → trade execution
paper_trader.py — Account balance + portfolio state
position.py — Trade lifecycle management
risk_manager.py — Position sizing + risk constraints
strategy_filter.py — Kill-switch + per-coin performance gating
Persistence & Recovery
trade_log.py — Trade history (trades.csv)
bootstrap.py — Reconstructs:
Balance
Open positions
Strategy state on restart
Orchestration
agent.py — Multi-coin scanning loop + Telegram alerts
alerts.py — Flow + funding anomaly alerts
Analytics
analyze_winrate.py — Signal-level performance
trades_recap.py — Trade-level PnL tracking (daily + all-time)
Signal Engine

Cuan Sniffer evaluates two setup families:

1. Continuation
Break of structure (BOS)
Order block / FVG alignment
Trend + HTF confirmation
2. Reversal
Liquidity sweep
CHoCH (structure flip)
VWAP deviation + flow confirmation
Scoring System

Signals are scored across multiple dimensions:

Market structure (BOS / CHoCH)
Trend alignment (1h + 4h EMA slope)
VWAP positioning
Volume & volatility context
On-chain whale pressure (SOL)
Funding rate (non-linear)
Open interest delta
RSI (context-aware)

Only signals above a strict threshold proceed.

Data-Driven Filters

Built from live performance analysis:

❌ Continuation in HTF chop → removed (negative expectancy)
❌ Reversal in HTF chop + macro down → removed (0% win rate)
✅ Session weighting (London/NY favored)
✅ Volatility filters (low-vol regimes blocked)
Execution Engine (Paper Trading)

Cuan Sniffer includes a fully integrated execution system:

Features
💰 Persistent account balance (no reset on restart)
📊 Real-time PnL tracking
⚖️ Dynamic position sizing (based on signal confidence)
🚫 Strategy kill-switch (auto-disable underperforming setups)
🔄 State recovery (positions + balance restored on boot)
Performance
Signal-Level Performance

Measured via real market replay (not backtests):

R-multiples per trade
Sharpe / Sortino ratios
Regime-level breakdowns
Walk-forward validation
Autocorrelation (signal independence)
Trade-Level Performance (NEW)
Real executed trades
Stored in trades.csv
Tracks:
Balance growth
Equity curve
Drawdowns
Win rate
Daily Recap

Automated Telegram recap includes:

Signal performance summary
Equity chart
📈 Trade PnL:
Today’s PnL
All-time PnL
Regime insights
Example Signal
🟢 SOL LONG SIGNAL 📈

🎯 Setup: Liquidity Sweep
🧭 Regime: reversal | htf up | macro up
⏱ TF: 15m

💰 Entry: 87.42
🛑 SL: 85.91
🎯 TP: 91.05
📊 R/R: 2.10R

⚡ Strength: 88%
📈 Funding: Shorts crowded
📦 OI: 48,291,033
🐋 Flow: Buyers stepping in
Setup
Requirements
Python 3.13+
Hyperliquid API access
Solana RPC (Alchemy recommended)
Install
git clone https://github.com/YOUR_USERNAME/cuan-sniffer.git
cd cuan-sniffer

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
Configure
cp .env.template .env

Set:

Telegram bot + chat ID
RPC URL
Coins to track
Run
# Terminal 1 — optional (on-chain tracker)
python main.py

# Terminal 2 — trading system
python agent.py
Analysis
# Evaluate signal performance
python analyze_winrate.py

# With slippage + walk-forward
python analyze_winrate.py --slippage-bps 20 --walk-forward --autocorr

# Trade PnL recap
python trades_recap.py
Roadmap
 Real-time multi-coin signal engine
 On-chain + perp data fusion
 Execution layer (paper trading)
 Persistent portfolio state
 Strategy-level risk controls
 Live execution (Hyperliquid integration)
 Portfolio-level optimization
 ML-based signal refinement
Positioning

Cuan Sniffer is designed as:

A quant-informed discretionary system, bridging raw market structure with systematic execution.

Not a black box. Not pure TA. Not pure quant.

A hybrid system focused on:

Explainability
Robustness
Real-world performance validation
Disclaimer

⚠️ High-risk leveraged trading system.
Not financial advice. For research and development purposes only.