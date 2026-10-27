"""Walk-forward backtesting on real Hyperliquid candles.

Read-only by construction, like research/levels/: nothing here can place,
modify or cancel an order, and test_backtest_isolation.py enforces it. The
only network access is the public `info` endpoint, through
research.levels.venue_client.
"""
