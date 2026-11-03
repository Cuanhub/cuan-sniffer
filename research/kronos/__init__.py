"""Kronos, the candlestick foundation model, as a research instrument.

Runs only in its own environment, research/kronos/.venv (PyTorch). That
environment has no Hyperliquid SDK and no signing libraries, so nothing run
from it can place an order. Nothing in core/ or tools/ may import this
package; test_kronos_isolation checks both.

  vendor/model   upstream model code, byte for byte (vendor/PROVENANCE.md)
  forecast.py    sampled future candles for a decision bar (needs torch)
  score.py       what a set of sampled futures says about one trade (numpy)
"""
