"""
Tests for features.py — RSI/ATR Wilder-smoothing fix (2026-08-25).

Locks in the actual Wilder alpha (1/period) against the previous,
mislabeled ewm(span=period) behavior (alpha=2/(period+1)) so a future
edit can't silently drift back.
"""

import unittest

import numpy as np
import pandas as pd

from features import rsi, atr, vwap_daily, vwap_session, vwap_weekly, _typical_price


def _rsi_with_alpha(series: pd.Series, alpha: float) -> pd.Series:
    """Same computation as features.rsi(), but with an explicit alpha
    instead of period -- used to derive both the correct (Wilder) and old
    (wrong, span-based) reference values for comparison."""
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=alpha, adjust=False).mean()
    roll_down = down.ewm(alpha=alpha, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-9)
    return 100.0 - (100.0 / (1.0 + rs))


class TestRsiWilderSmoothing(unittest.TestCase):

    def _mixed_series(self):
        # Deliberately irregular gains/losses so the two alpha conventions
        # produce measurably different smoothed values, not just a
        # coincidentally-matching edge case.
        return pd.Series([100.0, 103.0, 101.0, 105.0, 102.0, 108.0, 104.0, 110.0])

    def test_matches_wilder_alpha_exactly(self):
        period = 14
        s = self._mixed_series()
        got = rsi(s, period=period)
        expected = _rsi_with_alpha(s, alpha=1.0 / period)
        pd.testing.assert_series_equal(got, expected, check_names=False)

    def test_differs_measurably_from_old_span_based_behavior(self):
        # Proves the fix actually changed behavior (not a no-op refactor).
        period = 14
        s = self._mixed_series()
        got = rsi(s, period=period)
        old_wrong = _rsi_with_alpha(s, alpha=2.0 / (period + 1))
        diff = (got - old_wrong).abs().iloc[1:]  # index 0 is NaN either way
        self.assertGreater(float(diff.max()), 1.0)


def _tr(df: pd.DataFrame) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    return pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)


class TestAtrWilderSmoothing(unittest.TestCase):

    def _irregular_df(self):
        # Irregular true-range magnitudes so span vs Wilder alpha produce
        # measurably different smoothed values (a constant-TR series
        # converges to the same steady state under either convention and
        # wouldn't catch a regression).
        highs = [101, 108, 102, 112, 103, 118, 104, 122]
        lows = [99, 100, 100, 101, 101, 102, 102, 103]
        closes = [100, 104, 101, 106, 102, 110, 103, 115]
        return pd.DataFrame({"high": highs, "low": lows, "close": closes}, dtype=float)

    def test_matches_wilder_alpha_exactly(self):
        period = 14
        df = self._irregular_df()
        got = atr(df, period=period)
        expected = _tr(df).ewm(alpha=1.0 / period, adjust=False).mean()
        pd.testing.assert_series_equal(got, expected, check_names=False)

    def test_differs_measurably_from_old_span_based_behavior(self):
        period = 14
        df = self._irregular_df()
        got = atr(df, period=period)
        old_wrong = _tr(df).ewm(alpha=2.0 / (period + 1), adjust=False).mean()
        diff = (got - old_wrong).abs()
        self.assertGreater(float(diff.max()), 0.1)


class TestVwapTypicalPrice(unittest.TestCase):
    """VWAP must use typical price (H+L+C)/3, not close alone
    (2026-08-25 fix)."""

    def _df_with_wide_ranges(self, n=10):
        # Wide, asymmetric high/low ranges around each close so typical
        # price differs measurably from close alone.
        times = pd.date_range("2026-01-01T00:00:00Z", periods=n, freq="1h")
        closes = [100.0 + i for i in range(n)]
        return pd.DataFrame({
            "time": times,
            "high": [c + 5.0 for c in closes],
            "low": [c - 1.0 for c in closes],
            "close": closes,
            "volume": [1000.0] * n,
        })

    def test_typical_price_formula(self):
        df = self._df_with_wide_ranges(3)
        tp = _typical_price(df)
        expected = (df["high"] + df["low"] + df["close"]) / 3.0
        pd.testing.assert_series_equal(tp, expected, check_names=False)

    def test_vwap_session_uses_typical_price_not_close(self):
        df = self._df_with_wide_ranges()
        got = vwap_session(df)
        close_only_pv = (df["close"] * df["volume"]).cumsum()
        close_only_vwap = close_only_pv / (df["volume"].cumsum() + 1e-9)
        # Must differ from the old close-only computation given the
        # asymmetric high/low ranges built above.
        self.assertGreater(float((got - close_only_vwap).abs().max()), 0.5)

    def test_vwap_daily_and_weekly_also_fixed(self):
        df = self._df_with_wide_ranges()
        close_only_pv = (df["close"] * df["volume"]).cumsum()
        close_only_vwap = close_only_pv / (df["volume"].cumsum() + 1e-9)
        for fn in (vwap_daily, vwap_weekly):
            got = fn(df)
            self.assertGreater(float((got - close_only_vwap).abs().max()), 0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
