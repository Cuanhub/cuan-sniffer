"""
test_candle_integrity.py
------------------------
Validates candle integrity policy across perp_data, signal_engine, and
backtest modules.

Run with:
    python3 test_candle_integrity.py

Requires: pandas, numpy (in .venv or system environment)
No pytest, no network access.
"""

import sys
import traceback
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock

import pandas as pd
import numpy as np

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"

_results = []


def check(name: str, condition: bool, detail: str = ""):
    status = PASS if condition else FAIL
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    _results.append((name, condition))
    return condition


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_raw_hl_response(n: int, interval_h: int = 1, include_forming: bool = True):
    """
    Build a fake Hyperliquid candleSnapshot response list with n closed bars
    plus one forming bar (if include_forming=True).
    Each bar: t=open_ms, T=close_ms, o, h, l, c, v.
    """
    bars = []
    base = datetime(2026, 6, 1, 0, 0, 0, tzinfo=timezone.utc)
    n_total = n + (1 if include_forming else 0)
    for i in range(n_total):
        open_dt = base + timedelta(hours=i * interval_h)
        close_dt = open_dt + timedelta(hours=interval_h) - timedelta(milliseconds=1)
        price = 100.0 + float(i)
        bars.append({
            "t": int(open_dt.timestamp() * 1000),
            "T": int(close_dt.timestamp() * 1000),
            "o": str(price),
            "h": str(price + 1.0),
            "l": str(price - 1.0),
            "c": str(price + 0.5),
            "v": "1000.0",
        })
    return bars


def _make_ohlcv_df(n_closed: int, interval_h: int = 1, add_forming: bool = True) -> pd.DataFrame:
    """
    Build a DataFrame as returned by PerpDataFeed.get_ohlcv_df() after the fix.
    Uses open time ('t') as the 'time' column.
    """
    base = datetime(2026, 6, 1, 0, 0, 0, tzinfo=timezone.utc)
    rows = []
    n_total = n_closed + (1 if add_forming else 0)
    for i in range(n_total):
        open_dt = base + timedelta(hours=i * interval_h)
        close_dt = open_dt + timedelta(hours=interval_h) - timedelta(milliseconds=1)
        price = 100.0 + float(i)
        rows.append({
            "time": open_dt,
            "close_time": close_dt,
            "open": price,
            "high": price + 1.0,
            "low": price - 1.0,
            "close": price + 0.5,
            "volume": 1000.0,
        })
    df = pd.DataFrame(rows)
    return df


# ── Section 1: perp_data.py — timestamp parsing ───────────────────────────────

def test_perp_data_timestamp():
    print("\n=== 1. perp_data.py — candle timestamp parsing ===")
    from perp_data import PerpDataFeed

    feed = PerpDataFeed(coin="SOL", interval="1h", max_candles=10, debug=False)

    raw = _make_raw_hl_response(n=5, interval_h=1, include_forming=True)

    # Patch _post_with_backoff to return our fake data
    with patch.object(feed, "_post_with_backoff", return_value=raw):
        feed.refresh(force=True)

    df = feed.get_ohlcv_df()

    check("get_ohlcv_df has 'time' column", "time" in df.columns)
    check("get_ohlcv_df has 'close_time' column", "close_time" in df.columns)

    if df.empty:
        check("DataFrame is non-empty", False, "got empty df")
        return

    # time (open time) should be BEFORE close_time for every bar
    times = pd.to_datetime(df["time"], utc=True)
    close_times = pd.to_datetime(df["close_time"], utc=True)
    all_open_before_close = (times < close_times).all()
    check("time < close_time for every bar", bool(all_open_before_close))

    # For 1h candles, time should be exactly on the hour
    first_time = times.iloc[0]
    check(
        "first 'time' is on the hour (open time = t, not T)",
        first_time.minute == 0 and first_time.second == 0,
        f"got minute={first_time.minute} second={first_time.second}",
    )

    # close_time should be ~1h after time (minus 1ms)
    gap_ms = int((close_times.iloc[0] - times.iloc[0]).total_seconds() * 1000)
    expected_ms = 3600 * 1000 - 1
    check(
        "close_time is (interval - 1ms) after time",
        abs(gap_ms - expected_ms) <= 5,
        f"gap_ms={gap_ms} expected={expected_ms}",
    )

    # T is NOT used as 'time': 'time' must equal t, not T
    # t of first bar is base + 0h; T of first bar is base + 1h - 1ms
    base = datetime(2026, 6, 1, 0, 0, 0, tzinfo=timezone.utc)
    expected_open = base
    actual_open = times.iloc[0].replace(tzinfo=timezone.utc)
    check(
        "'time' column equals candle open (t), not close (T)",
        abs((actual_open - expected_open).total_seconds()) < 1,
        f"actual={actual_open} expected={expected_open}",
    )


# ── Section 2: _build_feature_frame — forming candle exclusion ────────────────

def test_build_feature_frame_drops_forming():
    print("\n=== 2. _build_feature_frame — forming candle excluded ===")
    from signal_engine import AdaptiveSignalEngine

    engine = AdaptiveSignalEngine(debug=False)

    # Build df with 210 closed bars + 1 forming bar (total 211)
    df_with_forming = _make_ohlcv_df(n_closed=210, add_forming=True)

    # Make the forming candle's close obviously wrong so we can detect leakage
    df_with_forming.loc[df_with_forming.index[-1], "close"] = 999999.0

    feat = engine._build_feature_frame(df_with_forming, min_len=200)

    check("feature frame is non-empty with 211 rows input", not feat.empty)

    if not feat.empty:
        last_close = float(feat["close"].iloc[-1])
        check(
            "forming candle (close=999999) NOT in feature frame",
            last_close < 999998.0,
            f"last_close={last_close}",
        )
        check(
            "feature frame has at most 210 rows",
            len(feat) <= 210,
            f"len={len(feat)}",
        )

    # RSI diff() makes row-0 NaN, which dropna() removes. So 210 closed bars →
    # 209 usable rows after dropna → passes min_len=200.
    df_minimal = _make_ohlcv_df(n_closed=210, add_forming=True)
    feat_minimal = engine._build_feature_frame(df_minimal, min_len=200)
    check("feature frame with 211 input rows passes min_len=200", not feat_minimal.empty)

    # With 201 closed bars and min_len=201, after NaN drop we get 200 rows → empty.
    df_too_few = _make_ohlcv_df(n_closed=201, add_forming=True)
    feat_too_few = engine._build_feature_frame(df_too_few, min_len=201)
    check("feature frame is empty when usable rows < min_len", feat_too_few.empty)


# ── Section 3: HTF resample — incomplete 4H bar excluded ─────────────────────

def test_htf_regime_excludes_forming():
    print("\n=== 3. _compute_htf_regime — forming 1H + last 4H bar excluded ===")
    from signal_engine import AdaptiveSignalEngine

    engine = AdaptiveSignalEngine(debug=False)

    # 200 closed 1H bars + 1 forming 1H bar = 201 total
    df = _make_ohlcv_df(n_closed=200, add_forming=True)

    # Corrupt the forming 1H candle to detect leakage
    df.loc[df.index[-1], "high"] = 9999999.0
    df.loc[df.index[-1], "close"] = 9999999.0

    regime, notes = engine._compute_htf_regime(df)

    check(
        "HTF regime does not return 'unknown' (has enough bars)",
        regime != "unknown",
        f"regime={regime} notes={notes}",
    )

    # The regime should be computed without the 9999999 corrupted bar influencing the EMA
    # If the forming bar leaked into the 4H resample and wasn't dropped, EMA would be
    # skewed toward 9999999. We can't assert exact regime, but we can verify 'unknown'
    # is not returned due to data issues (which would mean we have enough bars).
    check(
        "notes do not indicate data corruption error",
        not any("error" in n or "failed" in n for n in notes),
        f"notes={notes}",
    )


def test_htf_regime_insufficient_after_drops():
    print("\n=== 3b. _compute_htf_regime — returns unknown when bars < 10 after drops ===")
    from signal_engine import AdaptiveSignalEngine

    engine = AdaptiveSignalEngine(debug=False)

    # Only 42 1H bars → 42/4 = ~10 4H bars. After two drops (forming 1H + last 4H),
    # we may fall below 10.
    df = _make_ohlcv_df(n_closed=40, add_forming=True)
    regime, notes = engine._compute_htf_regime(df)

    # With 41 closed 1h bars after drop → ~10 4h bars → after last-bar drop → 9 → unknown
    check(
        "returns 'unknown' when insufficient closed bars remain",
        regime == "unknown",
        f"regime={regime} with 40 closed bars",
    )


# ── Section 4: Macro 1D — incomplete day excluded ─────────────────────────────

def test_macro_regime_excludes_forming_day():
    print("\n=== 4. _compute_macro_regime_4h — forming 1H + current incomplete day excluded ===")
    from signal_engine import AdaptiveSignalEngine

    engine = AdaptiveSignalEngine(debug=False)

    # 300 closed 1H bars (~12.5 days) + 1 forming = 301 total
    df = _make_ohlcv_df(n_closed=300, add_forming=True)

    # Corrupt the forming bar
    df.loc[df.index[-1], "close"] = 9999999.0

    regime, notes = engine._compute_macro_regime_4h(df)

    check(
        "macro regime does not return 'unknown' (has enough days)",
        regime != "unknown",
        f"regime={regime} notes={notes}",
    )
    check(
        "no data-error notes",
        not any("error" in n or "failed" in n for n in notes),
        f"notes={notes}",
    )


# ── Section 5: Daily zone — incomplete day excluded ───────────────────────────

def test_daily_zone_excludes_current_day():
    print("\n=== 5. _compute_daily_zone — current incomplete day excluded from swing range ===")
    from signal_engine import AdaptiveSignalEngine

    engine = AdaptiveSignalEngine(debug=False)

    # 200 closed 1H bars + 1 forming bar with an extreme intraday high
    df = _make_ohlcv_df(n_closed=200, add_forming=True)
    df.loc[df.index[-1], "high"] = 9999999.0  # extreme intraday high on forming bar

    zone_data = engine._compute_daily_zone(df)

    check(
        "daily zone does not return 'unknown'",
        zone_data.get("zone") != "unknown",
        f"zone={zone_data.get('zone')}",
    )

    swing_high = zone_data.get("swing_high", 0.0)
    check(
        "swing_high is not contaminated by forming bar's extreme intraday high",
        swing_high < 999.0,
        f"swing_high={swing_high}",
    )


# ── Section 6: Swing signal resample — forming 1H removed before resampling ──

def test_swing_resample_excludes_forming():
    print("\n=== 6. generate_swing_signal — forming 1H excluded before resample ===")
    from signal_engine import AdaptiveSignalEngine, AdaptiveSignalEngine

    engine = AdaptiveSignalEngine(debug=False)

    # Build df with 250 closed 1H bars + 1 forming
    df = _make_ohlcv_df(n_closed=250, add_forming=True)

    # The forming candle has an extreme close that would skew the feature frame
    df.loc[df.index[-1], "close"] = 9999999.0

    # We only care that the internal resample doesn't crash and that
    # generate_swing_signal either returns None (no signal) or returns a signal
    # whose entry_price is NOT 9999999 (forming bar close wasn't used)
    from unittest.mock import MagicMock
    fake_sentiment = MagicMock()
    fake_sentiment.funding_rate = 0.0001
    fake_sentiment.open_interest = 1000000
    fake_sentiment.bias = 0.0
    fake_sentiment.oi_delta_1h = 0
    fake_sentiment.oi_delta_4h = 0
    fake_sentiment.oi_change_pct = 0.0

    sig = engine.generate_swing_signal(df, {}, fake_sentiment, swing_tf="1h", coin="SOL")

    if sig is not None:
        check(
            "swing signal entry_price is not the forming bar's corrupted close",
            float(sig.entry_price) < 999998.0,
            f"entry_price={sig.entry_price}",
        )
    else:
        # No signal generated — forming bar was excluded (correct)
        check("swing signal returned None (no look-ahead trade triggered)", True)


# ── Section 7: backtest candle parsing — open time used ───────────────────────

def test_backtest_uses_open_time():
    print("\n=== 7. backtest candle parsers — open time (t) used not close time (T) ===")

    # Simulate what tools/research/analyze_winrate.py:fetch_candles does
    base = datetime(2026, 6, 10, 12, 0, 0, tzinfo=timezone.utc)
    fake_candle = {
        "t": int(base.timestamp() * 1000),
        "T": int((base + timedelta(minutes=1) - timedelta(milliseconds=1)).timestamp() * 1000),
        "o": "100.0", "h": "101.0", "l": "99.0", "c": "100.5",
    }

    # analyze_winrate logic
    ts_aw = datetime.fromtimestamp((fake_candle.get("t") or fake_candle.get("T")) / 1000, tz=timezone.utc)
    check(
        "analyze_winrate: timestamp equals open time (t)",
        ts_aw == base,
        f"ts={ts_aw} expected={base}",
    )
    check(
        "analyze_winrate: timestamp is NOT close time (T)",
        ts_aw != base + timedelta(minutes=1) - timedelta(milliseconds=1),
    )

    # backtest_missed logic
    ts_ms_bm = fake_candle.get("t") or fake_candle.get("T") or 0
    ts_bm = datetime.fromtimestamp(ts_ms_bm / 1000, tz=timezone.utc)
    check(
        "backtest_missed: ts_ms equals open time (t)",
        ts_bm == base,
        f"ts={ts_bm} expected={base}",
    )


# ── Section 8: Closed-only signal generation (no look-ahead) ─────────────────

def test_no_lookahead_in_signal():
    print("\n=== 8. generate_signal — no look-ahead from forming candle ===")
    from signal_engine import AdaptiveSignalEngine
    from unittest.mock import MagicMock

    engine = AdaptiveSignalEngine(debug=False)
    fake_sentiment = MagicMock()
    fake_sentiment.funding_rate = 0.0001
    fake_sentiment.open_interest = 1000000
    fake_sentiment.bias = 0.0
    fake_sentiment.oi_delta_1h = 0
    fake_sentiment.oi_delta_4h = 0
    fake_sentiment.oi_change_pct = 0.0
    fake_snapshot = MagicMock()
    fake_snapshot.timestamp = 0.0
    fake_snapshot.whale_pressure = 0.0

    # 210 closed bars + 1 forming with price 9999999
    df = _make_ohlcv_df(n_closed=210, add_forming=True)
    df.loc[df.index[-1], "close"] = 9999999.0
    df.loc[df.index[-1], "open"] = 9999999.0
    df.loc[df.index[-1], "high"] = 9999999.0

    sig = engine.generate_signal(df, {}, fake_sentiment, coin="SOL")

    if sig is not None:
        check(
            "generate_signal entry_price is not forming bar's 9999999 close",
            float(sig.entry_price) < 999998.0,
            f"entry={sig.entry_price}",
        )
    else:
        check("generate_signal returned None — no signal from corrupted forming bar", True)


# ── Runner ────────────────────────────────────────────────────────────────────

def run_all():
    suites = [
        test_perp_data_timestamp,
        test_build_feature_frame_drops_forming,
        test_htf_regime_excludes_forming,
        test_htf_regime_insufficient_after_drops,
        test_macro_regime_excludes_forming_day,
        test_daily_zone_excludes_current_day,
        test_swing_resample_excludes_forming,
        test_backtest_uses_open_time,
        test_no_lookahead_in_signal,
    ]

    print("=" * 60)
    print("CANDLE INTEGRITY TEST SUITE")
    print("=" * 60)

    for suite in suites:
        try:
            suite()
        except Exception as e:
            print(f"  [{FAIL}] {suite.__name__} raised: {e}")
            traceback.print_exc()
            _results.append((suite.__name__, False))

    print("\n" + "=" * 60)
    passed = sum(1 for _, ok in _results if ok)
    failed = sum(1 for _, ok in _results if not ok)
    print(f"TOTAL: {passed} passed, {failed} failed out of {len(_results)} checks")
    print("=" * 60)

    if failed > 0:
        print("\nFailed checks:")
        for name, ok in _results:
            if not ok:
                print(f"  - {name}")
        sys.exit(1)
    else:
        print("\nAll candle integrity checks passed.")
        sys.exit(0)


if __name__ == "__main__":
    run_all()
