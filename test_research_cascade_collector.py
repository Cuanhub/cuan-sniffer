"""
Functional tests for the cascade-fade forward collector -- detection,
resolution, and storage idempotency. No real network calls: candles are
synthetic so the suite stays deterministic and offline.
"""

import csv
import shutil
import unittest
from pathlib import Path

from research.cascade import storage
from research.cascade.detector import detect_cascade, try_fill_and_resolve, CascadeEvent


def _flat_candles(n, price=100.0, volume=10.0, start_ts=0):
    return [{"ts_ms": start_ts + i * 60_000, "open": price, "high": price,
              "low": price, "close": price, "volume": volume} for i in range(n)]


class TestDetectCascade(unittest.TestCase):

    def test_no_signal_below_move_threshold(self):
        candles = _flat_candles(61)
        candles[-1]["close"] = 100.1  # 0.1% move, below 0.3% threshold
        candles[-1]["volume"] = 100.0
        self.assertIsNone(detect_cascade("SOL", candles))

    def test_no_signal_below_volume_threshold(self):
        candles = _flat_candles(61)
        candles[-1]["close"] = 101.0  # 1% move, above threshold
        candles[-1]["volume"] = 11.0  # only 1.1x median, below 3x threshold
        self.assertIsNone(detect_cascade("SOL", candles))

    def test_up_cascade_detected_as_sell_fade(self):
        candles = _flat_candles(61)
        candles[-1]["close"] = 101.0
        candles[-1]["high"] = 101.0
        candles[-1]["volume"] = 50.0
        event = detect_cascade("SOL", candles)
        self.assertIsNotNone(event)
        self.assertEqual(event.fade_side, "sell")
        self.assertGreater(event.entry_price, candles[-1]["close"])

    def test_down_cascade_detected_as_buy_fade(self):
        candles = _flat_candles(61)
        candles[-1]["close"] = 99.0
        candles[-1]["low"] = 99.0
        candles[-1]["volume"] = 50.0
        event = detect_cascade("SOL", candles)
        self.assertIsNotNone(event)
        self.assertEqual(event.fade_side, "buy")
        self.assertLess(event.entry_price, candles[-1]["close"])

    def test_insufficient_history_returns_none(self):
        candles = _flat_candles(10)
        self.assertIsNone(detect_cascade("SOL", candles))


class TestResolution(unittest.TestCase):

    def _event(self):
        return CascadeEvent(
            symbol="SOL", detected_ts_ms=0, cascade_move_pct=1.0, cascade_vol_ratio=5.0,
            fade_side="sell", entry_price=101.0, stop_price=101.505,
            tp1_price=100.495, tp2_price=99.99,
        )

    def test_not_enough_time_returns_none(self):
        event = self._event()
        after = _flat_candles(2, price=101.0, start_ts=60_000)
        self.assertIsNone(try_fill_and_resolve(event, after))

    def test_never_filled_within_window(self):
        event = self._event()
        # price never reaches entry_price=101.0 (stays at 100)
        after = _flat_candles(20, price=100.0, start_ts=60_000)
        result = try_fill_and_resolve(event, after)
        self.assertIsNotNone(result)
        self.assertFalse(result["filled"])
        self.assertEqual(result["exit_reason"], "not_filled")

    def test_fills_and_hits_stop(self):
        event = self._event()
        after = _flat_candles(20, price=100.0, start_ts=60_000)
        after[0]["high"] = 101.0  # fills the sell
        for c in after[1:]:
            c["high"] = 102.0  # then runs up through stop
        result = try_fill_and_resolve(event, after)
        self.assertIsNotNone(result)
        self.assertTrue(result["filled"])
        self.assertIn(result["exit_reason"], ("stop", "tp1+stop"))
        self.assertLess(result["net_pnl_pct"], 0)


class TestSchemaWidening(unittest.TestCase):
    """Regression: on 2026-09-16 the resolution schema grew from 11 to 17
    columns mid-day. append_* only writes a header when it creates the file,
    so that day's already-open files kept an 11-column header while 17-field
    rows were appended under it -- ragged CSV that DictReader misreads
    silently. Five files were damaged before it was caught."""

    def setUp(self):
        self._orig = storage.DATA_DIR
        self._tmp = Path("/tmp/_test_cascade_widen")
        if self._tmp.exists():
            shutil.rmtree(self._tmp)
        self._tmp.mkdir(parents=True)
        storage.DATA_DIR = self._tmp

    def tearDown(self):
        storage.DATA_DIR = self._orig
        if self._tmp.exists():
            shutil.rmtree(self._tmp)

    def _narrow_file(self):
        """A file as the pre-change collector would have left it."""
        old_fields = storage.RESOLUTION_FIELDNAMES[:11]
        path = self._tmp / "SOL_2026-09-16_resolutions.csv"
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(old_fields)
            w.writerow(["SOL_1", "SOL", "t", "True", "stop", "1.0", "-0.5", "0.09", "-0.59", "-1.0", "4"])
        return path

    def test_widening_preserves_every_original_value(self):
        path = self._narrow_file()
        self.assertTrue(storage.widen_header_if_needed(path, storage.RESOLUTION_FIELDNAMES))
        with open(path, newline="") as f:
            rows = list(csv.reader(f))
        self.assertEqual(rows[0], list(storage.RESOLUTION_FIELDNAMES))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1][:11],
                         ["SOL_1", "SOL", "t", "True", "stop", "1.0", "-0.5", "0.09", "-0.59", "-1.0", "4"])
        self.assertTrue(all(v == "" for v in rows[1][11:]))

    def test_widening_is_idempotent(self):
        path = self._narrow_file()
        self.assertTrue(storage.widen_header_if_needed(path, storage.RESOLUTION_FIELDNAMES))
        self.assertFalse(storage.widen_header_if_needed(path, storage.RESOLUTION_FIELDNAMES))

    def test_append_to_narrow_file_does_not_produce_ragged_rows(self):
        """The actual failure: a wide row appended to a narrow file."""
        self._narrow_file()
        wide = {k: "" for k in storage.RESOLUTION_FIELDNAMES} | {
            "event_id": "SOL_2", "symbol": "SOL",
            "resolution_id": storage.make_resolution_id("SOL_2", 0.005),
            "arm_stop_pct": 0.005,
        }
        self.assertTrue(storage.append_resolution(wide, date_str="2026-09-16"))
        with open(self._tmp / "SOL_2026-09-16_resolutions.csv", newline="") as f:
            rows = list(csv.reader(f))
        width = len(storage.RESOLUTION_FIELDNAMES)
        self.assertTrue(all(len(r) == width for r in rows),
                        f"ragged rows: {[len(r) for r in rows]}")

    def test_unknown_header_is_refused_not_guessed(self):
        path = self._tmp / "SOL_2026-09-16_resolutions.csv"
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(["totally", "different", "header"])
        with self.assertRaises(ValueError):
            storage.widen_header_if_needed(path, storage.RESOLUTION_FIELDNAMES)


class TestStorageIdempotency(unittest.TestCase):

    def setUp(self):
        self._orig = storage.DATA_DIR
        self._tmp = Path("/tmp/_test_cascade_storage")
        if self._tmp.exists():
            shutil.rmtree(self._tmp)
        storage.DATA_DIR = self._tmp

    def tearDown(self):
        storage.DATA_DIR = self._orig
        if self._tmp.exists():
            shutil.rmtree(self._tmp)

    def _event_row(self, event_id="SOL_123"):
        return {k: "" for k in storage.EVENT_FIELDNAMES} | {
            "event_id": event_id, "symbol": "SOL", "fade_side": "sell",
        }

    def test_append_event_then_duplicate_is_noop(self):
        self.assertTrue(storage.append_event(self._event_row()))
        self.assertFalse(storage.append_event(self._event_row()))

    def test_resolution_matches_by_event_id(self):
        storage.append_event(self._event_row("SOL_999"))
        res_row = {k: "" for k in storage.RESOLUTION_FIELDNAMES} | {
            "event_id": "SOL_999", "symbol": "SOL",
        }
        self.assertTrue(storage.append_resolution(res_row))
        self.assertFalse(storage.append_resolution(res_row))

    def test_find_unresolved_excludes_resolved(self):
        storage.append_event(self._event_row("SOL_1"))
        storage.append_event(self._event_row("SOL_2"))
        storage.append_resolution({k: "" for k in storage.RESOLUTION_FIELDNAMES} | {
            "event_id": "SOL_1", "symbol": "SOL",
        })
        unresolved = storage.find_unresolved_event_ids(days_back=1)
        self.assertNotIn("SOL_1", unresolved)
        self.assertIn("SOL_2", unresolved)


if __name__ == "__main__":
    unittest.main()
