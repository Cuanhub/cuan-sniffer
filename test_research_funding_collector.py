"""
Functional tests for the funding collector -- storage idempotency, universe
selection, and row construction. No real network calls: venue responses are
faked so the suite stays deterministic and offline.
"""

import csv
import shutil
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from research.funding import storage
from research.funding.collector import _build_row
from research.funding.universe import select_universe


class TestStorageIdempotency(unittest.TestCase):

    def setUp(self):
        self._orig_data_dir = storage.DATA_DIR
        self._tmp_dir = Path("/tmp/_test_funding_storage")
        if self._tmp_dir.exists():
            shutil.rmtree(self._tmp_dir)
        storage.DATA_DIR = self._tmp_dir

    def tearDown(self):
        storage.DATA_DIR = self._orig_data_dir
        if self._tmp_dir.exists():
            shutil.rmtree(self._tmp_dir)

    def _row(self, ts, symbol="SOL"):
        return {k: "" for k in storage.FIELDNAMES} | {
            "timestamp_utc": ts, "symbol": symbol, "funding_rate": 0.0001,
        }

    def test_append_writes_header_once_and_appends_rows(self):
        self.assertTrue(storage.append_row(self._row("2026-09-01T00:00:00Z")))
        self.assertTrue(storage.append_row(self._row("2026-09-01T00:05:00Z")))
        path = storage._file_path("SOL", storage.datetime.fromisoformat("2026-09-01T00:00:00+00:00"))
        with open(path) as f:
            lines = f.readlines()
        self.assertEqual(lines[0].strip().split(",")[0], "timestamp_utc")
        self.assertEqual(len(lines), 3)  # header + 2 rows

    def test_restart_does_not_duplicate_existing_timestamp(self):
        storage.append_row(self._row("2026-09-01T00:00:00Z"))
        result = storage.append_row(self._row("2026-09-01T00:00:00Z"))
        self.assertFalse(result, "re-appending the same timestamp must be a no-op")
        path = storage._file_path("SOL", storage.datetime.fromisoformat("2026-09-01T00:00:00+00:00"))
        with open(path) as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 1)

    def test_different_symbols_get_different_files(self):
        storage.append_row(self._row("2026-09-01T00:00:00Z", symbol="SOL"))
        storage.append_row(self._row("2026-09-01T00:00:00Z", symbol="WIF"))
        sol_path = storage._file_path("SOL", storage.datetime.fromisoformat("2026-09-01T00:00:00+00:00"))
        wif_path = storage._file_path("WIF", storage.datetime.fromisoformat("2026-09-01T00:00:00+00:00"))
        self.assertTrue(sol_path.exists())
        self.assertTrue(wif_path.exists())

    def test_different_days_get_different_files(self):
        storage.append_row(self._row("2026-09-01T23:59:00Z"))
        storage.append_row(self._row("2026-09-02T00:01:00Z"))
        day1 = storage._file_path("SOL", storage.datetime.fromisoformat("2026-09-01T23:59:00+00:00"))
        day2 = storage._file_path("SOL", storage.datetime.fromisoformat("2026-09-02T00:01:00+00:00"))
        self.assertNotEqual(day1, day2)
        self.assertTrue(day1.exists())
        self.assertTrue(day2.exists())


class TestUniverseSelection(unittest.TestCase):

    def test_selects_top_n_by_volume_descending(self):
        fake_client = MagicMock()
        fake_client.get_meta_and_asset_ctxs.return_value = (
            {"universe": [{"name": "A"}, {"name": "B"}, {"name": "C"}, {"name": "D"}]},
            [
                {"dayNtlVlm": "100"},
                {"dayNtlVlm": "500"},
                {"dayNtlVlm": "10"},
                {"dayNtlVlm": "250"},
            ],
        )
        result = select_universe(fake_client, top_n=2)
        self.assertEqual([r["symbol"] for r in result], ["B", "D"])

    def test_handles_missing_or_malformed_volume_gracefully(self):
        fake_client = MagicMock()
        fake_client.get_meta_and_asset_ctxs.return_value = (
            {"universe": [{"name": "A"}, {"name": "B"}]},
            [{"dayNtlVlm": "bad"}, {}],
        )
        result = select_universe(fake_client, top_n=5)
        self.assertEqual(len(result), 2)
        self.assertTrue(all(r["day_ntl_vlm"] == 0.0 for r in result))


class TestBuildRow(unittest.TestCase):

    def test_computes_open_interest_usd_and_spread(self):
        ctx = {
            "funding": "0.0000125", "markPx": "100.0", "oraclePx": "100.5",
            "openInterest": "1000", "dayNtlVlm": "5000000",
        }
        book = {"levels": [[{"px": "99.9"}], [{"px": "100.1"}]]}
        row = _build_row("SOL", ctx, book, realized_funding_rate=None)
        self.assertAlmostEqual(row["open_interest_usd"], 100_000.0)
        self.assertAlmostEqual(row["mid"], 100.0)
        self.assertAlmostEqual(row["spread_bps"], 20.0, places=3)
        self.assertEqual(row["realized_funding_rate"], "")

    def test_missing_book_levels_leaves_bid_ask_empty_not_zero(self):
        ctx = {"funding": "0.0", "markPx": "100.0", "oraclePx": "100.0",
               "openInterest": "0", "dayNtlVlm": "0"}
        row = _build_row("SOL", ctx, {}, realized_funding_rate=None)
        self.assertEqual(row["best_bid"], "")
        self.assertEqual(row["best_ask"], "")
        self.assertEqual(row["spread_bps"], "")

    def test_realized_funding_rate_populated_when_provided(self):
        ctx = {"funding": "0.0001", "markPx": "1", "oraclePx": "1", "openInterest": "0", "dayNtlVlm": "0"}
        row = _build_row("SOL", ctx, {}, realized_funding_rate=0.00013)
        self.assertEqual(row["realized_funding_rate"], 0.00013)


if __name__ == "__main__":
    unittest.main()
