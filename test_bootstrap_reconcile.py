"""
Tests for boot-time venue reconciliation of stale open positions.

A position the venue disowns at boot must be written back to trades.csv as
closed (not just silently dropped from in-memory tracking) — otherwise the
CSV permanently lies about what's open.
"""

import csv
import os
import tempfile
import unittest
from pathlib import Path

import bootstrap
import trade_log
from position import CloseReason, Position, PositionState


class TestBootstrapVenueReconcile(unittest.TestCase):

    def setUp(self):
        self.tmp_path = tempfile.mktemp(suffix=".csv")
        self._old_bootstrap_trades_file = bootstrap.TRADES_FILE
        self._old_trade_log_trades_file = trade_log.TRADES_FILE
        bootstrap.TRADES_FILE = self.tmp_path
        trade_log.TRADES_FILE = self.tmp_path

    def tearDown(self):
        bootstrap.TRADES_FILE = self._old_bootstrap_trades_file
        trade_log.TRADES_FILE = self._old_trade_log_trades_file
        if os.path.exists(self.tmp_path):
            os.unlink(self.tmp_path)
        lock_path = f"{self.tmp_path}.lock"
        if os.path.exists(lock_path):
            os.unlink(lock_path)

    def _seed_open_position(self, coin="SOL", side="SHORT"):
        pos = Position(
            position_id=f"{coin}_test123",
            coin=coin, side=side, signal_id=1,
            entry_price=100.0, stop_price=102.0, tp_price=94.0, atr=1.0,
            state=PositionState.OPEN,
        )
        trade_log.upsert_trade_row(pos.to_dict())
        return pos

    def _read_row(self, position_id):
        with open(self.tmp_path, newline="") as f:
            for row in csv.DictReader(f):
                if row["position_id"] == position_id:
                    return row
        return None

    def test_venue_rejected_position_is_written_closed(self):
        self._seed_open_position(coin="SOL", side="SHORT")

        result = bootstrap.bootstrap_state(
            starting_balance=1000.0,
            paper_mode=False,
            venue_checker=lambda coin, side: False,  # venue says: not open
        )

        self.assertEqual(result.open_positions, [])

        row = self._read_row("SOL_test123")
        self.assertIsNotNone(row)
        self.assertEqual(row["state"], PositionState.CLOSED.value)
        self.assertEqual(row["close_reason"], CloseReason.SYSTEM_EXIT_RECONCILE.value)
        self.assertEqual(row["exit_trigger_source"], "bootstrap_venue_reject")
        self.assertEqual(row["reconciled_from_venue"], "true")
        self.assertEqual(float(row["realized_r"]), 0.0)
        self.assertEqual(float(row["pnl_usd"]), 0.0)

    def test_venue_confirmed_position_is_left_open(self):
        self._seed_open_position(coin="JTO", side="LONG")

        result = bootstrap.bootstrap_state(
            starting_balance=1000.0,
            paper_mode=False,
            venue_checker=lambda coin, side: True,  # venue confirms: still open
        )

        self.assertEqual(len(result.open_positions), 1)
        row = self._read_row("JTO_test123")
        self.assertEqual(row["state"], PositionState.OPEN.value)


if __name__ == "__main__":
    unittest.main()
