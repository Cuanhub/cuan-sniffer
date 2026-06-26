import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import executor
import executor_modules.telemetry as executor_telemetry
import smc_live_log


class TestSmcTelemetrySchemaMigration(unittest.TestCase):

    def test_smc_live_log_contains_retirable_telemetry_fields_and_aliases(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "smc_live_log.csv"
            path.write_text("timestamp_utc,event_type\n2026-06-01T00:00:00Z,old\n")

            old_path = smc_live_log.LOG_PATH
            old_initialized = smc_live_log._HEADER_INITIALIZED
            try:
                smc_live_log.LOG_PATH = str(path)
                smc_live_log._HEADER_INITIALIZED = False
                smc_live_log.init_smc_live_log()
                smc_live_log.append_smc_live_event(
                    event_type="score_candidate",
                    coin="SOL",
                    score=0.72,
                    rr=1.8,
                    htf_regime="up",
                    macro_regime="chop",
                    slippage_bps=2.4,
                )

                with path.open(newline="", encoding="utf-8") as fh:
                    reader = csv.DictReader(fh)
                    rows = list(reader)

                for field in (
                    "timestamp",
                    "symbol",
                    "raw_score",
                    "threshold",
                    "effective_threshold",
                    "metadata",
                    "setup_family",
                    "executor_result",
                    "rr_planned",
                    "total_score",
                    "reason_text",
                    "regime",
                    "regime_local",
                    "regime_htf_1h",
                    "regime_macro_4h",
                    "stop_dist",
                    "tp_dist",
                    "vol_state",
                    "vol_ratio",
                    "whale_pressure",
                    "flow_momentum",
                    "funding_rate",
                    "open_interest",
                    "long_short_bias",
                    "fill_slippage_bps",
                    "fill_ratio",
                ):
                    self.assertIn(field, reader.fieldnames)

                latest = rows[-1]
                self.assertEqual(latest["symbol"], "SOL")
                self.assertAlmostEqual(float(latest["raw_score"]), 0.72)
                self.assertAlmostEqual(float(latest["total_score"]), 0.72)
                self.assertAlmostEqual(float(latest["rr_planned"]), 1.8)
                self.assertEqual(latest["regime_htf_1h"], "up")
                self.assertEqual(latest["regime_macro_4h"], "chop")
                self.assertAlmostEqual(float(latest["fill_slippage_bps"]), 2.4)
            finally:
                smc_live_log.LOG_PATH = old_path
                smc_live_log._HEADER_INITIALIZED = old_initialized


class TestExecutorRejectSchemaMigration(unittest.TestCase):

    def test_executor_rejects_absorb_missed_signal_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            reject_path = Path(tmp) / "executor_rejects.csv"
            missed_path = Path(tmp) / "missed_signals.csv"
            reject_path.write_text(
                "timestamp,symbol,side,confidence,required_confidence,rr,"
                "required_rr,reject_reason,session,setup_family,market_regime,timeframe\n"
            )

            old_reject_path = executor.EXECUTOR_REJECTS_PATH
            old_telemetry_reject_path = executor_telemetry.EXECUTOR_REJECTS_PATH
            old_missed_path = executor.MISSED_LOG_FILE
            old_log_missed = executor.LOG_MISSED
            try:
                executor.EXECUTOR_REJECTS_PATH = str(reject_path)
                executor_telemetry.EXECUTOR_REJECTS_PATH = str(reject_path)
                executor.MISSED_LOG_FILE = str(missed_path)
                executor.LOG_MISSED = True
                executor_telemetry._MISSED_CONTEXTS.clear()

                ex = object.__new__(executor.Executor)
                ex.backend = SimpleNamespace(get_cached_mid_price=lambda coin: 103.0)
                ex._missed_log_memory = {}

                signal = SimpleNamespace(
                    coin="SOL",
                    side="LONG",
                    entry_price=100.0,
                    stop_price=98.0,
                    tp_price=104.0,
                    confidence=0.91,
                    regime="continuation|htf_up",
                    meta={"total_score": 0.88, "session": "ny_open"},
                )

                executor.Executor._log_missed(ex, signal, 42, "rr_too_low")
                executor.log_executor_reject(
                    symbol="SOL",
                    side="LONG",
                    confidence=0.91,
                    rr=1.4,
                    reject_reason="rr_too_low:1.40<1.55",
                    session="ny_open",
                    setup_family="continuation",
                    market_regime="weak_trend",
                    timeframe="1h",
                )

                with reject_path.open(newline="", encoding="utf-8") as fh:
                    reader = csv.DictReader(fh)
                    rows = list(reader)

                for field in (
                    "signal_id",
                    "coin",
                    "entry_price",
                    "stop_price",
                    "tp_price",
                    "total_score",
                    "regime",
                    "current_price",
                    "price_move_r",
                ):
                    self.assertIn(field, reader.fieldnames)

                row = rows[-1]
                self.assertEqual(row["signal_id"], "42")
                self.assertEqual(row["coin"], "SOL")
                self.assertAlmostEqual(float(row["entry_price"]), 100.0)
                self.assertAlmostEqual(float(row["stop_price"]), 98.0)
                self.assertAlmostEqual(float(row["tp_price"]), 104.0)
                self.assertAlmostEqual(float(row["total_score"]), 0.88)
                self.assertEqual(row["regime"], "continuation|htf_up")
                self.assertAlmostEqual(float(row["current_price"]), 103.0)
                self.assertAlmostEqual(float(row["price_move_r"]), 1.5)
            finally:
                executor.EXECUTOR_REJECTS_PATH = old_reject_path
                executor_telemetry.EXECUTOR_REJECTS_PATH = old_telemetry_reject_path
                executor.MISSED_LOG_FILE = old_missed_path
                executor.LOG_MISSED = old_log_missed
                executor_telemetry._MISSED_CONTEXTS.clear()


if __name__ == "__main__":
    unittest.main()
