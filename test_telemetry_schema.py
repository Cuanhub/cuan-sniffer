import csv
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import executor
import executor_modules.telemetry as executor_telemetry
import smc_live_log
import signal_engine_modules.telemetry as signal_telemetry


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
                    confidence=0.72,
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
                    "score_v1",
                    "score_v2",
                    "score_v3",
                    "active_quality_model",
                    "active_quality_score",
                    "signal_confidence",
                    "active_quality_score_source",
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
                self.assertAlmostEqual(float(latest["score_v1"]), 0.72)
                self.assertAlmostEqual(float(latest["signal_confidence"]), 0.72)
                self.assertAlmostEqual(float(latest["rr_planned"]), 1.8)
                self.assertEqual(latest["regime_htf_1h"], "up")
                self.assertEqual(latest["regime_macro_4h"], "chop")
                self.assertAlmostEqual(float(latest["fill_slippage_bps"]), 2.4)
            finally:
                smc_live_log.LOG_PATH = old_path
                smc_live_log._HEADER_INITIALIZED = old_initialized

    def test_smc_live_log_infers_v3_active_quality_from_score_v3(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "smc_live_log.csv"
            old_path = smc_live_log.LOG_PATH
            old_initialized = smc_live_log._HEADER_INITIALIZED
            old_model = os.environ.get("LIVE_ELIGIBILITY_MODEL")
            try:
                os.environ["LIVE_ELIGIBILITY_MODEL"] = "v3"
                smc_live_log.LOG_PATH = str(path)
                smc_live_log._HEADER_INITIALIZED = False
                smc_live_log.append_smc_live_event(
                    event_type="gate_reject",
                    symbol="SOL",
                    score=0.66,
                    score_v3=0.91,
                    confidence=0.66,
                    reject_reason="unit_test",
                )

                with path.open(newline="", encoding="utf-8") as fh:
                    rows = list(csv.DictReader(fh))

                latest = rows[-1]
                self.assertEqual(latest["active_quality_model"], "v3")
                self.assertAlmostEqual(float(latest["active_quality_score"]), 0.91)
                self.assertEqual(latest["active_quality_score_source"], "score_v3")
                self.assertAlmostEqual(float(latest["signal_confidence"]), 0.91)
            finally:
                if old_model is None:
                    os.environ.pop("LIVE_ELIGIBILITY_MODEL", None)
                else:
                    os.environ["LIVE_ELIGIBILITY_MODEL"] = old_model
                smc_live_log.LOG_PATH = old_path
                smc_live_log._HEADER_INITIALIZED = old_initialized


class TestSignalTelemetryActiveQuality(unittest.TestCase):

    def test_score_candidate_marks_pending_v3_without_mislabeling_raw_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            score_path = Path(tmp) / "score_distribution.csv"
            smc_path = Path(tmp) / "smc_live_log.csv"
            old_score_path = signal_telemetry.SCORE_DIST_PATH
            old_smc_path = smc_live_log.LOG_PATH
            old_initialized = smc_live_log._HEADER_INITIALIZED
            old_model = os.environ.get("LIVE_ELIGIBILITY_MODEL")
            try:
                os.environ["LIVE_ELIGIBILITY_MODEL"] = "v3"
                signal_telemetry.SCORE_DIST_PATH = str(score_path)
                smc_live_log.LOG_PATH = str(smc_path)
                smc_live_log._HEADER_INITIALIZED = False

                signal_telemetry.log_score_candidate(
                    symbol="JTO",
                    timeframe="1h",
                    side="SHORT",
                    score=0.67,
                    threshold=0.80,
                    setup_family="continuation",
                    market_regime="weak_trend",
                    htf_regime="down",
                    macro_regime="down",
                    session="ny_open",
                )

                with score_path.open(newline="", encoding="utf-8") as fh:
                    reader = csv.DictReader(fh)
                    rows = list(reader)

                for field in (
                    "score_v1", "score_v2", "score_v3", "active_quality_model",
                    "active_quality_score", "active_quality_score_source", "session",
                ):
                    self.assertIn(field, reader.fieldnames)

                latest = rows[-1]
                self.assertEqual(latest["active_quality_model"], "")
                self.assertAlmostEqual(float(latest["score_v1"]), 0.67)
                self.assertAlmostEqual(float(latest["score_v3"]), 0.0)
                self.assertAlmostEqual(float(latest["active_quality_score"]), 0.0)
                self.assertEqual(latest["active_quality_score_source"], "pending_score_v3")
                self.assertEqual(latest["session"], "ny_open")

                with smc_path.open(newline="", encoding="utf-8") as fh:
                    smc_rows = list(csv.DictReader(fh))
                smc_latest = smc_rows[-1]
                self.assertEqual(smc_latest["active_quality_model"], "")
                self.assertAlmostEqual(float(smc_latest["active_quality_score"]), 0.0)
                self.assertEqual(smc_latest["active_quality_score_source"], "pending_score_v3")
            finally:
                if old_model is None:
                    os.environ.pop("LIVE_ELIGIBILITY_MODEL", None)
                else:
                    os.environ["LIVE_ELIGIBILITY_MODEL"] = old_model
                signal_telemetry.SCORE_DIST_PATH = old_score_path
                smc_live_log.LOG_PATH = old_smc_path
                smc_live_log._HEADER_INITIALIZED = old_initialized

    def test_score_candidate_records_explicit_v3_active_quality(self):
        with tempfile.TemporaryDirectory() as tmp:
            score_path = Path(tmp) / "score_distribution.csv"
            smc_path = Path(tmp) / "smc_live_log.csv"
            old_score_path = signal_telemetry.SCORE_DIST_PATH
            old_smc_path = smc_live_log.LOG_PATH
            old_initialized = smc_live_log._HEADER_INITIALIZED
            old_model = os.environ.get("LIVE_ELIGIBILITY_MODEL")
            try:
                os.environ["LIVE_ELIGIBILITY_MODEL"] = "v3"
                signal_telemetry.SCORE_DIST_PATH = str(score_path)
                smc_live_log.LOG_PATH = str(smc_path)
                smc_live_log._HEADER_INITIALIZED = False

                signal_telemetry.log_score_candidate(
                    symbol="SUI",
                    timeframe="1h",
                    side="SHORT",
                    score=0.71,
                    threshold=0.64,
                    rr=1.6923,
                    setup_family="reversal",
                    market_regime="chop",
                    htf_regime="chop",
                    macro_regime="up",
                    session="ny_open",
                    score_v1=0.71,
                    score_v2=0.62,
                    score_v3=0.82,
                    active_quality_model="v3",
                    active_quality_score=0.82,
                    signal_confidence=0.82,
                    active_quality_score_source="score_v3",
                )

                with score_path.open(newline="", encoding="utf-8") as fh:
                    latest = list(csv.DictReader(fh))[-1]

                self.assertEqual(latest["active_quality_model"], "v3")
                self.assertAlmostEqual(float(latest["score_v1"]), 0.71)
                self.assertAlmostEqual(float(latest["score_v2"]), 0.62)
                self.assertAlmostEqual(float(latest["score_v3"]), 0.82)
                self.assertAlmostEqual(float(latest["active_quality_score"]), 0.82)
                self.assertAlmostEqual(float(latest["signal_confidence"]), 0.82)
                self.assertEqual(latest["active_quality_score_source"], "score_v3")

                with smc_path.open(newline="", encoding="utf-8") as fh:
                    smc_latest = list(csv.DictReader(fh))[-1]
                self.assertEqual(smc_latest["active_quality_model"], "v3")
                self.assertAlmostEqual(float(smc_latest["score_v3"]), 0.82)
                self.assertAlmostEqual(float(smc_latest["active_quality_score"]), 0.82)
                self.assertEqual(smc_latest["active_quality_score_source"], "score_v3")
            finally:
                if old_model is None:
                    os.environ.pop("LIVE_ELIGIBILITY_MODEL", None)
                else:
                    os.environ["LIVE_ELIGIBILITY_MODEL"] = old_model
                signal_telemetry.SCORE_DIST_PATH = old_score_path
                smc_live_log.LOG_PATH = old_smc_path
                smc_live_log._HEADER_INITIALIZED = old_initialized

    def test_gate_reject_infers_v3_active_quality_from_score_v3(self):
        with tempfile.TemporaryDirectory() as tmp:
            reject_path = Path(tmp) / "gate_rejects.csv"
            smc_path = Path(tmp) / "smc_live_log.csv"
            old_reject_path = signal_telemetry.GATE_REJECTS_PATH
            old_smc_path = smc_live_log.LOG_PATH
            old_initialized = smc_live_log._HEADER_INITIALIZED
            old_model = os.environ.get("LIVE_ELIGIBILITY_MODEL")
            try:
                os.environ["LIVE_ELIGIBILITY_MODEL"] = "v3"
                signal_telemetry.GATE_REJECTS_PATH = str(reject_path)
                smc_live_log.LOG_PATH = str(smc_path)
                smc_live_log._HEADER_INITIALIZED = False

                signal_telemetry.log_gate_reject(
                    symbol="ZEC",
                    timeframe="1h",
                    side="SHORT",
                    reject_reason="v3_score_below_threshold:0.620<0.800",
                    raw_score=0.53,
                    threshold=0.80,
                    confidence=0.53,
                    score_v3=0.62,
                    market_regime="chop",
                    htf_regime="down",
                    macro_regime="down",
                    session="ny_pm",
                    setup_family="reversal",
                )

                with reject_path.open(newline="", encoding="utf-8") as fh:
                    reader = csv.DictReader(fh)
                    rows = list(reader)

                latest = rows[-1]
                self.assertEqual(latest["active_quality_model"], "v3")
                self.assertAlmostEqual(float(latest["active_quality_score"]), 0.62)
                self.assertEqual(latest["active_quality_score_source"], "score_v3")
                self.assertAlmostEqual(float(latest["score_v1"]), 0.53)
                self.assertAlmostEqual(float(latest["score_v3"]), 0.62)
            finally:
                if old_model is None:
                    os.environ.pop("LIVE_ELIGIBILITY_MODEL", None)
                else:
                    os.environ["LIVE_ELIGIBILITY_MODEL"] = old_model
                signal_telemetry.GATE_REJECTS_PATH = old_reject_path
                smc_live_log.LOG_PATH = old_smc_path
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
                    meta={
                        "total_score": 0.88,
                        "session": "ny_open",
                        "active_quality_model": "v3",
                        "active_quality_score": 0.91,
                    },
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
                    "active_quality_model",
                    "active_quality_score",
                    "signal_confidence",
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
                self.assertEqual(row["active_quality_model"], "v3")
                self.assertAlmostEqual(float(row["active_quality_score"]), 0.91)
                self.assertAlmostEqual(float(row["signal_confidence"]), 0.91)
                self.assertEqual(row["regime"], "continuation|htf_up")
                self.assertAlmostEqual(float(row["current_price"]), 103.0)
                self.assertAlmostEqual(float(row["price_move_r"]), 1.5)
            finally:
                executor.EXECUTOR_REJECTS_PATH = old_reject_path
                executor_telemetry.EXECUTOR_REJECTS_PATH = old_telemetry_reject_path
                executor.MISSED_LOG_FILE = old_missed_path
                executor.LOG_MISSED = old_log_missed
                executor_telemetry._MISSED_CONTEXTS.clear()

    def test_executor_reject_accepts_direct_active_quality_without_missed_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            reject_path = Path(tmp) / "executor_rejects.csv"
            old_telemetry_reject_path = executor_telemetry.EXECUTOR_REJECTS_PATH
            try:
                executor_telemetry.EXECUTOR_REJECTS_PATH = str(reject_path)
                executor_telemetry._MISSED_CONTEXTS.clear()

                executor_telemetry.log_executor_reject(
                    symbol="JTO",
                    coin="JTO",
                    side="SHORT",
                    signal_id=99,
                    entry_price=0.78,
                    stop_price=0.80,
                    tp_price=0.74,
                    confidence=0.93,
                    total_score=0.66,
                    active_quality_model="v3",
                    active_quality_score=0.93,
                    signal_confidence=0.93,
                    rr=1.69,
                    reject_reason="research_only_mode",
                    session="ny_open",
                    setup_family="reversal",
                    market_regime="weak_trend",
                    regime="reversal|htf_chop|macro_chop|mkt_weak_trend",
                    timeframe="1h",
                )

                with reject_path.open(newline="", encoding="utf-8") as fh:
                    rows = list(csv.DictReader(fh))

                row = rows[-1]
                self.assertEqual(row["signal_id"], "99")
                self.assertEqual(row["coin"], "JTO")
                self.assertEqual(row["active_quality_model"], "v3")
                self.assertAlmostEqual(float(row["active_quality_score"]), 0.93)
                self.assertAlmostEqual(float(row["signal_confidence"]), 0.93)
                self.assertAlmostEqual(float(row["entry_price"]), 0.78)
                self.assertEqual(row["regime"], "reversal|htf_chop|macro_chop|mkt_weak_trend")
            finally:
                executor_telemetry.EXECUTOR_REJECTS_PATH = old_telemetry_reject_path
                executor_telemetry._MISSED_CONTEXTS.clear()


if __name__ == "__main__":
    unittest.main()
