import csv
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from score_v2 import compute_shadow_score_v2
from shadow_research import (
    CANDIDATE_FIELDS,
    append_shadow_candidate,
    append_shadow_execution,
    build_shadow_candidate,
    compute_forward_outcome,
    update_shadow_outcomes_from_df,
)


class TestShadowResearchIds(unittest.TestCase):

    def test_shadow_id_is_stable_and_geometry_sensitive(self):
        base = build_shadow_candidate(
            symbol="SOL",
            timeframe="1h",
            bar_time="2026-06-20T00:00:00Z",
            side="LONG",
            setup_family="continuation",
            market_regime="chop",
            htf_regime="up",
            macro_regime="chop",
            entry_price=100,
            stop_price=95,
            tp_price=110,
            engine_decision="accepted",
        )
        same = build_shadow_candidate(
            symbol="SOL",
            timeframe="1h",
            bar_time="2026-06-20T00:00:00Z",
            side="LONG",
            setup_family="continuation",
            market_regime="chop",
            htf_regime="up",
            macro_regime="chop",
            entry_price=100,
            stop_price=95,
            tp_price=110,
            engine_decision="accepted",
        )
        changed = build_shadow_candidate(
            symbol="SOL",
            timeframe="1h",
            bar_time="2026-06-20T00:00:00Z",
            side="LONG",
            setup_family="continuation",
            market_regime="chop",
            htf_regime="up",
            macro_regime="chop",
            entry_price=100,
            stop_price=95,
            tp_price=111,
            engine_decision="accepted",
        )

        self.assertEqual(base["shadow_id"], same["shadow_id"])
        self.assertNotEqual(base["shadow_id"], changed["shadow_id"])

    def test_append_shadow_candidate_dedupes_by_shadow_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "candidates.csv")
            row = build_shadow_candidate(
                symbol="SOL",
                timeframe="1h",
                bar_time="2026-06-20T00:00:00Z",
                side="LONG",
                setup_family="continuation",
                entry_price=100,
                stop_price=95,
                tp_price=110,
                engine_decision="accepted",
            )
            with patch.dict(os.environ, {"SHADOW_RESEARCH_ENABLED": "true"}, clear=False):
                self.assertTrue(append_shadow_candidate(row, path=path))
                self.assertFalse(append_shadow_candidate(row, path=path))

            with open(path, newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["shadow_id"], row["shadow_id"])

    def test_candidate_header_migration_preserves_inserted_v3_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "candidates.csv")
            old_fields = [
                field for field in CANDIDATE_FIELDS
                if field not in {
                    "score_v3",
                    "score_v3_version",
                    "score_v3_tags",
                    "score_v3_reason",
                }
            ]

            old_row = {field: "" for field in old_fields}
            old_row.update({
                "timestamp_utc": "2026-06-21T13:33:15Z",
                "shadow_id": "old36",
                "setup_key": "old-setup",
                "symbol": "SOL",
                "timeframe": "1h",
                "bar_time": "2026-06-21T13:00:00Z",
                "side": "LONG",
                "score_v2_reason": "old v2 reason",
                "engine_decision": "accepted",
                "setup_family": "reversal",
                "session": "london_open",
                "source": "signal_engine",
            })

            stale_new_row = {field: "" for field in CANDIDATE_FIELDS}
            stale_new_row.update({
                "timestamp_utc": "2026-06-21T14:33:15Z",
                "shadow_id": "new40",
                "setup_key": "new-setup",
                "symbol": "SUI",
                "timeframe": "1h",
                "bar_time": "2026-06-21T14:00:00Z",
                "side": "SHORT",
                "score_v2": "0.76",
                "score_v2_reason": "new v2 reason",
                "score_v3": "0.75",
                "score_v3_version": "v3_2026_06_factor_shadow",
                "score_v3_tags": "+fvg,+ob",
                "score_v3_reason": "+fvg, +ob",
                "engine_decision": "rejected",
                "engine_reject_reason": "score_below_threshold",
                "setup_family": "continuation",
                "session": "ny_open",
                "price": "123.45",
                "vol_ratio": "0.0123",
                "source": "signal_engine",
            })

            with open(path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(old_fields)
                writer.writerow([old_row.get(field, "") for field in old_fields])
                writer.writerow([stale_new_row.get(field, "") for field in CANDIDATE_FIELDS])

            appended = build_shadow_candidate(
                symbol="JTO",
                timeframe="1h",
                bar_time="2026-06-21T15:00:00Z",
                side="LONG",
                setup_family="continuation",
                score_v3=0.92,
                score_v3_version="v3_2026_06_factor_shadow",
                score_v3_tags="+fvg,+ob,v3_full_recipe",
                score_v3_reason="+fvg, +ob",
                engine_decision="accepted",
            )

            with patch.dict(os.environ, {"SHADOW_RESEARCH_ENABLED": "true"}, clear=False):
                self.assertTrue(append_shadow_candidate(appended, path=path))

            with open(path, newline="", encoding="utf-8") as fh:
                raw_rows = list(csv.reader(fh))
            self.assertEqual(raw_rows[0], CANDIDATE_FIELDS)
            self.assertTrue(all(len(row) == len(CANDIDATE_FIELDS) for row in raw_rows[1:]))

            with open(path, newline="", encoding="utf-8") as fh:
                rows = {row["shadow_id"]: row for row in csv.DictReader(fh)}

            self.assertEqual(rows["old36"]["engine_decision"], "accepted")
            self.assertEqual(rows["old36"]["setup_family"], "reversal")
            self.assertEqual(rows["old36"]["score_v3"], "")

            self.assertEqual(rows["new40"]["score_v3"], "0.75")
            self.assertEqual(rows["new40"]["score_v3_version"], "v3_2026_06_factor_shadow")
            self.assertEqual(rows["new40"]["score_v3_tags"], "+fvg,+ob")
            self.assertEqual(rows["new40"]["engine_decision"], "rejected")
            self.assertEqual(rows["new40"]["setup_family"], "continuation")
            self.assertEqual(rows["new40"]["session"], "ny_open")
            self.assertEqual(rows["new40"]["source"], "signal_engine")

            self.assertEqual(rows[appended["shadow_id"]]["score_v3"], "0.92")

    def test_candidate_migration_repairs_previously_shifted_v3_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "candidates.csv")
            shifted_row = {field: "" for field in CANDIDATE_FIELDS}
            shifted_row.update({
                "timestamp_utc": "2026-06-21T14:36:51Z",
                "shadow_id": "shifted40",
                "setup_key": "shifted-setup",
                "symbol": "JTO",
                "timeframe": "1h",
                "bar_time": "2026-06-21T14:00:00Z",
                "side": "SHORT",
                "entry_price": "0.70489000",
                "score_v2": "0.46",
                "score_v2_version": "v2_2026_06_factor_shadow",
                "score_v2_tags": "+fvg,+macro_chop",
                "score_v2_reason": "+fvg, +macro_chop",
                "score_v3": "0.56",
                "score_v3_version": "v3_2026_06_factor_shadow",
                "score_v3_tags": "missing_macro_regime,+preferred_symbol",
                "score_v3_reason": "+preferred_symbol",
                "engine_decision": "0.72",
                "engine_reject_reason": "v3_2026_06_factor_shadow",
                "setup_family": "+fvg,+macro_chop,+htf_down",
                "swing_family": "+fvg, +macro_chop, +htf_down",
                "session": "rejected",
                "market_regime": "score_below_threshold:0.490<0.640",
                "htf_regime": "reversal",
                "macro_regime": "",
                "edge_buckets": "asia_late",
                "edge_bucket_count": "weak_trend",
                "independent_bucket_count": "down",
                "governance_reason": "chop",
                "triggers": "execution_quality,orderflow,price_location",
                "stop_method": "5",
                "atr": "4",
                "price": "pass",
                "vol_state": "fvg_bull,fvg_bear",
                "vol_ratio": "",
                "source": "0.01695289",
            })

            with open(path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=CANDIDATE_FIELDS)
                writer.writeheader()
                writer.writerow(shifted_row)

            appended = build_shadow_candidate(
                symbol="SOL",
                timeframe="1h",
                bar_time="2026-06-21T15:00:00Z",
                side="LONG",
                setup_family="reversal",
                engine_decision="accepted",
            )
            with patch.dict(os.environ, {"SHADOW_RESEARCH_ENABLED": "true"}, clear=False):
                self.assertTrue(append_shadow_candidate(appended, path=path))

            with open(path, newline="", encoding="utf-8") as fh:
                rows = {row["shadow_id"]: row for row in csv.DictReader(fh)}

            repaired = rows["shifted40"]
            self.assertEqual(repaired["score_v3"], "0.72")
            self.assertEqual(repaired["score_v3_tags"], "+fvg,+macro_chop,+htf_down")
            self.assertEqual(repaired["engine_decision"], "rejected")
            self.assertEqual(repaired["engine_reject_reason"], "score_below_threshold:0.490<0.640")
            self.assertEqual(repaired["setup_family"], "reversal")
            self.assertEqual(repaired["session"], "asia_late")
            self.assertEqual(repaired["market_regime"], "weak_trend")
            self.assertEqual(repaired["htf_regime"], "down")
            self.assertEqual(repaired["macro_regime"], "chop")
            self.assertEqual(repaired["edge_bucket_count"], "5")
            self.assertEqual(repaired["independent_bucket_count"], "4")
            self.assertEqual(repaired["governance_reason"], "pass")
            self.assertEqual(repaired["triggers"], "fvg_bull,fvg_bear")
            self.assertEqual(repaired["atr"], "0.01695289")
            self.assertEqual(repaired["source"], "signal_engine")


class TestShadowForwardOutcome(unittest.TestCase):

    def test_long_outcome_labels_tp_first(self):
        times = pd.date_range("2026-06-20T01:00:00Z", periods=3, freq="h")
        df = pd.DataFrame({
            "time": times,
            "high": [105.0, 111.0, 109.0],
            "low": [99.0, 101.0, 102.0],
            "close": [104.0, 110.5, 108.0],
        })

        outcome = compute_forward_outcome(
            df,
            side="LONG",
            entry_price=100.0,
            stop_price=95.0,
            tp_price=110.0,
            bar_time="2026-06-20T00:00:00Z",
            horizon_bars=3,
        )

        self.assertEqual(outcome["first_touch"], "tp")
        self.assertEqual(outcome["bars_to_first_touch"], 2)
        self.assertEqual(outcome["hit_tp"], "true")
        self.assertEqual(outcome["hit_stop"], "false")
        self.assertAlmostEqual(outcome["outcome_r"], 2.0)
        self.assertAlmostEqual(outcome["mfe_r"], 2.2)
        self.assertAlmostEqual(outcome["mae_r"], -0.2)

    def test_short_same_bar_tp_and_stop_is_conservative(self):
        times = pd.date_range("2026-06-20T01:00:00Z", periods=2, freq="h")
        df = pd.DataFrame({
            "time": times,
            "high": [106.0, 103.0],
            "low": [89.0, 97.0],
            "close": [94.0, 100.0],
        })

        outcome = compute_forward_outcome(
            df,
            side="SHORT",
            entry_price=100.0,
            stop_price=105.0,
            tp_price=90.0,
            bar_time="2026-06-20T00:00:00Z",
            horizon_bars=2,
        )

        self.assertEqual(outcome["first_touch"], "stop_first_assumed")
        self.assertEqual(outcome["hit_tp"], "true")
        self.assertEqual(outcome["hit_stop"], "true")
        self.assertEqual(outcome["ambiguous_same_bar"], "true")
        self.assertAlmostEqual(outcome["outcome_r"], -1.0)


class TestShadowOutcomeFiles(unittest.TestCase):

    def test_update_outcomes_writes_ready_horizons_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidates_path = str(Path(tmp) / "candidates.csv")
            outcomes_path = str(Path(tmp) / "outcomes.csv")
            row = build_shadow_candidate(
                symbol="SOL",
                timeframe="1h",
                bar_time="2026-06-20T00:00:00Z",
                side="LONG",
                setup_family="continuation",
                entry_price=100,
                stop_price=95,
                tp_price=110,
                engine_decision="accepted",
            )
            times = pd.date_range("2026-06-20T01:00:00Z", periods=4, freq="h")
            df = pd.DataFrame({
                "time": times,
                "high": [104.0, 111.0, 112.0, 113.0],
                "low": [99.0, 101.0, 102.0, 103.0],
                "close": [103.0, 110.0, 111.0, 112.0],
            })

            with patch.dict(os.environ, {"SHADOW_RESEARCH_ENABLED": "true"}, clear=False):
                self.assertTrue(append_shadow_candidate(row, path=candidates_path))
                written = update_shadow_outcomes_from_df(
                    symbol="SOL",
                    timeframe="1h",
                    df=df,
                    candidate_path=candidates_path,
                    outcome_path=outcomes_path,
                    horizons=(2, 5),
                )
                written_again = update_shadow_outcomes_from_df(
                    symbol="SOL",
                    timeframe="1h",
                    df=df,
                    candidate_path=candidates_path,
                    outcome_path=outcomes_path,
                    horizons=(2, 5),
                )

            self.assertEqual(written, 1)
            self.assertEqual(written_again, 0)
            with open(outcomes_path, newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["shadow_id"], row["shadow_id"])
            self.assertEqual(rows[0]["horizon_bars"], "2")


class TestShadowExecutionEvents(unittest.TestCase):

    def test_execution_event_uses_existing_shadow_id_and_final_geometry(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "executions.csv")
            signal = SimpleNamespace(
                coin="SOL",
                side="LONG",
                entry_price=100.0,
                stop_price=94.0,
                tp_price=110.5,
                confidence=0.91,
                meta={
                    "shadow_id": "abc123",
                    "shadow_setup_key": "SOL|1h|bar|LONG|continuation",
                    "timeframe": "1h",
                    "total_score": 0.91,
                    "rr_planned": 1.8,
                    "final_entry": 100.0,
                    "final_stop": 94.0,
                    "final_tp": 110.5,
                    "final_rr": 1.75,
                    "final_stop_method": "atr+executor_redesign",
                    "stop_was_redesigned": True,
                },
            )
            result = SimpleNamespace(
                traded=True,
                reason="submitted",
                position_id="SOL_1",
                fill_price=100.1,
                fill_slippage_bps=3.5,
                fill_ratio=1.0,
                size_usd=500.0,
                risk_usd=30.0,
                entry_fee_usd=0.2,
                protection_status="ok",
            )

            with patch.dict(os.environ, {"SHADOW_RESEARCH_ENABLED": "true"}, clear=False):
                self.assertTrue(append_shadow_execution(signal, result, path=path))

            with open(path, newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(rows[0]["shadow_id"], "abc123")
            self.assertEqual(rows[0]["executor_decision"], "accepted")
            self.assertEqual(rows[0]["final_rr"], "1.75")
            self.assertEqual(rows[0]["stop_was_redesigned"], "true")

    def test_rejected_execution_event_records_reason_family(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "executions.csv")
            signal = SimpleNamespace(
                coin="SOL",
                side="LONG",
                entry_price=100.0,
                stop_price=95.0,
                tp_price=110.0,
                confidence=0.88,
                meta={
                    "shadow_id": "reject123",
                    "shadow_setup_key": "SOL|1h|bar|LONG|continuation",
                    "timeframe": "1h",
                    "total_score": 0.88,
                    "rr_planned": 2.0,
                    "setup_family": "continuation",
                },
            )
            result = SimpleNamespace(
                traded=False,
                reason="rr_too_low:1.40<1.55",
                position_id="",
                fill_price=0.0,
                fill_slippage_bps=0.0,
                fill_ratio=0.0,
                size_usd=0.0,
                risk_usd=0.0,
                entry_fee_usd=0.0,
                protection_status="",
            )

            with patch.dict(os.environ, {"SHADOW_RESEARCH_ENABLED": "true"}, clear=False):
                self.assertTrue(append_shadow_execution(signal, result, path=path))

            with open(path, newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(rows[0]["shadow_id"], "reject123")
            self.assertEqual(rows[0]["executor_decision"], "rejected")
            self.assertEqual(rows[0]["executor_reject_reason"], "rr_too_low:1.40<1.55")
            self.assertEqual(rows[0]["executor_reject_family"], "rr")


class TestShadowScoreContext(unittest.TestCase):

    def test_pre_threshold_context_does_not_emit_missing_tags(self):
        result = compute_shadow_score_v2({
            "symbol": "SOL",
            "coin": "SOL",
            "side": "LONG",
            "setup_family": "continuation",
            "session": "ny_open",
            "market_regime": "chop",
            "htf_regime": "up",
            "macro_regime": "chop",
            "score": 0.70,
            "total_score": 0.70,
            "timeframe": "1h",
            "stop_method": "pre_trade_levels_unbuilt",
            "fvg_bull": True,
        })
        missing = [tag for tag in result["score_v2_tags"] if tag.startswith("missing_")]
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
