import sys
import unittest
from pathlib import Path

from executor_modules.execution_policy import (
    ExecutionPolicyConfig,
    evaluate_execution_policy,
)

sys.path.insert(0, str(Path(__file__).resolve().parent / "tools" / "research"))
from replay_shadow_trades import apply_replay_policy  # type: ignore[import-not-found]  # noqa: E402


def _base_trade(**overrides):
    trade = {
        "timestamp": "2026-06-21T09:00:00Z",
        "symbol": "SOL",
        "side": "LONG",
        "entry": 100.0,
        "stop": 98.7,
        "tp": 102.2,
        "atr": 1.0,
        "timeframe": "1h",
        "session": "ny_open",
        "market_regime": "weak_trend",
        "htf_regime": "down",
        "macro_regime": "down",
        "setup_family": "continuation",
        "signal_confidence": 0.86,
        "regime": "continuation|htf_down|macro_down|mkt_weak_trend",
        "execution_track": "",
    }
    trade.update(overrides)
    return trade


class TestExecutionPolicyParity(unittest.TestCase):
    def test_engine_rr_169_widened_preserves_original_rr(self):
        result = evaluate_execution_policy(
            entry=100.0,
            stop=98.7,
            tp=102.2,
            side="LONG",
            atr=1.0,
            timeframe="1h",
            session="ny_open",
            market_regime="strong_trend",
            htf_regime="down",
            macro_regime="down",
            setup_family="continuation",
            confidence=0.86,
            current_price=100.0,
        )

        self.assertTrue(result.approved)
        self.assertIsNone(result.reject_reason)
        self.assertAlmostEqual(result.original_rr, 1.69, places=2)
        self.assertAlmostEqual(result.final_rr, result.original_rr, places=3)
        self.assertTrue(result.metadata["stop_redesign_tp_adjusted"])
        self.assertGreater(result.final_tp, result.original_tp)

    def test_stop_widening_beyond_max_rejects_too_wide(self):
        result = evaluate_execution_policy(
            entry=100.0,
            stop=99.8,
            tp=104.0,
            side="LONG",
            atr=1.0,
            timeframe="1h",
            session="ny_open",
            market_regime="strong_trend",
            htf_regime="down",
            macro_regime="down",
            setup_family="continuation",
            confidence=0.86,
            current_price=100.0,
        )

        self.assertFalse(result.approved)
        self.assertIn("stop_redesign_too_wide", result.reject_reason)

    def test_tp_cap_below_rr_floor_rejects_impossible(self):
        result = evaluate_execution_policy(
            entry=100.0,
            stop=99.0,
            tp=103.0,
            side="LONG",
            atr=1.0,
            timeframe="15m",
            session="ny_open",
            market_regime="weak_trend",
            htf_regime="down",
            macro_regime="down",
            setup_family="continuation",
            confidence=0.86,
            current_price=100.0,
            config=ExecutionPolicyConfig(
                regime_tp_cap_r=1.50,
                min_execution_effective_rr=1.55,
                apply_stop_redesign=False,
                apply_tp_cap=True,
                apply_effective_rr=False,
                apply_chop_block=False,
                apply_dual_chop_block=False,
                block_continuation_in_weak_trend=False,
            ),
        )

        self.assertFalse(result.approved)
        self.assertIn("regime_tp_cap_rr_impossible", result.reject_reason)
        self.assertTrue(result.tp_capped)

    def test_engine_only_valid_signal_passes_unchanged(self):
        trade = _base_trade(stop=98.0, tp=104.0)
        result = apply_replay_policy(trade, "engine_only")

        self.assertTrue(result.approved)
        self.assertEqual(result.redesigned_stop, trade["stop"])
        self.assertEqual(result.final_tp, trade["tp"])
        self.assertAlmostEqual(result.final_rr, 2.0, places=3)

    def test_replay_policy_matches_direct_policy(self):
        trade = _base_trade(
            stop=96.0,
            tp=108.0,
            market_regime="strong_trend",
            macro_regime="down",
            regime="continuation|htf_down|macro_down|mkt_strong_trend",
        )

        replay_result = apply_replay_policy(trade, "production_executor")
        direct_result = evaluate_execution_policy(
            entry=trade["entry"],
            stop=trade["stop"],
            tp=trade["tp"],
            side=trade["side"],
            atr=trade["atr"],
            timeframe=trade["timeframe"],
            session=trade["session"],
            market_regime=trade["market_regime"],
            htf_regime=trade["htf_regime"],
            macro_regime=trade["macro_regime"],
            setup_family=trade["setup_family"],
            confidence=trade["signal_confidence"],
            current_price=trade["entry"],
            regime=trade["regime"],
            track=trade["execution_track"],
        )

        self.assertEqual(replay_result.reject_reason, direct_result.reject_reason)
        self.assertAlmostEqual(replay_result.final_rr, direct_result.final_rr, places=6)

    def test_production_executor_blocks_continuation_in_weak_trend(self):
        trade = _base_trade(
            setup_family="continuation",
            market_regime="weak_trend",
            regime="continuation|htf_down|macro_down|mkt_weak_trend",
        )

        result = apply_replay_policy(trade, "production_executor")

        self.assertFalse(result.approved)
        self.assertEqual(
            result.reject_reason,
            "market_regime_block:continuation_in_weak_trend",
        )

    def test_production_executor_blocks_reversal_in_local_weak_trend(self):
        trade = _base_trade(
            setup_family="reversal",
            market_regime="mkt_weak_trend",
            htf_regime="chop",
            macro_regime="chop",
            regime="reversal|htf_chop|macro_chop|mkt_weak_trend",
        )

        result = apply_replay_policy(trade, "production_executor")

        self.assertFalse(result.approved)
        self.assertEqual(
            result.reject_reason,
            "market_regime_block:reversal_in_weak_trend",
        )

    def test_no_stop_redesign_differs_only_when_selected(self):
        trade = _base_trade(market_regime="strong_trend", regime="continuation|mkt_strong_trend")

        production_result = apply_replay_policy(trade, "production_executor")
        no_stop_result = apply_replay_policy(trade, "no_stop_redesign")

        self.assertTrue(production_result.approved)
        self.assertTrue(production_result.metadata["stop_redesign_tp_adjusted"])
        self.assertGreater(
            abs(trade["entry"] - production_result.redesigned_stop),
            abs(trade["entry"] - trade["stop"]),
        )
        self.assertTrue(no_stop_result.approved)
        self.assertEqual(no_stop_result.redesigned_stop, trade["stop"])
        self.assertEqual(no_stop_result.final_tp, trade["tp"])

    def test_chop_hard_block_active_in_production_executor_policy(self):
        # HARD_BLOCK_CHOP flipped false->true on 2026-08-25 (real
        # full-pipeline replay: chop-regime reversals showed n=94,
        # WR=23.4%, PF=0.678 -- worse than weak_trend, which was already
        # hard-blocked). This test previously asserted the opposite
        # (chop NOT blocked) because that used to be true; it no longer
        # is, by design. The chop block has no exception path (the old
        # CHOP_REVERSAL_EXCEPTION carve-out was removed 2026-08-28 as
        # dead code -- htf==up is unconditionally blocked upstream at the
        # signal_engine eligibility gate, so it could never fire).
        trade = _base_trade(
            stop=96.0,
            tp=108.0,
            market_regime="chop",
            macro_regime="down",
            setup_family="reversal",
            regime="reversal|htf_down|macro_down|mkt_chop",
        )

        result = apply_replay_policy(trade, "production_executor")

        self.assertFalse(result.approved)
        self.assertEqual(result.reject_reason, "market_regime_block:chop")

    def test_chop_block_does_not_apply_in_no_chop_block_policy(self):
        trade = _base_trade(
            stop=96.0,
            tp=108.0,
            market_regime="chop",
            macro_regime="chop",
            regime="continuation|htf_down|macro_chop|mkt_chop",
        )

        result = apply_replay_policy(trade, "no_chop_block")

        self.assertTrue(result.approved)
        self.assertNotEqual(result.reject_reason, "market_regime_block:chop")


if __name__ == "__main__":
    unittest.main()
