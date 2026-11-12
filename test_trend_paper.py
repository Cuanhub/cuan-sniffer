"""The paper trend pass: one definition of the strategy, a ledger that adds
up, and a pass that changes nothing when it cannot read the venue.
Synthetic prices and a fake client only; nothing here touches the network."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from research.backtest.experiments import trend_daily as T
from research.trend import paper as P

DAY = P.DAY_MS
T0 = T.day_ms("2026-01-01")


def closes(days=260, C=10, seed=1, drift=0.0):
    rng = np.random.default_rng(seed)
    return 100.0 * np.exp(np.cumsum(drift + rng.normal(0, 0.03, (days, C)), axis=0))


class FakeClient:
    """Serves daily candles up to `now`, mids at the last close, and a flat
    hourly funding rate. `broken` names what to fail."""

    def __init__(self, close, broken=""):
        self.close, self.broken, self.posts = close, broken, 0

    def bars(self, j, now_ms):
        n = min(len(self.close), int((now_ms - T0) // DAY) + 1)   # incl. the forming bar
        return [{"t": T0 + i * DAY, "o": 1, "h": 1, "l": 1, "c": float(self.close[i, j]),
                 "v": 1, "n": 1} for i in range(n)]

    def get_candles(self, coin, interval, start_ms, end_ms):
        assert interval == "1d"
        if self.broken == "candles":
            raise RuntimeError("500")
        j = P.COINS.index(coin)
        rows = self.bars(j, end_ms)
        if self.broken == "hole" and coin == "SOL":
            rows = rows[:-40] + rows[-39:]
        return rows

    def get_all_mids(self):
        if self.broken == "mids":
            return {}
        return {c: float(self.mid_row[j]) for j, c in enumerate(P.COINS)}

    def _post(self, payload, label):
        self.posts += 1
        if self.broken == "funding":
            raise RuntimeError("429")
        hours = range(payload["startTime"] // 3_600_000 + 1, payload["endTime"] // 3_600_000 + 1)
        return [{"coin": payload["coin"], "fundingRate": "0.0000125", "time": h * 3_600_000}
                for h in hours]


def now_for(day_index, minutes=5):
    """A few minutes into the UTC day after row `day_index` closed."""
    return T0 + (day_index + 1) * DAY + minutes * 60_000


class OneDefinition(unittest.TestCase):
    def test_replay_is_the_backtest_engine_day_for_day(self):
        close = closes(330, 4, seed=3, drift=0.001)
        s, v, e = T.features(close)
        days = np.arange(T.MIN_HISTORY - 1, len(close) - 1)
        engine = T.simulate(s, v, e, T.next_returns(close), days)["net"][0]
        model = P.replay(close, list(range(T.MIN_HISTORY - 1, len(close))))
        np.testing.assert_allclose(model[:-1], engine, atol=1e-12)

    def test_the_paper_ledger_is_the_model_when_fills_are_closes_and_funding_is_flat(self):
        close = closes(200, 10, seed=5)
        state, rows = P.fresh_state(), []
        first = T.MIN_HISTORY
        for i in range(first, len(close)):
            mids = {c: float(close[i, j]) for j, c in enumerate(P.COINS)}
            fund = {c: (T.FUND_LONG_BPS / 1e4 if w > 0 else 0.0)
                    for c, w in state["held"].items()}
            state, row = P.step(state, now_for(i, 0), close[: i + 1], mids, fund)
            rows.append(row)
        model = P.replay(close, list(range(first, len(close))))
        paper = sum(r["pnl_price"] - r["funding"] for r in rows) - sum(r["cost"] for r in rows[:-1])
        self.assertAlmostEqual(paper, float(model[:-1].sum()), places=10)
        st = P.status(rows, now_for(len(close) - 1, 0) // DAY * DAY, model)
        self.assertAlmostEqual(st["tracking_bps_day"], 0.0, places=6)

    def test_a_short_receives_positive_funding_and_a_long_pays_it(self):
        close = closes(130, 10, seed=7)
        mids = {c: float(close[-1, j]) for j, c in enumerate(P.COINS)}
        base = {"equity": 1.0, "peak": 1.0, "marks": mids, "last_day": "2026-05-09",
                "first_day": "2026-05-01", "funding_through_ms": 0}
        rate = {c: 0.001 for c in P.COINS}
        _, long_row = P.step({**base, "held": {"BTC": 0.1}}, now_for(129), close, mids, rate)
        _, short_row = P.step({**base, "held": {"BTC": -0.1}}, now_for(129), close, mids, rate)
        self.assertAlmostEqual(long_row["funding"], 0.0001)
        self.assertAlmostEqual(short_row["funding"], -0.0001)


class ThePass(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.close = closes(260, 10, seed=11)
        self.client = FakeClient(self.close)

    def run_day(self, i, minutes=5, **kw):
        self.client.mid_row = self.close[i]
        return P.run_pass(self.client, now_for(i, minutes), self.dir, **kw)

    def test_it_acts_once_per_utc_day(self):
        self.assertIsNotNone(self.run_day(200))
        self.assertIsNone(self.run_day(200, minutes=65))
        self.assertIsNotNone(self.run_day(201))
        rows = P.read_ledger(self.dir)
        self.assertEqual([r["day"] for r in rows], [T.iso(T0 + 201 * DAY), T.iso(T0 + 202 * DAY)])
        self.assertEqual(rows[0]["decision_close"], T.iso(T0 + 200 * DAY))

    def test_the_first_pass_only_opens_and_pays_its_costs(self):
        row = self.run_day(200)
        self.assertEqual(row["pnl_price"], 0.0)
        self.assertEqual(row["funding"], 0.0)
        self.assertAlmostEqual(row["cost"], row["gross"] * T.COST_BPS / 1e4)
        self.assertAlmostEqual(row["equity_after"], 1.0 - row["cost"])

    def test_the_forming_bar_is_never_read_as_a_close(self):
        row = self.run_day(200)
        j = P.COINS.index("BTC")
        t = next(x for x in row["trades"] if x["coin"] == "BTC")
        self.assertEqual(t["close"], float(self.close[200, j]))

    def test_an_unreadable_venue_changes_nothing(self):
        self.run_day(200)
        before = (self.dir / P.STATE_NAME).read_text(), (self.dir / P.LEDGER_NAME).read_text()
        for what in ("candles", "hole", "mids", "funding"):
            self.client.broken = what
            with self.assertRaises(P.Unreadable, msg=what):
                self.run_day(201)
        after = (self.dir / P.STATE_NAME).read_text(), (self.dir / P.LEDGER_NAME).read_text()
        self.assertEqual(before, after)

    def test_yesterdays_bar_must_exist_before_the_pass_acts(self):
        self.client.mid_row = self.close[200]
        late = T0 + 203 * DAY + 60_000              # the venue is two days behind
        self.client.close = self.close[:201]
        with self.assertRaises(P.Unreadable):
            P.run_pass(self.client, late, self.dir)

    def test_a_dry_run_writes_nothing(self):
        self.assertIsNotNone(self.run_day(200, dry_run=True))
        self.assertFalse((self.dir / P.LEDGER_NAME).exists())

    def test_a_missed_day_is_recorded_and_marked_through(self):
        self.run_day(200)
        row = self.run_day(203)
        self.assertEqual(row["missed_days"], 2)
        self.assertNotEqual(row["pnl_price"], 0.0)
        # funding covers the whole gap: 3 days of hourly rates on the held book
        self.assertNotEqual(row["funding"], 0.0)

    def test_a_repeated_day_in_the_ledger_reads_as_its_last_row(self):
        self.run_day(200)
        row = json.loads((self.dir / P.LEDGER_NAME).read_text().splitlines()[0])
        row["equity_after"] = 0.5
        with open(self.dir / P.LEDGER_NAME, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        self.assertEqual(P.read_ledger(self.dir)[0]["equity_after"], 0.5)


class Status(unittest.TestCase):
    def rows(self, equities, gap_at=None):
        out = []
        for k, e in enumerate(equities):
            d = k + (5 if gap_at is not None and k >= gap_at else 0)
            out.append({"day": T.iso(T0 + d * DAY), "equity_after": e, "funding": 0.0001,
                        "pnl_price": 0.0, "cost": 0.0, "held": {}, "gross": 0.0, "net": 0.0,
                        "decision_close": T.iso(T0 + (d - 1) * DAY),
                        "trades": [{"adverse_bps": 4.0}, {"adverse_bps": 8.0}]})
        return out

    def test_drawdown_is_from_the_peak_including_the_start(self):
        st = P.status(self.rows([0.9, 1.2, 0.84]), T0 + 2 * DAY)
        self.assertAlmostEqual(st["max_drawdown_pct"], -30.0)
        self.assertTrue(st["stop"]["S1_drawdown"])
        self.assertFalse(P.status(self.rows([0.9, 1.0]), T0 + DAY)["stop"]["S1_drawdown"])

    def test_gaps_drift_and_funding_are_measured(self):
        st = P.status(self.rows([1.0] * 12, gap_at=6), T0 + 16 * DAY)
        self.assertEqual(st["longest_gap_days"], 5)
        self.assertAlmostEqual(st["mean_adverse_bps"], 6.0)
        self.assertAlmostEqual(st["funding_pct_year"], 12 * 0.0001 / 16 * 365 * 100)
        self.assertTrue(st["go"]["G2_drift"])
        self.assertFalse(st["go"]["G1_rows"])

    def test_an_empty_ledger_has_not_started(self):
        self.assertEqual(P.status([], T0), {"started": False})


if __name__ == "__main__":
    unittest.main()
