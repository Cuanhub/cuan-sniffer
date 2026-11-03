"""research/kronos inputs and scoring, and the Kronos experiment's analysis --
all without PyTorch.

The forecaster is pluggable, so the analysis is tested with two stand-ins:
an ORACLE that is handed the real future (the analysis must find it, or it
could not find a real effect either) and NOISE that knows nothing (it must
not find that).
"""

import json
import math
from datetime import datetime, timezone

import numpy as np
import pytest

from research.backtest import band
from research.backtest import common as C
from research.backtest.costs import MEASURED
from research.backtest.engine import LIMIT, MARKET_CLOSE, Order, simulate
from research.backtest.experiments import kronos_filter as X
from research.kronos import inputs as I
from research.kronos import score as K

H = X.HORIZON


def bars_of(n, start=1_782_000_000_000):
    return [{"t": start + k * I.HOUR_MS, "o": 100.0 + k, "h": 101.0 + k, "l": 99.0 + k,
             "c": 100.5 + k, "v": 10.0 + k} for k in range(n)]


# ------------------------------------------------------------------ inputs --
def test_the_window_ends_at_the_decision_bar():
    bars = bars_of(1000)
    w = I.window(bars, 600, 464)
    assert len(w) == 464 and w[-1] is bars[600] and w[0] is bars[137]


def test_the_window_refuses_short_history_and_bars_past_the_end():
    bars = bars_of(1000)
    with pytest.raises(ValueError):
        I.window(bars, 400, 464)
    with pytest.raises(IndexError):
        I.window(bars, 1000, 464)


def test_amount_is_upstreams_fill_in_volume_times_mean_price():
    x = I.features(bars_of(3))
    b = bars_of(3)[2]
    assert x.shape == (3, 6)
    assert x[2, 5] == pytest.approx(b["v"] * (b["o"] + b["h"] + b["l"] + b["c"]) / 4)


def test_normalisation_is_upstreams():
    x = I.features(bars_of(50))
    xn, m, s = I.normalise(x)
    np.testing.assert_allclose(xn, np.clip((x - x.mean(0)) / (x.std(0) + 1e-5), -5, 5))
    spike = x.copy()
    spike[-1, 3] = 1e9
    assert I.normalise(spike)[0].max() == 5.0


def test_stamps_are_the_utc_fields_of_the_open_time():
    t = int(datetime(2026, 10, 27, 6, 0, tzinfo=timezone.utc).timestamp() * 1000)
    assert list(I.stamps([t])[0]) == [0, 6, 1, 27, 10]        # a Tuesday


def test_future_times_are_hourly_after_the_last_bar():
    assert I.future_times(1000, 3) == [1000 + I.HOUR_MS, 1000 + 2 * I.HOUR_MS,
                                       1000 + 3 * I.HOUR_MS]


# ------------------------------------------------------------------- score --
def path_of(rows):
    """(H, 6) from (o, h, l, c) rows, volume columns zero."""
    return np.array([[*r, 0.0, 0.0] for r in rows], dtype=float)


def test_repair_makes_each_candle_self_consistent():
    p = K.repair(path_of([(100, 99, 101, 102)]))       # high below close, low above open
    o, h, l, c = p[0]
    assert h == 102 and l == 100 and o == 100 and c == 102


def test_repair_drops_an_impossible_path():
    assert K.repair(path_of([(100, 101, 99, float("nan"))])) is None
    assert K.repair(path_of([(100, 101, -1, 100)])) is None


def test_direction_reads_the_close_step_bars_after_the_decision():
    paths = [K.repair(path_of([(100, 101, 99, 100 + k) for k in range(1, 30)]))] * 2
    assert K.direction(paths, 100.0, 24) == pytest.approx(0.24)


DEC = {"t": 0, "o": 100.0, "h": 100.5, "l": 99.5, "c": 100.0}


def test_a_market_trade_that_reaches_its_target_scores_the_engines_net_r():
    o = Order("X", 0, MARKET_CLOSE, 98.0, 104.0)
    path = K.path_bars(path_of([(100, 101, 99.5, 100.5), (100.5, 105, 100, 104.5)]), 0)
    want = simulate([DEC] + path, o, MEASURED)
    assert want.outcome == "tp"
    assert K.path_r(o, DEC, path) == pytest.approx(want.r_net)


def test_a_bid_the_path_never_reaches_scores_zero():
    o = Order("X", 0, LIMIT, 93.0, 110.0, limit=95.0, ttl=12)
    path = K.path_bars(path_of([(100, 101, 99, 100)] * 20), 0)
    assert K.path_r(o, DEC, path) == 0.0


def test_a_trade_open_at_the_paths_end_is_marked_to_market_and_pays_the_exit():
    o = Order("X", 0, MARKET_CLOSE, 90.0, 120.0)
    path = K.path_bars(path_of([(100, 101, 99, 101)] * 5), 0)
    tr = simulate([DEC] + path, o, MEASURED)
    assert tr.outcome == "open"
    exit_fee_r = MEASURED.taker_fee_pct / 100 * 101 / (100 - 90)
    assert K.path_r(o, DEC, path) == pytest.approx(tr.r_net - exit_fee_r)
    assert K.path_r(o, DEC, path) < tr.r_gross - tr.cost_r + 1e-12


def test_spearman():
    assert K.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert K.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert math.isnan(K.spearman([1, 1, 1, 1], [1, 2, 3, 4]))


# ---------------------------------------------------------------- analysis --
def test_the_sign_rule_falls_back_to_the_median_when_a_side_is_thin():
    g, mode = X.assign([1.0] * 60 + [-1.0] * 60)
    assert mode == "sign" and sum(g) == 60
    g, mode = X.assign([1.0] * 110 + [-1.0] * 10)
    assert mode == "median"


@pytest.fixture(scope="module")
def walk():
    """Random-walk bars and the strategy's resolved trades on them."""
    bars = C.random_walks(41, coins=8, n=2600)
    trades = X.ordered([t for c in sorted(bars)
                        for t in band.run(c, bars[c], MEASURED, min_rr=X.MIN_RR) if t.resolved])
    assert len(trades) >= 60, len(trades)
    return bars, trades


def oracle(bars_by, decisions, seed):
    """The REAL next H bars, as every sampled path. Cheating on purpose."""
    out = np.full((len(decisions), 4, H, 6), np.nan)
    for r, (c, i) in enumerate(decisions):
        fut = bars_by[c][i + 1:i + 1 + H]
        if len(fut) == H:
            out[r, :] = np.array([[b["o"], b["h"], b["l"], b["c"], 0, 0] for b in fut])
    return out


def noise(bars_by, decisions, seed):
    """Paths that know nothing: a random walk from the decision close."""
    rng = np.random.default_rng(seed)
    out = np.empty((len(decisions), 4, H, 6))
    for r, (c, i) in enumerate(decisions):
        c0 = bars_by[c][i]["c"]
        steps = np.exp(np.cumsum(rng.normal(0, 0.01, (4, H)), axis=1)) * c0
        out[r, :, :, 0] = out[r, :, :, 3] = steps
        out[r, :, :, 1] = steps * 1.004
        out[r, :, :, 2] = steps * 0.996
        out[r, :, :, 4:] = 0
    return out


def run_with(walk, fn, monkeypatch):
    bars, trades = walk
    monkeypatch.setattr(X, "MIN_VALID", 4)
    sc = X.score_trades(bars, trades, fn, seed_base=1)
    return X.analyse(trades, sc, {}, B=500)["hypotheses"]


def test_an_oracle_forecaster_is_found(walk, monkeypatch):
    h = run_with(walk, oracle, monkeypatch)
    k2 = h["K2_expected_r"]["raw"]
    assert k2["estimate"] > 0.8 and k2["p"] < 0.001, k2
    assert h["K1_direction"]["raw"]["estimate"] > 0.2


def test_a_forecaster_that_knows_nothing_is_not_found(walk, monkeypatch):
    """Not significant, and well short of the oracle's ~1.5R. Only a smoke
    check on ~100 trades: the null calibration measures bias properly."""
    h = run_with(walk, noise, monkeypatch)
    for name in X.HYPOTHESES:
        st = h[name]["raw"]
        assert st["p"] > 0.05 and abs(st["estimate"]) < 1.0, (name, st)


def test_a_resumed_run_draws_exactly_what_an_uninterrupted_one_does(walk, tmp_path, monkeypatch):
    bars, trades = walk
    monkeypatch.setattr(X, "MIN_VALID", 4)
    calls = []

    def counted(b, decisions, seed):
        calls.append(seed)
        return noise(b, decisions, seed)

    whole = X.score_trades(bars, trades, counted, 7)
    cache = tmp_path / "s.json"
    X.score_trades(bars, trades, counted, 7, cache=cache, cfg={"c": 1})
    saved = json.loads(cache.read_text())
    later = X.ordered(trades)[len(trades) // 2:]
    for t in later:
        del saved["scores"][X.key_of(t)]
    cache.write_text(json.dumps(saved))
    calls.clear()
    resumed = X.score_trades(bars, trades, counted, 7, cache=cache, cfg={"c": 1})
    assert resumed == whole
    assert 0 < len(calls) < math.ceil(len(trades) / X.BATCH)


def test_a_cache_from_another_configuration_is_not_reused(walk, tmp_path, monkeypatch):
    bars, trades = walk
    monkeypatch.setattr(X, "MIN_VALID", 4)
    cache = tmp_path / "s.json"
    X.score_trades(bars, trades, noise, 7, cache=cache, cfg={"c": 1})
    calls = []
    X.score_trades(bars, trades, lambda *a: calls.append(1) or noise(*a), 7,
                   cache=cache, cfg={"c": 2})
    assert len(calls) == math.ceil(len(trades) / X.BATCH)


FRAMINGS = [f"{n}:{s}" for n in X.HYPOTHESES for s in ("raw", "excess")]
CALIBRATED = {f: False for f in FRAMINGS}
NO_WIDEN = {f: 1.0 for f in FRAMINGS}


def stat(estimate=0.2, p=0.001, se_2way=0.05):
    return {"estimate": estimate, "p": p, "se_2way": se_2way, "halves": [0.1, 0.3]}


def hyps(raw, excess=None):
    return {"hypotheses": {n: {"raw": raw, "excess": excess or raw, "approved_mean_net_r": 0.05}
                           for n in X.HYPOTHESES}}


def test_decide_uses_raw_when_it_calibrated_and_excess_when_only_that_did():
    a = hyps(stat())
    miscal = {"K1_direction:raw": False, "K1_direction:excess": False,
              "K2_expected_r:raw": True, "K2_expected_r:excess": False}
    d = X.decide(a, miscal, NO_WIDEN)
    assert d["K1_direction"]["statistic"] == "raw"
    assert d["K2_expected_r"]["statistic"] == "excess"
    assert d["K1_direction"]["deployable"]
    assert d["K1_direction"]["as_first_registered"]["deployable"]
    miscal["K2_expected_r:excess"] = True
    d = X.decide(a, miscal, NO_WIDEN)
    assert d["K2_expected_r"]["statistic"] is None
    assert not d["K2_expected_r"]["as_first_registered"]["information"]


# ---------------------------------------------- PREREG Amendment 1 (spread) --
def test_the_amendment_can_only_take_a_verdict_away():
    """D that clears Holm on the week-clustered SE but not on the two-way SE
    is information as first registered, and not under Amendment 1."""
    d = X.decide(hyps(stat(se_2way=0.2)), CALIBRATED, NO_WIDEN)
    for n in X.HYPOTHESES:
        assert d[n]["as_first_registered"]["deployable"]
        assert not d[n]["criteria"]["holm_rejected_two_way"]
        assert not (d[n]["information"] or d[n]["material"] or d[n]["deployable"])


def test_the_two_way_se_is_widened_by_the_null_spread_and_never_narrowed():
    a = hyps(stat(se_2way=0.08))          # z 2.5: Holm p 0.025 over the two
    assert X.decide(a, CALIBRATED, NO_WIDEN)["K1_direction"]["information"]
    wide = X.decide(a, CALIBRATED, {f: 1.5 for f in FRAMINGS})["K1_direction"]
    assert wide["widened_by"] == 1.5 and not wide["information"]
    narrow = X.decide(a, CALIBRATED, {f: 0.5 for f in FRAMINGS})["K1_direction"]
    assert narrow["widened_by"] == 1.0
    assert narrow["p_holm_two_way"] == X.decide(a, CALIBRATED, NO_WIDEN)["K1_direction"][
        "p_holm_two_way"]


def test_a_statistic_without_a_two_way_se_can_never_pass():
    assert X.p_two_way({"estimate": 0.5, "p": 1e-9, "halves": [0.5, 0.5]}) == 1.0
    assert not X.decide(hyps({"estimate": 0.5, "p": 1e-9, "halves": [0.5, 0.5]}),
                        CALIBRATED, NO_WIDEN)["K1_direction"]["information"]


def coin_luck(seed, coins=30, weeks=20, per=4):
    """A null in which a coin's luck outlasts a week, as a random walk's drift
    does: each coin's results carry its own lasting luck u, while approval
    follows a separate coin trait a, independent of u. D is zero on average
    but swings with which coins happened to be lucky."""
    rng = np.random.default_rng(seed)
    week = 7 * 24 * 3600 * 1000
    v, g, t, c = [], [], [], []
    for k in range(coins):
        u, a = rng.normal(), rng.normal()
        for w in range(weeks):
            for _ in range(per):
                v.append(0.5 * u + rng.normal())
                g.append(int(rng.random() < (0.8 if a > 0 else 0.2)))
                t.append(1_782_000_000_000 + w * week + int(rng.integers(0, week)))
                c.append(f"C{k}")
    return v, g, t, c


def test_the_two_way_se_is_never_below_either_one_way_se():
    v, g, t, c = coin_luck(1)
    st = X.diff_stats(v, g, t, B=400, coins=c)
    assert st["se_2way"] >= max(st["se"], st["se_coin"])
    assert "se_2way" not in X.diff_stats(v, g, t, B=50)


def test_the_two_way_se_is_the_size_of_ds_real_swing_where_the_week_se_is_not():
    """The failure Amendment 1 fixes, on a null built to have it."""
    rows = []
    for s in range(30):
        v, g, t, c = coin_luck(100 + s)
        rows.append(X.diff_stats(v, g, t, B=300, coins=c))
    sd = float(np.std([r["estimate"] for r in rows], ddof=1))
    rms = {k: math.sqrt(np.mean([r[k] ** 2 for r in rows])) for k in ("se", "se_2way")}
    lo, hi = X.chance_band(len(rows))
    assert sd / rms["se"] > hi, (sd, rms)
    assert lo < sd / rms["se_2way"] < hi, (sd, rms)


def test_the_chance_band_is_chi_squares():
    lo, hi = X.chance_band(20)            # chi-square(19): 8.907 and 32.852
    assert lo == pytest.approx(math.sqrt(8.907 / 19), abs=0.005)
    assert hi == pytest.approx(math.sqrt(32.852 / 19), abs=0.005)


def test_the_spread_summary_sets_the_spread_against_the_reported_se():
    est = (-0.1, 0.0, 0.1, 0.2)
    per = [{"K1_direction:raw": {"estimate": e, "se": 0.1, "se_2way": 0.2,
                                 "fires": e > 0.15, "fires_2way": False}} for e in est]
    s = X.spread_summary(per)["K1_direction:raw"]
    sd = float(np.std(est, ddof=1))
    assert s["spread"] == pytest.approx(sd)
    assert (s["ratio_week"], s["ratio_2way"]) == (pytest.approx(sd / 0.1), pytest.approx(sd / 0.2))
    assert (s["datasets"], s["fired"], s["fired_2way"]) == (4, 1, 0)


def test_null_spread_reanalyses_finished_null_datasets_and_waits_for_the_rest(
        tmp_path, monkeypatch):
    nc, nb = 3, 1700
    for name, value in (("NULL_DATASETS", 2), ("NULL_COINS", nc), ("NULL_BARS", nb),
                        ("SCORES", tmp_path / "scores"), ("MIN_VALID", 4)):
        monkeypatch.setattr(X, name, value)
    cfg = {"c": 1}

    def null_trades(k):
        bars = X.null_bars(k, nc, nb)
        return bars, X.ordered([t for c in sorted(bars) for t in X._null_run((k, nc, nb, c))])

    for k in range(2):
        bars, trades = null_trades(k)
        X.score_trades(bars, trades, noise, 5, X.SCORES / f"null_{k:02}.json",
                       {**cfg, "coins": nc, "bars": nb})
    p1 = X.SCORES / "null_01.json"
    whole = p1.read_text()
    partial = json.loads(whole)
    partial["scores"].pop(next(iter(partial["scores"])))
    p1.write_text(json.dumps(partial))
    path = tmp_path / "spread.json"
    out = X.null_spread(cfg, B=100, path=path, workers=2, log=lambda *a: None)
    assert list(out["per_dataset"]) == ["0"] and not out["complete"]

    bars, trades = null_trades(0)
    sc = X.score_trades(bars, trades, noise, 5, X.SCORES / "null_00.json",
                        {**cfg, "coins": nc, "bars": nb})
    excess = dict(X._null_ctrl((0, nc, nb, X.key_of(t), t)) for t in trades)
    a = X.analyse(trades, sc, excess, 100)
    for n in X.HYPOTHESES:
        for s in ("raw", "excess"):
            got = out["per_dataset"]["0"][f"{n}:{s}"]
            assert got["estimate"] == a["hypotheses"][n][s]["estimate"]
            assert got["se_2way"] == a["hypotheses"][n][s]["se_2way"]

    p1.write_text(whole[:len(whole) // 2])          # caught mid-write by --null
    assert not X.null_spread(cfg, B=100, path=path, workers=2, log=lambda *a: None)["complete"]
    p1.write_text(whole)
    out = X.null_spread(cfg, B=100, path=path, workers=2, log=lambda *a: None)
    assert sorted(out["per_dataset"]) == ["0", "1"] and out["complete"]
    assert X.null_spread({"c": 2}, B=100, path=path, workers=2,
                         log=lambda *a: None)["datasets"] == 0
