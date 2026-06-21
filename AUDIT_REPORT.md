# Hyperliquid Momentum Bot Audit

Date: 2026-06-16
Repo: `/Users/myair/Documents/PROJECTS 2025/Sol Flow/Cuan_Sniffer`

This audit covers the active perps path I found in this repo: `agent.py`, `perp_data.py`, `perp_sentiment.py`, `signal_engine.py`, `executor.py`, `risk_manager.py`, execution backends, trade logging, and the available CSV/log artifacts. I did not find an active `bot.py` or `bot_v4_2.py`.

External validation used:

- Hyperliquid info endpoint docs: `candleSnapshot` exists, supports the listed intervals, and only the most recent 5000 candles are available: <https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/info-endpoint>
- Live Hyperliquid REST sample via `curl`: `candleSnapshot` returns both `t` and `T`; `t` is candle open time and `T` is candle close/end time minus 1 ms. A request made at `2026-06-16T01:43:43Z` returned a 1h candle with `t=1781571600000` and `T=1781575199999`, so the endpoint includes the currently forming 1h candle.
- Hyperliquid funding docs: funding is hourly; positive funding means longs pay shorts; payment is `position_size * oracle_price * funding_rate`: <https://hyperliquid.gitbook.io/hyperliquid-docs/trading/funding>
- Hyperliquid fee docs: base perps taker fee is 0.045% and maker is 0.015%: <https://hyperliquid.gitbook.io/hyperliquid-docs/trading/fees>
- Hyperliquid margining/liquidation docs: leverage and required margin are venue-level concepts, and liquidation occurs when account equity falls below maintenance: <https://hyperliquid.gitbook.io/hyperliquid-docs/trading/margining>, <https://hyperliquid.gitbook.io/hyperliquid-docs/trading/liquidations>
- CCXT manual: the last/current OHLCV candle can be incomplete until the next candle starts, and OHLCV is slower secondary data: <https://github.com/ccxt/ccxt/wiki/Manual#ohlcv-candlestick-charts>
- Anchored VWAP reference: anchored VWAP should begin from a chosen anchor point/event, not merely reset by daily/session clock: <https://trendspider.com/learning-center/anchored-vwap-trading-strategies/>

## 1. Executive Verdict

Blunt verdict: the bot is not structurally ready for live trading. It is safe enough to continue controlled paper/testnet investigation after fixing the candle and accounting issues, but the current live data does not show edge.

- Structurally sound: partially. There is real engineering around execution, native protection, stale entry checks, position state, and logging. But several strategy claims do not match implementation, and there are serious candle timestamp/partial-candle alignment issues.
- Likely profitable: not proven; current evidence argues no. `trades.csv` has 107 closed trades from 2026-04-17 to 2026-06-09, 34.6% win rate, -37.99R total, average -0.355R, profit factor 0.484. That is not a small sample of "looks fine."
- Safe to paper trade: yes only if paper/testnet means research mode, not performance belief. Paper mode lacks real funding accrual and the current analytics environment cannot run the shipped analyzers without installing requirements.
- Safe to live trade: no. Disable live until at least candle timestamps, incomplete-candle handling for all regime layers, funding/PnL accounting, leverage/margin semantics, and reporting are fixed.
- Biggest failure mode: the strategy is not the strategy it claims to be. The current agent feeds 1h candles, labels the main signal as 1h/swing, hard-blocks continuation by default, lacks prior-day high/low AVWAP, and uses raw/forming candles for HTF/macro/daily context.

## 2. System Map

Actual pipeline:

1. `agent.py` starts one `CoinState` per tracked coin.
2. Each state uses `PerpDataFeed(coin, interval="1h", max_candles=400)` at `agent.py:573`. This is not a 15m live feed.
3. `PerpDataFeed` calls Hyperliquid `POST /info` with `type="candleSnapshot"` in `perp_data.py:265-272`, parses candle timestamps with `c.get("T") or c.get("t")` in `perp_data.py:292-300`, and caches/fallbacks snapshots in `perp_data.py:231-287`.
4. `agent.py` calls `generate_signal(df, ...)` and passes `tf_label="1h"` at `agent.py:670-682`. The variable name `sig_15m` is stale.
5. If swing mode is enabled, `agent.py` also calls `generate_swing_signal` for `1h,4h` at `agent.py:687-712`.
6. `_execute_signal` mutates metadata to `timeframe=tf_label` and sets `execution_track="swing"` for `1h` and `4h` at `agent.py:731-735`.
7. `signal_engine.py` builds features from closed candles by dropping the last row at `signal_engine.py:193-265`, then scores continuation/reversal/fallback setups.
8. HTF, macro, and daily zone are computed from the full raw feed in `signal_engine.py:2360-2362`, not from the closed-only feature frame.
9. `executor.py` applies hard strategy gates, cooldowns, risk sizing, score sizing overlay, available-margin cap, live entry validation, backend execution, and protection orders.
10. Position management runs in `_evaluate_live_position`: stop first, then TP/partial/BE/trailing depending on mode at `executor.py:2645-2754`.
11. `trade_log.py` upserts `trades.csv` by `position_id` at `trade_log.py:231` and appends event rows to `smc_live_log.csv`.

## 3. Gate-by-Gate Audit

| Gate | Intended behavior | Actual behavior | Status | Issue / fix |
|---|---|---:|---|---|
| Higher-timeframe regime | Trade only with validated HTF/macro context | 4h and daily regimes are resampled from full raw 1h feed at `signal_engine.py:459-602`; main call uses raw df at `signal_engine.py:2360-2362` | Fail/partial | Uses forming 1h/current 4h/current day data. Compute HTF/macro from closed-only source and drop incomplete resampled bars. |
| Breakout filter | Momentum breakout filter | Continuation uses SMC triggers and trend alignment, not a strict lookback breakout. Executor hard-blocks continuation by default at `executor.py:743-747` | Fail/spec mismatch | Define whether the strategy is SMC confluence or momentum breakout. If breakout, add explicit `BREAKOUT_LOOKBACK` and false-breakout tests. |
| RSI filter | Enforce momentum or reversal RSI bounds | RSI is a soft score only, not a hard gate | Pass if intended soft, fail if intended hard | Make it explicit in config/tests. |
| Volume filter | Require expansion / reject dead volume | `vol_spike` adds +0.08, `vol_collapse` -0.06 at `signal_engine.py:1189-1200` | Partial | No hard `VOL_RATIO_MIN`; validate whether soft scoring ranks outcomes. |
| ATR% volatility gate | Reject too-low/too-high vol | `_compute_vol_state` is mostly soft; low vol gets a score penalty at `signal_engine.py:2505-2512` | Partial/fail | Implement explicit `ATR_PCT_MIN/MAX` if this is supposed to be a gate. |
| AVWAP proximity | Prior UTC day high/low anchored VWAP proximity | Code implements session/daily/weekly reset VWAP in `features.py:69-143` and scores deviation in `signal_engine.py:1202-1268` | Fail | No prior-day high/low anchors. Add true anchored series from prior UTC high/low event candles and log distance. |
| Funding penalty | Penalize longs when positive funding; shorts when negative | Direction is correct: positive funding subtracts score, then short receives inverse in callers; Hyperliquid docs confirm positive funding means longs pay shorts | Partial | Funding thresholds may be too high for hourly rates, and actual funding PnL is not accounted. Add hourly funding accrual. |
| Sentiment penalty | Penalize crowded/weak sentiment | `perp_sentiment.py:249-330` extracts funding/OI/premium and creates a simple bias; OI deltas need warmup | Partial | Cold starts produce zero OI deltas; no proven calibration. Persist OI history and add tests. |
| Loss-streak score bump | Increase/reduce score after streak | Actual code applies a 3-loss continuation penalty at `signal_engine.py:1556-1603` | Spec mismatch | Rename config/docs. A loss bump would be martingale-ish; keep penalty, but test it. |
| Score-based sizing | Larger size for higher-quality setups | Risk manager sizes by confidence; executor then multiplies risk/size again at `executor.py:1019-1040` | High risk | This increases account risk, not just margin. Cap final risk per trade after all overlays. |
| Cooldowns/dedup | Prevent duplicate entries | Agent duplicate/rejected setup cooldown, executor reject memory, pending/open coin checks | Mostly pass | Open-position short circuit at `agent.py:741-745` is unlogged, so skipped signal data undercounts this reject family. |
| Daily drawdown stop | Halt after daily loss | `risk_manager.py:250-253` halts on future entries only | Partial | Does not de-risk existing positions and can be exceeded by simultaneous open positions/slippage. |
| HWM halt | Halt after max drawdown | `risk_manager.py:255-262` halts on future entries only | Partial | Not a circuit breaker for open risk. Add portfolio-level exposure reduction policy. |
| TP1 partial | Close partial at +R | Partial path exists at `executor.py:2725-2737`; `trades.csv` upsert avoids duplicate trade rows | Pass/partial | Needs tests for accounting and fill ratios. |
| Breakeven move | Move stop to entry after TP1 or full mode BE | Implemented in full mode at `executor.py:2680-2705`; partial close also moves BE in `_live_take_partial` | Pass | Test native stop replacement failure path. |
| Trailing runner | Trail after partial | In `ENABLE_PARTIAL_TP` mode, runner holds static BE stop; no trailing at `executor.py:2750-2753` | Fail/spec mismatch | Either document static runner or add trailing in partial mode. |
| Time stop | Exit after max bars/time | No active max-hold/time-stop execution path found | Fail | Add `TIME_STOP_BARS`/`MAX_HOLD_SEC`, tests, and explicit close reason. |
| Closed-candle handling | Never trade on forming candle | Direct feature frame drops last row at `signal_engine.py:212-213` | Partial/fail | HTF/macro/daily use raw data; feed labels candles by `T` and includes forming candle. Fix timestamp and resampled contexts. |

## 4. Data Integrity Audit

Reliable enough:

- `trades.csv`: 107 closed rows, all current upsert schema. It is usable for live result direction, but includes changing code/config and manual/system exits. It says the system has lost -37.99R.
- `orders.csv`: useful for fill/order-state audit. Entry raw fill/request bps is noisy because it is not side-normalized in my quick pass, but order states show many failed/cancelled/reconciled states worth deeper review.
- `smc_live_log.csv`: useful for current-ish engine/executor frequencies from 2026-05-14 to 2026-06-16, but not sufficient for historical expectancy because it mixes versions and event types.

Contaminated or limited:

- `signals.csv`: 16,249 rows from 2026-04-10 to 2026-06-16. It mixes 15m, 1h, and 4h eras and many changing reject reasons. Use only after adding `code_version`, config hash, and git commit to every row.
- `missed_signals.csv`: 13,195 rows from 2026-04-17 to 2026-06-16. It records `price_move_r`, which is a directional proxy at logging time, not a TP/SL outcome. It is not a missed-signal win rate.
- `signals_evaluated.csv`: 452 rows from 2026-03-20 to 2026-04-03. This predates much of the current execution logic, and the nonzero outcomes are concentrated in rows with blank `executor_result`. Not trustworthy for current taken-vs-skipped comparisons.
- `archive/*` and `reset_backup/*`: useful for forensics, not for current edge.

Candle integrity:

- Hyperliquid returns forming candles. Verified live.
- The code parses `T` before `t`, so candle time is close/end time rather than open/start time (`perp_data.py:295`). This can shift session labels, resampling boundaries, and daily calculations.
- The last row is dropped for feature generation, but HTF/macro/daily context uses raw data, so the bot can still be influenced by incomplete candles.
- Cache fallback in `perp_data.py:277-287` can serve stale snapshots after fetch failure. Executor live-mid validation helps, but stale engine outputs still pollute signal/missed logs.

Data to discard/separate:

- Do not blend pre-2026-05-14 SMC logs with later logs.
- Do not use `signals_evaluated.csv` as proof of current edge.
- Split `trades.csv` by code/config regimes: at least before/after continuation hard-block, 4h blocking, full-TP vs partial mode, and any reset date.
- Add `git_commit`, `config_hash`, `code_schema_version`, `paper/live`, and `feed_interval` to all future signals/trades.

## 5. Profitability Analysis Using Available Data

I ran stdlib CSV analysis because the active Python environment does not have `pandas`, `requests`, `numpy`, or `matplotlib`, even though they are listed in `requirements.txt`.

Closed trades:

- Trades: 107 closed, all `paper_mode=false`.
- Period: 2026-04-17 to 2026-06-09.
- Win rate: 34.6%.
- Total R: -37.99R.
- Average R: -0.355R.
- Profit factor: 0.484.
- Max drawdown on CSV order: -37.99R.
- Fees logged: $19.79 total, $0.185 average per trade.
- Avg hold: 393 minutes; median 140 minutes; max 5909 minutes.

By timeframe:

- 15m: 47 trades, 38.3% WR, -18.25R, avg -0.388R.
- 1h: 43 trades, 44.2% WR, -8.70R, avg -0.202R.
- 4h: 17 trades, 0% WR, -11.04R. Blocking 4h is justified by available live data.

By setup:

- Swing: 53 trades, -13.24R, avg -0.250R.
- Continuation: 42 trades, -19.77R, avg -0.471R.
- Reversal: 12 trades, -4.98R, avg -0.415R.

Partial/runner:

- `partial_closed=true`: 42 trades, 85.7% WR, +30.95R, avg +0.737R.
- `partial_closed=false`: 65 trades, 1.5% WR, -68.94R, avg -1.061R.
- Interpretation: TP1 is a strong survival line, but the entry filter is poor. The bot is taking too many trades that never reach +1R.

Missed signals proxy:

- `missed_signals.csv` overall `price_move_r`: n=13,195, mean +0.155R, 55.3% positive. This is not a win rate.
- Interesting reject proxies: `market_regime_block:chop` n=342, mean +0.684R; `weak_trend_conf_gate:conf=0.67<0.80` n=227, mean +1.667R; `hard_blocked_setup_family:continuation` n=315, mean -0.376R.
- `suggested_params.json` recent 7-day sample found no actionable suggestions; it also warns that `price_move_r` is only a proxy.

Conclusion from available data: taken trades are not demonstrably higher quality than skipped trades. The only honest next step is a clean replay with current code, versioned logs, realistic costs, and proper candle handling.

## 6. Backtest / Walk-Forward Review

Existing analyzers:

- `tools/research/analyze_winrate.py` evaluates `signals.csv` against Hyperliquid 1m candles and supports slippage, walk-forward split, and autocorrelation. Weaknesses: depends on missing packages here; uses `T` timestamp; does not model exit fees, stop exit slippage, funding, partial TP, BE, trailing, native-order latency, executor gate ordering, or duplicate suppression exactly.
- `tools/research/backtest_missed.py` evaluates missed signals against 5m candles. Weaknesses: no fees/slippage/funding; no partial/runner/BE; no execution delay; only 5m bar path with stop priority; dedup key is price-based; `--interval 1h` breaks because it assumes minute suffix at `tools/research/backtest_missed.py:334`.

Exact walk-forward implementation needed:

1. Freeze current code and config with git commit/config hash.
2. Build a deterministic replay harness that calls the real `PerpDataFeed` parser, `AdaptiveSignalEngine`, `Executor` gates in dry-run mode, and the same position manager.
3. Use only closed candles: `t` as open time, drop any candle with `T >= now_ms`, and drop incomplete resampled 4h/daily bars.
4. Use event-based fills: entry delay 1-2 polling cycles, taker fee 4.5 bps, side-aware slippage from order book or conservative bps model, stop gap/slippage, hourly funding, and protection-order failure modeling.
5. Split by time, not random rows: rolling windows such as 21d train / 7d test, then 28d / 7d, with purging around split boundaries.
6. Report parameter stability zones, not best point estimates.

Parameter families to test:

- `BREAKOUT_LOOKBACK`: 12, 24, 48 bars.
- `RSI_LONG_MIN` / `RSI_SHORT_MAX`: trend-following bands such as 50/50, 55/45, 60/40.
- `VOL_RATIO_MIN`: volume ratio or volume spike tiers; test hard vs soft.
- `ATR_PCT_MIN/MAX`: e.g. 0.3%-0.8% min and 2%-6% max depending symbol.
- `BASE_MIN_SCORE_TO_TRADE`: 0.64-0.78, but only after score calibration.
- `STOP_ATR`: 0.8, 1.0, 1.3, 1.6, 2.0.
- `TP1_ATR` / `PARTIAL_TP_R`: 0.8R, 1.0R, 1.2R.
- `TRAIL_ATR`: 0.8, 1.0, 1.5, 2.0.
- `TIME_STOP_BARS`: 3, 6, 12, 24 bars depending timeframe.

Minimum proof threshold:

- OOS expectancy after costs > +0.10R/trade.
- Profit factor > 1.25 OOS.
- No single symbol/session/regime contributes more than 50% of total positive R.
- Degradation from IS to OOS < 40%.
- Max drawdown compatible with account size after 20x liquidation assumptions.

## 7. Code Bugs and Logic Bugs

Critical:

1. Candle timestamp bug: `perp_data.py:295` uses `T` before `t`, labeling candles by close/end time. Fix to use `t` for candle index and keep `T` as `close_time`.
2. Partial-candle leakage into HTF/macro/daily context: `signal_engine.py:2360-2362` passes raw df into context functions. Fix to pass closed-only df and explicitly drop incomplete resampled bars.
3. Strategy mismatch: active feed is 1h (`agent.py:573`), main generated signal is labeled 1h (`agent.py:681`), and executor marks it swing (`agent.py:735`). The claimed 15m momentum bot is not what is running.
4. AVWAP prior-day high/low logic is absent. `features.py:69-143` is session/daily/weekly VWAP, not anchored to prior UTC high/low.
5. Live funding accounting is zero: `live_execution_backend.py:592-600` always returns zero funding. Actual Hyperliquid funding is hourly.

High:

1. Executor hard-blocks continuation by default at `executor.py:743-747`; the bot cannot live-trade the core momentum/continuation thesis unless config overrides it.
2. Available-margin sizing treats requested notional as required margin at `executor.py:3093-3095`. Hyperliquid margin is notional/leverage. This is conservative for notional cap but does not actually manage venue leverage/liquidation risk.
3. Score sizing overlay increases `decision.risk_usd` after risk manager approval at `executor.py:1019-1040`. Final risk can exceed intended per-trade risk.
4. Reporting double-counts fees in `trades_recap.py:74-78`: `realized_r` already includes exit fees in execution paths, then recap subtracts `total_fees_usd/r_value` again. `pnl_usd` also does not clearly include entry fees.
5. No time stop, despite `CloseReason.SYSTEM_EXIT_TIMEOUT` existing in `position.py`.
6. Partial mode has no ATR runner trail (`executor.py:2750-2753`), despite the requested gate list.
7. Full-loss accounting clamps losses to `MAX_FULL_LOSS_R` in `risk_manager.py:351-364` and `risk_manager.py:372-400`, which can understate tail losses from gaps/liquidations.

Medium:

1. `generate_signal` checks `abs(chosen_score) < threshold` at `signal_engine.py:2537`. If a future negative score somehow reaches selection, absolute value could pass it.
2. `market_regime` gate in executor can conflate market, HTF, and macro chop through `_signal_market_regime`; this may over-block setups.
3. `signals_evaluated.csv` has stale rows with previous `StrategyFilter.is_allowed()` argument errors, proving past schema/code drift.
4. `PerpDataFeed` can fail-soft to stale cached candles after network failure. Add max stale age and log `snapshot_age_sec` into every signal.
5. `tools/research/backtest_missed.py` assumes interval suffix `m`.

Low:

1. Misleading names/comments: `sig_15m` in `agent.py:670` is a 1h signal.
2. `.env.example` documents partial mode, but runtime defaults in code differ unless env is loaded.
3. No test files were found.

## 8. Risk/Ruin Audit

For a $100 account at 20x leverage, the live danger is not only nominal R. It is simultaneous exposure, liquidation distance, stop failure, and accounting optimism.

- Baseline `RISK_PCT_PER_TRADE` is 1%. Confidence sizing can lift that to 1.25R, and executor score overlay can multiply again up to 1.5x. That is up to 1.875% account risk before weak-trend reductions and margin caps.
- `MAX_OPEN_POSITIONS` defaults to 4. If multiple positions open before daily halt triggers, intended exposure can exceed the daily 3R stop.
- Daily/HWM halts only block future entries, not existing positions.
- The code does not set Hyperliquid leverage or verify current asset leverage before entries.
- The code does not compute liquidation price or reject trades whose stop is too near liquidation.
- Native protection exists and many rows show `protected`, but software/accounting assumes stops close as modeled. Gap/slippage beyond `MAX_FULL_LOSS_R` is clamped in risk accounting.
- Funding is ignored in live PnL, material for multi-hour holds. Median hold is 140 minutes and mean hold is 393 minutes, so hourly funding matters.
- Live trade log shows 107 closed live rows and -37.99R. On a $100 account, that kind of drawdown would likely have ended the experiment unless size was much smaller than the current risk model.

Stop live trading until:

- final post-overlay risk is capped,
- current leverage is fetched/set and logged,
- liquidation distance is checked,
- daily/HWM circuit breakers include open exposure,
- funding and tail slippage are represented in PnL.

## 9. Recommended Fixes

Must fix before more paper testing:

1. Change candle parsing to store both `open_time=t` and `close_time=T`; use `open_time` as the DataFrame timestamp.
2. Drop forming candles by comparing `close_time < now_ms`, not blindly `iloc[:-1]`.
3. Build HTF/macro/daily context from closed candles only and drop incomplete 4h/daily resamples.
4. Add `code_version`, `git_commit`, `config_hash`, `feed_interval`, and `candle_closed_at` to signal/trade logs.
5. Fix `trades_recap.py` net fee math and separate gross R, exit-fee R, entry-fee R, funding R, and net R.
6. Install/use `requirements.txt` in the run environment so analysis scripts are actually runnable.

Must fix before live mode:

1. Disable score overlay from increasing risk, or cap final risk after all overlays.
2. Fetch/set Hyperliquid leverage per coin and calculate liquidation distance.
3. Add real hourly funding accrual for paper and live.
4. Add true time stop.
5. Add partial-mode runner trail or document that partial runner is static BE-to-TP.
6. Remove/justify `MAX_FULL_LOSS_R` clamping for reporting; never hide tail losses.
7. Decide strategy identity: momentum breakout or SMC swing/reversal. Remove contradictory gates.

Nice-to-have:

1. Add true prior-day high/low AVWAP anchors.
2. Persist OI history across restarts.
3. Add order-book spread/depth snapshot at signal time.
4. Add a clean data mart: one row per candidate, one row per order, one row per position, all with version/config hashes.

## 10. Test Suite

No existing pytest/test suite was found. Add tests before another live run.

Unit tests:

- `PerpDataFeed` parses `t` as `time`, stores `T` as `close_time`, and rejects/drop-forming candles.
- A live-style candle list with current forming last candle produces features from the previous closed candle.
- HTF 4h resample drops incomplete 4h bar.
- Daily macro/daily zone excludes the current incomplete UTC day.
- Prior-day high/low AVWAP starts exactly at the selected anchor event candle.
- Funding score direction: positive funding penalizes LONG and benefits SHORT; negative funding does opposite.
- RSI/volume/ATR gates behave as hard or soft according to explicit config.
- Score sizing overlay cannot exceed final max risk.
- Cooldown/dedup prevents duplicate entries but logs the skip reason.

Integration tests:

- Full signal to executor dry-run with mocked candles and mocked mid price.
- Stale entry rejection when live mid has moved beyond ATR buffer.
- TP1 partial close records one trade row, not duplicate trades.
- Runner close after partial updates final R exactly once.
- BE native stop replacement failure raises protection status and does not pretend safety.
- Daily drawdown halt rejects new entries after threshold.
- HWM halt rejects new entries after threshold.
- Time stop closes after configured bars.
- No duplicate open position per coin under concurrent signals.

Data tests:

- `signals.csv` schema includes version/config fields.
- `trades.csv` net R equals gross R minus entry fee, exit fees, and funding exactly once.
- Backtest replay output matches executor logs on a small fixed fixture.

## 11. Final 48-Hour Action Plan

First 12 hours:

1. Freeze branch and copy `.env` to a redacted config hash.
2. Fix candle parser: use `t`, preserve `T`, drop incomplete by `T < now_ms`.
3. Fix HTF/macro/daily to use closed-only data.
4. Add tests for candle closure and resampling.

Next 12 hours:

1. Fix trade accounting and recap net R.
2. Add funding accrual.
3. Add final risk cap after all score overlays.
4. Add leverage/liquidation checks or force paper/testnet only.

Next 24 hours:

1. Run clean current-code replay from 2026-05-14 onward using the repaired candle pipeline.
2. Re-run missed-signal simulation with current code/config only and realistic costs.
3. Compare taken vs skipped by coin, side, session, timeframe, setup family, market regime, funding bucket, VWAP/AVWAP distance, volume ratio, ATR% bucket, and false-breakout frequency.
4. Decide continue/stop:
   - Continue only if OOS net expectancy > +0.10R, PF > 1.25, max DD tolerable for $100 account, and no single regime/symbol dominates.
   - Stop if repaired replay remains negative, if edge vanishes after costs/funding, or if profitability depends on one narrow parameter peak.

Current conclusion: keep researching in paper/testnet after fixes. Do not run live capital on the current implementation.
