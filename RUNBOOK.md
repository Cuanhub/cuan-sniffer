# Cuan Sniffer — Operations Runbook

Live operations for the **level trader** (`tools/live_levels_trader.py`), the
only thing in this repo that places orders.

> Everything this file described before 2026-09-18 belonged to the SMC bot
> retired in `b20820e`. Its commands referenced `main.py`, `executor.py` and
> `signal_engine.py`, none of which exist. If you are reading an old copy,
> stop.

---

## Read this before arming anything

The gate this trades measured **−0.0413R against random entry** over 688
filled backfill trades (z = −0.93). That is no edge distinguishable from
throwing darts. Its apparent +203R over the backfill came from a window in
which all 28 universe coins rose; random entries with the identical
stop/TP geometry returned *more*.

It is armed anyway, deliberately, to buy information at roughly $24 a trade.
Every default is chosen to lose slowly and stop early. Do not widen a rail
because it refused a trade you liked.

---

## Start, stop, status

```sh
# status — the state file, no side effects
.venv/bin/python -m tools.live_levels_trader --status

# one dry pass, prints intents, places nothing
.venv/bin/python -m tools.live_levels_trader --once

# install / start under launchd (this is what trades)
mkdir -p ~/Library/Logs/cuan-sniffer
cp deploy/launchd/com.cuansniffer.trader.plist ~/Library/LaunchAgents/
launchctl load -w ~/Library/LaunchAgents/com.cuansniffer.trader.plist

# stop
launchctl unload ~/Library/LaunchAgents/com.cuansniffer.trader.plist

# is it running? PID in column 1, exit code in column 2
launchctl list | grep cuansniffer
```

The `mkdir` is not optional. launchd does not create a missing
`StandardOutPath` directory — it starts the job and drops the output.

A bare `kill` will not stick: `KeepAlive` restarts anything exiting non-zero.
Exit 0 is treated as deliberate, so a clean SIGTERM shutdown stays down.

**After editing the plist**, unload and load again. launchd caches the copy
it read at load time. `deploy/launchd/` is the source of truth;
`~/Library/LaunchAgents/` holds a deployment artifact.

---

## Arming takes two independent gates

```
--arm                    on the command line
LEVELS_LIVE_ARMED=yes    in the environment (the plist sets it)
```

Both are required. Removing either leaves the process running harmlessly in
dry mode rather than stopping it dead, so you can disarm without losing the
scan. Dry mode skips the trigger check entirely — it prints sized intents,
never `wait`/`TRIGGER` lines.

---

## Where things are

| what | where |
|---|---|
| stdout | `~/Library/Logs/cuan-sniffer/trader.log` |
| stderr | `~/Library/Logs/cuan-sniffer/trader.err.log` |
| state | `research/levels/data/_live_state.json` |
| every decision | `research/levels/data/live_trades.csv` |
| labels (tracked in git) | `research/levels/data/labels.csv` |
| rails | `research/levels/config.py` |

Logs are **not rotated** — launchd does not, and neither does the trader.
A repo-root `live_trader.log` is a dead artifact from before the move; the
live output is in `~/Library/Logs`.

---

## Halts

A halt is the one condition the trader cannot leave on its own. `preflight`
refuses to open anything while `halted` is true, so it idles loudly until a
human clears it. Every halt sends a Telegram alert on the transition.

Causes, all of them:

| reason | meaning |
|---|---|
| `venue holds [...] which this module did not open` | a position exists that the trader has no record of |
| `venue holds N positions, cap is 2` | concurrency cap breached at the venue |
| `notional $X exceeds 10x equity` | leverage backstop |
| `filled=True but fill_price=...` | a fill it cannot parse; the position may well exist |
| `... emergency exit did not fill` | protection failed **and** the close was rejected: a position is open and unprotected |
| `daily loss stop: ...` | realised losses breached 15% of equity |

```sh
# see it
.venv/bin/python -m tools.live_levels_trader --status

# clear it — only after you have looked at the venue yourself
.venv/bin/python -m tools.live_levels_trader --resume
```

`--resume` clears the flag and nothing else. It does not close positions, it
does not cancel orders, and it does not check that the cause is gone. If the
cause persists, the next pass halts again.

**The `emergency exit did not fill` halt means go look at Hyperliquid now.**
There is an open position with no stop on it, and the trader deliberately
kept its record of it rather than forgetting it.

---

## The rails

```
risk 3%/trade · max 2 concurrent · 5 new/day · 10x leverage cap · 15% daily loss stop
```

All defined once, in `research/levels/config.py`, each tagged with whether it
rests on a measurement or a guess. Thirteen are tagged `GUESSED` — that count
is the honest measure of how much of this is assumption. Change them there,
not in `.env`, and record why in `DECISIONS.md`.

The daily loss stop and the daily trade cap are deliberately consistent: at
3% risk a 15% stop halts after five losing trades, so `MAX_NEW_PER_DAY=5`
never refuses a trade the risk budget would have allowed. It bounds churn
the budget cannot see — opening and closing without losing enough to trip
anything.

---

## Alerts

`notify.py` sends to Telegram via `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID`
in `.env`. Three things are sent and nothing else:

- the trader armed (rate limited to 5 minutes, so a restart loop is one message)
- a halt, on the transition only
- a position opened and closed immediately because protection failed

Sending never raises — a Telegram outage is not a trading condition. If
credentials are missing it prints `[ALERT unsent ...]` and carries on.

To check alerting end to end without waiting for an incident:

```sh
.venv/bin/python -c "
from dotenv import load_dotenv; load_dotenv()
import notify; print(notify.send('runbook test'))"
```

---

## Common procedures

**Restart after a crash.** Nothing to do — `KeepAlive` restarts on non-zero
exit and `preflight` refuses immediately on a persisted halt flag, so it
idles rather than trading blind. Confirm with `--status`.

**Disarm without stopping the scan.** Remove `LEVELS_LIVE_ARMED` from the
plist, then unload and load. The process keeps scanning in dry mode.

**Check what it thinks it holds.** `--status`. Compare against Hyperliquid
directly; `venue_sanity` does this every pass and after every entry, but it
is worth doing by hand after any incident.

**Label setups to improve the filter.**

```sh
.venv/bin/python -m research.levels.review                       # ranked shortlist
.venv/bin/python -m research.levels.review --label <id> take --note "..."
.venv/bin/python -m research.levels.review --stats               # gate vs your labels
```

The current filter was fitted to 18 labels; `labels.csv` now holds 25. More
labels is the only thing that moves this system off guesses.

---

## Tests

```sh
.venv/bin/python -m pytest -q
```

100 tests, no network. `conftest.py` stubs the Telegram transport for the
whole suite — `.env` is loaded at import, so a test process holds real
credentials and halt tests would otherwise send real messages.

Run them before deploying any change to a rail or to the protection path.

---

## Why the project lives at `~/CuanSniffer`

A launchd job does not inherit Terminal's Full Disk Access. Measured on
2026-09-18: with the project under `~/Documents`, the venv interpreter could
not read its own `pyvenv.cfg` and nothing started — it fails silently, with
an empty log.

**If this is ever moved under `~/Documents`, `~/Desktop` or `~/Downloads`,
every launchd job here stops working.** See `deploy/launchd/README.md` for
the probe that re-verifies this after a move.
