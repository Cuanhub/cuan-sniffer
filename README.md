# Cuan Sniffer

A support/resistance level scanner, a labelling loop, and a small live trader
on Hyperliquid.

> Everything this file described before 2026-09-18 belonged to an SMC signal
> engine retired in `b20820e` — a candle-integrity layer, a confidence model,
> a risk manager, a score filter. None of those modules exist. The code is in
> git history; `archive/v3_2026_06/POSTMORTEM.md` says why it was retired.

---

## What this is actually for

The bot's 48 recorded trades sat **7.2pp below the random-walk baseline** for
their own reward/risk — no demonstrated entry edge. The same account's
September book, traded by hand, returned **+4.885% per closing fill** on n=9,
with alpha surviving a beta check on two of four symbols.

The gap between those two numbers is judgement that exists nowhere in the
codebase. This repo is an attempt to find out what it is: log every setup the
rules would fire on, have a human label them take/skip/maybe, and measure
where the rules and the judgement disagree.

The live trader is a small, deliberately over-braked side effect of that. It
has **no measured edge** — the gate it trades scored −0.0413R against random
entry over 688 backfill trades (z = −0.93). It is armed to buy information at
roughly $24 a trade, not because it is expected to make money.

---

## Layout

```
research/levels/          logging only -- cannot place an order
  levels.py               support/resistance detection and quality scoring
  gate.py                 "is this level live right now?" + feature capture
  trigger.py              entry trigger: has the level actually held?
  config.py               every tunable, tagged with what it rests on
  scanner.py              long-lived scan loop, writes episodes
  call.py                 live call sheet for a human verdict
  review.py               ranked shortlist + labelling CLI
  storage.py              append-only, idempotent writes
  universe.py             symbol selection
  venue_client.py         read-only venue access

tools/live_levels_trader.py   the ONLY path that places orders
live_execution_backend.py     Hyperliquid order placement and protection
execution_backend.py          thin execution interface + fill result type
live_data_guard.py            data freshness and API backoff
notify.py                     Telegram alerts for halts and failures
```

### The isolation guarantee

`research/levels/` suggests entry, stop and take-profit prices, which are
numbers a reader could mistake for instructions. Nothing in that package can
act on them: it cannot import any order-capable module, enforced two ways —
a runtime import-graph check and a static AST scan — by
`test_research_levels_isolation.py`.

`tools/live_levels_trader.py` imports both sides, which is exactly why it
lives in `tools/` and not in the package.

---

## Quick start

```sh
cp .env.example .env          # RPC_URL, HL_ACCOUNT_ADDRESS, HL_SECRET_KEY,
                              # TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
.venv/bin/pip install -r requirements.txt

# scan and log, places nothing
.venv/bin/python -m research.levels.scanner --dry-run

# label what it found
.venv/bin/python -m research.levels.review
.venv/bin/python -m research.levels.review --label <episode_id> take --note "..."

# one dry trading pass, prints sized intents
.venv/bin/python -m tools.live_levels_trader --once
```

Arming requires `--arm` **and** `LEVELS_LIVE_ARMED=yes` — two independent
gates, so neither a stray flag in a shell history nor a leftover environment
variable can trade on its own. See `RUNBOOK.md` before doing that.

```sh
.venv/bin/python -m pytest -q      # 101 tests, no network
```

---

## The data

| file | one row per | answers |
|---|---|---|
| `episodes.csv` | gated (coin, level, bar) | what the rules fired on |
| `outcomes.csv` | episode, after ~7d | whether it made money |
| `labels.csv` | episode | whether *you* would have taken it |

Outcomes and labels answer different questions. A gate that is profitable but
fires on setups you would never take has not captured the strategy; a gate
that misses setups you would take is leaving edge behind. Neither shows up in
outcomes alone.

`labels.csv` is tracked in git. The other three are not: episodes and
outcomes can be replayed from price history, but a judgement call cannot be
replayed from anything.

Everything is measured in **sigma** (stdev of 4h log returns over 120 bars),
not percent. A first hand-encoding used raw percent and fired on 0 of 108
defensible parameter sets, because the same setup measured 124% on CASHCAT
and 9.8% on BTC. Normalised, those are 9.9σ and 13.2σ — the same event.

---

## Calibration status

Thirteen parameters in `config.py` are tagged `GUESSED`. That count is the
honest measure of how much of this system is assumption.

The gate fires roughly **25× more often** than the account trades by hand.
That gap is the thing being measured, not a bug to tune away — fitting
thresholds to the same two charts that produced them is how this project
generated four earlier false positives. Collect labels first.

---

## Where to look next

| document | for |
|---|---|
| `RUNBOOK.md` | running the live trader; halts, alerts, procedures |
| `research/levels/SCHEMA.md` | every column, and why it is measured that way |
| `DECISIONS.md` | what was tried, what it measured, what was concluded |
| `HYPOTHESES.md`, `CASCADE_HYPOTHESES.md` | pre-registrations from the retired funding and cascade lines. Kept unedited on purpose — they record what was predicted before the data arrived |
| `deploy/launchd/README.md` | keeping it running across reboots |
