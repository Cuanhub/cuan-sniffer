# Cuan Lab for OpenClaw

A research lab an OpenClaw agent can run unattended: it backtests ideas for
the Cuan Sniffer band strategy on real Hyperliquid data, keeps an honest
record of everything it tries, and sends you proposals. **It cannot trade.**

Build it from this repo:

```bash
python -m tools.build_openclaw_package      # -> dist/cuan-lab/ and dist/cuan-lab.tar.gz
```

`dist/cuan-lab/` is an OpenClaw workspace. Its `lab/` folder holds the lab,
plus copies of this repo's research code taken at build time, so it always
backtests the strategy that is actually live.

## What it can and cannot do

| can | cannot |
|---|---|
| fetch public 1h candles from Hyperliquid (read-only, no key) | place, cancel or modify an order -- no exchange client or signing library is shipped, and the build refuses to include one |
| keep a growing archive of real bars past the venue's 208-day limit | read or write the live bot's state, halt file or control queue |
| read a read-only COPY of the live strategy's research records | write anything back to the live machine |
| design, pre-register and run experiments | run an experiment before its design is committed, change it afterwards, or open the holdout twice |
| propose a change, with evidence | apply a change -- a human ships it |

The build refuses to produce a package containing `core/`, the live trader,
the Telegram control, the protection loop, `notify.py`, the live data
directory, a `.env`, the account address, or anything shaped like a private
key. The lab's own tests check the same things again on the machine it runs
on.

## How it decides anything

The agent writes hypotheses and features. The lab's fixed, tested code
(`lab/labcore/`) computes every number and verdict:

1. **Pre-registration** -- the design is git-committed before real data.
2. **Null calibration** -- the experiment runs on 20 random-walk datasets;
   if it "finds edge" there, or its average there isn't zero, the test is
   broken and nothing it says on real data is believed.
3. **Development run** -- real data up to a seal 45 days back.
   Week-clustered bootstrap (crypto coins move together), Holm across
   framings, and an effect must be positive, significant, present in both
   halves of the period, and agree with its controls.
4. **Holdout** -- the most recent 45 days, never seen in development,
   opened once per experiment at a threshold that tightens with every
   opening.
5. **Proposal** -- only after a holdout pass with at least +0.10R per trade.

Every step, including every failure, goes into a hash-chained ledger, so
"we tested 40 ideas and 1 worked" can't quietly become "we found an edge".

## Install (on the machine OpenClaw runs on)

1. Copy `dist/cuan-lab.tar.gz` there and unpack it where your OpenClaw
   workspace should live, or point OpenClaw at it
   (`agents.defaults.workspace` in your OpenClaw config).
2. `cd cuan-lab/lab && ./install.sh` -- makes a venv, initialises the lab's
   own git repository (separate from the trading repo), runs the selftest,
   and does the first data refresh.
3. **Optional, same machine only:** let the lab read the live records by
   setting `CUAN_LIVE_DATA=/path/to/CuanSniffer/research/levels/data` in
   the environment OpenClaw runs commands with. The lab copies five CSVs
   from there and never writes to it.
4. **Restrict the agent's shell.** In OpenClaw, set `tools.exec.mode` to
   `allowlist` (or `ask`) and allow only the lab's interpreter and git:
   `<workspace>/lab/.venv/bin/python` and `git`. Don't add `python3`,
   `bash` or `node` as safe binaries.
5. **Use a separate Telegram bot** for the lab (BotFather). The live bot's
   chat carries `/halt` and `/closeall`; the lab must never be in it.
6. Create the schedule in `AUTOMATIONS.md` (confirm flags with `openclaw
   automations create --help` -- the CLI changed between versions).

Then ask it, in chat: "Run lab-status", "Run lab-cycle", "What did the lab
learn this week?"

## Updating

When the live strategy changes, rebuild and copy over `lab/research/` and
`lab/tools/` only; keep the lab's `ledger/`, `data/`, `experiments/`,
`results/` and `.git`. Experiments registered under the old code will refuse
to run (the harness hash changed): that's correct, as their designs now
describe a different strategy.

## Promoting a proposal

A proposal is a file in `lab/proposals/`. Read its evidence block, then make
the change in the live repo yourself (or with Claude Code), with tests, and
deploy it the usual way. Record the decision in the lab so the ledger knows:
add a `ledger.append("operator_decision", ...)` entry, or ask the agent to
record "accepted" or "declined".
