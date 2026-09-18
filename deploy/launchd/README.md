# launchd deployment — the level trader

Keeps `tools/live_levels_trader.py` running across logout and reboot.
Canonical copies live here; `~/Library/LaunchAgents/` holds the deployed
copies.

> This file described the funding and cascade collectors until 2026-09-18.
> Both were retired in `3d4d89e`, and their plists are gone. What remains is
> one job that places real orders, plus a diagnostic probe.

**`com.cuansniffer.trader.plist` is the only job here that places real
orders.** Read the header of `tools/live_levels_trader.py` before installing
it: no edge has been measured for this strategy, and the filter is fitted to
a couple of dozen labels. `RUNBOOK.md` is the operations reference.

---

## Why this project lives at `~/CuanSniffer`

A launchd-spawned job does **not** inherit Terminal's Full Disk Access. This
was measured here on 2026-09-18, not assumed: with the project under
`~/Documents`, a probe job could not read a single source file —

    READ_SOURCE : FAIL -> Operation not permitted
    LIST_DATA   : FAIL
    IMPORT      : FAIL -> PermissionError: '.../.venv/pyvenv.cfg'

The venv interpreter could not read its own `pyvenv.cfg`, so nothing could
start. The same wall had already cost `~/TrenchKit` 46 consecutive check-in
failures over six days before it was moved out.

The project was moved to `~/CuanSniffer` on 2026-09-18 for exactly this
reason. **If it is ever moved back under `~/Documents`, `~/Desktop` or
`~/Downloads`, the job here silently stops working** — it does not error
visibly, it starts, fails to import, and leaves an empty log.

Step 1 re-verifies this after any move.

### Step 1 — probe (do this before anything else)

```sh
cp deploy/launchd/tcc_probe.sh /tmp/
cp deploy/launchd/com.cuansniffer.tccprobe.plist ~/Library/LaunchAgents/
launchctl load -w ~/Library/LaunchAgents/com.cuansniffer.tccprobe.plist
sleep 5
cat /tmp/cuan_tcc_probe.out
```

Then unload it — it is a diagnostic, not a service:

```sh
launchctl unload ~/Library/LaunchAgents/com.cuansniffer.tccprobe.plist
rm ~/Library/LaunchAgents/com.cuansniffer.tccprobe.plist
```

**All four lines `OK`** → TCC is not in the way. Go to step 2.

**Any line `FAIL`** → launchd cannot reach the project. Pick one:

- **Grant Full Disk Access to the venv interpreter.** System Settings →
  Privacy & Security → Full Disk Access → add `.venv/bin/python` *and*
  `/bin/zsh`. Fastest path, but FDA on a Python binary applies to *every*
  script that interpreter ever runs. Worth understanding before clicking.
- **Move the project out of `~/Documents`**, which is what TrenchKit did.
  Most robust, and the permission stops being a live dependency, but every
  absolute path — this plist, and anything else pointing at the current
  location — has to be updated.

### Step 2 — stop whatever is already running

An interactive or `nohup`'d trader will otherwise keep running alongside the
launchd job. Two armed processes share no state file locking, so both would
claim slots independently and `MAX_CONCURRENT` would mean nothing.

```sh
pkill -TERM -f "tools.live_levels_trader"
pgrep -fl "tools\.live_levels_trader"     # expect no output
```

`SIGTERM` is correct, not `-9`: the trader traps it, finishes the current
pass and exits 0.

### Step 3 — install

```sh
mkdir -p ~/Library/Logs/cuan-sniffer
cp deploy/launchd/com.cuansniffer.trader.plist ~/Library/LaunchAgents/
launchctl load -w ~/Library/LaunchAgents/com.cuansniffer.trader.plist
```

The `mkdir` is not optional. launchd does not create a missing
`StandardOutPath` directory; it starts the job and drops the output.

### Step 4 — verify it is actually trading

```sh
launchctl list | grep cuansniffer          # PID in col 1, 0 in col 2
tail -f ~/Library/Logs/cuan-sniffer/trader.log
```

A healthy log opens with `mode: ARMED -- PLACING REAL ORDERS`, a rails line
and the measured-edge note, then prints a sizing table and a `wait` or
`TRIGGER` line per candidate roughly every 15 minutes. An armed start also
sends a Telegram message; if that did not arrive, fix alerting before
leaving it unattended.

```sh
.venv/bin/python -m tools.live_levels_trader --status
```

---

## Operating notes

**Stopping.** `launchctl unload ~/Library/LaunchAgents/com.cuansniffer.trader.plist`.
A bare `kill` will not stick: `KeepAlive` restarts anything that exits
non-zero. Exit 0 is treated as deliberate, so a clean `SIGTERM` shutdown
stays down.

**Disarming without stopping.** Arming needs `--arm` *and*
`LEVELS_LIVE_ARMED=yes`. Drop the `EnvironmentVariables` block, reload, and
the process keeps scanning in dry mode instead of dying — which is the point
of requiring two independent gates.

**A halt survives a restart.** It is persisted in
`research/levels/data/_live_state.json`, and the trader returns 1 on a fatal
venue-sanity halt, which `KeepAlive` restarts. Preflight then refuses again
on the flag, so it idles loudly rather than trading blind. Clear it with
`--resume` only after looking at the venue.

**After editing a plist**, unload and load again — launchd caches the copy it
read at load time. Edit the copy here, re-copy to `~/Library/LaunchAgents/`,
then reload. The deployed copy is a deployment artifact; this one is the
source of truth.

**Logs are not rotated.** launchd does not, and neither does the trader. It
writes roughly 30 lines per 15-minute pass — about 1.5 MB a day with the
current universe. Fine for a live-test window, not fine left for months.

**Config.** Unlike the retired collectors, the trader **does** load `.env`
(`load_dotenv` at import), and the plist sets `LEVELS_LIVE_ARMED` on top of
it. Rails belong in `research/levels/config.py`, which reads env vars with
committed defaults; putting a rail override in `.env` or in the plist makes
a deployed job silently diverge from the code. If you pin one for an
experiment, say so in `DECISIONS.md`.

**The venv path matters.** `ProgramArguments` points at `.venv/bin/python`,
not at the homebrew binary it symlinks to. CPython only adds the venv's
`site-packages` when invoked through the venv directory, because that is
where it looks for `pyvenv.cfg`; the resolved path would import a bare
interpreter with no `hyperliquid` package.
