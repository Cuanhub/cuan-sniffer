# launchd deployment — research collectors

Keeps `research/funding/collector.py` and `research/cascade/collector.py`
running across logout and reboot, so the forward sample
`CASCADE_HYPOTHESES.md` requires can actually reach two continuous weeks.
Canonical copies live here; `~/Library/LaunchAgents/` holds the deployed
copies.

Two jobs, not one supervisor: the collectors are already independent
processes with separate universes, cadences and output directories, and a
crash in one must not take the other's sample with it.

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
`~/Downloads`, every launchd job here silently stops working** — it does not
error visibly, it starts, fails to import, and leaves an empty log.

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
  Privacy & Security → Full Disk Access → add
  `.venv/bin/python` *and* `/bin/zsh`. Fastest path, but FDA on a Python
  binary applies to *every* script that interpreter ever runs, not just
  these collectors. Worth understanding before clicking.
- **Move the project out of `~/Documents`** (e.g. `~/CuanSniffer`), which is
  what TrenchKit did. Most robust and the permission stops being a live
  dependency, but every absolute path — these plists, and anything else
  pointing at the current location — has to be updated.
- **Stay on `nohup`** and accept that the run does not survive a reboot.
  Honest fallback, and fine if the machine stays up; just know that a
  restart silently ends the collection window.

### Step 2 — stop whatever is already running

`nohup`'d collectors from an interactive session will otherwise keep running
alongside the launchd jobs: double the API load, and duplicate funding
snapshot rows. Cascade resolutions dedupe on `resolution_id`, funding
snapshots do not.

```sh
pkill -TERM -f "research.funding.collector"
pkill -TERM -f "research.cascade.collector"
pgrep -fl "research\.(funding|cascade)\.collector"   # expect no output
```

`SIGTERM` is correct, not `-9`: both collectors trap it, finish the current
cycle and persist pending state before exiting.

### Step 3 — install

```sh
mkdir -p ~/Library/Logs/cuan-sniffer
cp deploy/launchd/com.cuansniffer.funding.plist ~/Library/LaunchAgents/
cp deploy/launchd/com.cuansniffer.cascade.plist ~/Library/LaunchAgents/
launchctl load -w ~/Library/LaunchAgents/com.cuansniffer.funding.plist
launchctl load -w ~/Library/LaunchAgents/com.cuansniffer.cascade.plist
```

The `mkdir` is not optional. launchd does not create a missing
`StandardOutPath` directory; it starts the job and drops the output.

### Step 4 — verify it is actually collecting

```sh
launchctl list | grep cuansniffer          # PID in col 1, 0 in col 2
tail -f ~/Library/Logs/cuan-sniffer/cascade.log
```

A healthy cascade log shows `cycle complete: pending=N` about once a minute
and names its arms at startup; funding shows `cycle complete: N rows written`
every five minutes. Confirm a data file appears for today:

```sh
ls -la research/funding/data/*_$(date -u +%F).csv | head
```

---

## Operating notes

**Stopping.** `launchctl unload ~/Library/LaunchAgents/com.cuansniffer.<job>.plist`.
A bare `kill` will not stick: `KeepAlive` restarts anything that exits
non-zero. Exit 0 is treated as deliberate, so a clean `SIGTERM` shutdown
stays down.

**After editing a plist**, unload and load again — launchd caches the copy it
read at load time. Edit the copy here, re-copy to `~/Library/LaunchAgents/`,
then reload. The deployed copy is a deployment artifact; this one is the
source of truth.

**Logs are not rotated.** launchd does not rotate, and neither collector
does. Cascade writes roughly a line a minute (~1.5 MB over two weeks),
funding a line per five minutes. Fine for one collection window, not fine
left running for months.

**Config.** Neither collector loads `.env`; both read env vars with defaults
baked into the code (`CASCADE_STOP_ARMS`, `FUNDING_UNIVERSE_TOP_N`,
`FUNDING_UNIVERSE_EXCLUDE`). The plists deliberately set no
`EnvironmentVariables`, so the committed defaults are the single source of
truth and a deployed job cannot silently diverge from the repo. To pin a
value for an experiment, add it to the plist and say so in `DECISIONS.md` —
an undocumented override in a deployed plist is invisible to anyone reading
the code later.

**The venv path matters.** `ProgramArguments` points at `.venv/bin/python`,
not at the homebrew binary it symlinks to. CPython only adds the venv's
`site-packages` when invoked through the venv directory, because that is
where it looks for `pyvenv.cfg`; the resolved path would import a bare
interpreter with no `hyperliquid` package.
