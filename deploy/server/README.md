# Moving CuanSniffer from the Mac to a cloud server

Why: the Mac sleeps when its lid closes or it travels, and the trend book's
pass then runs late or not at all (2026-11-14 ran 23 hours late). A server
that never sleeps keeps the live record clean.

What moves: every job the Mac runs — protect, trader (recording only),
scanner, telegram, and the hourly trend live and trend paper runs, plus daily
retention. The keep-awake job is not needed.

**The one rule: never run the jobs on both machines at once.** Two copies
would trade the same account twice. The Mac's jobs are stopped and disabled
before the server's start, and a rollback reverses that order.

## 1. Create the server (operator)

- Any provider; the smallest plan is enough (the jobs use ~120 MB of memory).
- Ubuntu 24.04, in a region where you are allowed to use Hyperliquid.
- Log in with an SSH key, not a password.

## 2. Install (as root on the server)

```
curl -LsSf https://raw.githubusercontent.com/Cuanhub/cuan-sniffer/simplify-execution-core/deploy/server/setup.sh -o setup.sh
less setup.sh        # read it first
bash setup.sh
```

It sets the clock to UTC, opens SSH only, creates the user `cuan`, installs
Python 3.13.5 and the Mac's exact package versions, and installs the systemd
units. It starts nothing.

## 3. Cutover, just after a daily trend pass

Pick a time after that day's pass and its stops are confirmed in the Mac's
trend log (on the Mac that is about 00:55 UTC). The stops rest on the venue
throughout, so the positions are protected while nothing runs.

On the Mac — stop every job, and move the files so a reboot cannot restart them:

```
for j in protect trader scanner telegram trendlive trendpaper retention awake; do
  launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.cuansniffer.$j.plist
done
launchctl list | grep cuansniffer          # must print nothing
mkdir -p ~/Library/LaunchAgents.disabled
mv ~/Library/LaunchAgents/com.cuansniffer.*.plist ~/Library/LaunchAgents.disabled/
```

Copy the secrets and the live state (these are never in git):

```
cd ~/CuanSniffer
scp .env cuan@SERVER:CuanSniffer/.env
scp -r research/levels/data/. cuan@SERVER:CuanSniffer/research/levels/data/
scp -r research/trend/data/.  cuan@SERVER:CuanSniffer/research/trend/data/
ssh cuan@SERVER 'chmod 600 CuanSniffer/.env'
```

## 4. Start (as root on the server)

```
systemctl enable --now cuan-protect cuan-trader cuan-scanner cuan-telegram
systemctl enable --now cuan-trendlive.timer cuan-trendpaper.timer cuan-retention.timer
systemctl start cuan-trendlive        # one stop check now, rather than at :10
```

The trend live run fires at 10 past every hour (UTC); the first one after
00:00 UTC that can read the day's candles is the daily pass, as
PREREG_live.md defines it. On the Mac it happened to run near :55.

## 5. Check

```
systemctl --no-pager status 'cuan-*'
systemctl list-timers 'cuan-*'
tail -n 5 ~cuan/logs/*.log
sudo -u cuan -H bash -c 'cd ~/CuanSniffer && .venv/bin/python -m tools.trend_live --status'
```

Expect: protect listing all ten coins as `trend`; the trader logging `max 0`;
the trend log ending `stops checked`; the status showing the same value, peak
and positions as on the Mac.

## Rollback

On the server: `systemctl disable --now 'cuan-*'` and confirm nothing runs.
Copy both data folders back to the Mac (`.env` there is unchanged), then on the
Mac move the plists back from `~/Library/LaunchAgents.disabled/` and
`launchctl bootstrap gui/$(id -u)` each one.

## Updating the code later

```
sudo -u cuan -H bash -c 'cd ~/CuanSniffer && git pull --ff-only'
systemctl restart cuan-protect cuan-trader cuan-scanner cuan-telegram
```

A restart is a production deploy; the operator runs it.
