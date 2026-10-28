# Schedule

Two kinds of job, on purpose:

- **Command jobs** run a script with NO model call. Collection is
  deterministic and needs no judgement, so no LLM touches it.
- **Agent jobs** run a model turn in an isolated session: interpretation
  and research, under AGENTS.md.

| when (UTC) | job | kind | delivers |
|---|---|---|---|
| daily 00:20 | `lab.py refresh --quiet-ok` | command | only failures (prints NO_REPLY when fine) |
| daily 01:00 | "Run the lab-cycle skill once." | agent, isolated | a 3-6 line summary |
| Sunday 13:00 | "Run the lab-weekly skill." | agent, isolated | the weekly digest |
| 1st of month 02:00 | `lab.py selftest` | command | only failures |

00:20 UTC sits after the hourly candle close and before the agent job, so
the agent always sees fresh bars.

A refresh takes about 15 minutes, and that is deliberate: it fetches 1h AND
15m bars for 45 coins one request every 10 seconds (CUAN_LAB_FETCH_GAP_SEC).
If the lab shares a machine or IP with the live bot, they share the venue's
per-IP request budget, and a full-speed refresh is what rate-limited the live
trader on 2026-10-27. Do not lower the gap to make it faster.

## Creating them

The automation CLI has changed across OpenClaw versions (`openclaw cron add`
became `openclaw automations create`). **Confirm every flag with
`openclaw automations create --help` on your install before running these.**
As documented at the time of writing (command payloads need operator.admin):

```bash
LAB="$HOME/.openclaw/workspace/lab"      # wherever the workspace lives

openclaw automations create --name "lab refresh" --cron "20 0 * * *" --tz UTC \
  --command "$LAB/.venv/bin/python lab.py refresh --quiet-ok" \
  --command-cwd "$LAB" --timeout-seconds 1800 \
  --announce --channel telegram --to "<lab chat id>"

openclaw automations create --name "lab cycle" --cron "0 1 * * *" --tz UTC \
  --session isolated --message "Run the lab-cycle skill once." \
  --announce --channel telegram --to "<lab chat id>"

openclaw automations create --name "lab weekly" --cron "0 13 * * 0" --tz UTC \
  --session isolated --message "Run the lab-weekly skill." \
  --announce --channel telegram --to "<lab chat id>"

openclaw automations create --name "lab selftest" --cron "0 2 1 * *" --tz UTC \
  --command "$LAB/.venv/bin/python lab.py selftest" \
  --command-cwd "$LAB" --timeout-seconds 1800 \
  --announce --channel telegram --to "<lab chat id>"
```

## Telegram: use a SEPARATE bot

The live trader's bot token is already polled by one process, and a second
poller on the same token steals its updates (409 Conflict, seen 2026-10).
Create a new bot with BotFather for the lab and give OpenClaw that token.
The lab must never be able to send to, or read from, the live bot's chat:
that chat carries `/halt` and `/closeall`.
