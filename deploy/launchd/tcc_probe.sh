#!/bin/zsh
# Decides whether a launchd job can actually reach this project.
#
# A launchd-spawned job does not inherit Terminal's Full Disk Access. On this
# machine that was proven to matter on 2026-09-15 for ~/TrenchKit: a probe job
# read a file under ~/ fine and the identical read under ~/Documents returned
# "Operation not permitted", and a check-in agent had failed 46/46 times over
# six days for that reason. Cuan_Sniffer lives under ~/Documents, so the same
# wall is expected here -- expected, not assumed, which is what this measures.
#
# Run it as a launchd job (not from a terminal -- a terminal inherits the very
# access this is testing) and read /tmp/cuan_tcc_probe.out.
P="/Users/myair/Documents/PROJECTS 2025/Sol Flow/Cuan_Sniffer"
echo "=== cuan-sniffer TCC probe $(date -u +%FT%TZ) ==="
echo "uid=$(id -u) euid=$(id -un)"

if head -c 60 "$P/research/cascade/detector.py" >/dev/null 2>/tmp/cuan_tcc_err.txt; then
  echo "READ_SOURCE : OK"
else
  echo "READ_SOURCE : FAIL -> $(cat /tmp/cuan_tcc_err.txt)"
fi

if ls "$P/research/funding/data" >/dev/null 2>&1; then
  echo "LIST_DATA   : OK"
else
  echo "LIST_DATA   : FAIL (cannot list the collector's output directory)"
fi

if touch "$P/research/funding/data/_tcc_probe.tmp" 2>/dev/null; then
  echo "WRITE_DATA  : OK"
  rm -f "$P/research/funding/data/_tcc_probe.tmp"
else
  echo "WRITE_DATA  : FAIL (collector could start but never persist a row)"
fi

# The decisive one: can the venv interpreter import the collector package?
"$P/.venv/bin/python" -u -c "
import sys
sys.path.insert(0, '$P')
try:
    import research.cascade.detector as d
    print('IMPORT      : OK  stop_arms=%s' % (d.STOP_ARMS,))
except Exception as e:
    print('IMPORT      : FAIL -> %s: %s' % (type(e).__name__, e))
" 2>&1 | tail -3

echo "=== end probe ==="
