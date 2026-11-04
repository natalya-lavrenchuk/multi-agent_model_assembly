#!/bin/bash
# One progress check: counts results, restarts run_all.sh if it died before finishing, appends a status line
# to FINDINGS.md, commits and pushes.
cd "$(dirname "$0")/.."
cnt() { [ -f "$1" ] && wc -l < "$1" || echo 0; }
S="stats:"; for t in andrews aiba monod contois; do S="$S $t=$(cnt runs/stats/$t.jsonl)"; done
D="design: $(ls runs/design_study/*.json 2>/dev/null | wc -l) files, $(cat runs/design_study/*.jsonl 2>/dev/null | wc -l) summary lines"
STAGE=$(grep -E "start|done|restart" cloud_tasks/run_all.log | tail -1)
NOTE=""
if ! pgrep -f run_all.sh >/dev/null && ! grep -q "all done" cloud_tasks/run_all.log; then
  echo "--- restart $(date -u +%FT%TZ) (runner was not running)" >> cloud_tasks/run_all.log
  nohup cloud_tasks/run_all.sh >> cloud_tasks/run_all.log 2>&1 &
  NOTE=" RUNNER RESTARTED."
fi
pgrep -f autocommit.sh >/dev/null || (cd .. && nohup agents/cloud_tasks/autocommit.sh >/dev/null 2>&1 &)
ERR=$(grep -lE "Traceback|LLM_API_KEY" runs/stats/*.log runs/design_study/*.log 2>/dev/null | tr '\n' ' ')
LINE="- $(date -u +%FT%TZ) $S; $D; stage: $STAGE.${ERR:+ errors in: $ERR}$NOTE"
echo "$LINE" >> cloud_tasks/FINDINGS.md
echo "$LINE"
cd .. && git add agents/runs/stats agents/runs/design_study agents/cloud_tasks agents/tests 2>/dev/null
git commit -q -m "Cloud runs: progress check" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011nwP6BPV6FEGwA4YMwcbvA"
for d in 2 4 8 16 0; do git push -q -u origin "$(git rev-parse --abbrev-ref HEAD)" && break; sleep $d; done
