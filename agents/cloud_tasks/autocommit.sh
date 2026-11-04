#!/bin/bash
# Commits and pushes the run results every 50 minutes.
cd "$(dirname "$0")/../.."
B=$(git rev-parse --abbrev-ref HEAD)
while true; do
  sleep 3000
  git add agents/runs/stats agents/runs/design_study agents/cloud_tasks 2>/dev/null
  git commit -q -m "Cloud runs: periodic results checkpoint" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011nwP6BPV6FEGwA4YMwcbvA" && \
  for d in 2 4 8 16 0; do git push -q -u origin "$B" && break; sleep $d; done
done
