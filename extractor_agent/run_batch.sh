#!/bin/bash
# One command for many papers: every LLM step for each paper whose boundary is approved, then the graph.
#   ./run_batch.sh papers/xing_2010 papers/nagy_2008 ...      (needs LLM_API_KEY exported)
# A paper with a proposed (not approved) boundary is skipped and listed at the end. Output: logs/batch_<time>.log
mkdir -p logs
log="logs/batch_$(date +%Y%m%d_%H%M).log"
skipped=()
for p in "$@"; do
  status=$(python3 -c "import json;print(json.load(open('$p/paper.json')).get('status') or 'approved')" 2>/dev/null)
  if [ "$status" != "approved" ]; then skipped+=("$p ($status)"); continue; fi
  echo "=== $p" | tee -a "$log"
  for step in symbols laws fixes supmeta graph; do
    python3 -m extractor.$step "$p" 2>&1 | tee -a "$log" | grep -E "^[0-9]+ (symbols|laws)|->|Error|Traceback|to look at" 
  done
done
[ ${#skipped[@]} -gt 0 ] && printf 'skipped, boundary not approved: %s\n' "${skipped[@]}" | tee -a "$log"
echo "full output: $log"
