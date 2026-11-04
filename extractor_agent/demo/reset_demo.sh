#!/bin/bash
# Reset the demo paper to "just uploaded the PDF": its results go to _to_delete/demo_runs/<time>/, while the
# saved LLM answers (llm_cache) stay, so every LLM step in the demo answers instantly with the rehearsal's answers.
#   ./demo/reset_demo.sh            (paper folder: papers/anane_demo)
# Rehearse once (needs the API key), then reset; on the day, click the same steps and approve the boundary unchanged.
d=papers/${1:-anane_demo}
[ -d "$d" ] || { echo "no $d"; exit 1; }
out=_to_delete/demo_runs/$(date +%Y%m%d_%H%M%S); mkdir -p "$out"
for f in "$d"/*; do
  case "$(basename "$f")" in paper.pdf|llm_cache) ;; *) mv -n "$f" "$out"/ ;; esac
done
echo "reset $d (kept paper.pdf and llm_cache; results moved to $out)"
echo "next: in the app, open $(basename "$d") and upload demo/paddle.json and demo/paddle_blocks.json"
