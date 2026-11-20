#!/usr/bin/env bash
# Metrics for VTM/ECM/AV1/AVM (4 processes in parallel), then the combined CSVs, then the 1080p decode benchmark alone.
# Usage (inside `conda activate compression`):  nohup scripts/run_ctc_all.sh > results/ctc_all.log 2>&1 &
cd "$(dirname "$0")/.."
# Parallel chunk decoders per codec (ECM decodes slowest, so it gets the most).
for cj in vtm:2 ecm:8 av1:2 avm:4; do
  python scripts/eval_ctc_chunks.py --codecs ${cj%:*} --dec-jobs ${cj#*:} >> results/ctc_${cj%:*}.log 2>&1 &
done
wait
python scripts/eval_ctc_chunks.py                  # aggregate only (everything cached)
grep -h FAIL results/ctc_{vtm,ecm,av1,avm}.log && echo "some points failed; rerun this script to retry them"
python scripts/bench_ctc_decode.py                 # nothing else running now -> clean throughput numbers
