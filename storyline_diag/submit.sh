#!/bin/bash
# Submit all diagnostic plots: one array task per event, then the
# multi-event plots once every task has finished (pass or fail).
#   ./submit.sh            all five events
#   ./submit.sh 1,4        only EVENTS[1] and EVENTS[4] (1986, 2024)
set -e
cd "$(dirname "$0")"
mkdir -p slurm_logs
ARRAY_OPT=()
[ -n "$1" ] && ARRAY_OPT=(--array="$1")
JOB=$(sbatch --parsable "${ARRAY_OPT[@]}" submit_diagnostics.sbatch)
echo "Per-event plots: job ${JOB}"
MULTI=$(sbatch --parsable --dependency=afterany:${JOB} submit_multi_event.sbatch)
echo "Multi-event plots: job ${MULTI} (starts after ${JOB})"
echo "Logs: slurm_logs/ (one per task) and logs/<event_label>/<script>.log"
