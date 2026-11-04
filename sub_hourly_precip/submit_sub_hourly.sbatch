#!/bin/bash
#SBATCH --job-name=sub_hourly
#SBATCH --account=m2637
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/sub_hourly_%j.out
# Run from sub_hourly_precip/:  mkdir -p slurm_logs && sbatch submit_sub_hourly.sbatch
# fetch_asos.py and fetch_ar_storms.py need network access: run them on a login node first.
# Walltime is a placeholder; the build step (h5 reads) dominates.

source /global/common/software/e3sm/anaconda_envs/load_latest_e3sm_unified_pm-cpu.sh
module load conda
conda activate earth

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
status=0
for s in build_blocks_nc plot_depth_duration plot_asos_compare plot_hf_characteristics; do
    echo "=== $s $(date +%H:%M:%S)"
    if python "$s.py" > "logs/$s.log" 2>&1; then
        echo "    ok"
    else
        echo "    FAILED, last lines of logs/$s.log:"; tail -n 15 "logs/$s.log"; status=1
    fi
done
exit $status
