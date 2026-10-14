#!/bin/bash
#SBATCH --account=m2637
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --job-name=storyline_multi
#SBATCH --output=slurm_logs/%x-%j.out

# Plots that combine all events (MULTI_EVENT in run_diagnostics.py).
# Queued by submit.sh to start after the per-event array finishes.

SCRIPT_DIR="/global/u2/j/jsallen/projects/ar_storylines_2026/ar-storyline-catalog/storyline_diag"

source /global/common/software/e3sm/anaconda_envs/load_latest_e3sm_unified_pm-cpu.sh
module load conda
conda activate earth

cd "${SCRIPT_DIR}"
STORYLINE_STAGE=multi python -u run_diagnostics.py
