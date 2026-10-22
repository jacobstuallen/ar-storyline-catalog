#!/bin/bash
#SBATCH --account=m2637
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --job-name=process_sf
#SBATCH --output=slurm_logs/%x-%j.out

# Structure functions for every event/resolution in process.yaml, then all
# plots for every event (run_process.py).
# Walltime is a placeholder; tighten after the first run.
# Submit from process_diag/:  mkdir -p slurm_logs && sbatch submit_process.sbatch

SCRIPT_DIR="/global/u2/j/jsallen/projects/ar_storylines_2026/ar-storyline-catalog/process_diag"

source /global/common/software/e3sm/anaconda_envs/load_latest_e3sm_unified_pm-cpu.sh
module load conda
conda activate earth

cd "${SCRIPT_DIR}"
python -u compute_structure_functions.py
python -u run_process.py
