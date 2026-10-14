#!/bin/bash
#SBATCH --account=m2637
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --array=0-4
#SBATCH --job-name=storyline_plots
#SBATCH --output=slurm_logs/%x-%A_%a.out
##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=jacob.stu.allen@gmail.com

# One array task per event (index into EVENTS in run_diagnostics.py).
# Submit with ./submit.sh, which also queues the multi-event job after this.
# Walltime is a placeholder: check the per-script minutes in the summary at
# the end of each task's log and tighten it.

SCRIPT_DIR="/global/u2/j/jsallen/projects/ar_storylines_2026/ar-storyline-catalog/storyline_diag"

source /global/common/software/e3sm/anaconda_envs/load_latest_e3sm_unified_pm-cpu.sh
module load conda
conda activate earth

cd "${SCRIPT_DIR}"
echo "Job ${SLURM_JOB_ID}, array task ${SLURM_ARRAY_TASK_ID}, on $(hostname) at $(date)"
python -u run_diagnostics.py
STATUS=$?
echo "Finished at $(date), exit ${STATUS}"
exit ${STATUS}
