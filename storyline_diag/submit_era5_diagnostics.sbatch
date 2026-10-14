#!/bin/bash
#SBATCH --account=m2637
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --array=0-4
#SBATCH --job-name=storyline_era5
#SBATCH --output=slurm_logs/%x-%A_%a.out

# storyline_diagnostics.py (ERA5-vs-E3SM frames + MP4s), one array task per
# event. Separate from the plot job because it takes hours. Submit with:
#   mkdir -p slurm_logs && sbatch submit_era5_diagnostics.sbatch
# One event only:  sbatch --array=4 submit_era5_diagnostics.sbatch

SCRIPT_DIR="/global/homes/j/jsallen/projects/ar_storylines_2026/storyline_diag_03"
CONFIGS=(config.1964.WUS.G-NDG.yaml config.1986.WUS.G-NDG.yaml config.2005.WUS.G-NDG.yaml
         config.2019.WUS.G-NDG.yaml config.2024.WUS.G-NDG.yaml)
CONFIG="${CONFIGS[${SLURM_ARRAY_TASK_ID}]}"

source /global/common/software/e3sm/anaconda_envs/load_latest_e3sm_unified_pm-cpu.sh
module load conda
conda activate earth

cd "${SCRIPT_DIR}"
echo "Job ${SLURM_JOB_ID}, config ${CONFIG}, on $(hostname) at $(date)"
python -u storyline_diagnostics.py "${CONFIG}"
STATUS=$?
echo "Finished at $(date), exit ${STATUS}"
exit ${STATUS}
