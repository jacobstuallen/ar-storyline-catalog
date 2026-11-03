#!/bin/bash
#SBATCH --job-name=rain_snow
#SBATCH --account=m2637
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --time=03:00:00
#SBATCH --output=slurm_logs/rain_snow_%j.out
# Run from rain_snow_diag/:  mkdir -p slurm_logs && sbatch submit_rain_snow.sbatch
# Walltime is a placeholder; the build step dominates (h2 3D reads).

source /global/common/software/e3sm/anaconda_envs/load_latest_e3sm_unified_pm-cpu.sh
module load conda
conda activate earth

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
status=0
for s in build_phase_nc plot_elevband_timeseries plot_partition_curve plot_snowline_maps plot_transect \
         plot_threshold_summary plot_snowfrac_decomposition plot_snowfrac_vs_meltlayer plot_snowfrac_vs_rate; do
    echo "=== $s $(date +%H:%M:%S)"
    if python "$s.py" > "logs/$s.log" 2>&1; then
        echo "    ok"
    else
        echo "    FAILED, last lines of logs/$s.log:"; tail -n 15 "logs/$s.log"; status=1
    fi
done
exit $status
