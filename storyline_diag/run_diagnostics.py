"""
run_diagnostics.py

Runs the diagnostic scripts for every storyline event, each as its own
subprocess, with a log per script and a pass/fail table at the end.

    python run_diagnostics.py         # every event, then the multi-event plots
    (inside a SLURM array task)       # only EVENTS[SLURM_ARRAY_TASK_ID]
    STORYLINE_STAGE=multi python run_diagnostics.py    # only the multi-event plots

Comment out entries below to skip them. Each script picks its event from
STORYLINE_CONFIG, which this runner sets.
Logs: logs/<event_label>/<script>.log and logs/multi_event/<script>.log
"""

import os
import subprocess
import sys
import time
from datetime import timedelta

from storyline_config import load_event, event_window
from storyline_io import case_coverage

EVENTS = [
    "config.1964.WUS.G-NDG.yaml",
    "config.1986.WUS.G-NDG.yaml",
    "config.2005.WUS.G-NDG.yaml",
    "config.2019.WUS.G-NDG.yaml",
    "config.2024.WUS.G-NDG.yaml",
]

PER_EVENT = [                      # run in this order (overlay needs the netCDFs)
    #"build_comparison_nc.py",
    #"plot_multicase_overlay.py",
    #"plot_accum_precip.py",
    #"plot_max_precip.py",
    #"plot_county_accum.py",
    #"plot_huc10_accum.py",
    #"plot_huc10_24hr.py",
    #"print_basin_totals.py",
    #"plot_idf_overlay.py",
    #"plot_ivt_ar.py",
    #"plot_precip_partition.py",
    "plot_ncfr_maps.py",
]

MULTI_EVENT = [
    #"plot_county_timeseries.py",
]

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)


def run(script, log_dir, env):
    os.makedirs(log_dir, exist_ok=True)
    log = os.path.join(log_dir, script.replace('.py', '.log'))
    t = time.time()
    with open(log, 'w') as f:
        rc = subprocess.call([sys.executable, '-u', script], stdout=f, stderr=subprocess.STDOUT, env=env)
    dt = time.time() - t
    status = "ok" if rc == 0 else f"FAILED (exit {rc})"
    print(f"  {script:<30} {status:<18} {dt / 60:5.1f} min   {log}", flush=True)
    return (script, rc, dt, log)


def preflight(cfg):
    """Print what output each case has inside the event window."""
    t0, t1 = event_window(cfg)
    print(f"  window {t0} to {t1}")
    step = timedelta(hours=cfg['time'].get('hours_step', 3))
    for case in cfg['cases']:
        cov = case_coverage(case, t0, t1)
        if cov is None:
            msg = "NO OUTPUT in window (skipped by the plots)"
        else:
            first, last, n = cov
            full = first <= t0 + step and last >= t1 - step
            msg = f"{first} to {last}, {n} records" + ("" if full else "  PARTIAL (skipped by the plots)")
        print(f"  {case['resolution']:>6}  {case['name']}: {msg}")


results = []
stage = os.environ.get("STORYLINE_STAGE", "")
task = os.environ.get("SLURM_ARRAY_TASK_ID")

if stage != "multi":
    events = [EVENTS[int(task)]] if task is not None else EVENTS
    for config in events:
        cfg = load_event(config)
        label = cfg['event_label']
        print(f"\n===== {label} ({config}) =====", flush=True)
        preflight(cfg)
        env = {**os.environ, "STORYLINE_CONFIG": config}
        for script in PER_EVENT:
            results.append((label,) + run(script, os.path.join("logs", label), env))

if stage == "multi" or (task is None and stage == ""):
    print("\n===== multi-event =====", flush=True)
    for script in MULTI_EVENT:
        results.append(("multi_event",) + run(script, os.path.join("logs", "multi_event"), dict(os.environ)))

print("\n===== summary =====")
failed = [r for r in results if r[2] != 0]
for label, script, rc, dt, log in results:
    print(f"  {label:<14} {script:<30} {'ok' if rc == 0 else 'FAILED':<7} {dt / 60:5.1f} min")
if failed:
    print(f"\n{len(failed)} failed. Last lines of each log:")
    for label, script, rc, dt, log in failed:
        with open(log) as f:
            tail = f.readlines()[-8:]
        print(f"\n--- {label} / {script} ({log}) ---\n" + "".join(tail).rstrip())
sys.exit(1 if failed else 0)
