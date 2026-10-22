"""
run_process.py

Runs the plot scripts for every event in process.yaml. Each script runs as a
separate process with PROCESS_EVENT set to the event year, so one crash
doesn't stop the rest. Logs: logs/<year>/<script>.log. Ends with a
pass/fail table.

Run compute_structure_functions.py first (or use submit_process.sbatch,
which runs both).

Run:  python run_process.py
"""

import os
import subprocess
import sys
import time

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))

SCRIPTS = [
    "plot_nudge_profile.py",
    "plot_sf_single.py",
    "plot_sf_resolutions.py",
]

# Event years from process.yaml (config.<year>.WUS.G-NDG.yaml). Replace with a
# list such as [2019, 2024] to run a subset.
with open(os.path.join(HERE, "process.yaml")) as f:
    EVENTS = [int(p.split('.')[1]) for p in yaml.safe_load(f)['events']]

results = []
for year in EVENTS:
    log_dir = os.path.join(HERE, "logs", str(year))
    os.makedirs(log_dir, exist_ok=True)
    for script in SCRIPTS:
        log = os.path.join(log_dir, script.replace(".py", ".log"))
        print(f"{year}  {script} ...", flush=True)
        t = time.time()
        with open(log, "w") as fh:
            rc = subprocess.run([sys.executable, "-u", script], cwd=HERE,
                                env=dict(os.environ, PROCESS_EVENT=str(year)),
                                stdout=fh, stderr=subprocess.STDOUT).returncode
        results.append((year, script, rc, (time.time() - t) / 60, log))

print(f"\n{'event':6} {'script':28} {'status':6} {'min':>5}")
for year, script, rc, minutes, log in results:
    with open(log) as fh:
        skipped = any(line.startswith("SKIP") for line in fh)
    status = "FAIL" if rc else ("skip" if skipped else "ok")
    print(f"{year:<6} {script:28} {status:6} {minutes:5.1f}")

failed = [r for r in results if r[2] != 0]
for year, script, rc, minutes, log in failed:
    print(f"\n--- {year} {script} (exit {rc}), last lines of {log}:")
    with open(log) as fh:
        print("".join(fh.readlines()[-15:]))
sys.exit(1 if failed else 0)
