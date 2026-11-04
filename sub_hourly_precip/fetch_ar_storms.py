"""AR3+ North Coast storms from the CW3E catalog, and ASOS 1-minute data for
each. Needs network access: run on a login node, after fetch_asos.py (which
writes data/asos/stations.csv).

1. Reads the CW3E per-point event CSVs listed in ar_catalog.files, keeps
   events with AR scale >= ar_catalog.min_scale inside ar_catalog.years, and
   merges events that overlap or are less than merge_gap_hours apart (across
   points) into storms.
   Written to climatology.dir/storms.csv (storm id, start, end, max scale,
   points, and which simulated event it overlaps, if any).
2. For each storm and each climatology.stations station, downloads 1-minute
   data for AR start - hours_before to AR end + hours_after:
   climatology.dir/STORM_ID/ID.csv (IEM, NCEI fallback). Existing files are
   kept unless asos.overwrite is true.

Run:  python fetch_ar_storms.py
"""
import csv
import os
from datetime import timedelta

import sh_common as sh

opts = sh.load_options()
AC = opts["ar_catalog"]
CL = opts["climatology"]
OUT = sh.rel(CL["dir"])
os.makedirs(OUT, exist_ok=True)

# ── 1. storms ────────────────────────────────────────────────────────────
events = []
for lat, fn in AC["files"].items():
    path = os.path.join(sh.rel(AC["dir"]), fn)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} missing: download the CW3E event CSV for {lat}N "
                                f"and set ar_catalog.files")
    rows = sh.read_ar_catalog(path, AC["columns"], int(AC.get("header_rows", 1)))
    y0, y1 = AC["years"]
    months = AC.get("months")
    keep = [dict(r, point=float(lat)) for r in rows
            if r["scale"] >= AC["min_scale"] and y0 <= r["start"].year <= y1
            and (not months or r["start"].month in months)]
    print(f"  {lat}N: {len(rows)} events, {len(keep)} AR{AC['min_scale']}+ in {y0}-{y1}"
          + (f", months {months}" if months else ""))
    events += keep

events.sort(key=lambda e: e["start"])
gap = timedelta(hours=float(AC.get("merge_gap_hours", 0)))
storms = []
for e in events:
    if storms and e["start"] <= storms[-1]["end"] + gap:
        s = storms[-1]
        s["end"] = max(s["end"], e["end"])
        s["scale"] = max(s["scale"], e["scale"])
        s["points"].add(e["point"])
    else:
        storms.append(dict(start=e["start"], end=e["end"], scale=e["scale"], points={e["point"]}))

sims = []
for cfg_name in opts["events"]:
    cfg = sh.load_event(opts, cfg_name)
    sims.append((cfg["event_label"], *sh.analysis_window(opts, cfg)))
pre, post = timedelta(hours=AC["hours_before"]), timedelta(hours=AC["hours_after"])
for s in storms:
    s["id"] = s["start"].strftime("%Y%m%d%H")
    s["w0"], s["w1"] = s["start"] - pre, s["end"] + post
    s["sim"] = ";".join(lab for lab, a, b in sims if a < s["w1"] and b > s["w0"])

with open(os.path.join(OUT, "storms.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["id", "ar_start", "ar_end", "window_start", "window_end", "max_scale", "points", "simulated_event"])
    for s in storms:
        w.writerow([s["id"], f"{s['start']:%Y-%m-%d %H:%M}", f"{s['end']:%Y-%m-%d %H:%M}",
                    f"{s['w0']:%Y-%m-%d %H:%M}", f"{s['w1']:%Y-%m-%d %H:%M}", s["scale"],
                    ";".join(f"{p:g}" for p in sorted(s["points"])), s["sim"]])
n_by = {k: sum(s["scale"] == k for s in storms) for k in range(AC["min_scale"], 6)}
print(f"\n{len(storms)} storms ({', '.join(f'AR{k}: {n}' for k, n in n_by.items())}); "
      f"overlapping simulated events: {[s['id'] + '=' + s['sim'] for s in storms if s['sim']]}")

# ── 2. ASOS per storm ────────────────────────────────────────────────────
meta = sh.asos_stations(opts)
missing = [sid for sid in CL["stations"] if sid not in meta]
if missing:
    raise KeyError(f"{missing} not in data/asos/stations.csv; widen asos.bbox and rerun fetch_asos.py")
for i, s in enumerate(storms):
    print(f"\n=== [{i + 1}/{len(storms)}] {s['id']} AR{s['scale']}: {s['w0']} to {s['w1']} UTC")
    for sid in CL["stations"]:
        path = os.path.join(OUT, s["id"], f"{sid}.csv")
        if os.path.isfile(path) and not opts["asos"].get("overwrite", False):
            print(f"  {sid}: exists, skip")
            continue
        sh.fetch_station_window(opts, dict(meta[sid], id=sid), s["w0"], s["w1"], path)
print("\nDone.")
