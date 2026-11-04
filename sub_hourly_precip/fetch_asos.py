"""Download ASOS 1-minute data from the Iowa Environmental Mesonet (a copy of
NCEI DSI-6405/6406). Needs network access: run on a login node.

1. Station list: every station in IEM's ASOS1MIN network (sites with 1-minute
   data) inside asos.bbox (default region_bbox), or asos.stations if set.
   Written to data/asos/stations.csv (id, name, lat, lon, elev_m).
2. For each event, 1-minute asos.vars over the analysis window +- pad_hours,
   one request per station: data/asos/EVENT/ID.csv. Existing files are kept
   unless asos.overwrite is true.

IEM's own `sample=` option subsamples (every Nth minute), it does not sum, so
this always requests 1-minute data. Precip is inches per minute; 0.00 is
reported explicitly and missing comes back as M.

Run:  python fetch_asos.py
"""
import csv
import json
import os
import time
import urllib.parse
import urllib.request
from datetime import timedelta

import numpy as np

import sh_common as sh

opts = sh.load_options()
A = opts["asos"]
OUT = sh.asos_dir(opts)
os.makedirs(OUT, exist_ok=True)
UA = {"User-Agent": "ar-storyline-catalog sub_hourly_precip (python urllib)"}


def get(url, timeout=300):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")


# ── 1. stations ──────────────────────────────────────────────────────────
print(f"Station list: {A['network_url']}")
feats = json.loads(get(A["network_url"]))["features"]
a, b, c, d = A.get("bbox") or opts["region_bbox"]
want = [s.upper() for s in A["stations"]] if A.get("stations") else None
stations = []
for f in feats:
    lon, lat = f["geometry"]["coordinates"][:2]
    sid = f["id"]
    if want is not None:
        if sid not in want:
            continue
    elif not (a <= lat <= c and b <= lon <= d):
        continue
    p = f["properties"]
    stations.append(dict(id=sid, name=p.get("sname", ""), lat=lat, lon=lon,
                         elev_m=p.get("elevation"), archive_begin=p.get("archive_begin"),
                         network=p.get("network")))
if want is not None:
    missing = sorted(set(want) - {s["id"] for s in stations})
    if missing:
        print(f"  not in ASOS1MIN network (no 1-min data at IEM): {missing}")
stations.sort(key=lambda s: -s["lat"])
with open(os.path.join(OUT, "stations.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["id", "name", "lat", "lon", "elev_m", "archive_begin", "network"])
    w.writeheader()
    w.writerows(stations)
print(f"  {len(stations)} stations:")
for s in stations:
    print(f"    {s['id']:5s} {s['name'][:30]:30s} {s['lat']:8.4f} {s['lon']:10.4f} "
          f"{s['elev_m'] if s['elev_m'] is not None else 'nan':>7} m  archive from {s['archive_begin']}")

# ── 2. data per event ────────────────────────────────────────────────────
pad = timedelta(hours=float(A.get("pad_hours", 0)))
for cfg_name in opts["events"]:
    cfg = sh.load_event(opts, cfg_name)
    label = cfg["event_label"]
    t0, t1 = sh.analysis_window(opts, cfg)
    s0, s1 = t0 - pad, t1 + pad
    if s1.year < 2000:
        print(f"\n=== {label}: before 2000, no ASOS 1-minute archive; skip")
        continue
    print(f"\n=== {label}: {s0} to {s1} UTC")
    edir = os.path.join(OUT, label)
    os.makedirs(edir, exist_ok=True)
    for s in stations:
        path = os.path.join(edir, f"{s['id']}.csv")
        if os.path.isfile(path) and not A.get("overwrite", False):
            print(f"  {s['id']}: exists, skip")
            continue
        q = urllib.parse.urlencode({
            "station": s["id"], "vars": ",".join(A["vars"]),
            "sts": s0.strftime("%Y-%m-%dT%H:%MZ"), "ets": s1.strftime("%Y-%m-%dT%H:%MZ"),
            "sample": "1min", "what": "download", "tz": "UTC", "gis": "1", "delim": "comma"})
        try:
            txt = get(f"{A['request_url']}?{q}")
        except Exception as e:
            print(f"  {s['id']}: FAILED {type(e).__name__}: {e}")
            time.sleep(float(A.get("sleep_s", 2)))
            continue
        lines = txt.strip().splitlines()
        if len(lines) <= 1:
            print(f"  {s['id']}: no records")
        else:
            with open(path, "w") as fh:
                fh.write(txt)
            times, v = sh.read_asos_csv(path)
            p = v.get("precip", np.array([]))
            nmin = int((s1 - s0).total_seconds() // 60)
            print(f"  {s['id']}: {len(times)} of {nmin} minutes, precip valid "
                  f"{np.isfinite(p).sum()} min, total {np.nansum(p) * sh.MM_PER_IN:.1f} mm")
            if len(times) and "precip" not in v:
                print(f"    header: {lines[0]}")
        time.sleep(float(A.get("sleep_s", 2)))
print("\nDone.")
