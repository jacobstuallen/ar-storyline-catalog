"""Download ASOS 1-minute data for the simulated events. Needs network
access: run on a login node.

1. Station list: every station in IEM's ASOS1MIN network (sites with 1-minute
   data) inside asos.bbox (default region_bbox), or asos.stations if set.
   Written to data/asos/stations.csv (id, name, lat, lon, elev_m).
2. For each event, 1-minute asos.vars over the analysis window +- pad_hours,
   one request per station: data/asos/EVENT/ID.csv. Existing files are kept
   unless asos.overwrite is true.
3. If IEM returns nothing for a station (or the request fails) and
   asos.ncei_fallback is true, NCEI's monthly page-2 files
   (asos.ncei_url/YYYY/MM/asos-1min-pg2-KXXX-YYYYMM.dat, cached in
   data/asos/ncei_raw/) are parsed instead and written in the same CSV layout.
   Page 2 has precip, ptype, pres1 and tmpf only (no wind).

IEM's own `sample=` option subsamples (every Nth minute), it does not sum, so
this always requests 1-minute data. Precip is inches per minute; 0.00 is
reported explicitly and missing comes back as M.

Run:  python fetch_asos.py
"""
import csv
import json
import os
from datetime import timedelta

import sh_common as sh

opts = sh.load_options()
A = opts["asos"]
OUT = sh.asos_dir(opts)
os.makedirs(OUT, exist_ok=True)

# ── 1. stations ──────────────────────────────────────────────────────────
print(f"Station list: {A['network_url']}")
feats = json.loads(sh.http_get(A["network_url"]))["features"]
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
    for s in stations:
        path = os.path.join(OUT, label, f"{s['id']}.csv")
        if os.path.isfile(path) and not A.get("overwrite", False):
            print(f"  {s['id']}: exists, skip")
            continue
        sh.fetch_station_window(opts, s, s0, s1, path)
print("\nDone.")
