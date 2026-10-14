"""
prefetch.py

Downloads everything the plots fetch from the web, into
~/.cache/storyline_viz/ (and cartopy's own cache), so batch jobs never need
network access. Run once on a login node, and again after adding locations,
HU2 regions or counties:

    python prefetch.py
"""

import glob
import os

import yaml

import storyline_plot as sp
from storyline_config import load_event

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

cfgs = [load_event(p) for p in sorted(glob.glob("config.*.WUS.G-NDG.yaml"))]
failures = []


def attempt(what, fn, *args):
    try:
        fn(*args)
        print(f"  ok      {what}")
    except Exception as e:
        failures.append(what)
        print(f"  FAILED  {what}: {type(e).__name__}: {e}")


print("\nCensus counties")
attempt("cb_2018_us_county_20m.zip", sp.counties_gdf)

print("\nUSGS WBD (HU2 zips; large, can take a while)")
hu2 = {"18"}
for cfg in cfgs:
    hu2 |= set(cfg['plots'].get('huc10_accum', {}).get('hu2_regions', []))
    hu2 |= set(cfg['plots'].get('accum_precip', {}).get('hu2_regions', []))
    oc = cfg.get('observed_comparison', {})
    hu2 |= set(oc.get('huc10_24hr', {}).get('hu2_regions', []))
    hu2 |= {b.get('hu2', '18') for b in oc.get('basins', {}).values()}
for r in sorted(hu2):
    attempt(f"WBD HU2 {r}", sp.cached_download, sp.WBD_HU2_ZIP_TMPL.format(region=r),
            "boundaries", f"WBD_{r}_HU2_Shape.zip")

print("\nNOAA Atlas 14 (one CSV per location)")
locs = {}
for cfg in cfgs:
    locs.update({k: v for k, v in cfg.get('locations', {}).items() if v is not None})
for name, (lat, lon) in locs.items():
    attempt(f"Atlas 14 {name}", sp.fetch_atlas14, name, lat, lon)

print("\nCartopy Natural Earth 50m (coastline, states, borders)")
try:
    import cartopy.io.shapereader as shpreader
    for cat, name in (("physical", "coastline"),
                      ("cultural", "admin_1_states_provinces_lakes"),
                      ("cultural", "admin_0_boundary_lines_land")):
        attempt(f"natural earth {name}", shpreader.natural_earth, '50m', cat, name)
except ImportError as e:
    failures.append("cartopy")
    print(f"  FAILED  cartopy import: {e}")

with open("county_timeseries.yaml") as f:
    ct = yaml.safe_load(f)
print("\nCounties used by plot_county_timeseries.py")
for n in ct.get('counties', []):
    attempt(f"county {n}", sp.county_geom, n, str(ct.get('state_fips', '06')))

print(f"\nDone. {len(failures)} failure(s)" + (f": {failures}" if failures else "."))
