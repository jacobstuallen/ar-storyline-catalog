"""
plot_idf_overlay.py

For each named location: NOAA Atlas 14 depth-duration-frequency curves,
with this event's max depth at each duration (1 hr to 10 days) from the
nearest E3SM cell for every case overlaid. One panel per location.
Options: plots.idf_overlay in defaults.yaml (locations, return_periods).

E3SM output is 3-hourly, so durations of 3 hr or less use peak rate x
duration.

VERIFY: the PFDS CSV layout assumed here (an "ARI (years):" header line,
then "5-min:, ..." rows, with later tables for the confidence bounds) is
from memory of the format. Cached CSVs are in ~/.cache/storyline_viz/atlas14/;
open one to confirm. If parsing fails the raw text is printed.

Run:  python plot_idf_overlay.py      (event: STORYLINE_CONFIG, see storyline_config.py)
"""

import re

import numpy as np
from scipy.spatial import cKDTree

import storyline_plot as sp
import matplotlib.pyplot as plt

from storyline_config import load_event, event_window, plot_opts, resolve_locations
from storyline_io import read_e3sm_precip, cumulative_depth, available_cases

DURATIONS_HR = {"60-min": 1, "2-hr": 2, "3-hr": 3, "6-hr": 6, "12-hr": 12, "24-hr": 24,
                "2-day": 48, "4-day": 96, "7-day": 168, "10-day": 240}
DURATION_ROW_RE = re.compile(
    r"^\s*(5-min|10-min|15-min|30-min|60-min|2-hr|3-hr|6-hr|12-hr|24-hr|"
    r"2-day|3-day|4-day|7-day|10-day|20-day|30-day|45-day|60-day)\s*:?\s*,(.+)$")
ATLAS14_DEFAULT_YEARS = [1, 2, 5, 10, 25, 50, 100, 200, 500, 1000]

cfg = load_event()
opt = plot_opts(cfg, 'idf_overlay')
t0, t1 = event_window(cfg)
hours_step = cfg['time'].get('hours_step', 3)
locs = resolve_locations(cfg, opt.get('locations'))
years_wanted = opt.get('return_periods', [2, 10, 25, 100, 500, 1000])


def parse_atlas14(raw, name):
    """{duration: {return_period: depth_in}} from the first table (estimates)."""
    header, table = None, {}
    for line in raw.splitlines():
        if header is None and "ari" in line.lower() and ":" in line:
            yrs = [int(v) for v in re.findall(r"\b(\d+)\b", line.split(":", 1)[1])]
            if yrs:
                header = yrs
            continue
        m = DURATION_ROW_RE.match(line)
        if not m or m.group(1) in table:
            continue
        depths = []
        for v in m.group(2).split(","):
            try:
                depths.append(float(v))
            except ValueError:
                pass
        years = header or ATLAS14_DEFAULT_YEARS
        if header is None and not table:
            print(f"  WARNING ({name}): no ARI header; assuming {ATLAS14_DEFAULT_YEARS}.")
        if len(depths) != len(years):
            print(f"  WARNING ({name}, {m.group(1)}): {len(depths)} values vs {len(years)} periods.")
        table[m.group(1)] = dict(zip(years, depths))
    if not table:
        raise ValueError(f"No duration rows parsed for {name}. Raw:\n{raw[:2000]}")
    return table


def max_depths(times, rate):
    """{duration: max depth (in)} at one cell."""
    cum = cumulative_depth(times, rate)
    out = {}
    for label, hr in DURATIONS_HR.items():
        if hr <= hours_step:
            out[label] = np.nanmax(rate) * hr / 25.4
            continue
        steps = int(round(hr / hours_step))
        out[label] = np.nanmax(cum[steps:] - cum[:-steps]) / 25.4 if steps < len(cum) else np.nan
    return out


# ── read each case once, over a box around all locations ─────────────────
lats = [v[0] for v in locs.values()]
lons = [v[1] for v in locs.values()]
bbox = (min(lats) - 0.5, min(lons) - 0.5, max(lats) + 0.5, max(lons) + 0.5)
event_points = {name: {} for name in locs}      # name -> {case_label: depths}
cases = available_cases(cfg, t0, t1)
for case in cases:
    print(f"Reading E3SM: {case['name']} ...")
    times, lat, lon, prect = read_e3sm_precip(case, t0, t1, bbox)
    tree = cKDTree(np.column_stack([lat, lon]))
    for name, (la, lo) in locs.items():
        _, j = tree.query([la, lo])
        event_points[name][case['label']] = max_depths(times, prect[:, j])
        print(f"  {name}: 24-hr max {event_points[name][case['label']]['24-hr']:.1f} in")

# ── figure ───────────────────────────────────────────────────────────────
ncols = min(3, len(locs))
nrows = int(np.ceil(len(locs) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.5 * nrows), squeeze=False)
axes = axes.ravel()
for ax, (name, (la, lo)) in zip(axes, locs.items()):
    table = parse_atlas14(sp.fetch_atlas14(name, la, lo), name)
    durs = [d for d in DURATIONS_HR if d in table]
    x = [DURATIONS_HR[d] for d in durs]
    avail = {y for d in durs for y in table[d]}
    years = [y for y in years_wanted if y in avail]
    for yr, c in zip(years, plt.cm.Blues(np.linspace(0.35, 1.0, len(years)))):
        ax.plot(x, [table[d].get(yr, np.nan) for d in durs], color=c, lw=1.3, marker='o',
                ms=3, label=f"{yr}-yr")
    for case in cases:
        dep = event_points[name][case['label']]
        ax.plot(x, [dep.get(d, np.nan) for d in durs], color=sp.case_color(case), lw=2.2,
                marker='^', ms=7, zorder=10, label=case['label'])
    ax.set_xscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(durs, rotation=45, ha='right')
    ax.minorticks_off()
    ax.set_ylabel("Precipitation depth (in)")
    ax.set_title(name.replace('_', ' '), loc='left')
    ax.grid(True, which='major', ls=':', alpha=0.5)
    ax.legend(fontsize=8, ncol=2)
for ax in axes[len(locs):]:
    ax.axis('off')
fig.tight_layout()
sp.save_png(fig, sp.out_path(cfg, "idf_overlay"))
