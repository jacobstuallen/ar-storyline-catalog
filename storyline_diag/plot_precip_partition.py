"""
plot_precip_partition.py

Convective (PRECC) vs large-scale (PRECL) precipitation during the peak
24-hr episode over Northern California land, for every storyline event and
resolution. Multi-event figure.

Data: the h1 tape, 3-hourly MEAN PRECC / PRECL / PRECT (m/s). Each record is
the mean over the interval ending at its timestamp, so depth = rate x step
(no trapezoid, unlike the instantaneous h0 PRECT used elsewhere).

Regions (REGIONS below), one figure + CSV each:
  norcal       NORCAL box intersected with California (Natural Earth 50m).
  north_coast  Humboldt + Mendocino + Sonoma + Marin (Census counties).
Data are read once per run over READ_BOX, then masked per region. Means are
weighted by the h1 'area' variable when present. The peak 24-hr window is
picked separately for each region.

Episode: the 24-hr window (8 records) with the largest area-mean PRECT.
COMMON window (default): picked from REF_RESOLUTION and applied to every
resolution of that event, so all runs are compared over the same hours.
Each run's OWN peak window is also computed and written to the CSV.

Storm totals (bottom row): the same partition summed over each config's
observed_comparison.storm_window (UTC), or the full event window if a config
has none. Window edges that fall inside a 3-hr record (e.g. 08 UTC) take the
overlapping fraction of that record.

Output: plots/multi_event/precip_partition_<region>.png and .csv

Run:  python plot_precip_partition.py
"""

import csv
import glob
import os
from datetime import datetime, timedelta

import numpy as np
import netCDF4 as nc
import shapely.geometry as sgeom
from shapely.ops import unary_union
import cartopy.io.shapereader as shpreader

import storyline_plot as sp
import matplotlib.pyplot as plt

from storyline_config import load_event, event_window
from storyline_io import read_e3sm, available_cases

# ── options ──────────────────────────────────────────────────────────────
# (config, title, optional time override with the same keys as a config's time: block)
EVENTS = [
    ("config.1964.WUS.G-NDG.yaml", "1964", None),
    ("config.1986.WUS.G-NDG.yaml", "1986", None),
    ("config.2005.WUS.G-NDG.yaml", "2005-06", None),
    ("config.2019.WUS.G-NDG.yaml", "2019", None),
    ("config.2024.WUS.G-NDG.yaml", "2024", None),
]
STREAM = "eam.h1"
READ_BOX = "NORCAL"                # box read from the files; every region must sit inside it
STATE_FIPS = "06"
# One figure + CSV per region. A region is either a state clipped to READ_BOX
# ('state') or a union of Census counties ('counties').
REGIONS = {
    "norcal":      {"label": "NorCal land", "state": "California"},
    "north_coast": {"label": "North Coast", "counties": ["Humboldt", "Mendocino", "Sonoma", "Marin"]},
}
WINDOW_HOURS = 24
REF_RESOLUTION = "30x16"           # None = every run uses its own peak window in the figure
RESOLUTIONS = ["30x8", "30x16", "30x32"]
RES_KM = {"30x8": "14", "30x16": "7", "30x32": "3.5"}
COMP_COLORS = {"PRECL": "#4C78A8", "PRECC": "#F58518"}
EVENT_COLORS = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e"]
CLOSURE_TOL = 0.01                 # warn if |PRECC + PRECL - PRECT| / PRECT exceeds this
OUT_DIR = os.path.join("plots", "multi_event")
OUT_PREFIX = "precip_partition"    # -> precip_partition_<region>.png / .csv


# ── helpers ──────────────────────────────────────────────────────────────
def region_geom(spec, bbox):
    box = sgeom.box(bbox[1], bbox[0], bbox[3], bbox[2])
    if 'state' in spec:
        reader = shpreader.Reader(shpreader.natural_earth('50m', 'cultural', 'admin_1_states_provinces_lakes'))
        g = next(r.geometry for r in reader.records() if r.attributes.get('name') == spec['state'])
        return g.intersection(box)
    g = unary_union([sp.county_geom(c, STATE_FIPS) for c in spec['counties']])
    if not box.contains(g):
        print(f"  WARNING: {spec['label']} extends outside {READ_BOX}; cells outside are dropped")
    return g


def bbox_mask(lat, lon, bbox):
    min_lat, min_lon, max_lat, max_lon = bbox
    return (lat >= min_lat) & (lat <= max_lat) & (lon >= min_lon) & (lon <= max_lon)


def cell_areas(case, bbox):
    """h1 'area' for the cells read_e3sm returns (same bbox selection and order), or None."""
    files = sorted(glob.glob(os.path.join(case['e3sm_dir'], f"*.{STREAM}.*.nc")))
    with nc.Dataset(files[0]) as ds:
        if 'area' not in ds.variables:
            return None
        lat = np.asarray(ds['lat'][:])
        lon = np.asarray(ds['lon'][:])
        lon = np.where(lon > 180, lon - 360, lon)
        area = np.asarray(ds['area'][:])
    return area[bbox_mask(lat, lon, bbox)]


def peak_index(prect_mm, n):
    """Start index of the n-record window with the largest sum."""
    return int(np.argmax(np.convolve(prect_mm, np.ones(n), mode='valid')))


def window_sums(dep, i, n):
    return {k: float(np.sum(v[i:i + n])) for k, v in dep.items()}


def storm_window(cfg, t0, t1):
    """(start, end) from observed_comparison.storm_window, else the event window."""
    sw = (cfg.get('observed_comparison') or {}).get('storm_window')
    if not sw:
        print("  no storm_window in config; using the full event window")
        return t0, t1
    fmt = "%Y-%m-%d %H:%M"
    return datetime.strptime(sw['start'], fmt), datetime.strptime(sw['end'], fmt)


def overlap_sums(times, dep, start, end, step_h):
    """Depth over [start, end]. Record i covers (times[i] - step, times[i]];
    records cut by an edge contribute their overlapping fraction."""
    step = timedelta(hours=step_h)
    frac = np.array([max(0.0, (min(t, end) - max(t - step, start)).total_seconds())
                     for t in times]) / step.total_seconds()
    if times[0] - step > start or times[-1] < end:
        print(f"  WARNING: storm window {start} to {end} extends past the data "
              f"({times[0] - step} to {times[-1]})")
    return {k: float(np.sum(v * frac)) for k, v in dep.items()}


def frac(d):
    return d['PRECC'] / d['PRECT'] if d['PRECT'] > 0 else np.nan


# ── read and reduce ──────────────────────────────────────────────────────
rows = {r: [] for r in REGIONS}       # region -> one row per event x resolution
plotted = {r: {} for r in REGIONS}    # region -> (title, resolution) -> peak 24-hr sums
storm = {r: {} for r in REGIONS}      # region -> (title, resolution) -> storm-window sums
geoms = {}

for config, title, time_override in EVENTS:
    cfg = load_event(config)
    if time_override:
        cfg['time'] = {**cfg['time'], **time_override}
    t0, t1 = event_window(cfg)
    step_h = cfg['time'].get('hours_step', 3)
    n = int(round(WINDOW_HOURS / step_h))
    bbox = tuple(cfg['regions'][READ_BOX])
    if not geoms:
        geoms = {r: region_geom(spec, bbox) for r, spec in REGIONS.items()}

    cfg_h1 = {**cfg, 'cases': [{**c, 'e3sm_stream': STREAM} for c in cfg['cases']
                               if c['resolution'] in RESOLUTIONS]}
    print(f"\n===== {title}: {t0} to {t1} =====")
    s0, s1 = storm_window(cfg, t0, t1)
    print(f"  storm window {s0} to {s1}")
    cases = available_cases(cfg_h1, t0, t1, required=False)

    series = {r: {} for r in REGIONS}
    for case in cases:
        res = case['resolution']
        try:
            times, lat, lon, v = read_e3sm(case, t0, t1, bbox, ('PRECC', 'PRECL', 'PRECT'))
        except KeyError:
            print(f"  {res}: no PRECT in {STREAM}; using PRECC + PRECL")
            times, lat, lon, v = read_e3sm(case, t0, t1, bbox, ('PRECC', 'PRECL'))
            v['PRECT'] = v['PRECC'] + v['PRECL']

        dt = np.unique([(b - a).total_seconds() / 3600 for a, b in zip(times[:-1], times[1:])])
        if len(dt) != 1 or dt[0] != step_h:
            print(f"  WARNING {res}: record spacing {dt} hr, expected {step_h}")

        area = cell_areas(case, bbox)
        if area is None:
            print(f"  {res}: no 'area' in {STREAM}; equal weights")
        to_mm = 1000.0 * 3600.0 * step_h          # m/s mean rate -> mm per record

        for rkey, g in geoms.items():
            m = sp.points_in(g, lat, lon)
            if not m.any():
                print(f"  {res} / {rkey}: no cells inside; skipped")
                continue
            w = np.ones(int(m.sum())) if area is None else area[m]
            w = w / w.sum()
            dep = {k: (v[k][:, m] * to_mm) @ w for k in ('PRECC', 'PRECL', 'PRECT')}
            resid = np.abs(dep['PRECC'] + dep['PRECL'] - dep['PRECT']).sum() / max(dep['PRECT'].sum(), 1e-9)
            if resid > CLOSURE_TOL:
                print(f"  WARNING {res} / {rkey}: PRECC + PRECL differs from PRECT by {100 * resid:.1f}%")
            print(f"  {res} / {rkey}: {int(m.sum())} cells, event total {dep['PRECT'].sum():.1f} mm")
            series[rkey][res] = (times, dep, int(m.sum()))

    for rkey in REGIONS:
        ser = series[rkey]
        if not ser:
            print(f"  no usable runs for {title} / {rkey}")
            continue

        # Peak window is chosen per region: the coast can peak on a different day.
        ref = REF_RESOLUTION if REF_RESOLUTION in ser else next(iter(ser))
        ref_times, ref_dep, _ = ser[ref]
        common_t = ref_times[peak_index(ref_dep['PRECT'], n)]

        for res in RESOLUTIONS:
            if res not in ser:
                continue
            times, dep, ncell = ser[res]
            i_own = peak_index(dep['PRECT'], n)
            own = window_sums(dep, i_own, n)
            hit = np.where(times == common_t)[0]
            if len(hit) and hit[0] + n <= len(times):
                i_com = int(hit[0])
                com = window_sums(dep, i_com, n)
            else:
                print(f"  {res} / {rkey}: common window start {common_t} not in this run; using its own peak")
                i_com, com = i_own, own

            plotted[rkey][(title, res)] = com if REF_RESOLUTION else own
            tot = overlap_sums(times, dep, s0, s1, step_h)
            storm[rkey][(title, res)] = tot
            rows[rkey].append({
                'event': title, 'resolution': res, 'km': RES_KM[res], 'cells': ncell,
                'common_ref': ref,
                'common_start_utc': times[i_com] - timedelta(hours=step_h),
                'common_end_utc': times[i_com + n - 1],
                'common_PRECC_mm': com['PRECC'], 'common_PRECL_mm': com['PRECL'],
                'common_PRECT_mm': com['PRECT'], 'common_conv_frac': frac(com),
                'own_start_utc': times[i_own] - timedelta(hours=step_h),
                'own_end_utc': times[i_own + n - 1],
                'own_PRECC_mm': own['PRECC'], 'own_PRECL_mm': own['PRECL'],
                'own_PRECT_mm': own['PRECT'], 'own_conv_frac': frac(own),
                'storm_start_utc': s0, 'storm_end_utc': s1,
                'storm_PRECC_mm': tot['PRECC'], 'storm_PRECL_mm': tot['PRECL'],
                'storm_PRECT_mm': tot['PRECT'], 'storm_conv_frac': frac(tot),
            })

if not any(rows.values()):
    raise SystemExit("No event had usable h1 output.")


# ── figure ───────────────────────────────────────────────────────────────
def draw_row(ax1, ax2, data, titles, ylabel, title_bar, title_frac, legends=True):
    # stacked depths
    gap = len(RESOLUTIONS) + 1
    xt, xl = [], []
    for e, title in enumerate(titles):
        for j, res in enumerate(RESOLUTIONS):
            s = data.get((title, res))
            if s is None:
                continue
            x = e * gap + j
            ax1.bar(x, s['PRECL'], color=COMP_COLORS['PRECL'], width=0.85,
                    label='PRECL (large-scale)' if not xt else None)
            ax1.bar(x, s['PRECC'], bottom=s['PRECL'], color=COMP_COLORS['PRECC'], width=0.85,
                    label='PRECC (convective)' if not xt else None)
            xt.append(x)
            xl.append(RES_KM[res])
        ax1.text(e * gap + (len(RESOLUTIONS) - 1) / 2, -0.11, title, transform=ax1.get_xaxis_transform(),
                 ha='center', va='top', fontsize=11)
    ax1.set_xticks(xt)
    ax1.set_xticklabels(xl, fontsize=8)
    ax1.set_ylabel(ylabel)
    ax1.set_title(title_bar, loc='left')
#    ax1.text(1.0, 1.01, "bar labels: grid spacing (km)", transform=ax1.transAxes, ha='right',
#             va='bottom', fontsize=8, color='0.4')
    if legends:
        ax1.legend(frameon=False, loc='upper right')
    ax1.spines[['top', 'right']].set_visible(False)

    # convective fraction vs resolution
    for e, title in enumerate(titles):
        xs, ys = [], []
        for j, res in enumerate(RESOLUTIONS):
            s = data.get((title, res))
            if s is not None and s['PRECT'] > 0:
                xs.append(j)
                ys.append(100 * s['PRECC'] / s['PRECT'])
        ax2.plot(xs, ys, '-o', color=EVENT_COLORS[e % len(EVENT_COLORS)], label=title, lw=1.5, ms=5)
    ax2.set_xticks(range(len(RESOLUTIONS)))
    ax2.set_xticklabels([f"{RES_KM[r]} km" for r in RESOLUTIONS])
    ax2.set_xlim(-0.3, len(RESOLUTIONS) - 0.7)
    ax2.set_ylim(bottom=-1)
    ax2.axhline(0, color='dimgray', lw=0.5)
    ax2.set_ylabel("PRECC / PRECT (%)")
    ax2.set_title(title_frac, loc='left')
    if legends:
        ax2.legend(frameon=False, fontsize=9)
    ax2.spines[['top', 'right']].set_visible(False)


os.makedirs(OUT_DIR, exist_ok=True)
for rkey, spec in REGIONS.items():
    if not rows[rkey]:
        continue
    label = spec['label']

    # table
    csv_path = os.path.join(OUT_DIR, f"{OUT_PREFIX}_{rkey}.csv")
    with open(csv_path, 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[rkey][0]))
        wr.writeheader()
        for r in rows[rkey]:
            wr.writerow({k: (f"{x:.3f}" if isinstance(x, float) else x) for k, x in r.items()})
    print(f"\nWritten: {csv_path}")

    for kind, head in (('common', f"Peak {WINDOW_HOURS}-hr (common window)"), ('storm', "Storm total")):
        print(f"\n{head} area-mean depth, {label}:")
        print(f"  {'event':<8} {'res':>6} {'window (UTC)':<35} {'PRECC':>7} {'PRECL':>7} {'conv %':>7}")
        for r in rows[rkey]:
            win = f"{r[kind + '_start_utc']:%Y-%m-%d %H} to {r[kind + '_end_utc']:%Y-%m-%d %H}"
            print(f"  {r['event']:<8} {r['resolution']:>6} {win:<35} {r[kind + '_PRECC_mm']:7.1f} "
                  f"{r[kind + '_PRECL_mm']:7.1f} {100 * r[kind + '_conv_frac']:7.1f}")

    # figure
    titles = [t for _, t, _ in EVENTS if any((t, res) in plotted[rkey] for res in RESOLUTIONS)]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.0), gridspec_kw={'width_ratios': [2.3, 1]})
    fig.subplots_adjust(wspace=0.25, hspace=0.35, bottom=0.06, left=0.08, right=0.92, top=0.96)
    draw_row(axes[0, 0], axes[0, 1], plotted[rkey], titles, f"Area-mean {WINDOW_HOURS}-hr depth (mm)",
             f"a) Peak 24-hr precipitation, {label}", "b) Convective fraction, peak 24 hr")
    draw_row(axes[1, 0], axes[1, 1], storm[rkey], titles, "Area-mean storm-total depth (mm)",
             f"c) Storm-total precipitation, {label}", "d) Convective fraction, storm total",
             legends=False)

    # Same fraction axis in both rows so b and d compare directly.
    top = max(axes[0, 1].get_ylim()[1], axes[1, 1].get_ylim()[1])
    for ax in axes[:, 1]:
        ax.set_ylim(-1, top)

    sp.save_png(fig, os.path.join(OUT_DIR, f"{OUT_PREFIX}_{rkey}.png"))
