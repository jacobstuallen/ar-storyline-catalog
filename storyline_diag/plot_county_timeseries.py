"""
plot_county_timeseries.py

County area-mean precipitation for every storyline event: rate (thin lines,
left axis) and running accumulation (thick lines, right axis), one panel per
event, E3SM resolutions and AORC overlaid. One figure per region, where a
region is one county or the union of the listed counties (region_mode).

Run:  python plot_county_timeseries.py
All options live in county_timeseries.yaml (next to this script).

Method:
  - E3SM: simple mean of PRECT over cells whose centers are inside the
    region, accumulated with a trapezoidal integral.
  - AORC: mean over grid points inside the region of each hourly value,
    averaged into `aorc.rate_hours` blocks for the rate line, running sum
    for the accumulation. Hours are stamped at their end (storyline_io.py).
  - Data are read once per event/case over a box around all counties.
  - A case or AORC read that fails is skipped with a printed reason.
"""

import os
import re
from datetime import timedelta

import numpy as np
import yaml
from shapely.ops import unary_union

import storyline_plot as sp
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

from storyline_config import load_event, event_window, data_dir
from storyline_io import (read_e3sm_precip, cumulative_depth, read_aorc_stack, aorc_available,
                          available_cases)

OPTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "county_timeseries.yaml")
with open(OPTS_FILE) as f:
    opt = yaml.safe_load(f)

resolutions = opt.get('resolutions', ["30x8", "30x16"])
units = opt.get('units', 'in')
conv = 25.4 if units == 'in' else 1.0
rate_unit = f"{units}/hr"
aorc_on = opt.get('aorc', {}).get('enabled', True)
aorc_block = int(opt.get('aorc', {}).get('rate_hours', 3))
FALLBACK_COLORS = {"30x8": "#d95f02", "30x16": "#1b9e77", "30x32": "#7570b3"}
AORC_COLOR = "black"


def res_color(res, cfg):
    if res in (opt.get('colors') or {}):
        return opt['colors'][res]
    style = cfg.get('overlay', {}).get('styles', {}).get(res, {})
    if 'color' in style:
        return tuple(c / 255.0 for c in style['color'])
    return FALLBACK_COLORS.get(res, "grey")


def slug(s):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


# ── regions ──────────────────────────────────────────────────────────────
fips = str(opt.get('state_fips', '06'))
names = opt.get('counties', ["Sonoma"])
mode = opt.get('region_mode', 'separate')
geoms = {n: sp.county_geom(n, fips) for n in names}
regions = []
if mode in ('separate', 'both'):
    regions += [(f"{n} County", g) for n, g in geoms.items()]
if mode in ('combined', 'both') and len(geoms) > 1:
    regions.append((" + ".join(names) + " Counties", unary_union(list(geoms.values()))))
if not regions:
    raise ValueError(f"region_mode '{mode}' produced no regions.")
bx0, by0, bx1, by1 = unary_union([g for _, g in regions]).bounds
bbox = (by0 - 0.25, bx0 - 0.25, by1 + 0.25, bx1 + 0.25)


def region_masks(lat, lon):
    masks = []
    for rname, g in regions:
        m = sp.points_in(g, lat, lon)
        if not m.any():
            rp = g.representative_point()
            m = np.zeros(np.shape(lat), dtype=bool)
            m[np.unravel_index(np.argmin((lat - rp.y) ** 2 + (lon - rp.x) ** 2), np.shape(lat))] = True
            print(f"    {rname}: no points inside; using nearest")
        masks.append(m)
    return masks


def e3sm_series(case, t0, t1):
    times, lat, lon, prect = read_e3sm_precip(case, t0, t1, bbox)
    out = []
    for (rname, _), m in zip(regions, region_masks(lat, lon)):
        rate = np.nanmean(prect[:, m], axis=1)
        acc = cumulative_depth(times, rate)
        print(f"    {rname}: {int(m.sum())} cells, total {acc[-1]:.1f} mm")
        out.append((times, rate, times, acc))
    return out


def aorc_series(cfg, t0, t1):
    times, lat, lon, vals = read_aorc_stack(data_dir(cfg, 'aorc_dir'), t0, t1, bbox)
    lon2d, lat2d = np.meshgrid(lon, lat)
    acc_t = np.concatenate([[t0], times])
    k = ((np.array([(t - t0).total_seconds() for t in times]) / 3600.0 - 1e-6) // aorc_block)
    blocks = np.unique(k)
    r_t = np.array([t0 + timedelta(hours=aorc_block * (b + 1)) for b in blocks])
    out = []
    for (rname, _), m in zip(regions, region_masks(lat2d, lon2d)):
        hourly = np.nanmean(vals[:, m], axis=1)
        acc = np.concatenate([[0.0], np.nancumsum(hourly)])
        rate = np.array([np.nanmean(hourly[k == b]) for b in blocks])
        print(f"    {rname}: {int(m.sum())} grid points, total {acc[-1]:.1f} mm")
        out.append((r_t, rate, acc_t, acc))
    return out


# ── read everything ──────────────────────────────────────────────────────
panels = [[] for _ in regions]
used_res, style_cfg = [], None
for ev in opt['events']:
    cpath = ev['config'] if os.path.isabs(ev['config']) else \
        os.path.join(os.path.dirname(OPTS_FILE), ev['config'])
    cfg = load_event(cpath)
    if 'time' in ev:
        cfg['time'] = {**cfg['time'], **ev['time']}
    style_cfg = style_cfg or cfg
    t0, t1 = event_window(cfg)
    title = ev.get('title', cfg.get('event_label'))
    print(f"\n=== {title}: {t0} to {t1} ===")

    series = [[] for _ in regions]
    notes = []
    covered = {c['name'] for c in available_cases(cfg, t0, t1, required=False)}
    for res in resolutions:
        case = next((c for c in cfg['cases'] if c.get('resolution') == res), None)
        if case is None or case['name'] not in covered:
            continue
        print(f"  {res}: {case['name']}")
        try:
            per_region = e3sm_series(case, t0, t1)
        except Exception as e:
            print(f"    SKIPPED ({type(e).__name__}: {e})")
            continue
        label = cfg.get('resolution_labels', {}).get(res, res)
        for r, s in enumerate(per_region):
            series[r].append((res, label, res_color(res, cfg)) + s)
        if res not in used_res:
            used_res.append(res)

    if aorc_on:
        if not aorc_available(t0):
            notes.append("AORC: n/a (pre-1979)")
        else:
            print("  AORC")
            try:
                for r, s in enumerate(aorc_series(cfg, t0, t1)):
                    series[r].append(("AORC", "AORC", AORC_COLOR) + s)
            except Exception as e:
                print(f"    SKIPPED ({type(e).__name__}: {e})")
                notes.append("AORC: n/a")

    for r in range(len(regions)):
        panels[r].append({'title': title, 't0': t0, 't1': t1, 'series': series[r], 'notes': notes})


# ── summary table ────────────────────────────────────────────────────────
for (rname, _), rp in zip(regions, panels):
    print(f"\n{rname} event totals ({units}):")
    for p in rp:
        tot = ", ".join(f"{s[1]} {s[6][-1] / conv:.2f}" for s in p['series']) or "no data"
        print(f"  {p['title']}: {tot}")


def ylims(region_panels):
    rates = [np.nanmax(s[4]) / conv for rp in region_panels for p in rp for s in p['series']]
    accs = [s[6][-1] / conv for rp in region_panels for p in rp for s in p['series']]
    return (opt.get('rate_ymax') or (1.1 * max(rates) if rates else 1.0),
            opt.get('accum_ymax') or (1.1 * max(accs) if accs else 1.0))


shared = ylims(panels) if opt.get('share_ylim_across_regions', True) else None

# ── one figure per region ────────────────────────────────────────────────
ncols = int(opt.get('ncols', 2))
out_dir = opt.get('out_dir', "plots/multi_event")
for (rname, _), rp in zip(regions, panels):
    rate_ymax, accum_ymax = shared or ylims([rp])
    nrows = int(np.ceil(len(rp) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 3.4 * nrows), squeeze=False)
    axes = axes.ravel()
    for i, (ax, p) in enumerate(zip(axes, rp)):
        ax2 = ax.twinx()
        for key, label, color, rt, r, at, a in p['series']:
            if key == "AORC":
                ax.step(rt, r / conv, where='pre', color=color, lw=0.9, alpha=0.8, zorder=2)
            else:
                ax.plot(rt, r / conv, color=color, lw=0.9, alpha=0.8, zorder=2)
            ax2.plot(at, a / conv, color=color, lw=2.4, zorder=3)
        ax.set_ylim(0, rate_ymax)
        ax2.set_ylim(0, accum_ymax)
        ax.set_xlim(p['t0'], p['t1'])
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=8))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        ax.grid(axis='x', color='0.9', lw=0.6)
        ax.set_axisbelow(True)
        ax.set_title(p['title'], loc='left', fontsize=11)
        if i % ncols == 0:
            ax.set_ylabel(f"Rate ({rate_unit})")
        if i % ncols == ncols - 1 or i == len(rp) - 1:
            ax2.set_ylabel(f"Accumulated ({units})")
        if not p['series']:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha='center', va='center', color='0.5')
        if opt.get('show_totals', True) and (p['series'] or p['notes']):
            lines = [f"{s[1]}: {s[6][-1] / conv:.1f} {units}" for s in p['series']] + p['notes']
            ax.text(0.02, 0.96, "\n".join(lines), transform=ax.transAxes, ha='left', va='top',
                    fontsize=7.5, color='0.25', linespacing=1.4,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=2))

    res_labels = style_cfg.get('resolution_labels', {})
    handles = [Line2D([], [], color=res_color(r, style_cfg), lw=2.4, label=res_labels.get(r, r))
               for r in used_res]
    if aorc_on:
        handles.append(Line2D([], [], color=AORC_COLOR, lw=2.4, label="AORC"))
    handles += [Line2D([], [], color='0.4', lw=0.9, label=f"Rate ({rate_unit}), left axis"),
                Line2D([], [], color='0.4', lw=2.4, label=f"Accumulated ({units}), right axis")]
    spare = axes[len(rp):]
    for ax in spare:
        ax.axis('off')
    fig.tight_layout()
    title = f"{rname} mean (UTC)"
    if len(spare):
        spare[0].legend(handles=handles, loc='center', frameon=False, fontsize=10,
                        title=title, title_fontsize=10)
    else:
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.0),
                   ncol=min(len(handles), 4), frameon=False, fontsize=9, title=title)
    sp.save_png(fig, os.path.join(out_dir, f"{slug(rname)}_precip_timeseries.png"))
