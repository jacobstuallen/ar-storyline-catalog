"""
plot_ncfr_maps.py

Lat/lon maps of cold-front structure leading up to the peak North Coast
rainfall hour, to look for narrow cold frontal rainbands (NCFRs).

For each case in the event:
  1. Timing. Area-mean precip over the North Coast counties (cells inside the
     union of COUNTIES, simple mean). The max 24-h window is first located
     with h0 PRECT (3-hourly instantaneous), then refined with h5 PRECT
     aggregated to hourly means. T0 = end of the wettest hour inside that
     24-h window. Falls back to h0 alone if a case has no h5.
  2. h0 frames, T0-48 h to T0+24 h, every 3 h. Six panels:
       PSL + 850 wind  | T850 + |grad T850| | V850
       OMEGA850        | 850 rel. vorticity | PRECT (instantaneous)
     850 fields are masked where PS < 850 hPa (if h0 has PS).
     |grad T850| contours are hidden over land (GRADT_MASK_LAND). Land comes
     from h0 LANDFRAC, else 1 - OCNFRAC, else Natural Earth 50m land.
  3. A static copy of the frame nearest T0 - 6 h.
  4. h5 hourly precip frames, T0-12 h to T0+24 h.
  Both windows are clipped to the event's time: block.
  5. One timing figure and CSV for all cases.

Fields are regridded (linear, Delaunay) to a regular lat/lon grid with
spacing GRID_DX[resolution], so gradients and vorticity can be computed.
Colour levels are shared across all cases in the event.

Run:
    STORYLINE_CONFIG=config.2019.WUS.G-NDG.yaml python plot_ncfr_maps.py

Outputs in plots/<event_label>/ncfr/:
    ncfr_timing.png, ncfr_timing.csv
    ncfr_Tminus6h_<res>.png
    ncfr_h0_<res>.gif     (frames in h0_<res>/)
    ncfr_h5_<res>.gif     (frames in h5_<res>/)
"""

import csv
import glob
import os
from datetime import timedelta

import numpy as np
import netCDF4 as nc
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay, cKDTree
from shapely.ops import unary_union

import storyline_plot as sp
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

import storyline_io as sio
from storyline_config import load_event, event_window

# ── options ──────────────────────────────────────────────────────────────
COUNTIES = ["Humboldt", "Mendocino", "Sonoma", "Marin"]
STATE_FIPS = "06"
MAP_EXTENT = [-132.0, -119.0, 34.0, 44.0]       # [min_lon, max_lon, min_lat, max_lat]
GRID_DX = {"30x8": 0.125, "30x16": 0.0625, "30x32": 0.03125}   # regrid spacing, deg
H0_HOURS_BEFORE = 48             # h0 animation starts this many hours before T0
H0_HOURS_AFTER = 24              # ... and ends this many hours after T0
H5_HOURS_BEFORE = 12             # h5 hourly animation, same idea
H5_HOURS_AFTER = 24
STATIC_OFFSET_H = -6             # static snapshot relative to T0
H5_STREAM = "eam.h5"
PSL_INTERVAL_HPA = 4
GRADT_CONTOURS = [8, 14, 20]     # K / 100 km
GRADT_MASK_LAND = True           # hide |grad T850| contours over land (terrain artifacts)
LAND_THRESHOLD = 0.5             # grid point is land if land fraction > this
BARBS_ACROSS = 22                # approx. number of barbs across the map
GIF_WIDTH_PX = 1800              # frames are downsampled to this for the GIFs
GIF_SECONDS_PER_FRAME = 0.6

R_EARTH = 6.371e6

# ── setup ────────────────────────────────────────────────────────────────
cfg = load_event()
t0_ev, t1_ev = event_window(cfg)
cases = sio.available_cases(cfg, t0_ev, t1_ev)
out_dir = os.path.join("plots", cfg['event_label'], "ncfr")
os.makedirs(out_dir, exist_ok=True)

county_union = unary_union([sp.county_geom(n, STATE_FIPS) for n in COUNTIES])
county_geoms = [sp.county_geom(n, STATE_FIPS) for n in COUNTIES]
bx0, by0, bx1, by1 = county_union.bounds
COUNTY_BBOX = (by0 - 0.25, bx0 - 0.25, by1 + 0.25, bx1 + 0.25)
MAP_BBOX = sp.pad_bbox(sp.extent_to_bbox(MAP_EXTENT), 1.0)
county_label = ", ".join(COUNTIES)


# ── helpers ──────────────────────────────────────────────────────────────
def county_mask(lat, lon):
    m = sp.points_in(county_union, lat, lon)
    if not m.any():
        raise ValueError("No cells inside the county union.")
    return m


def h0_vars(case):
    f = sorted(glob.glob(os.path.join(case['e3sm_dir'], f"*.{case.get('e3sm_stream', 'eam.h0')}.*.nc")))[0]
    with nc.Dataset(f) as ds:
        return set(ds.variables)


def h5_case(case):
    c = dict(case)
    c['e3sm_stream'] = H5_STREAM
    return c


def read_h5_hourly(case, t0, t1, bbox):
    """Hourly mean PRECT (mm/hr) for hours ending in (t0, t1], cells in bbox.
    Each h5 record goes to the hour containing its averaging-interval midpoint
    (from time_bnds if present, else time minus half the record spacing).
    Columns are read as one contiguous block per file, not the full mesh.
    Hours with missing records are NaN. Returns (hour_ends, lat, lon, rate)."""
    c = h5_case(case)
    idx = sio._e3sm_file_index(c, t0 - timedelta(hours=1), t1 + timedelta(hours=1))
    if not idx:
        raise FileNotFoundError(f"No {H5_STREAM} records for {case['name']} in {t0} to {t1}")
    min_lat, min_lon, max_lat, max_lon = bbox
    cols = lat = lon = None
    mids, vals = [], []
    for f, times, keep in idx:
        with nc.Dataset(f) as ds:
            if cols is None:
                la = np.asarray(ds.variables['lat'][:])
                lo = np.asarray(ds.variables['lon'][:])
                lo = np.where(lo > 180, lo - 360, lo)
                cols = np.where((la >= min_lat) & (la <= max_lat) &
                                (lo >= min_lon) & (lo <= max_lon))[0]
                lat, lon = la[cols], lo[cols]
                c0, c1 = cols.min(), cols.max() + 1
            if 'time_bnds' in ds.variables:
                tb = ds.variables['time_bnds'][keep]
                tv = ds.variables['time']
                b0 = sio.nc_times(_fake_tvar(tb[:, 0], tv))
                b1 = sio.nc_times(_fake_tvar(tb[:, 1], tv))
                mid = np.array([a + (b - a) / 2 for a, b in zip(b0, b1)])
            else:
                tt = times[keep]
                dt = np.median(np.diff(times)) if len(times) > 1 else timedelta(minutes=10)
                mid = tt - dt / 2
            sl = slice(keep[0], keep[-1] + 1)
            block = np.asarray(ds.variables['PRECT'][sl, c0:c1])[keep - keep[0]]
            vals.append(block[:, cols - c0])
            mids.append(mid)
    mids = np.concatenate(mids)
    vals = np.concatenate(vals, axis=0) * 3.6e6
    mids, first = np.unique(mids, return_index=True)       # restart overlap
    vals = vals[first]

    hour_of = np.array([(m.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)) for m in mids])
    h_start = t0.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    hours = []
    h = h_start
    while h <= t1:
        hours.append(h)
        h += timedelta(hours=1)
    hours = np.array(hours)
    counts = np.array([(hour_of == h).sum() for h in hours])
    n_exp = int(np.median(counts[counts > 0])) if (counts > 0).any() else 0
    rate = np.full((len(hours), len(cols)), np.nan)
    for i, h in enumerate(hours):
        sel = hour_of == h
        if n_exp and sel.sum() == n_exp:
            rate[i] = vals[sel].mean(axis=0)
    n_bad = int((counts != n_exp).sum())
    if n_bad:
        print(f"    h5: {n_bad} of {len(hours)} hours incomplete (expected {n_exp} records/hr), set NaN")
    return hours, lat, lon, rate


class _fake_tvar:
    """Minimal stand-in so sio.nc_times can decode time_bnds with time's units."""
    def __init__(self, values, tv):
        self._v = np.asarray(values)
        self.units = tv.units
        self.calendar = getattr(tv, 'calendar', 'standard')

    def __getitem__(self, k):
        return self._v[k]


def rolling24(times, depth_per_step, steps):
    """(end_index, 24-h depth) of the wettest `steps`-long run."""
    s = np.convolve(np.nan_to_num(depth_per_step), np.ones(steps), mode='valid')
    i = int(np.argmax(s))
    return i + steps - 1, float(s[i])


def report_spacing(lat, lon, name):
    """Median nearest-neighbour spacing (km) in the four quadrants of the map,
    to check the domain lies inside the refined region."""
    phi, lam = np.radians(lat), np.radians(lon)
    xyz = np.c_[np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi)]
    d, _ = cKDTree(xyz).query(xyz, k=2)
    km = d[:, 1] * R_EARTH / 1e3
    lon_c = 0.5 * (MAP_EXTENT[0] + MAP_EXTENT[1])
    lat_c = 0.5 * (MAP_EXTENT[2] + MAP_EXTENT[3])
    parts = []
    for tag, m in (("NW", (lon < lon_c) & (lat >= lat_c)), ("NE", (lon >= lon_c) & (lat >= lat_c)),
                   ("SW", (lon < lon_c) & (lat < lat_c)), ("SE", (lon >= lon_c) & (lat < lat_c))):
        parts.append(f"{tag} {np.median(km[m]):.1f}" if m.any() else f"{tag} n/a")
    print(f"    {name} median cell spacing (km): " + ", ".join(parts))


def regridder(lat, lon, dx):
    glon = np.arange(MAP_EXTENT[0], MAP_EXTENT[1] + dx / 2, dx)
    glat = np.arange(MAP_EXTENT[2], MAP_EXTENT[3] + dx / 2, dx)
    tri = Delaunay(np.c_[lon, lat])
    GX, GY = np.meshgrid(glon, glat)

    def f(v):
        return LinearNDInterpolator(tri, v)(GX, GY)
    return glon, glat, f


def ddx_ddy(fld, glon, glat):
    dlam = np.radians(glon[1] - glon[0])
    dphi = np.radians(glat[1] - glat[0])
    coslat = np.cos(np.radians(glat))[:, None]
    dfdy = np.gradient(fld, axis=0) / (R_EARTH * dphi)
    dfdx = np.gradient(fld, axis=1) / (R_EARTH * coslat * dlam)
    return dfdx, dfdy


def land_mask(case, t, flat, flon, vars_avail, rg, glon, glat):
    """Boolean land mask on the regular grid."""
    for v, conv in (('LANDFRAC', lambda x: x), ('OCNFRAC', lambda x: 1.0 - x)):
        if v in vars_avail:
            _, la, lo, d = sio.read_e3sm(case, t, t, MAP_BBOX, (v,))
            if len(la) == len(flat) and np.allclose(la, flat) and np.allclose(lo, flon):
                frac = rg(conv(d[v][0]))
                print(f"    land mask from {v}")
                return np.nan_to_num(frac, nan=1.0) > LAND_THRESHOLD
    import cartopy.io.shapereader as shpreader
    geoms = list(shpreader.Reader(shpreader.natural_earth('50m', 'physical', 'land')).geometries())
    GX, GY = np.meshgrid(glon, glat)
    print("    land mask from Natural Earth 50m land")
    return sp.points_in(unary_union(geoms), GY, GX)


def h_label(t, T0):
    return f"{t:%Y-%m-%d %H:%M} UTC (T{(t - T0).total_seconds() / 3600:+.0f} h)"


# ── pass 1: timing and data ──────────────────────────────────────────────
results = []
for case in cases:
    res = case['resolution']
    print(f"\n=== {case['name']} ({res}) ===")

    # coarse: h0 3-hourly
    times, lat, lon, prect = sio.read_e3sm_precip(case, t0_ev, t1_ev, COUNTY_BBOX)
    cm = county_mask(lat, lon)
    rate0 = np.nanmean(prect[:, cm], axis=1)
    step_h = np.diff(sio.hours_since(times, times[0]))
    incr = 0.5 * (rate0[:-1] + rate0[1:]) * step_h
    n24 = int(round(24 / np.median(step_h)))
    j, d24_h0 = rolling24(times, incr, n24)
    w1_h0 = times[j + 1]
    w0_h0 = w1_h0 - timedelta(hours=24)
    print(f"  h0 max 24-h county mean: {d24_h0:.1f} mm, {w0_h0} to {w1_h0} ({int(cm.sum())} cells)")

    # fine: h5 hourly, refine inside a padded window
    has_h5 = True
    try:
        pad = timedelta(hours=6)
        hrs, hlat, hlon, hrate = read_h5_hourly(case, max(t0_ev, w0_h0 - pad), min(t1_ev, w1_h0 + pad),
                                                COUNTY_BBOX)
        hcm = county_mask(hlat, hlon)
        r1 = np.nanmean(hrate[:, hcm], axis=1)
        k, d24 = rolling24(hrs, r1, 24)
        w1 = hrs[k]
        w0 = w1 - timedelta(hours=24)
        inwin = np.arange(k - 23, k + 1)
        kpk = inwin[np.nanargmax(r1[inwin])]
        T0, peak = hrs[kpk], float(r1[kpk])
        series_h5 = (hrs, r1)
        print(f"  h5 max 24-h county mean: {d24:.1f} mm, {w0} to {w1}")
    except (FileNotFoundError, KeyError, ValueError) as e:
        has_h5 = False
        print(f"  WARNING: no usable h5 ({e}); timing from h0 only")
        w0, w1, d24 = w0_h0, w1_h0, d24_h0
        inwin = np.where((times >= w0) & (times <= w1))[0]
        kpk = inwin[np.argmax(rate0[inwin])]
        T0, peak = times[kpk], float(rate0[kpk])
        series_h5 = None
    print(f"  T0 (peak hour end): {T0}, {peak:.2f} mm/hr")

    # h0 frames
    vars_avail = h0_vars(case)
    want = ['PSL', 'U850', 'V850', 'T850', 'OMEGA850', 'PRECT']
    missing = [v for v in want if v not in vars_avail]
    if missing:
        raise KeyError(f"h0 for {case['name']} is missing {missing}")
    use_ps = 'PS' in vars_avail
    if not use_ps:
        print("  WARNING: h0 has no PS; 850 hPa fields are not masked under terrain")
    ta = max(t0_ev, T0 - timedelta(hours=H0_HOURS_BEFORE))
    tb = min(t1_ev, T0 + timedelta(hours=H0_HOURS_AFTER))
    ft, flat, flon, fv = sio.read_e3sm(case, ta, tb, MAP_BBOX, want + (['PS'] if use_ps else []))
    report_spacing(flat, flon, "h0")
    glon, glat, rg = regridder(flat, flon, GRID_DX.get(res, 0.125))
    land = (land_mask(case, ft[0], flat, flon, vars_avail, rg, glon, glat)
            if GRADT_MASK_LAND else np.zeros((len(glat), len(glon)), dtype=bool))
    frames = []
    for i, t in enumerate(ft):
        g = {v: rg(fv[v][i]) for v in want + (['PS'] if use_ps else [])}
        if use_ps:
            below = g['PS'] < 85000.0
            for v in ('U850', 'V850', 'T850', 'OMEGA850'):
                g[v] = np.where(below, np.nan, g[v])
        dTdx, dTdy = ddx_ddy(g['T850'], glon, glat)
        dvdx, _ = ddx_ddy(g['V850'], glon, glat)
        _, dudy = ddx_ddy(g['U850'], glon, glat)
        frames.append({
            't': t, 'psl': g['PSL'] / 100.0, 'u': g['U850'], 'v': g['V850'],
            't850': g['T850'] - 273.15, 'gradt': np.where(land, np.nan, np.hypot(dTdx, dTdy) * 1e5),
            'omega': g['OMEGA850'], 'vort': (dvdx - dudy) * 1e4, 'prect': g['PRECT'] * 3.6e6,
        })
    print(f"  h0 frames: {len(frames)} ({ft[0]} to {ft[-1]}), grid {len(glat)} x {len(glon)}")

    # h5 hourly maps
    h5_frames = []
    if has_h5:
        try:
            h5a = max(t0_ev, T0 - timedelta(hours=H5_HOURS_BEFORE + 1))
            h5b = min(t1_ev, T0 + timedelta(hours=H5_HOURS_AFTER))
            mh, mlat, mlon, mrate = read_h5_hourly(case, h5a, h5b, MAP_BBOX)
            _, _, rg5 = regridder(mlat, mlon, GRID_DX.get(res, 0.125))
            for i, h in enumerate(mh):
                if h >= T0 - timedelta(hours=H5_HOURS_BEFORE):
                    h5_frames.append({'t': h, 'prect': rg5(mrate[i])})
            print(f"  h5 hourly frames: {len(h5_frames)}")
        except (FileNotFoundError, KeyError, ValueError) as e:
            print(f"  WARNING: h5 map read failed ({e})")

    results.append(dict(case=case, res=res, label=case['label'], w0=w0, w1=w1, d24=d24,
                        T0=T0, peak=peak, has_h5=has_h5, glon=glon, glat=glat,
                        frames=frames, h5_frames=h5_frames,
                        series_h0=(times, rate0), series_h5=series_h5))

# ── shared colour levels ─────────────────────────────────────────────────
def cat(key, src='frames'):
    return [f[key] for r in results for f in r[src]]


psl_all = np.concatenate([np.ravel(a) for a in cat('psl')])
psl_lo = np.floor(np.nanmin(psl_all) / PSL_INTERVAL_HPA) * PSL_INTERVAL_HPA
psl_hi = np.ceil(np.nanmax(psl_all) / PSL_INTERVAL_HPA) * PSL_INTERVAL_HPA
PSL_LEV = np.arange(psl_lo, psl_hi + PSL_INTERVAL_HPA, PSL_INTERVAL_HPA)

t_all = np.concatenate([np.ravel(a) for a in cat('t850')])
T_LEV = np.arange(np.floor(np.nanpercentile(t_all, 0.5)), np.ceil(np.nanpercentile(t_all, 99.5)) + 1, 1.0)

def sym_levels(arrays, n=21):
    m = sp.nice_max(arrays)
    return np.linspace(-m, m, n)

V_LEV = sym_levels(cat('v'))
WSPD_LEV = np.arange(0, 36, 2.5)   # m/s, fixed across events
OMEGA_LEV = sym_levels(cat('omega'))
VORT_LEV = sym_levels(cat('vort'))
P_MAX = sp.nice_max(cat('prect') + cat('prect', 'h5_frames'))
P_LEV = np.linspace(0, P_MAX, 21)
P_LEV[0] = 0.1                    # leave dry areas white
P_CMAP = sp.PRECIP_CMAP


# ── plotting ─────────────────────────────────────────────────────────────
def add_map(ax):
    sp.plot_background(ax, MAP_EXTENT)
    ax.add_geometries(county_geoms, crs=ccrs.PlateCarree(), facecolor='none',
                      edgecolor='black', linewidth=0.4, zorder=6)


def cbar(fig, ax, m, label):
    cb = fig.colorbar(m, ax=ax, orientation='horizontal', shrink=0.85, pad=0.04, aspect=35)
    cb.set_label(label, fontsize=8)
    cb.ax.tick_params(labelsize=7)


def precip_panel(fig, ax, glon, glat, fld, label):
    m = ax.contourf(glon, glat, np.ma.masked_less(fld, P_LEV[0]), levels=P_LEV, cmap=P_CMAP, extend='max', transform=ccrs.PlateCarree())
    cbar(fig, ax, m, label)


def h0_figure(r, f):
    glon, glat = r['glon'], r['glat']
    proj = ccrs.PlateCarree()
    fig, axs = plt.subplots(2, 3, figsize=(10.5, 7), subplot_kw={'projection': proj})
    fig.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.10, hspace=0.12, wspace=0.06)
    axs = axs.ravel()
    for ax in axs:
        add_map(ax)
    tr = dict(transform=proj)

    # PSL + barbs
    ax = axs[0]
    m = ax.contourf(glon, glat, np.hypot(f['u'], f['v']), levels=WSPD_LEV, cmap='Greys',
                    extend='max', alpha=0.6, **tr)
    cbar(fig, ax, m, "850 hPa wind speed (m/s)")
    cs = ax.contour(glon, glat, f['psl'], levels=PSL_LEV, colors='k', linewidths=0.6, **tr)
    ax.clabel(cs, fmt='%d', fontsize=6)
    sk = max(1, len(glon) // BARBS_ACROSS)
    ax.barbs(glon[::sk], glat[::sk], f['u'][::sk, ::sk] * 1.944, f['v'][::sk, ::sk] * 1.944,
             length=4.5, linewidth=0.4, color='tab:blue', **tr)
    ax.set_title("PSL (hPa), 850 hPa wind (barbs in kt)", fontsize=9)

    # T850 + |grad T|
    ax = axs[1]
    m = ax.contourf(glon, glat, f['t850'], levels=T_LEV, cmap='RdYlBu_r', extend='both', **tr)
    ax.contour(glon, glat, f['gradt'], levels=GRADT_CONTOURS, colors='k', linewidths=0.5, **tr)
    cbar(fig, ax, m, "T850 (°C)")
    ax.set_title(rf"T850 + |$\nabla$T850| contours {GRADT_CONTOURS} K/100 km", fontsize=9)

    # V850
    ax = axs[2]
    m = ax.contourf(glon, glat, f['v'], levels=V_LEV, cmap='PuOr_r', extend='both', **tr)
    cbar(fig, ax, m, "V850 (m/s)")
    ax.set_title("850 hPa meridional wind", fontsize=9)

    # OMEGA850
    ax = axs[3]
    m = ax.contourf(glon, glat, f['omega'], levels=OMEGA_LEV, cmap='RdBu', extend='both', **tr)
    cbar(fig, ax, m, "OMEGA850 (Pa/s), negative = ascent")
    ax.set_title("850 hPa vertical velocity", fontsize=9)

    # vorticity
    ax = axs[4]
    m = ax.contourf(glon, glat, f['vort'], levels=VORT_LEV, cmap='RdBu_r', extend='both', **tr)
    cbar(fig, ax, m, r"$\zeta$850 ($10^{-4}$ s$^{-1}$)")
    ax.set_title("850 hPa relative vorticity", fontsize=9)

    # PRECT
    ax = axs[5]
    precip_panel(fig, ax, glon, glat, f['prect'], "PRECT (mm/hr)")
    ax.set_title("Precipitation rate (instantaneous)", fontsize=9)

    fig.suptitle(f"{cfg['event_label']}  {r['label']}  {h_label(f['t'], r['T0'])}", fontsize=12,
                 x=0.05, ha='left')
    return fig


def h5_figure(r, f):
    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=(8, 7), subplot_kw={'projection': proj})
    add_map(ax)
    precip_panel(fig, ax, r['glon'], r['glat'], f['prect'], "hourly mean precip (mm/hr)")
    ax.set_title(f"{cfg['event_label']}  {r['label']}\nhour ending {h_label(f['t'], r['T0'])}", fontsize=10)
    return fig


def make_gif(pngs, path):
    try:
        from PIL import Image
    except ImportError:
        print(f"  Pillow not available; skipping {path}")
        return
    imgs = []
    for p in pngs:
        im = Image.open(p).convert('RGB')
        w, h = im.size
        if w > GIF_WIDTH_PX:
            im = im.resize((GIF_WIDTH_PX, int(h * GIF_WIDTH_PX / w)), Image.LANCZOS)
        imgs.append(im.quantize(colors=256, method=Image.Quantize.MEDIANCUT))
    imgs[0].save(path, save_all=True, append_images=imgs[1:], loop=0,
                 duration=int(GIF_SECONDS_PER_FRAME * 1000))
    print(f"Written: {path}")


for r in results:
    res = r['res']
    print(f"\nPlotting {r['label']}")

    fdir = os.path.join(out_dir, f"h0_{res}")
    pngs = []
    for i, f in enumerate(r['frames']):
        p = os.path.join(fdir, f"frame_{i:03d}.png")
        sp.save_png(h0_figure(r, f), p)
        pngs.append(p)
    make_gif(pngs, os.path.join(out_dir, f"ncfr_h0_{res}.gif"))

    target = r['T0'] + timedelta(hours=STATIC_OFFSET_H)
    i6 = int(np.argmin([abs((f['t'] - target).total_seconds()) for f in r['frames']]))
    sp.save_png(h0_figure(r, r['frames'][i6]),
                os.path.join(out_dir, f"ncfr_Tminus{abs(STATIC_OFFSET_H)}h_{res}.png"))

    if r['h5_frames']:
        fdir = os.path.join(out_dir, f"h5_{res}")
        pngs = []
        for i, f in enumerate(r['h5_frames']):
            p = os.path.join(fdir, f"frame_{i:03d}.png")
            sp.save_png(h5_figure(r, f), p)
            pngs.append(p)
        make_gif(pngs, os.path.join(out_dir, f"ncfr_h5_{res}.gif"))


# ── timing figure + CSV ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
for r in results:
    col = sp.case_color(r['case'])
    t, v = r['series_h0']
    ax.plot(t, v, color=col, lw=0.8, ls='--', alpha=0.7)
    if r['series_h5'] is not None:
        ax.plot(*r['series_h5'], color=col, lw=1.4, label=f"{r['label']} (h5 hourly)")
    else:
        ax.plot([], [], color=col, ls='--', label=f"{r['label']} (h0 only)")
    ax.axvspan(r['w0'], r['w1'], color=col, alpha=0.08)
    ax.axvline(r['T0'], color=col, lw=1.0)
ax.plot([], [], color='0.4', ls='--', lw=0.8, label="h0 3-hourly instantaneous")
ax.set_ylabel("county-mean precip (mm/hr)")
ax.set_title(f"{cfg['event_label']}: {county_label} mean. Shading = max 24-h window, line = T0", fontsize=10)
ax.legend(fontsize=7, frameon=False)
fig.autofmt_xdate()
sp.save_png(fig, os.path.join(out_dir, "ncfr_timing.png"))

csv_path = os.path.join(out_dir, "ncfr_timing.csv")
with open(csv_path, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["event", "case", "resolution", "win24_start_utc", "win24_end_utc",
                "depth24_mm", "T0_peak_hour_end_utc", "peak_rate_mmhr", "timing_source"])
    for r in results:
        w.writerow([cfg['event_label'], r['case']['name'], r['res'], r['w0'], r['w1'],
                    f"{r['d24']:.1f}", r['T0'], f"{r['peak']:.2f}", "h5" if r['has_h5'] else "h0"])
print(f"Written: {csv_path}")
