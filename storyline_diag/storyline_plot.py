"""
storyline_plot.py

Shared plotting style, map helpers, boundary downloads (cached), and PNG
saving. Import this BEFORE matplotlib.pyplot in a script, so the Agg backend
and the Avenir/white style are set first:

    import storyline_plot as sp
    import matplotlib.pyplot as plt

Everything downloaded from the web is cached under ~/.cache/storyline_viz/.
Run prefetch.py once on a login node so batch jobs never need the network.
"""

import os
import urllib.request

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import matplotlib.ticker

# ── style ────────────────────────────────────────────────────────────────
FONT_PATH = '/global/homes/j/jsallen/miniconda3/fonts/Avenir-Medium.otf'
try:
    font_manager.fontManager.addfont(FONT_PATH)
    plt.rcParams['font.sans-serif'] = font_manager.FontProperties(fname=FONT_PATH).get_name()
except Exception as e:
    print(f"WARNING: could not load Avenir ({e}); using default sans-serif")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

try:
    import colormaps as _cmaps
    PRECIP_CMAP = _cmaps.precip2_17lev
    DEEP_CMAP = _cmaps.deep
except Exception:
    PRECIP_CMAP = plt.get_cmap('YlGnBu')
    DEEP_CMAP = plt.get_cmap('YlGnBu')

DPI = 300

# Line colors for per-case curves (IDF, IVT), keyed by resolution so a
# missing case never shifts the others' colors.
CASE_COLORS = {"30x8": "darkorange", "30x16": "indianred", "30x32": "maroon"}


def case_color(case):
    return CASE_COLORS.get(case.get('resolution'), 'grey')


def out_path(cfg, name):
    """plots/<event_label>/<name>.png"""
    return os.path.join("plots", cfg.get('event_label', 'unlabeled_event'), f"{name}.png")


def save_png(fig, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"Written: {path}")


def nice_max(arrays, pct=99.5):
    """A round colorbar ceiling from the pct-th percentile of all finite
    values: 1, 2, 2.5, 5 x 10^k. Used when a vmax option is null."""
    vals = np.concatenate([np.ravel(np.asarray(a, dtype=float)) for a in arrays]) if arrays else []
    vals = vals[np.isfinite(vals)] if len(vals) else vals
    if len(vals) == 0:
        return 1.0
    v = float(np.percentile(np.abs(vals), pct))
    if v <= 0:
        return 1.0
    k = 10 ** np.floor(np.log10(v))
    for m in (1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        if m * k >= v:
            return m * k
    return 10 * k


# ── geometry helpers ─────────────────────────────────────────────────────
def bbox_to_extent(bbox):
    """(min_lat, min_lon, max_lat, max_lon) -> [min_lon, max_lon, min_lat, max_lat]"""
    a, b, c, d = bbox
    return [b, d, a, c]


def extent_to_bbox(extent):
    a, b, c, d = extent
    return (c, a, d, b)


def pad_bbox(bbox, pad):
    a, b, c, d = bbox
    return (a - pad, b - pad, c + pad, d + pad)


def points_in(geom, lat, lon):
    """Boolean mask (same shape as lat) of points inside a shapely geometry."""
    lat, lon = np.asarray(lat), np.asarray(lon)
    try:
        import shapely
        m = shapely.contains_xy(geom, lon.ravel(), lat.ravel())
    except AttributeError:          # shapely < 2
        from shapely.geometry import Point
        from shapely.prepared import prep
        p = prep(geom)
        m = np.array([p.contains(Point(x, y)) for x, y in zip(lon.ravel(), lat.ravel())])
    return m.reshape(lat.shape)


# ── downloads (cached) ───────────────────────────────────────────────────
CACHE_DIR = os.path.expanduser("~/.cache/storyline_viz")
CENSUS_COUNTY_ZIP = "https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_county_20m.zip"
WBD_HU2_ZIP_TMPL = ("https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/WBD/HU2/"
                    "Shape/WBD_{region}_HU2_Shape.zip")
ATLAS14_URL = ("https://hdsc.nws.noaa.gov/cgi-bin/hdsc/new/fe_text_mean.csv"
               "?lat={lat:.4f}&lon={lon:.4f}&data=depth&units=english&series=pds")


def cached_download(url, subdir, name):
    dest_dir = os.path.join(CACHE_DIR, subdir)
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, name)
    if not os.path.isfile(dest):
        print(f"  Downloading {url}")
        tmp = dest + ".part"
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, dest)
    return dest


def counties_gdf():
    import geopandas as gpd
    z = cached_download(CENSUS_COUNTY_ZIP, "boundaries", "cb_2018_us_county_20m.zip")
    return gpd.read_file(f"zip://{z}").to_crs(epsg=4326)


def county_geom(name, state_fips="06"):
    gdf = counties_gdf()
    m = gdf[(gdf["STATEFP"] == str(state_fips)) & (gdf["NAME"] == name)]
    if m.empty:
        raise ValueError(f"County '{name}' (STATEFP {state_fips}) not found.")
    return m.geometry.iloc[0]


def counties_in_extent(extent):
    from shapely.geometry import box
    gdf = counties_gdf()
    a, b, c, d = extent
    return gdf[gdf.intersects(box(a, c, b, d))].reset_index(drop=True)


def wbd_layer(hu2, layer):
    """One WBD layer ('WBDHU8', 'WBDHU10') for one HU2 region, EPSG:4326,
    lower-case column names."""
    import geopandas as gpd
    z = cached_download(WBD_HU2_ZIP_TMPL.format(region=hu2), "boundaries",
                        f"WBD_{hu2}_HU2_Shape.zip")
    gdf = gpd.read_file(f"zip://{z}!Shape/{layer}.shp").to_crs(epsg=4326)
    gdf.columns = [c.lower() if c != 'geometry' else c for c in gdf.columns]
    return gdf


def wbd_in_extent(layer, hu2_regions, extent):
    """All polygons of `layer` across the HU2 regions intersecting extent.
    A region that fails to download/read is skipped with a warning."""
    import pandas as pd
    import geopandas as gpd
    from shapely.geometry import box
    a, b, c, d = extent
    clip = box(a, c, b, d)
    pieces = []
    for r in hu2_regions:
        try:
            g = wbd_layer(r, layer)
        except Exception as e:
            print(f"  WARNING: could not read {layer} for HU2 {r} ({e}); skipped.")
            continue
        pieces.append(g[g.intersects(clip)])
    if not pieces:
        raise RuntimeError(f"No {layer} polygons for HU2 regions {hu2_regions}.")
    code = 'huc' + layer[5:]
    out = gpd.GeoDataFrame(pd.concat(pieces, ignore_index=True), crs='EPSG:4326')
    return out.drop_duplicates(subset=code).reset_index(drop=True)


def fetch_atlas14(name, lat, lon):
    path = cached_download(ATLAS14_URL.format(lat=lat, lon=lon), "atlas14",
                           f"{name}_{lat:.4f}_{lon:.4f}.csv")
    with open(path) as f:
        return f.read()


# ── maps ─────────────────────────────────────────────────────────────────
def plot_background(ax, extent):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.4)
    ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=0.5, alpha=0.7)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=0.5, alpha=0.7)


def load_boundaries(opt, extent):
    """{'county': geom, 'huc8': gdf} per opt['boundary'] (list of names)."""
    out = {}
    wanted = opt.get('boundary') or []
    if 'county' in wanted:
        out['county'] = county_geom(opt.get('county', 'Sonoma'), opt.get('state_fips', '06'))
    if 'huc8' in wanted:
        out['huc8'] = wbd_in_extent('WBDHU8', opt.get('hu2_regions', ['18']), extent)
    return out


def add_boundaries(ax, boundaries):
    import cartopy.crs as ccrs
    if boundaries.get('huc8') is not None:
        ax.add_geometries(boundaries['huc8'].geometry, crs=ccrs.PlateCarree(),
                          facecolor='none', edgecolor='black', linewidth=0.2,
                          linestyle='--', zorder=9)
    if boundaries.get('county') is not None:
        ax.add_geometries([boundaries['county']], crs=ccrs.PlateCarree(),
                          facecolor='none', edgecolor='red', linewidth=0.8, zorder=10)


def draw_field(ax, panel, levels, cmap):
    """panel: {'kind': 'grid'|'mesh', 'lat', 'lon', 'val'}. Regular grids
    (AORC) use contourf, the unstructured E3SM mesh uses tricontourf."""
    import cartopy.crs as ccrs
    from matplotlib.tri import Triangulation
    if panel['kind'] == 'grid':
        return ax.contourf(panel['lon'], panel['lat'], np.ma.masked_invalid(panel['val']),
                           levels=levels, cmap=cmap, extend='max',
                           transform=ccrs.PlateCarree())
    return ax.tricontourf(Triangulation(panel['lon'], panel['lat']), panel['val'],
                          levels=levels, cmap=cmap, extend='max', transform=ccrs.PlateCarree())


def na_panel(ax, text):
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha='center', va='center',
            color='0.4', fontsize=10)


def cbar_right(fig, ax, mappable, label, pad=0.02, width=0.015):
    """Vertical colorbar to the right of `ax`, matching its height."""
    try:
        ax.apply_aspect()           # cartopy axes shrink to their aspect at draw time
    except Exception:
        pass
    pos = ax.get_position()
    cax = fig.add_axes([pos.x1 + pad, pos.y0, width, pos.y1 - pos.y0])
    cb = fig.colorbar(mappable, cax=cax, orientation='vertical', label=label)
    cb.ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(6))
    cb.ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%g'))
    return cb


# ── polygons ─────────────────────────────────────────────────────────────
def zonal_stat(lat, lon, values, gdf, how='mean'):
    """Copy of gdf with 'value' = per-polygon `how` ('mean'|'max') of the
    point values inside it. Polygons with no point inside (grid coarser than
    the polygon) take the nearest point's value; 'nearest_fallback' flags them."""
    import geopandas as gpd
    from scipy.spatial import cKDTree
    lat, lon, values = np.ravel(lat), np.ravel(lon), np.ravel(values)
    ok = np.isfinite(values)
    lat, lon, values = lat[ok], lon[ok], values[ok]
    pts = gpd.GeoDataFrame({'value': values}, geometry=gpd.points_from_xy(lon, lat),
                           crs='EPSG:4326')
    joined = gpd.sjoin(pts, gdf[['geometry']], how='inner', predicate='within')
    stat = joined.groupby('index_right')['value'].agg(how)
    out = gdf.copy()
    out['value'] = stat.reindex(out.index)
    out['nearest_fallback'] = out['value'].isna()
    n_missing = int(out['nearest_fallback'].sum())
    if n_missing and len(values):
        rp = out.loc[out['nearest_fallback'], 'geometry'].representative_point()
        _, idx = cKDTree(np.column_stack([lat, lon])).query(np.column_stack([rp.y.values, rp.x.values]))
        out.loc[out['nearest_fallback'], 'value'] = values[idx]
        print(f"    {n_missing}/{len(out)} polygons had no points inside; nearest point used.")
    return out


def choropleth_row(panels, extent, cmap, norm, cbar_label, figsize, linewidth=0.15):
    """One row of polygon maps. panels: [(title, gdf_with_value or None)]."""
    import cartopy.crs as ccrs
    import matplotlib.cm as cm
    fig, axes = plt.subplots(1, len(panels), figsize=figsize, squeeze=False,
                             subplot_kw={'projection': ccrs.PlateCarree()})
    axes = axes[0]
    for ax, (title, g) in zip(axes, panels):
        plot_background(ax, extent)
        if g is None:
            na_panel(ax, "n/a")
        else:
            g.plot(ax=ax, column='value', cmap=cmap, norm=norm, edgecolor='white',
                   linewidth=linewidth, transform=ccrs.PlateCarree(),
                   missing_kwds={'color': 'lightgrey'})
        ax.set_title(title, loc='left')
    fig.subplots_adjust(right=0.90, wspace=0.05)
    cbar_right(fig, axes[-1], cm.ScalarMappable(norm=norm, cmap=cmap), cbar_label)
    return fig, axes
