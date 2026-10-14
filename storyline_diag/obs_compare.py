"""
obs_compare.py

Helpers for comparing storyline runs to observed event stats (storm totals,
calendar-day totals). Used by print_basin_totals.py and plot_huc10_24hr.py.
Options come from the `observed_comparison:` block (defaults.yaml plus the
event config's windows).
"""

import sys
from datetime import date, datetime, timedelta

import numpy as np

from storyline_config import load_event, data_dir
from storyline_io import (read_e3sm_precip, window_depth, read_aorc_sum,
                          aorc_available, available_cases)

MM_PER_IN = 25.4


def load_obs_cfg():
    """Event config, or exit cleanly (status 0) if it has no storm windows."""
    cfg = load_event()
    oc = cfg.get('observed_comparison') or {}
    if 'storm_window' not in oc or 'day_window' not in oc:
        print("SKIP: no observed_comparison storm_window/day_window in this event config.")
        sys.exit(0)
    return cfg, oc


def parse_window(win):
    """{start: 'YYYY-MM-DD HH:MM', end: ...} (UTC) -> (t0, t1)."""
    def _to_dt(v):
        if isinstance(v, datetime):
            return v
        if isinstance(v, date):
            return datetime.combine(v, datetime.min.time())
        return datetime.strptime(str(v).strip(), "%Y-%m-%d %H:%M")
    t0, t1 = _to_dt(win['start']), _to_dt(win['end'])
    if t1 <= t0:
        raise ValueError(f"Window end {t1} is not after start {t0}.")
    return t0, t1


def e3sm_window_accum(case, t0, t1, region_bbox):
    """(lat, lon, depth_mm) over [t0, t1]. Reads a few hours past each edge so
    records bracketing an 08 UTC edge are available."""
    pad = timedelta(hours=6)
    times, lat, lon, prect = read_e3sm_precip(case, t0 - pad, t1 + pad, region_bbox)
    return lat, lon, window_depth(times, prect, t0, t1)


def aorc_window_accum(cfg, t0, t1, region_bbox):
    """(lat1d, lon1d, depth_mm_2d), or None if AORC isn't available."""
    if not aorc_available(t0):
        print(f"  AORC n/a for {t0.year} (record starts 1979).")
        return None
    try:
        return read_aorc_sum(data_dir(cfg, 'aorc_dir'), t0, t1, region_bbox)
    except Exception as e:
        print(f"  AORC SKIPPED ({type(e).__name__}: {e})")
        return None
