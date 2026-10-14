"""
storyline_config.py

Config loading shared by every script. One event YAML is deep-merged over
defaults.yaml (event wins key by key, dicts merge recursively, lists and
scalars are replaced wholesale).

Which event a script runs on:
    STORYLINE_CONFIG=config.2024.WUS.G-NDG.yaml python plot_max_precip.py
or, with the variable unset, DEFAULT_CONFIG below. run_diagnostics.py sets
the variable for each event.

No heavy scientific imports here, so storyline_diagnostics.py can import it
cheaply.
"""

import copy
import os
from datetime import date, datetime, timedelta

import yaml

DEFAULT_CONFIG = "config.1986.WUS.G-NDG.yaml"
HERE = os.path.dirname(os.path.abspath(__file__))


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f) or {}


def deep_merge(base, override):
    """Recursively merge `override` into `base`; override wins. Neither input
    is mutated."""
    merged = copy.deepcopy(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], val)
        else:
            merged[key] = copy.deepcopy(val)
    return merged


def find_defaults_path(case_path, explicit=None):
    """Explicit path, else defaults.yaml next to the event config, else in cwd."""
    if explicit:
        return explicit
    sibling = os.path.join(os.path.dirname(os.path.abspath(case_path)), "defaults.yaml")
    if os.path.isfile(sibling):
        return sibling
    if os.path.isfile("defaults.yaml"):
        return "defaults.yaml"
    return None


def normalize_cases(cfg):
    if 'cases' in cfg:
        return cfg
    if 'case' in cfg:
        cfg = copy.deepcopy(cfg)
        cfg['cases'] = [cfg.pop('case')]
        return cfg
    raise KeyError("Config has neither a top-level 'case:' nor 'cases:' block.")


def resolve_scrip_file(case_cfg, cfg, resolution=None):
    res = resolution or case_cfg.get('resolution')
    lut = cfg.get('resolution_scrip_files', {})
    if res not in lut:
        raise KeyError(f"No scrip_file for resolution '{res}'. Known: {sorted(lut)}.")
    return lut[res]


def resolve_config(case_path, defaults_path=None, require_scrip=True):
    """Merge the event YAML over defaults.yaml and normalize `cases`."""
    case_cfg = load_config(case_path)
    defaults_path = find_defaults_path(case_path, defaults_path)
    base_cfg = load_config(defaults_path) if defaults_path else {}
    cfg = normalize_cases(deep_merge(base_cfg, case_cfg))

    for case in cfg['cases']:
        case.setdefault('e3sm_stream', 'eam.h0')
        if require_scrip and not case.get('scrip_file'):
            try:
                case['scrip_file'] = resolve_scrip_file(case, cfg)
            except KeyError:
                case['scrip_file'] = None

    cfg.setdefault('active_regions', ["ALL"])
    cfg.setdefault('locations', {})
    cfg['_config_path'] = os.path.abspath(case_path)
    return cfg


def validate_config(cfg, require_scrip=True):
    """Raise a clear error for missing fields. A case without `label:` gets
    resolution_labels[resolution] from defaults.yaml, else its name."""
    if not cfg.get('cases'):
        raise KeyError("Config resolved to an empty 'cases' list.")
    res_labels = cfg.get('resolution_labels', {})
    for case in cfg['cases']:
        case.setdefault('label', res_labels.get(case.get('resolution'), case.get('name')))
        missing = [k for k in ('name', 'e3sm_dir', 'resolution') if not case.get(k)]
        if missing:
            raise KeyError(f"case '{case.get('name', '?')}' is missing field(s): {missing}")
        if require_scrip and not case.get('scrip_file'):
            raise KeyError(
                f"No scrip_file for case '{case['name']}' (resolution "
                f"'{case.get('resolution')}'). Known: "
                f"{sorted(cfg.get('resolution_scrip_files', {}))}."
            )
    return cfg


def resolve_locations(cfg, names=None):
    """{name: (lat, lon)} for the requested names (default: all)."""
    all_locs = cfg.get('locations', {}) or {}
    if names is None:
        names = [k for k, v in all_locs.items() if v is not None]
    out, missing = {}, []
    for name in names:
        if all_locs.get(name) is None:
            missing.append(name)
            continue
        lat, lon = all_locs[name]
        out[name] = (float(lat), float(lon))
    if missing:
        raise KeyError(f"Location(s) not defined (or null): {missing}. "
                       f"Known: {sorted(k for k, v in all_locs.items() if v is not None)}")
    return out


# ── helpers used by the plot scripts ─────────────────────────────────────
def config_path_from_env(default=DEFAULT_CONFIG):
    """STORYLINE_CONFIG if set, else `default`. Relative paths are tried
    against the cwd first, then this file's directory."""
    path = os.environ.get("STORYLINE_CONFIG") or default
    if not os.path.isabs(path) and not os.path.isfile(path):
        alt = os.path.join(HERE, path)
        if os.path.isfile(alt):
            path = alt
    return path


def load_event(path=None):
    """Resolved + validated config for one event (no SCRIP file needed)."""
    path = path or config_path_from_env()
    cfg = validate_config(resolve_config(path, None, require_scrip=False), require_scrip=False)
    print(f"Event config: {path}  (event_label: {cfg.get('event_label')})")
    return cfg


def event_dates(cfg):
    t = cfg['time']
    start = date(t['start_year'], t['start_month'], t['start_day'])
    return [start + timedelta(days=i) for i in range(t['num_days'])]


def event_window(cfg, start_day=0, window_days=None):
    """(t0, t1) datetimes, 00 UTC on the first day to 00 UTC after the last.
    start_day/window_days pick a sub-window, in whole days."""
    t = cfg['time']
    n = t['num_days']
    window_days = n - start_day if window_days is None else window_days
    if start_day < 0 or window_days < 1 or start_day + window_days > n:
        raise ValueError(f"start_day={start_day}, window_days={window_days} does not fit "
                         f"inside the event's num_days={n}.")
    t0 = datetime(t['start_year'], t['start_month'], t['start_day']) + timedelta(days=start_day)
    return t0, t0 + timedelta(days=window_days)


def plot_opts(cfg, name):
    """The `plots.<name>` block from the merged config (empty dict if absent)."""
    return dict((cfg.get('plots', {}) or {}).get(name, {}) or {})


def data_dir(cfg, key):
    """`data.<key>` from the merged config, e.g. data_dir(cfg, 'aorc_dir')."""
    return (cfg.get('data', {}) or {}).get(key)
