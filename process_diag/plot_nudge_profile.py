"""
plot_nudge_profile.py  (block 1)

NORCAL vertical profiles for one event, one line per resolution:
  left:  area-weighted RMS of Nudge_U (nudging tendency), m/s per hour
  right: area-weighted RMS of U - Target_U, m/s
Shows which levels are nudged and how hard, for picking one free and one
nudged level for the structure functions.

Target_U is ERA5 (6-hourly, linearly interpolated in time by the model), so
U - Target_U also contains time-interpolation error between ERA5 times.

Run:  python plot_nudge_profile.py
"""

import numpy as np

import process_io as pio
import storyline_plot as sp
import matplotlib.pyplot as plt

EVENT_YEAR = pio.event_year(2019)     # run_process.py sets this per event

pcfg = pio.load_process_cfg()
opt = pcfg['nudge_profile']
cfg = pio.find_event(pcfg, EVENT_YEAR)
t0, t1 = pio.analysis_window(cfg, pcfg['window'])
print(f"Window: {t0} to {t1}, every {opt['every_hours']} h")

profiles = []          # (case, pressure_hPa, rms_nudge, rms_diff)
for case in cfg['cases']:
    recs = pio.records(case, pcfg['stream'], t0, t1, opt['every_hours'])
    if not pio.check_coverage(recs, t0, t1, case['name']):
        continue
    cols = pio.Columns(recs[0][0], pcfg['region'])
    w = cols.area / cols.area.sum()
    print(f"  {case['resolution']}: {cols.n} columns, {len(recs)} records")

    sq_nudge = np.zeros(len(cols.lev))
    sq_diff = np.zeros(len(cols.lev))
    p_mean = np.zeros(len(cols.lev))
    for path, rec, t in recs:
        nudge = cols.read(path, rec, 'Nudge_U') * 3600.0        # m/s/s -> m/s/h
        diff = cols.read(path, rec, 'U') - cols.read(path, rec, 'Target_U')
        ps = cols.read(path, rec, 'PS')
        sq_nudge += (nudge ** 2) @ w
        sq_diff += (diff ** 2) @ w
        p_mean += (cols.hyam * cols.p0 + cols.hybm * (ps @ w)) / 100.0
    n = len(recs)
    profiles.append((case, p_mean / n, np.sqrt(sq_nudge / n), np.sqrt(sq_diff / n)))

if not profiles:
    print("SKIP: no case covers the window.")
    raise SystemExit(0)

# printed table for picking levels
print("\n  idx   lev   p(hPa)  " + "  ".join(f"{c['resolution']:>14}" for c, *_ in profiles))
print("                     " + "  ".join(f"{'nudge  U-Tgt':>14}" for _ in profiles))
for k in range(len(profiles[0][1])):
    row = "  ".join(f"{pn[k]:6.3f} {pd[k]:6.2f}" for _, _, pn, pd in profiles)
    print(f"  {k:3d}  {cols.lev[k]:6.1f}  {profiles[0][1][k]:6.1f}  {row}")

fig, axes = plt.subplots(1, 2, figsize=(9, 6), sharey=True)
lo, hi = opt['namelist_window']
p_ref = profiles[0][1]
for case, p, rms_n, rms_d in profiles:
    kw = dict(color=sp.case_color(case), lw=1.6, label=case['label'])
    axes[0].plot(rms_n, p, **kw)
    axes[1].plot(rms_d, p, **kw)
for ax in axes:
    for idx in (lo, hi):
        ax.axhline(p_ref[idx - 1], color='0.5', ls='--', lw=0.8)
    ax.set_ylim(1020, 100)
    ax.set_xlim(left=0)
    ax.grid(alpha=0.3)
axes[0].text(0.98, p_ref[lo - 1], f"Vwin_Lindex={lo}", ha='right', va='bottom',
             fontsize=8, color='0.4', transform=axes[0].get_yaxis_transform())
axes[0].text(0.98, p_ref[hi - 1], f"Vwin_Hindex={hi}", ha='right', va='bottom',
             fontsize=8, color='0.4', transform=axes[0].get_yaxis_transform())
axes[0].set_xlabel("RMS Nudge_U (m s$^{-1}$ h$^{-1}$)")
axes[1].set_xlabel("RMS U $-$ Target_U (m s$^{-1}$)")
axes[0].set_ylabel("Pressure (hPa, NORCAL mean)")
axes[0].legend(frameon=False, fontsize=9)
fig.suptitle(f"{cfg['event_label']}  NORCAL  {t0:%Y-%m-%d %H}Z to {t1:%Y-%m-%d %H}Z")
fig.tight_layout()

sp.save_png(fig, f"{pio.out_dir(pcfg, 'plot_dir', cfg['event_label'])}/nudge_profile.png")
