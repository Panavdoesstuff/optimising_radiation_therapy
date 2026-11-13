"""Figures. Light-mode static PNGs (report/slides). Colours from a validated categorical palette:
structure identity = colour (fixed order), plan identity = line style."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

SURFACE = "#fcfcfb"; INK = "#0b0b0b"; INK2 = "#52514e"; GRID = "#e4e3df"
STRUCT_COLOR = {"PTV_68": "#2a78d6", "PTV_56": "#eb6834", "Rectum": "#1baf7a",
                "Bladder": "#eda100", "BODY": "#e87ba4"}
PLAN_STYLE = {"easy": "-", "bao": "--", "dvc": ":"}
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5",
                                                     "#256abf", "#184f95", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", ["#184f95", "#3987e5", "#f0efec", "#e66767", "#a52a2a"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
    "legend.frameon": False, "lines.linewidth": 2,
})


def dvh_curve(doses, n=600, dmax=None):
    dmax = dmax or doses.max()
    grid = np.linspace(0, dmax, n)
    s = np.sort(doses)
    frac = 1.0 - np.searchsorted(s, grid, side="right") / len(s)
    return grid, 100 * frac


LABEL_AT = {"PTV_68": (60, (6, 0)), "PTV_56": (75, (-52, -4)), "Rectum": (30, (6, 2)),
            "Bladder": (50, (6, 2)), "BODY": (12, (6, 2))}   # (volume %, text offset)


def fig_dvh(path, plans, sets, structures, constraints=(), title=None, xmax=100.0):
    """plans: list of (key, label, dose_vector). Doses beyond xmax are noted, not drawn."""
    fig, ax = plt.subplots(figsize=(9, 5.6))
    dmax = max(d[sets[s]].max() for _, _, d in plans for s in structures) * 1.02
    for key, label, d in plans:
        for s in structures:
            g, v = dvh_curve(d[sets[s]], dmax=dmax)
            ax.plot(g, v, PLAN_STYLE[key], color=STRUCT_COLOR[s], lw=1.8 if key == "easy" else 2.0)
    # direct labels: structure name at the easy plan's curve, near 50% volume
    key0, _, d0 = plans[0]
    for s in structures:
        g, v = dvh_curve(d0[sets[s]], dmax=dmax)
        vol_at, off = LABEL_AT.get(s, (50, (6, 2)))
        i = np.argmin(np.abs(v - vol_at))
        ax.annotate(s.replace("_", " "), (g[i], v[i]), xytext=off, textcoords="offset points",
                    color=INK, fontsize=9, fontweight="bold")
    for (organ, t, alpha) in constraints:
        ax.plot([t], [100 * alpha], marker="v", ms=10, color=STRUCT_COLOR[organ],
                markeredgecolor=INK, markeredgewidth=1.2, zorder=5, ls="none")
        ax.annotate(f"limit: {organ} ≤{alpha:.0%} above {t:g} Gy", (t, 100 * alpha), xytext=(8, -14),
                    textcoords="offset points", fontsize=9, color=INK)
    # legends: plan = line style (ink), structure = colour
    from matplotlib.lines import Line2D
    h1 = [Line2D([], [], color=INK, ls=PLAN_STYLE[k], label=l) for k, l, _ in plans]
    h2 = [Line2D([], [], color=STRUCT_COLOR[s], label=s.replace("_", " ")) for s in structures]
    leg1 = ax.legend(handles=h1, loc="upper right", title="Plan", fontsize=9)
    ax.add_artist(leg1)
    ax.legend(handles=h2, loc="center right", title="Structure", fontsize=9)
    ax.set_xlabel("Dose (Gy)"); ax.set_ylabel("Volume receiving ≥ dose (%)")
    over = [f"{l.split(':')[0]}: {d[sets['BODY']].max():.0f} Gy" for _, l, d in plans] if "BODY" in structures else []
    if over and dmax > xmax:
        fig.text(0.99, 0.01, "BODY maximum dose (beyond the axis) — " + ";  ".join(over),
                 ha="right", va="bottom", fontsize=8.5, color=INK2)
    ax.set_xlim(0, min(dmax, xmax)); ax.set_ylim(0, 102)
    ax.set_title(title or "Dose-volume histograms")
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(path, dpi=200); plt.close(fig)


def fig_dose_maps(path, vols, masks, z, diffs, crop, vmax):
    """vols: 3 x (label, 3D dose [ap, rl, si]); diffs: 2 x (label, 3D diff); masks: name -> 3D bool.
    Same colour scale for all plans; symmetric diverging scale for differences."""
    (a0, a1), (b0, b1) = crop
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.4), constrained_layout=True)
    def show(ax, img, **kw):
        im = ax.imshow(img[a0:a1, b0:b1], origin="upper", interpolation="nearest", **kw)
        for nm, mk in masks.items():
            m2 = mk[a0:a1, b0:b1, z]
            if m2.any():
                ax.contour(m2.astype(float), levels=[0.5], colors=[STRUCT_COLOR[nm]], linewidths=1.3)
        ax.set_axis_off()
        return im
    for ax, (label, vol) in zip(axes[0], vols):
        im = show(ax, vol[:, :, z], cmap=SEQ, vmin=0, vmax=vmax); ax.set_title(label)
    fig.colorbar(im, ax=list(axes[0]), shrink=0.85, label="Dose (Gy)")
    lim = max(np.abs(d[a0:a1, b0:b1, z]).max() for _, d in diffs) or 1.0
    for ax, (label, dif) in zip(axes[1, 1:], diffs):
        im2 = show(ax, dif[:, :, z], cmap=DIV, norm=TwoSlopeNorm(0, -lim, lim)); ax.set_title(label)
    fig.colorbar(im2, ax=list(axes[1, 1:]), shrink=0.85, label="Dose difference (Gy)")
    from matplotlib.lines import Line2D
    axes[1, 0].set_axis_off()
    axes[1, 0].legend(handles=[Line2D([], [], color=STRUCT_COLOR[n], label=n.replace("_", " ")) for n in masks],
                      loc="center", fontsize=10, title="Outlines")
    fig.suptitle(f"Axial dose, slice {z} (centre of PTV 68); anterior at top, patient right on the left")
    fig.savefig(path, dpi=200); plt.close(fig)


def fluence_image(bi, x):
    """Lay beamlet intensities on the BEAMINFO x/y grid."""
    xs = np.unique(np.round(bi["x"], 4)); ys = np.unique(np.round(bi["y"], 4))
    img = np.full((len(ys), len(xs)), np.nan)
    ix = np.searchsorted(xs, np.round(bi["x"], 4)); iy = np.searchsorted(ys, np.round(bi["y"], 4))
    img[iy, ix] = x
    return img, xs, ys


def fig_fluence(path, rows, vmax):
    """rows: list of (row_label, [(angle, beaminfo, x_beam), ...])."""
    ncol = max(len(r[1]) for r in rows)
    fig, axes = plt.subplots(len(rows), ncol, figsize=(2.6 * ncol, 2.9 * len(rows)), squeeze=False)
    for r, (label, beams) in enumerate(rows):
        for c in range(ncol):
            ax = axes[r, c]; ax.grid(False)
            if c >= len(beams):
                ax.axis("off"); continue
            ang, bi, xb = beams[c]
            img, xs, ys = fluence_image(bi, xb)
            im = ax.imshow(img, cmap=SEQ, vmin=0, vmax=vmax, origin="lower", interpolation="nearest",
                           extent=[xs.min() - .25, xs.max() + .25, ys.min() - .25, ys.max() + .25])
            ax.set_title(f"{ang}°  (Σ={xb.sum()/1e3:.0f}k)", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            for sp_ in ax.spines.values():
                sp_.set_visible(False)
            if c == 0:
                ax.set_ylabel(label)
    fig.colorbar(im, ax=axes, shrink=0.7, label="Beamlet intensity")
    fig.suptitle("Fluence maps (beamlet intensities on the BEAMINFO grid)")
    fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def fig_price(path, rows, ref_label):
    """rows: list of (label, {Rectum, Bladder, BODY weighted terms}, note). Horizontal stacked bars."""
    terms = ["Rectum", "Bladder", "BODY"]
    fig, ax = plt.subplots(figsize=(9.5, 0.9 * len(rows) + 1.4))
    ref = next(sum(t.values()) for l, t, _ in rows if l == ref_label)
    for i, (label, t, note) in enumerate(rows):
        left = 0.0
        for k in terms:
            ax.barh(i, t[k], left=left, color=STRUCT_COLOR[k], height=0.6,
                    edgecolor=SURFACE, linewidth=2, label=k if i == 0 else None)
            left += t[k]
        tot = sum(t.values())
        txt = f"{tot:.2f}" if label == ref_label else f"{tot:.2f}  ({tot-ref:+.2f}, {100*(tot-ref)/ref:+.1f}% vs easy)"
        ax.text(left + 0.5, i, txt + (f"   {note}" if note else ""), va="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows])
    ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    ax.set_xlabel("Weighted objective = 1·mean(Rectum) + 0.6·mean(Bladder) + 0.6·mean(BODY)   [Gy]")
    ax.set_xlim(0, max(sum(t.values()) for _, t, _ in rows) * 1.75)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), title="Objective term", ncol=3, fontsize=9)
    ax.set_title("Price of realism: objective by plan, split into its three terms")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def fig_sweep(path, Ks, objs, bounds, lb_all, easy_point):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(Ks, bounds, objs, color="#cde2fb", alpha=0.8, lw=0, label="Unproven gap (bound → plan)")
    ax.plot(Ks, objs, "-o", color="#2a78d6", ms=8, label="Best plan found (verified)")
    ax.plot(Ks, bounds, "--", color=INK2, lw=1.5, label="Best proven lower bound")
    ax.axhline(lb_all, color=INK2, lw=1, ls=":")
    ax.annotate(f"all 20 candidate beams (LP): {lb_all:.2f}", (Ks[-1], lb_all), xytext=(0, -14),
                textcoords="offset points", ha="right", fontsize=9, color=INK2)
    ax.plot([easy_point[0]], [easy_point[1]], "s", color="#eb6834", ms=9,
            markeredgecolor=SURFACE, markeredgewidth=2, label="Easy plan (5 equispaced beams)")
    for K, o in zip(Ks, objs):
        ax.annotate(f"{o:.1f}", (K, o), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9)
    ax.set_xlabel("Maximum number of beams K"); ax.set_ylabel("Weighted objective (Gy)")
    ax.set_xticks(Ks); ax.set_ylim(0, max(objs + [easy_point[1]]) * 1.12)
    ax.legend(loc="lower left", fontsize=9)
    ax.set_title("Beam angle selection: objective vs number of beams")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def fig_rectum_distance(path, dist_proj, ptv_proj, dist_r, dose_plans, t, alpha):
    """Why the rectum V_t limit cannot go much lower.
    dist_proj: 2D [si, ap] min distance (mm) of rectum voxels to the PTV along the left-right axis
    (NaN where no rectum); ptv_proj: 2D bool PTV projection; dist_r: per-voxel distances (mm);
    dose_plans: list of (label, per-voxel rectum dose)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True,
                             gridspec_kw=dict(width_ratios=[1.0, 1.25, 1.1]))
    # (a) where the rectum is, coloured by distance to the PTV
    ax = axes[0]; ax.grid(False)
    rows = np.where(np.isfinite(dist_proj).any(1) | ptv_proj.any(1))[0]
    cols = np.where(np.isfinite(dist_proj).any(0) | ptv_proj.any(0))[0]
    r0, r1 = max(rows.min() - 3, 0), rows.max() + 4
    c0, c1 = max(cols.min() - 3, 0), cols.max() + 4
    im = ax.imshow(dist_proj[r0:r1, c0:c1], cmap=SEQ.reversed(), vmin=0, vmax=np.nanpercentile(dist_r, 98),
                   origin="upper", interpolation="nearest")
    ax.contour(ptv_proj[r0:r1, c0:c1].astype(float), levels=[0.5], colors=[STRUCT_COLOR["PTV_68"]], linewidths=1.5)
    ax.set_xticks([]); ax.set_yticks([])
    for sp_ in ax.spines.values():
        sp_.set_visible(False)
    ax.set_title("Rectum (sagittal projection), coloured by\ndistance to the PTV; blue outline = PTV")
    ax.set_xlabel("anterior ←   → posterior"); ax.set_ylabel("superior ↑")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Distance to PTV (mm)")
    # (b) dose vs distance
    ax = axes[1]
    styles = ["o", "s"]
    for (label, dr), mk in zip(dose_plans, styles):
        jit = np.random.default_rng(0).uniform(-0.6, 0.6, size=len(dist_r))
        ax.scatter(dist_r + jit, dr, s=9, marker=mk, alpha=0.45, color="#2a78d6" if mk == "o" else "#eb6834",
                   edgecolors="none", label=label)
    ax.axhline(t, color=INK, lw=1.2, ls="--")
    ax.annotate(f"{t:g} Gy threshold", (np.percentile(dist_r, 97), t),
                xytext=(0, 5), textcoords="offset points", ha="right", fontsize=9, color=INK)
    ax.set_xlabel("Distance to PTV (mm, small jitter added)"); ax.set_ylabel("Rectum voxel dose (Gy)")
    ax.set_title("Dose falls with distance; voxels near the PTV stay hot")
    ax.legend(loc="upper right", fontsize=9, markerscale=2)
    # (c) cumulative share of the rectum within a distance
    ax = axes[2]
    ds = np.sort(dist_r); frac = 100 * np.arange(1, len(ds) + 1) / len(ds)
    ax.step(ds, frac, where="post", color="#1baf7a", lw=2)
    r_a = float(np.percentile(dist_r, 100 * alpha))
    ax.axhline(100 * alpha, color=INK2, lw=1, ls=":")
    ax.axvline(r_a, color=INK2, lw=1, ls=":")
    ax.plot([r_a], [100 * alpha], "o", color="#1baf7a", ms=9, markeredgecolor=SURFACE, markeredgewidth=2)
    ax.annotate(f"{alpha:.0%} of the rectum lies within\n{r_a:.1f} mm of the PTV", (r_a, 100 * alpha),
                xytext=(r_a + 12, 100 * alpha + 6), textcoords="data", fontsize=9, color=INK, va="bottom",
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax.set_xlabel("Distance to PTV (mm)"); ax.set_ylabel("Rectum volume within this distance (%)")
    ax.set_ylim(0, 102); ax.set_xlim(0, np.percentile(dist_r, 99))
    ax.set_title(f"Share of the rectum close to the PTV")
    fig.suptitle(f"Why the rectal limit (≤{alpha:.0%} above {t:g} Gy) cannot be pushed much lower")
    fig.savefig(path, dpi=200); plt.close(fig)


def fig_sweep_certified(path, Ks, opt10, cand_note, K20, ub20, lb20, lb_all, easy_point):
    """Certified optimum vs K on the 10-candidate grid, plus uncertified 20-candidate plans."""
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(Ks, opt10, "-o", color="#2a78d6", ms=8, zorder=3,
            label=f"Certified optimum, {cand_note} (enumeration, gap 0)")
    for K, o in zip(Ks, opt10):
        ax.annotate(f"{o:.1f}", (K, o), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=9)
    if K20:
        ax.vlines(K20, lb20, ub20, color=INK2, lw=1, alpha=0.6, zorder=1)
        ax.plot(K20, ub20, "D", color="#1baf7a", ms=7, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3,
                label="Best plan found, 20 candidates (not certified); line = gap to bound")
    ax.axhline(lb_all, color=INK2, lw=1, ls=":")
    ax.annotate(f"all 20 candidate beams (LP floor): {lb_all:.2f}", (max(Ks), lb_all), xytext=(0, -14),
                textcoords="offset points", ha="right", fontsize=9, color=INK2)
    ax.plot([easy_point[0]], [easy_point[1]], "s", color="#eb6834", ms=9, markeredgecolor=SURFACE,
            markeredgewidth=2, zorder=4, label="Easy LP: 5 equispaced beams")
    ax.set_xlabel("Maximum number of beams K"); ax.set_ylabel("Weighted objective (Gy)")
    ax.set_xticks(sorted(set(Ks) | set(K20)))
    ax.set_ylim(min(lb_all, min(lb20) if lb20 else lb_all) * 0.9, max(list(opt10) + list(ub20) + [easy_point[1]]) * 1.08)
    ax.legend(loc="upper right", fontsize=8.5)
    ax.set_title("Price of a beam budget (normal-tissue cap on)")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)
