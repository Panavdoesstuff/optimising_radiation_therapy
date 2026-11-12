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


def fig_dvh(path, plans, sets, structures, constraints=(), title=None):
    """plans: list of (key, label, dose_vector)."""
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
        i = np.argmin(np.abs(v - (88 if s.startswith("PTV") else 55)))
        ax.annotate(s.replace("_", " "), (g[i], v[i]), xytext=(4, 4), textcoords="offset points",
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
    ax.set_xlim(0, dmax); ax.set_ylim(0, 102)
    ax.set_title(title or "Dose-volume histograms")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def fig_dose_maps(path, vols, masks, z, diffs, crop, vmax):
    """vols: list of (label, 3D dose [ap, rl, si]); diffs: list of (label, 3D diff).
    masks: dict name -> 3D bool. Same colour scale for all plans; symmetric scale for diffs."""
    from matplotlib.gridspec import GridSpec
    n_top, n_bot = len(vols), len(diffs)
    ncol = max(n_top, n_bot)
    fig = plt.figure(figsize=(4.1 * ncol, 8.2))
    gs = GridSpec(2, ncol * 2, figure=fig)
    (a0, a1), (b0, b1) = crop
    def show(ax, img, cmap, norm=None, vmin=None, vmax_=None):
        im = ax.imshow(img[a0:a1, b0:b1], cmap=cmap, norm=norm, vmin=vmin, vmax=vmax_,
                       origin="upper", interpolation="nearest")
        for nm, mk in masks.items():
            m2 = mk[a0:a1, b0:b1, z]
            if m2.any():
                ax.contour(m2.astype(float), levels=[0.5], colors=[STRUCT_COLOR[nm]], linewidths=1.3)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        return im
    for i, (label, vol) in enumerate(vols):
        ax = fig.add_subplot(gs[0, 2 * i:2 * i + 2])
        im = show(ax, vol[:, :, z], SEQ, vmin=0, vmax_=vmax)
        ax.set_title(label)
    fig.colorbar(im, ax=fig.axes[:n_top], shrink=0.8, label="Dose (Gy)")
    lim = max(np.abs(d[a0:a1, b0:b1, z]).max() for _, d in diffs) or 1.0
    off = ncol - n_bot
    axs = []
    for i, (label, dif) in enumerate(diffs):
        ax = fig.add_subplot(gs[1, off + 2 * i:off + 2 * i + 2])
        im2 = show(ax, dif[:, :, z], DIV, norm=TwoSlopeNorm(0, -lim, lim))
        ax.set_title(label); axs.append(ax)
    fig.colorbar(im2, ax=axs, shrink=0.8, label="Dose difference (Gy)")
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color=STRUCT_COLOR[n], label=n.replace("_", " ")) for n in masks],
               loc="lower left", ncol=1, fontsize=9, title="Outlines")
    fig.suptitle(f"Axial dose, slice {z} (through the centre of PTV 68); anterior at top")
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
    ax.legend(loc="lower right", title="Objective term", ncol=3, fontsize=9)
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
