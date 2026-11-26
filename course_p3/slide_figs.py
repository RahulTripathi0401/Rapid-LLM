"""Presentation-sized figures (large text, deck palette). Output: course_p3/figures/slides/."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OUT = os.path.join(HERE, "figures", "slides")
os.makedirs(OUT, exist_ok=True)

BG = "#FBFBF8"
INK = "#14213D"
MUTED = "#5B6577"
AR = "#2F6FB3"      # blue = autoregressive
DIFF = "#D9711C"    # orange = block diffusion
DIFF2 = "#8A3B0A"   # dark orange
DIFF3 = "#F2B27A"   # light orange
GRID = "#D9DDE5"

plt.rcParams.update({
    "font.size": 20, "axes.titlesize": 22, "axes.labelsize": 20, "legend.fontsize": 17,
    "xtick.labelsize": 18, "ytick.labelsize": 18, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 1, "figure.facecolor": BG, "axes.facecolor": BG,
    "savefig.facecolor": BG, "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 3.2,
    "lines.markersize": 8,
})


def speedups(d, keys):
    ar = d[d.schedule == "AR"].set_index(keys).decode_s
    x = d[d.schedule != "AR"].copy()
    x["speedup"] = [ar[tuple(r[k] for k in keys)] / r.decode_s for _, r in x.iterrows()]
    return x


def call_time():
    d = pd.read_csv(os.path.join(RES, "crossover.csv"))
    d = d[(d.hw == "a100_80GB_1gpu") & (d.S == 512) & (d.fits.astype(str) == "True")]
    fig, ax = plt.subplots(figsize=(10, 6.4))
    for label, g, c in [("AR: 1 position per call", d[d.schedule == "AR"], AR),
                        ("Diffusion b = 8: 8 positions per call", d[(d.b == 8) & (d.R == 4)], DIFF)]:
        g = g.sort_values("B")
        ax.plot(g.B, g.ms_per_call, marker="o", color=c, label=label)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 4, 16, 64, 256], ["1", "4", "16", "64", "256"])
    ax.set_xlabel("batch B (sequences)")
    ax.set_ylabel("ms per model call")
    ax.set_ylim(0, 150)
    ax.annotate("memory-bound:\nextra positions are free", xy=(4, 13), xytext=(1.1, 55), color=MUTED, fontsize=17,
                arrowprops=dict(arrowstyle="->", color=MUTED))
    ax.annotate("compute-bound:\ncost grows with B·b", xy=(128, 72.6), xytext=(14, 105), color=MUTED, fontsize=17,
                arrowprops=dict(arrowstyle="->", color=MUTED))
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "call_time.png"), dpi=150)


def sweep():
    d = pd.read_csv(os.path.join(RES, "sweep.csv"))
    x = speedups(d, ["S", "B"])
    cfgs = [((8, 4), DIFF, "-"), ((16, 4), DIFF2, "-"), ((8, 2), DIFF3, "-"), ((16, 8), MUTED, "--")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.8), sharey=True)
    for ax, S, name in zip(axes, (2048, 8192, 32768), ("2K", "8K", "32K"), strict=True):
        sub = x[x.S == S]
        for (b, r), c, ls in cfgs:
            g = sub[(sub.b == b) & (sub.R == r)].sort_values("B")
            ax.plot(g.B, g.speedup, ls, marker="o", color=c, label=f"b = {b}, R = {r}")
        ax.axhline(1, color=INK, lw=1.6)
        ax.set_xscale("log", base=2)
        bs = [v for v in sorted(sub.B.unique()) if v != 64]
        ax.set_xticks(bs, [str(int(v)) for v in bs])
        ax.set_title(f"prompt {name}  (max batch {int(sub.Bmax.iloc[0])})", color=INK)
        ax.set_xlabel("batch B")
        ax.set_ylim(0, 4.4)
    axes[0].set_ylabel("decode speedup vs AR")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(os.path.join(OUT, "sweep.png"), dpi=150)


def crossover():
    d = pd.read_csv(os.path.join(RES, "crossover.csv"))
    d = d[(d.S == 512) & (d.fits.astype(str) == "True")]
    x = speedups(d, ["hw", "B"])
    fig, ax = plt.subplots(figsize=(10, 6.4))
    for hw, ls, hname in (("a100_80GB_1gpu", "-", "A100"), ("H100_SXM5_80GB", "--", "H100")):
        for (b, r), c in (((8, 4), DIFF), ((16, 4), DIFF2)):
            g = x[(x.hw == hw) & (x.b == b) & (x.R == r)].sort_values("B")
            ax.plot(g.B, g.speedup, ls, marker="o", color=c, label=f"{hname}  b = {b}, R = {r}")
    ax.axhline(1, color=INK, lw=1.6)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 4, 16, 64, 256], ["1", "4", "16", "64", "256"])
    ax.set_xlabel("batch B (prompt 512, output 128)")
    ax.set_ylabel("decode speedup vs AR")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "crossover.png"), dpi=150)


if __name__ == "__main__":
    call_time()
    sweep()
    crossover()
    print("written to", OUT)
