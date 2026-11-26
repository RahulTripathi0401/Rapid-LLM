"""Figures for the Project 3 report (reads course_p3/results/*.csv)."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"figure.dpi": 140, "font.size": 9, "axes.grid": True, "grid.alpha": 0.3})


def _key(row):
    return f"b={int(row.b)}, R={int(row.R)}"


def required():
    d = pd.read_csv(os.path.join(RES, "required.csv"))
    kv_save = d.kv_save.astype(str) == "True"
    d["variant"] = "diffusion, R calls"
    d.loc[kv_save, "variant"] = "diffusion, R+1 calls"
    d.loc[d.schedule == "AR", "variant"] = "AR"
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    for ax, o in zip(axes, (128, 256), strict=True):
        p = d[d.O == o].pivot(index="B", columns="variant", values="total_s")
        p[["AR", "diffusion, R calls", "diffusion, R+1 calls"]].plot.bar(ax=ax, rot=0)
        ax.set_title(f"Llama-3.1-8B, A100, prompt 512, output {o}")
        ax.set_xlabel("batch B")
        ax.set_ylabel("total time (s)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "required_total_time.png"))


def sweep():
    d = pd.read_csv(os.path.join(RES, "sweep.csv"))
    ar = d[d.schedule == "AR"].set_index(["S", "B"])
    dd = d[d.schedule != "AR"].copy()
    dd["speedup"] = dd.apply(lambda r: ar.loc[(r.S, r.B), "decode_s"] / r.decode_s, axis=1)
    dd["cfg"] = dd.apply(_key, axis=1)
    Ss = sorted(d.S.unique())
    fig, axes = plt.subplots(1, len(Ss), figsize=(4 * len(Ss), 3.4), sharey=True)
    for ax, S in zip(axes, Ss, strict=True):
        sub = dd[dd.S == S]
        for cfg, g in sub.groupby("cfg", sort=False):
            g = g.sort_values("B")
            ax.plot(g.B, g.speedup, marker="o", ms=3, label=cfg)
        ax.axhline(1.0, color="k", lw=1)
        ax.set_xscale("log", base=2)
        ax.set_title(f"prompt {S}, output 128 (max B = {int(sub.Bmax.iloc[0])})")
        ax.set_xlabel("batch B")
    axes[0].set_ylabel("decode speedup vs AR")
    axes[-1].legend(fontsize=7, ncol=1, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "sweep_speedup.png"))

    # tokens/s heat-table for the largest batch per S
    rows = []
    for S in Ss:
        bmax = d[d.S == S].B.max()
        for cfg, g in dd[(dd.S == S) & (dd.B == bmax)].groupby("cfg", sort=False):
            rows.append(dict(S=S, B=bmax, cfg=cfg, speedup=g.speedup.iloc[0]))
    pd.DataFrame(rows).pivot(index="cfg", columns="S", values="speedup").round(2).to_csv(
        os.path.join(RES, "sweep_speedup_at_max_batch.csv")
    )


def crossover():
    d = pd.read_csv(os.path.join(RES, "crossover.csv"))
    d = d[d.fits.astype(str) == "True"]  # only batches whose peak memory fits in 80 GiB
    ar = d[d.schedule == "AR"].set_index(["hw", "S", "B"])
    dd = d[d.schedule != "AR"].copy()
    dd["speedup"] = dd.apply(lambda r: ar.loc[(r.hw, r.S, r.B), "decode_s"] / r.decode_s, axis=1)
    dd["cfg"] = dd.apply(_key, axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4), sharey=True)
    styles = {"a100_80GB_1gpu": "-", "H100_SXM5_80GB": "--"}
    for ax, S in zip(axes, sorted(d.S.unique()), strict=True):
        for (hw, cfg), g in dd[dd.S == S].groupby(["hw", "cfg"], sort=False):
            g = g.sort_values("B")
            ax.plot(g.B, g.speedup, styles.get(hw, "-"), marker="o", ms=2.5,
                    label=f"{'A100' if 'a100' in hw else 'H100'} {cfg}")
        ax.axhline(1.0, color="k", lw=1)
        ax.set_xscale("log", base=2)
        ax.set_title(f"prompt {S}, output 128")
        ax.set_xlabel("batch B")
    axes[0].set_ylabel("decode speedup vs AR")
    axes[1].legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "crossover_a100_h100.png"))

    # crossover batch: first B where speedup < 1
    out = []
    for (hw, S, cfg), g in dd.groupby(["hw", "S", "cfg"], sort=False):
        g = g.sort_values("B")
        lose = g[g.speedup < 1.0]
        out.append(dict(hw=hw, S=S, cfg=cfg, max_fit_B=int(g.B.max()), speedup_B1=round(g.speedup.iloc[0], 2),
                        speedup_at_max_B=round(g.speedup.iloc[-1], 2),
                        first_losing_B=int(lose.B.iloc[0]) if len(lose) else None,
                        best_speedup=round(g.speedup.max(), 2)))
    pd.DataFrame(out).to_csv(os.path.join(RES, "crossover_summary.csv"), index=False)

    # per-call time vs batch: why it flips (memory-bound -> compute-bound)
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    a = d[(d.hw == "a100_80GB_1gpu") & (d.S == 512)]
    for label, g in [("AR (q=1)", a[a.schedule == "AR"]), ("b=8 (q=8)", a[(a.b == 8) & (a.R == 4)]),
                     ("b=32 (q=32)", a[(a.b == 32) & (a.R == 8)])]:
        g = g.sort_values("B")
        ax.plot(g.B, g.ms_per_call, marker="o", ms=3, label=label)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("batch B")
    ax.set_ylabel("ms per model call")
    ax.set_title("A100, prompt 512: cost of one call")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "call_time_vs_batch.png"))


if __name__ == "__main__":
    for f in (required, sweep, crossover):
        try:
            f()
            print(f"{f.__name__}: ok")
        except FileNotFoundError as e:
            print(f"{f.__name__}: skipped ({e})")
