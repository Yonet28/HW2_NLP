"""Produce every table and figure of the paper from the JSONL results.

    python -m patchbench.analyze --results results/ --out results/analysis/

Outputs: tables as CSV and LaTeX, figures as PDF and PNG, random failure examples (Markdown),
and summary.json with the hypothesis verdicts.
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from .experiments import EDGE, MIDDLE

B = 2000
MARGIN = 5.0          # equivalence margin for H3, in accuracy points
EN_COLOR, FR_COLOR = "#2a78d6", "#eb6834"
INK, MUTED = "#0b0b0b", "#52514e"


def load(results_dir):
    rows = []
    for p in sorted(glob.glob(os.path.join(results_dir, "*.jsonl"))):
        with open(p) as f:
            rows += [json.loads(line) for line in f if line.strip()]
    df = pd.DataFrame(rows)
    df["model_short"] = df["model"].str.split("/").str[-1]
    for c in ("correct", "trap", "trap_previous", "marker_found", "parsed"):
        df[c] = df[c].astype(float)
    return df


# --------------------------------------------------------------------------- statistics

def boot_ci(stat, units, rng, b=B):
    """Percentile CI of stat(resampled units). units: array of unit ids."""
    vals = []
    for _ in range(b):
        vals.append(stat(rng.choice(units, len(units), replace=True)))
    vals = np.array([v for v in vals if v is not None and not np.isnan(v)])
    return np.percentile(vals, [2.5, 97.5]) if len(vals) else (np.nan, np.nan)


def paired_gap(df, a, b_, key="cond", rng=None):
    """Mean of correct[a] - correct[b] over shared pair_ids, in points, with bootstrap CI."""
    p = df[df[key].isin([a, b_])].pivot_table(index="pair_id", columns=key, values="correct")
    p = p.dropna()
    if p.empty:
        return np.nan, np.nan, np.nan, 0
    d = (p[a] - p[b_]).values * 100
    lo, hi = boot_ci(lambda s: d[s].mean(), np.arange(len(d)), rng)
    return d.mean(), lo, hi, len(d)


def acc_ci(x, rng):
    x = np.asarray(x) * 100
    lo, hi = boot_ci(lambda s: x[s].mean(), np.arange(len(x)), rng)
    return x.mean(), lo, hi


def fmt(m, lo, hi):
    return f"{m:.1f} [{lo:.1f}, {hi:.1f}]" if not np.isnan(m) else "n/a"


# --------------------------------------------------------------------------- analyses

def table_main(df, rng):
    """Accuracy by model, language and number of distractors (native template), with the paired gap."""
    out = []
    m = df[(df.exp == "main") & (df.template == "native")]
    for (model, d), g in m.groupby(["model_short", "d"]):
        row = {"model": model, "d": d}
        for lang in ("en", "fr"):
            row[f"acc_{lang}"] = fmt(*acc_ci(g[g.ctx_lang == lang].correct, rng))
        gap = paired_gap(g, "en", "fr", key="ctx_lang", rng=rng)
        row["gap_en_minus_fr"] = fmt(*gap[:3]); row["n_pairs"] = gap[3]
        row["_gap"] = gap
        out.append(row)
    return out


def h2_interaction(df, rng):
    """Middle drop per language and interaction I = M_fr - M_en (native template), plus a logistic GEE
    with exchangeable correlation within items: correct ~ lang * zone + C(d)."""
    res = {}
    m = df[(df.exp == "main") & (df.template == "native") & df.pos.isin(EDGE | MIDDLE)].copy()
    m["zone"] = np.where(m.pos.isin(MIDDLE), "middle", "edge")
    for model, g in m.groupby("model_short"):
        units = g.pair_id.unique()
        by = {u: x for u, x in g.groupby("pair_id")}

        def stat(sample):
            s = pd.concat([by[u] for u in sample])
            a = s.groupby(["ctx_lang", "zone"]).correct.mean()
            try:
                return ((a["fr", "edge"] - a["fr", "middle"]) - (a["en", "edge"] - a["en", "middle"])) * 100
            except KeyError:
                return np.nan
        a = g.groupby(["ctx_lang", "zone"]).correct.mean() * 100
        I = stat(units)
        lo, hi = boot_ci(stat, units, rng, b=1000)
        r = {"M_en": a["en", "edge"] - a["en", "middle"], "M_fr": a["fr", "edge"] - a["fr", "middle"],
             "I": I, "I_ci": [lo, hi]}
        try:
            import statsmodels.api as sm
            import statsmodels.formula.api as smf
            g2 = g.assign(fr=(g.ctx_lang == "fr").astype(int), mid=(g.zone == "middle").astype(int))
            fit = smf.gee("correct ~ fr * mid + C(d)", groups="pair_id", data=g2,
                          family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()).fit()
            ci = fit.conf_int().loc["fr:mid"].tolist()
            r["gee_fr_x_middle_logodds_(negative=larger_FR_middle_drop)"] = float(fit.params["fr:mid"])
            r["gee_ci"] = ci
            r["gee_p"] = float(fit.pvalues["fr:mid"])
        except Exception as e:          # noqa: BLE001
            r["gee_error"] = str(e)
        res[model] = r
    return res


def decomposition(df, rng):
    """Oracle, template and token-length components of the gap, per model."""
    rows, verdicts = [], {}
    for model, g in df.groupby("model_short"):
        r = {"model": model}
        o = g[g.exp == "oracle"]
        for t in ("native", "arrow"):
            r[f"oracle_gap_{t}"] = paired_gap(o[o.cond == f"oracle_{t}"], "en", "fr", "ctx_lang", rng)
        cb = o[o.cond == "closed_book"]
        r["closed_book_acc"] = cb.correct.mean() * 100 if len(cb) else np.nan
        mm = g[(g.exp == "main") & (g.d == 0)]
        for t in ("native", "arrow"):
            r[f"full_gap_{t}_d0"] = paired_gap(mm[mm.template == t], "en", "fr", "ctx_lang", rng)
        L = g[g.exp == "length"]
        if len(L):
            r["len_content_matched"] = paired_gap(L, "en", "fr", rng=rng)
            r["len_en_long_vs_fr"] = paired_gap(L, "en_long", "fr", rng=rng)
            r["len_en_vs_fr_short"] = paired_gap(L, "en", "fr_short", rng=rng)
            r["len_budget_baseline"] = paired_gap(L, "en", "fr_budget", rng=rng)
            r["len_cost_25"] = paired_gap(L, "en", "en_long", rng=rng)
            r["len_cost_50"] = paired_gap(L, "en", "en_plus50", rng=rng)
            fs = L[L.cond == "fr_short"]
            r["fr_short_matched_rate"] = float(fs["matched"].astype(float).mean()) if "matched" in fs else np.nan
            # share of the content-matched gap explained by length
            p = L.pivot_table(index="pair_id", columns="cond", values="correct").dropna()
            arr = p.values; cols = list(p.columns)
            ci = {c: cols.index(c) for c in ("en", "fr", "en_long", "fr_short")}

            def share(s):
                x = arr[s]
                g0 = (x[:, ci["en"]] - x[:, ci["fr"]]).mean()
                tm = ((x[:, ci["en_long"]] - x[:, ci["fr"]]).mean() + (x[:, ci["en"]] - x[:, ci["fr_short"]]).mean()) / 2
                return np.nan if abs(g0) < 1e-9 else 1 - tm / g0
            r["share_explained_by_length"] = (share(np.arange(len(arr))),
                                              *boot_ci(share, np.arange(len(arr)), rng))
            # H3 verdict with the equivalence margin
            g1, g2 = r["len_en_long_vs_fr"], r["len_en_vs_fr_short"]
            inside = all(-MARGIN <= x[1] and x[2] <= MARGIN for x in (g1, g2))
            outside = all((x[1] > 0 or x[2] < 0) and (x[2] > MARGIN or x[1] < -MARGIN) for x in (g1, g2))
            verdicts[model] = "supported" if inside else ("contradicted" if outside else "inconclusive")
        if len(L):
            fs = L[L.cond == "fr_short"]
            r["fr_short_token_excess_pct"] = float(((fs.ctx_tokens / fs.L_en) - 1).mean() * 100)
        c = g[g.exp == "crosslang"]
        if len(c):
            r["crosslang_acc"] = {k: v * 100 for k, v in c.groupby("cond").correct.mean().items()}
            for cond in ("ctx_en_q_fr", "ctx_fr_q_en", "ctx_fr_q_fr"):
                r[f"crosslang_ctx_en_q_en_minus_{cond}"] = paired_gap(c, "ctx_en_q_en", cond, rng=rng)
        # prompt format and padding (explains the pilot): same items as main, d = 0, native template
        pf = g[g.exp == "promptfmt"]
        if len(pf):
            base = g[(g.exp == "main") & (g.template == "native") & (g.d == 0)].assign(cond="marker_pad1")
            both = pd.concat([pf, base[base.pair_id.isin(pf.pair_id)]])
            for cond in ("marker_pad1", "marker_pad0", "plain_pad1", "plain_pad0"):
                sub = both[both.cond == cond]
                r[f"promptfmt_{cond}_gap"] = paired_gap(sub, "en", "fr", "ctx_lang", rng)
                r[f"promptfmt_{cond}_acc_fr"] = sub[sub.ctx_lang == "fr"].correct.mean() * 100
                r[f"promptfmt_{cond}_prev_fr"] = sub[sub.ctx_lang == "fr"].trap_previous.mean() * 100
        rows.append(r)
    return rows, verdicts


def interference(df):
    m = df[(df.exp == "main") & (df.template == "native")]
    t = m.groupby(["model_short", "ctx_lang", "d"]).agg(
        accuracy=("correct", "mean"), trap_rate=("trap", "mean"), previous_value=("trap_previous", "mean"),
        n=("correct", "size"))
    return (t[["accuracy", "trap_rate", "previous_value"]] * 100).round(1).join(t["n"])


def parsing(df):
    return (df.groupby(["model_short", "q_lang"])[["marker_found", "parsed"]].mean() * 100).round(1)


# --------------------------------------------------------------------------- figures

def fig_position(df, out, rng):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    m = df[(df.exp == "main") & (df.template == "native")]
    models = sorted(m.model_short.unique()); D = sorted(m.d.unique())
    fig, axes = plt.subplots(len(D), len(models), figsize=(3.2 * len(models), 2.6 * len(D)),
                             sharey=True, squeeze=False)
    for i, d in enumerate(D):
        for j, model in enumerate(models):
            ax = axes[i][j]
            g = m[(m.model_short == model) & (m.d == d)]
            for lang, color, mk in (("en", EN_COLOR, "o"), ("fr", FR_COLOR, "s")):
                pts = []
                for pos, x in g[g.ctx_lang == lang].groupby("pos"):
                    pts.append((pos + 1, *acc_ci(x.correct, rng)))
                pts = np.array(pts)
                if not len(pts):
                    continue
                ax.fill_between(pts[:, 0], pts[:, 2], pts[:, 3], color=color, alpha=0.12, lw=0)
                ax.plot(pts[:, 0], pts[:, 1], color=color, lw=2, marker=mk, ms=5,
                        label="English" if lang == "en" else "French")
            ax.set_ylim(0, 100); ax.set_xticks([1, 4, 7, 10, 14, 17, 20])
            ax.grid(axis="y", color="#e6e5e0", lw=0.8); ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            ax.tick_params(colors=MUTED, labelsize=8)
            if i == 0:
                ax.set_title(model, fontsize=9, color=INK)
            if j == 0:
                ax.set_ylabel(f"Accuracy (%)\n{d} distractors", fontsize=8, color=INK)
            if i == len(D) - 1:
                ax.set_xlabel("Patch containing the answer", fontsize=8, color=INK)
    axes[0][0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"fig_position.{ext}"), dpi=200)
    plt.close(fig)


def fig_decomposition(rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = [("oracle_gap_native", "Oracle, native template"), ("oracle_gap_arrow", "Oracle, arrow template"),
            ("full_gap_native_d0", "Full context, native"), ("full_gap_arrow_d0", "Full context, arrow"),
            ("len_en_long_vs_fr", "Token-matched (EN-long vs FR)"), ("len_en_vs_fr_short", "Token-matched (EN vs FR-short)"),
            ("len_budget_baseline", "Fixed token budget (baseline)")]
    rows = [r for r in rows if "len_content_matched" in r]
    if not rows:
        return
    fig, axes = plt.subplots(1, len(rows), figsize=(2.6 * len(rows) + 1.8, 2.8), sharey=True, squeeze=False)
    y = np.arange(len(keys))[::-1]
    for ax, r in zip(axes[0], rows):
        vals = [r.get(k, (np.nan,) * 3) for k, _ in keys]
        m = np.array([v[0] for v in vals])
        err = np.array([[v[0] - v[1] for v in vals], [v[2] - v[0] for v in vals]])
        ax.barh(y, m, height=0.55, color=FR_COLOR)
        ax.errorbar(m, y, xerr=err, fmt="none", ecolor=INK, lw=1, capsize=2)
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.set_yticks(y); ax.set_yticklabels([l for _, l in keys], fontsize=7, color=INK)
        ax.set_title(r["model"], fontsize=9, color=INK)
        ax.set_xlabel("EN minus FR accuracy (points)", fontsize=7.5, color=INK)
        ax.grid(axis="x", color="#e6e5e0", lw=0.8); ax.set_axisbelow(True)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=7)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"fig_decomposition.{ext}"), dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------- examples

def examples(df, out, k=12, seed=0):
    """Randomly sampled failures (not cherry-picked), regenerating the prompt for context."""
    from .data import build_item, render_question
    f = df[(df.exp == "main") & (df.correct == 0)]
    if f.empty:
        return
    s = f.sample(min(k, len(f)), random_state=seed)
    lines = ["# Randomly sampled failures (seed 0)\n"]
    for _, r in s.iterrows():
        lines.append(f"- **{r.model_short}**, {r.ctx_lang.upper()}, {r.template}, patch {r.pos + 1}, "
                     f"{r.d} distractors. Gold {r.answer:g}, previous {r.previous:g}. "
                     f"Output: `{r.output}`. Trap: {bool(r.trap)}, previous value: {bool(r.trap_previous)}.")
    with open(os.path.join(out, "examples_failures.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- main

def to_jsonable(o):
    if isinstance(o, (np.floating, float)):
        return None if np.isnan(o) else round(float(o), 2)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (tuple, list, np.ndarray)):
        return [to_jsonable(x) for x in o]
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/analysis")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    df = load(args.results)

    main_rows = table_main(df, rng)
    t1 = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in main_rows])
    t1.to_csv(os.path.join(args.out, "table1_main.csv"), index=False)
    with open(os.path.join(args.out, "table1_main.tex"), "w") as fh:
        fh.write(t1.to_latex(index=False))

    dec, verdicts = decomposition(df, rng)
    with open(os.path.join(args.out, "table2_decomposition.json"), "w") as fh:
        json.dump(to_jsonable(dec), fh, indent=1)
    flat = []
    for r in dec:
        row = {"model": r["model"]}
        for k, v in r.items():
            if isinstance(v, tuple) and len(v) >= 3:
                row[k] = fmt(*v[:3])
            elif k not in ("model", "crosslang_acc"):
                row[k] = v
        flat.append(row)
    t2 = pd.DataFrame(flat).set_index("model").T
    t2.to_csv(os.path.join(args.out, "table2_decomposition.csv"))
    with open(os.path.join(args.out, "table2_decomposition.tex"), "w") as fh:
        fh.write(t2.to_latex())

    h2 = h2_interaction(df, rng)
    inter = interference(df); inter.to_csv(os.path.join(args.out, "table3_interference.csv"))
    pars = parsing(df); pars.to_csv(os.path.join(args.out, "table_parsing.csv"))
    fig_position(df, args.out, rng)
    fig_decomposition(dec, args.out)
    examples(df, args.out, seed=args.seed)

    summary = {
        "n_rows": len(df), "models": sorted(df.model_short.unique()),
        "H1_main_gap": {r["model"] + f"|d={r['d']}": r["_gap"] for r in main_rows},
        "H2": h2, "H3_verdict": verdicts, "equivalence_margin": MARGIN,
    }
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(to_jsonable(summary), fh, indent=1)
    print(t1.to_string(index=False)); print(); print(t2.to_string()); print()
    print(json.dumps(to_jsonable({"H2": h2, "H3": verdicts}), indent=1)); print(); print(inter); print(pars)


if __name__ == "__main__":
    main()
