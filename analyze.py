"""Turn generations + CLIP scores + human ratings into tables, charts, and a numbers report.

  python analyze.py                       # uses the default CLIP scorer
  python analyze.py --scorer ViT-B-32:laion2b_s34b_b79k

Outputs (in data/results/):
  summary_by_category.csv     model x category: CLIP mean, human pass rate, refusals
  agreement.csv               CLIP vs human agreement, overall / per category / per model
  disagreements.csv           images where CLIP and humans disagree most ("CLIP fooled" / "CLIP too harsh")
  rater_agreement.csv         inter-rater reliability (Krippendorff alpha, pairwise Cohen kappa)
  fig1_human_pass_by_category.png
  fig2_clip_by_category.png
  fig3_clip_vs_human.png      small multiples, one panel per category
  fig4_clip_agreement_by_category.png
  fig5_failure_gallery.png    each model's most embarrassing failures with CLIP score
  report.md                   every headline number, ready to paste into slides
"""
import argparse
import itertools
import json
import textwrap

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score, roc_auc_score

from mdb import style
from mdb.clip_scorer import DEFAULT_SCORER
from mdb.common import (CATEGORY_LABELS, CATEGORY_ORDER, CLIP_CSV, GENERATIONS_CSV, RATINGS_CSV, RESULTS,
                        ensure_dirs, load_prompts, read_csv_or_empty)

style.apply()
import matplotlib.pyplot as plt  # noqa: E402


# ----------------------------------------------------------------------------- data
def load(scorer):
    prompts = load_prompts()
    clip = read_csv_or_empty(CLIP_CSV)
    if clip.empty:
        raise SystemExit("No CLIP scores yet. Run score_clip.py first.")
    clip = clip[clip["scorer"] == scorer].copy()
    if clip.empty:
        raise SystemExit(f"No CLIP scores for scorer {scorer}.")
    # Percentile of each image's CLIP score within its model, so models with different score scales compare.
    clip["clip_pct"] = clip.groupby("model")["clip_score"].rank(pct=True)
    ratings = read_csv_or_empty(RATINGS_CSV)
    gens = read_csv_or_empty(GENERATIONS_CSV)
    return prompts, clip, ratings, gens


def human_per_image(ratings):
    if ratings.empty:
        return pd.DataFrame(columns=["key", "human_pass", "human_overall", "item_rate", "n_raters"])
    r = ratings.copy()
    r["item_frac"] = np.where(r["n_items"] > 0, r["n_passed"] / r["n_items"], np.nan)
    r.loc[r["broken"] == 1, "item_frac"] = 0.0
    return (r.groupby("key")
              .agg(human_pass=("all_pass", "mean"), human_overall=("overall", "mean"),
                   item_rate=("item_frac", "mean"), n_raters=("rater", "nunique"))
              .reset_index())


def se(x):
    x = pd.Series(x).dropna()
    return x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan


# ----------------------------------------------------------------------------- stats
def rater_reliability(ratings):
    rows = []
    if ratings.empty or ratings["rater"].nunique() < 2:
        return pd.DataFrame(rows)
    try:
        import krippendorff
        for col, level in (("all_pass", "nominal"), ("overall", "ordinal")):
            mat = ratings.pivot_table(index="rater", columns="key", values=col, aggfunc="first")
            mat = mat.loc[:, mat.notna().sum() >= 2]
            if mat.shape[1] >= 5:
                a = krippendorff.alpha(reliability_data=mat.to_numpy(dtype=float), level_of_measurement=level)
                rows.append({"measure": f"krippendorff_alpha_{col}", "raters": "all", "value": round(a, 3),
                             "n_images": mat.shape[1]})
    except ImportError:
        pass
    for a, b in itertools.combinations(sorted(ratings["rater"].unique()), 2):
        ra = ratings[ratings["rater"] == a].set_index("key")
        rb = ratings[ratings["rater"] == b].set_index("key")
        shared = ra.index.intersection(rb.index)
        if len(shared) < 10:
            continue
        x, y = ra.loc[shared, "all_pass"], rb.loc[shared, "all_pass"]
        k = cohen_kappa_score(x, y) if len(set(x) | set(y)) > 1 else np.nan
        rows.append({"measure": "cohen_kappa_all_pass", "raters": f"{a} vs {b}",
                     "value": round(k, 3), "n_images": len(shared),
                     "pct_agree": round(float((x.values == y.values).mean()), 3)})
    return pd.DataFrame(rows)


def pairwise_concordance(df):
    """Within each prompt: over image pairs where humans clearly differ, how often does CLIP order them the same?

    This removes prompt difficulty from the comparison (CLIP scores are not comparable across prompts).
    """
    agree = total = 0
    for _, g in df.groupby("prompt_id"):
        v = g[["clip_score", "human_overall"]].to_numpy()
        for i, j in itertools.combinations(range(len(v)), 2):
            dh = v[i, 1] - v[j, 1]
            if abs(dh) < 1:  # humans don't clearly prefer either image
                continue
            total += 1
            agree += np.sign(v[i, 0] - v[j, 0]) == np.sign(dh)
    return (agree / total if total else np.nan), total


def model_winner_agreement(df):
    """Per prompt and model pair: does CLIP pick the same winning model as the human raters?"""
    m = df.groupby(["prompt_id", "category", "model"])[["clip_score", "human_overall"]].mean().reset_index()
    rows = []
    for (pid, cat), g in m.groupby(["prompt_id", "category"]):
        g = g.set_index("model")
        for a, b in itertools.combinations(g.index, 2):
            dh = g.loc[a, "human_overall"] - g.loc[b, "human_overall"]
            if abs(dh) < 0.25:
                continue
            dc = g.loc[a, "clip_score"] - g.loc[b, "clip_score"]
            rows.append({"prompt_id": pid, "category": cat, "agree": int(np.sign(dc) == np.sign(dh))})
    return pd.DataFrame(rows, columns=["prompt_id", "category", "agree"])


def agreement_table(df):
    rows = []

    def add(scope, name, g):
        row = {"scope": scope, "group": name, "n_images": len(g)}
        if len(g) >= 5 and g["clip_score"].nunique() > 1 and g["human_overall"].nunique() > 1:
            row["spearman_clip_vs_overall"] = round(spearmanr(g["clip_score"], g["human_overall"]).statistic, 3)
        passed = (g["human_pass"] >= 0.5).astype(int)
        if passed.nunique() == 2:
            # AUC: probability CLIP ranks a human-passed image above a human-failed one. 0.5 = coin flip.
            row["auc_clip_predicts_pass"] = round(roc_auc_score(passed, g["clip_pct"]), 3)
        c, n = pairwise_concordance(g)
        row["within_prompt_concordance"] = round(c, 3) if n else np.nan
        row["n_pairs"] = n
        rows.append(row)

    add("overall", "all", df)
    for cat in CATEGORY_ORDER:
        if (df["category"] == cat).any():
            add("category", cat, df[df["category"] == cat])
    for m in sorted(df["model"].unique()):
        add("model", m, df[df["model"] == m])
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- charts
def grouped_bars(summary, value, err, ylabel, title, path, ylim=None, pct=False):
    models = sorted(summary["model"].unique())
    cats = [c for c in CATEGORY_ORDER if c in set(summary["category"])]
    fig, ax = plt.subplots(figsize=(12, 6.2))
    width = 0.8 / len(models)
    x = np.arange(len(cats))
    for i, m in enumerate(models):
        s = summary[summary["model"] == m].set_index("category").reindex(cats)
        pos = x - 0.4 + width * (i + 0.5)
        ax.bar(pos, s[value], width * 0.92, color=style.color(m), label=style.label(m), zorder=2)
        if err:
            ax.errorbar(pos, s[value], yerr=s[err], fmt="none", ecolor=style.INK_2, elinewidth=1, capsize=0, zorder=3)
    ax.set_xticks(x, [CATEGORY_LABELS[c] for c in cats])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="x", visible=False)
    if ylim:
        ax.set_ylim(*ylim)
    if pct:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.savefig(path)
    plt.close(fig)


def dot_plot(summary, path):
    """CLIP means as dots: CLIP scores live in a narrow band, and bars would force a misleading zero baseline."""
    models = sorted(summary["model"].unique())
    cats = [c for c in CATEGORY_ORDER if c in set(summary["category"])]
    fig, ax = plt.subplots(figsize=(12, 6.2))
    y = np.arange(len(cats))[::-1]
    off = np.linspace(-0.18, 0.18, len(models)) if len(models) > 1 else [0]
    for i, m in enumerate(models):
        s = summary[summary["model"] == m].set_index("category").reindex(cats)
        ax.errorbar(s["clip_mean"], y + off[i], xerr=s["clip_se"], fmt="o", ms=9, color=style.color(m),
                    ecolor=style.color(m), elinewidth=1.5, label=style.label(m), mec=style.SURFACE, mew=1.5)
    ax.set_yticks(y, [CATEGORY_LABELS[c] for c in cats])
    ax.set_xlabel("Mean CLIPScore (± 1 SE)")
    ax.set_title("CLIP's view: alignment score by category")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.savefig(path)
    plt.close(fig)


def scatter_panels(df, agree, path):
    cats = [c for c in CATEGORY_ORDER if c in set(df["category"])]
    ncol = 3
    nrow = int(np.ceil(len(cats) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 4.2 * nrow), sharex=True, sharey=True, squeeze=False)
    rng = np.random.default_rng(0)
    x_mid = df["clip_score"].median()
    for ax, cat in zip(axes.flat, cats):
        g = df[df["category"] == cat]
        for m in sorted(df["model"].unique()):
            gm = g[g["model"] == m]
            jitter = rng.uniform(-0.12, 0.12, len(gm))
            ax.scatter(gm["clip_score"], gm["human_overall"] + jitter, s=46, color=style.color(m),
                       edgecolor=style.SURFACE, linewidth=1.2, alpha=0.9, label=style.label(m))
        ax.axvline(x_mid, color=style.MUTED, lw=0.8, ls="--")
        ax.axhline(3, color=style.MUTED, lw=0.8, ls="--")
        a = agree.set_index("group")
        rho = a.loc[cat, "spearman_clip_vs_overall"] if cat in a.index and "spearman_clip_vs_overall" in a else np.nan
        ax.set_title(f"{CATEGORY_LABELS[cat]}   ρ = {rho:.2f}" if pd.notna(rho) else CATEGORY_LABELS[cat], fontsize=13)
        ax.text(0.98, 0.04, "CLIP fooled", transform=ax.transAxes, ha="right", va="bottom", fontsize=10,
                color=style.INK_2, style="italic")
    for ax in axes.flat[len(cats):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("Human overall (1–5)")
    for ax in axes[-1, :]:
        ax.set_xlabel("CLIPScore")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Where CLIP and humans disagree", x=0.01, ha="left", fontsize=16, fontweight="bold")
    fig.text(0.01, 0.935, "One dot per image. Dashed lines: CLIP median and human midpoint (3). "
             "Bottom-right = CLIP liked it, humans didn't.", ha="left", fontsize=12, color=style.INK_2)
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    fig.savefig(path)
    plt.close(fig)


def agreement_bars(agree, winners, path):
    a = agree[agree["scope"] == "category"].set_index("group").reindex(CATEGORY_ORDER).dropna(how="all")
    cats = list(a.index)
    win = winners.groupby("category")["agree"].mean().reindex(cats) if len(winners) else pd.Series(np.nan, cats)
    fig, ax = plt.subplots(figsize=(12, 6.2))
    x = np.arange(len(cats))
    # Neutral + violet on purpose: these bars are metrics, so they must not reuse the model colors.
    ax.bar(x - 0.2, a["within_prompt_concordance"], 0.37, color=style.INK_2, label="Image pairs, same prompt")
    ax.bar(x + 0.2, win, 0.37, color="#6250d6", label="Which model won, per prompt")
    ax.axhline(0.5, color=style.INK_2, lw=1.2, ls="--")
    ax.text(1.005, 0.5, "coin flip", transform=ax.get_yaxis_transform(), ha="left", va="center",
            color=style.INK_2, fontsize=11)
    ax.set_xticks(x, [CATEGORY_LABELS[c] for c in cats])
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_ylabel("How often CLIP agrees with humans")
    ax.set_title("Does CLIP rank images the way humans do?")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.savefig(path)
    plt.close(fig)


def failure_gallery(df, prompts, path, k=3):
    """Each model's most embarrassing failures: humans failed it, CLIP liked it most."""
    models = sorted(df["model"].unique())
    pmap = prompts.set_index("id")["prompt"]
    fig, axes = plt.subplots(len(models), k, figsize=(4.2 * k, 4.9 * len(models)), squeeze=False)
    for r, m in enumerate(models):
        g = df[(df["model"] == m) & (df["human_pass"] < 0.5)].sort_values(["clip_pct", "human_overall"],
                                                                          ascending=[False, True])
        g = g.drop_duplicates("prompt_id").head(k)
        for c in range(k):
            ax = axes[r, c]
            ax.axis("off")
            if c >= len(g):
                continue
            row = g.iloc[c]
            ax.imshow(Image.open(row["path"]).convert("RGB"))
            cap = textwrap.fill(f"“{pmap[row['prompt_id']]}”", 34)
            ax.set_title(f"{cap}\nCLIP {row['clip_score']:.1f} (top {max(1, round(100 * (1 - row['clip_pct']) + 0.5)):.0f}%) · "
                         f"human {row['human_overall']:.1f}/5", fontsize=10.5, loc="left", fontweight="normal")
        axes[r, 0].text(-0.06, 0.5, style.label(m), transform=axes[r, 0].transAxes, rotation=90,
                        ha="right", va="center", fontsize=15, fontweight="bold", color=style.color(m))
    fig.suptitle("Most embarrassing failures: humans said no, CLIP said yes", x=0.01, ha="left",
                 fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0.02, 0, 1, 0.97))
    fig.savefig(path)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scorer", default=DEFAULT_SCORER)
    args = ap.parse_args()
    ensure_dirs()
    prompts, clip, ratings, gens = load(args.scorer)
    from mdb.common import list_images
    paths = list_images().set_index("key")["path"]
    clip["path"] = clip["key"].map(paths)

    human = human_per_image(ratings)
    df = clip.merge(human, on="key", how="left")
    rated = df.dropna(subset=["human_overall"]).copy()

    # Summary by model x category
    summ = (df.groupby(["model", "category"])
              .agg(n_images=("key", "size"), clip_mean=("clip_score", "mean"), clip_se=("clip_score", se),
                   human_pass=("human_pass", "mean"), human_pass_se=("human_pass", se),
                   human_overall=("human_overall", "mean"), item_rate=("item_rate", "mean"))
              .reset_index())
    if not gens.empty:
        ref = gens[gens["status"] == "refused"].groupby(["model", "category"]).size().rename("refused")
        summ = summ.merge(ref.reset_index(), on=["model", "category"], how="left").fillna({"refused": 0})
    summ.round(4).to_csv(RESULTS / "summary_by_category.csv", index=False)
    dot_plot(summ, RESULTS / "fig2_clip_by_category.png")

    report = [f"# Mini-DrawBench results\n\nCLIP scorer: `{args.scorer}`\n",
              f"Images scored: {len(df)} · images with human ratings: {len(rated)} "
              f"· ratings: {len(ratings)} · raters: {ratings['rater'].nunique() if len(ratings) else 0}\n"]
    if not gens.empty:
        report.append("Refusals by model: " + json.dumps(gens[gens["status"] == "refused"]
                                                          .groupby("model").size().to_dict()) + "\n")

    if rated.empty:
        report.append("\nNo human ratings yet: run rate_app.py, then re-run this script.\n")
    else:
        grouped_bars(summ, "human_pass", "human_pass_se", "Images passing every checklist item",
                     "Human judgment: pass rate by category", RESULTS / "fig1_human_pass_by_category.png",
                     ylim=(0, 1), pct=True)
        agree = agreement_table(rated)
        agree.to_csv(RESULTS / "agreement.csv", index=False)
        winners = model_winner_agreement(rated)
        scatter_panels(rated, agree, RESULTS / "fig3_clip_vs_human.png")
        agreement_bars(agree, winners, RESULTS / "fig4_clip_agreement_by_category.png")
        failure_gallery(rated, prompts, RESULTS / "fig5_failure_gallery.png")

        dis = rated.assign(
            disagreement=np.where((rated["clip_pct"] >= 0.75) & (rated["human_pass"] < 0.5), "clip_fooled",
                                  np.where((rated["clip_pct"] <= 0.25) & (rated["human_pass"] >= 0.5),
                                           "clip_too_harsh", "")))
        dis = dis[dis["disagreement"] != ""].merge(prompts[["id", "prompt"]], left_on="prompt_id", right_on="id")
        dis.sort_values(["disagreement", "clip_pct"], ascending=[True, False])[
            ["disagreement", "model", "category", "prompt_id", "seed", "prompt", "clip_score", "clip_pct",
             "human_pass", "human_overall", "path"]].to_csv(RESULTS / "disagreements.csv", index=False)

        rel = rater_reliability(ratings)
        rel.to_csv(RESULTS / "rater_agreement.csv", index=False)

        o = agree.iloc[0]
        report += [
            "\n## Headline numbers\n",
            f"- Spearman ρ, CLIP vs human overall (all images): **{o.get('spearman_clip_vs_overall', float('nan')):.2f}**",
            f"- AUC, CLIP predicting a human pass (0.5 = coin flip): **{o.get('auc_clip_predicts_pass', float('nan')):.2f}**",
            f"- Within-prompt concordance (CLIP orders image pairs like humans): **{o['within_prompt_concordance']:.0%}** "
            f"over {int(o['n_pairs'])} pairs",
            f"- CLIP picks the same winning model as humans: **{winners['agree'].mean():.0%}** of "
            f"{len(winners)} prompt-level comparisons" if len(winners) else "- Model-winner agreement: n/a (one model)",
            f"- 'CLIP fooled' images (top-quartile CLIP, failed by humans): **{(dis['disagreement'] == 'clip_fooled').sum()}**",
            f"- 'CLIP too harsh' images (bottom-quartile CLIP, passed by humans): **{(dis['disagreement'] == 'clip_too_harsh').sum()}**",
            "\n## Human pass rate by model and category\n",
            summ.pivot(index="category", columns="model", values="human_pass").reindex(CATEGORY_ORDER)
                .map(lambda v: f"{v:.0%}" if pd.notna(v) else "").to_markdown(),
            "\n## CLIP vs human agreement\n", agree.to_markdown(index=False),
            "\n## Rater reliability\n", rel.to_markdown(index=False) if len(rel) else "Need 2+ raters on shared images.",
        ]
    report += ["\n## Mean CLIPScore by model and category\n",
               summ.pivot(index="category", columns="model", values="clip_mean").reindex(CATEGORY_ORDER)
                   .round(2).to_markdown()]
    (RESULTS / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))
    print(f"\nWrote results to {RESULTS}")


if __name__ == "__main__":
    main()
