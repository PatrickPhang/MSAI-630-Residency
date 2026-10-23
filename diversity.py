"""Stretch goal: how different are 10 samples of the same prompt? (mode collapse / quality-vs-diversity)

Diversity = mean pairwise cosine DISTANCE between image embeddings of the N samples of one prompt
(0 = identical images). Reported with two embedders: CLIP (semantic) and DINOv2 (visual/layout).

  python diversity.py                                                # SDXL-Turbo vs SD 1.5
  python diversity.py --models sd15 --guidance 2 5 7.5 12            # quality-vs-diversity sweep (CFG)
  python diversity.py --analyze-only                                 # re-make charts from saved images

Expectation to TEST, not assume: SDXL-Turbo is distilled with an adversarial (GAN-style) loss, which
tends to trade diversity for sharpness (SD 1.5 is plain diffusion, so it is the natural contrast),
and higher CFG pushes samples toward the mode.
"""
import argparse
import itertools

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from mdb import style
from mdb.clip_scorer import ClipScorer, clip_score
from mdb.common import DIVERSITY, RESULTS, ensure_dirs, load_prompts
from mdb.generators import free_gpu, load_backend

style.apply()
import matplotlib.pyplot as plt  # noqa: E402

DEFAULT_IDS = ["C02", "S02", "B01", "R02", "X05"]
DEFAULT_GUIDANCE = {"sdxl_turbo": 0.0, "sd15": 7.5}


def run_dir(model, g):
    return DIVERSITY / f"{model}_cfg{g:g}"


def generate(models, guidances, ids, n, seed0):
    prompts = load_prompts(ids)
    for model in models:
        gs = guidances if (guidances and model == "sd15") else [DEFAULT_GUIDANCE[model]]
        backend = None
        for g in gs:
            for _, p in prompts.iterrows():
                for s in range(seed0, seed0 + n):
                    path = run_dir(model, g) / f"{p['id']}_s{s}.png"
                    if path.exists():
                        continue
                    backend = backend or load_backend(model)
                    img, status, info = backend(p["prompt"], s, guidance=g)
                    if img is not None:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        img.save(path)
                    print(f"{model} cfg={g} {p['id']} seed={s} {status}")
        backend = None
        free_gpu()


class Dino:
    def __init__(self, device):
        from transformers import AutoImageProcessor, AutoModel
        self.proc = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
        self.model = AutoModel.from_pretrained("facebook/dinov2-small").to(device).eval()
        self.device = device

    @torch.no_grad()
    def embed(self, ims):
        x = self.proc(images=[im.convert("RGB") for im in ims], return_tensors="pt").to(self.device)
        return F.normalize(self.model(**x).pooler_output.float(), dim=-1)


def mean_pairwise_distance(emb):
    sims = emb @ emb.T
    iu = torch.triu_indices(len(emb), len(emb), 1)
    return float((1 - sims[iu[0], iu[1]]).mean())


def analyze(ids, use_dino=True):
    prompts = load_prompts(ids).set_index("id")
    scorer = ClipScorer()
    dino = None
    if use_dino:
        try:
            dino = Dino(scorer.device)
        except Exception as e:  # noqa: BLE001
            print(f"DINOv2 unavailable ({e}); CLIP-only diversity.")
    rows = []
    for d in sorted(p for p in DIVERSITY.iterdir() if p.is_dir()):
        model, _, cfg = d.name.partition("_cfg")
        if model not in DEFAULT_GUIDANCE:
            continue
        for pid in prompts.index:
            files = sorted(d.glob(f"{pid}_s*.png"))
            if len(files) < 2:
                continue
            ims = [Image.open(f) for f in files]
            ce = scorer.embed_images(ims)
            te = scorer.embed_texts([prompts.loc[pid, "prompt"]])
            row = {"run": d.name, "model": model, "cfg": float(cfg) if cfg else np.nan, "prompt_id": pid,
                   "category": prompts.loc[pid, "category"], "n": len(files),
                   "clip_diversity": mean_pairwise_distance(ce),
                   "clip_score_mean": float(np.mean([clip_score(c) for c in (ce @ te.T).squeeze(1).tolist()]))}
            if dino:
                row["dino_diversity"] = mean_pairwise_distance(dino.embed(ims))
            rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No diversity images found. Run without --analyze-only first.")
    df.round(4).to_csv(RESULTS / "diversity.csv", index=False)
    print(df.groupby("run")[[c for c in df.columns if c.endswith("diversity") or c == "clip_score_mean"]].mean().round(4))

    # Model comparison at default settings
    def is_default(r):
        d = DEFAULT_GUIDANCE.get(r["model"])
        return pd.isna(r["cfg"]) or d is None or r["cfg"] == d
    base = df[df.apply(is_default, axis=1)]
    if base["model"].nunique() >= 1:
        metric = "dino_diversity" if "dino_diversity" in base else "clip_diversity"
        m = base.groupby("model")[metric].agg(["mean", "sem"]).reset_index()
        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.bar([style.label(x) for x in m["model"]], m["mean"], 0.55, yerr=m["sem"],
               color=[style.color(x) for x in m["model"]], ecolor=style.INK_2)
        ax.set_ylabel(f"Mean pairwise distance ({'DINOv2' if metric.startswith('dino') else 'CLIP'})")
        ax.set_title(f"How different are {int(base['n'].max())} samples of the same prompt?")
        ax.grid(axis="x", visible=False)
        fig.savefig(RESULTS / "fig6_diversity_by_model.png")
        plt.close(fig)

    # CFG sweep: quality vs diversity
    sweep = df[df["model"] == "sd15"].dropna(subset=["cfg"])
    if sweep["cfg"].nunique() >= 2:
        metric = "dino_diversity" if "dino_diversity" in sweep else "clip_diversity"
        s = sweep.groupby("cfg")[[metric, "clip_score_mean"]].mean().reset_index()
        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.plot(s["clip_score_mean"], s[metric], "-o", color=style.color("sd15"), lw=2, ms=9)
        for _, r in s.iterrows():
            ax.annotate(f"CFG {r['cfg']:g}", (r["clip_score_mean"], r[metric]), textcoords="offset points",
                        xytext=(8, 6), fontsize=11, color=style.INK_2)
        ax.set_xlabel("Mean CLIPScore (alignment / 'quality')")
        ax.set_ylabel("Diversity (mean pairwise distance)")
        ax.set_title("SD 1.5: turning up guidance buys alignment with diversity")
        fig.savefig(RESULTS / "fig7_cfg_quality_vs_diversity.png")
        plt.close(fig)
    print(f"Wrote diversity results to {RESULTS}")


def contact_sheet(model_run, pid, out):
    files = sorted((DIVERSITY / model_run).glob(f"{pid}_s*.png"))[:10]
    if not files:
        return
    fig, axes = plt.subplots(2, 5, figsize=(15, 6.4))
    for ax in axes.flat:
        ax.axis("off")
    for ax, f in zip(axes.flat, files):
        ax.imshow(Image.open(f).convert("RGB"))
    fig.suptitle(f"{model_run} · {pid}: 10 seeds", x=0.01, ha="left")
    fig.savefig(out)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["sdxl_turbo", "sd15"])
    ap.add_argument("--guidance", nargs="*", type=float, help="CFG sweep values (applied to sd15 only)")
    ap.add_argument("--ids", nargs="*", default=DEFAULT_IDS)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed0", type=int, default=1000, help="Different seeds from the main benchmark")
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--no-dino", action="store_true")
    a = ap.parse_args()
    ensure_dirs()
    if not a.analyze_only:
        generate(a.models, a.guidance, a.ids, a.n, a.seed0)
    analyze(a.ids, use_dino=not a.no_dino)
    for d in sorted(p for p in DIVERSITY.iterdir() if p.is_dir()):
        contact_sheet(d.name, a.ids[0], RESULTS / f"contact_{d.name}_{a.ids[0]}.png")


if __name__ == "__main__":
    main()
