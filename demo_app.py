"""Demo app: live generation with CLIP scores, benchmark browser, and result charts."""
import argparse

import gradio as gr
import pandas as pd
import torch
from PIL import Image

from mdb import style
from mdb.clip_scorer import DEFAULT_SCORER, ClipScorer, clip_score
from mdb.common import CLIP_CSV, RATINGS_CSV, RESULTS, list_images, load_prompts, read_csv_or_empty
from mdb.generators import load_backend

PROMPTS = load_prompts()
CHOICES = [f"{r.id} · {r.prompt}" for r in PROMPTS.itertuples()]
_cache = {}


def available_models():
    return ["sdxl_turbo", "sd15"] if torch.cuda.is_available() else []


def warm_up():
    for m in available_models():
        print(f"Loading {m} ...")
        get(m)("warm-up", 0)
    if available_models():
        get("clip").score([Image.new("RGB", (64, 64))], ["warm-up"])
        print("Warm-up done.")


def get(name):
    if name not in _cache:
        _cache[name] = ClipScorer() if name == "clip" else load_backend(name)
    return _cache[name]


def live(prompt, seed):
    prompt = (prompt or "").strip()
    if not prompt:
        raise gr.Error("Type a prompt first.")
    out = []
    for m in available_models():
        img, status, info = get(m)(prompt, int(seed))
        if img is None:
            out.append((Image.new("RGB", (512, 512), "#e6e5e0"), f"{style.label(m)}: {status} ({info[:80]})"))
            continue
        cos = get("clip").score([img], [prompt])[0]
        out.append((img, f"{style.label(m)} · CLIPScore {clip_score(cos):.1f}"))
    if not out:
        raise gr.Error("No GPU in this runtime. Use the Browse tab (pre-generated images) or switch to a T4.")
    return out


def pick(choice):
    return choice.split(" · ", 1)[1] if choice else ""


def browse(choice):
    if not choice:
        return [], ""
    pid = choice.split(" · ", 1)[0]
    imgs = list_images()
    imgs = imgs[imgs["prompt_id"] == pid]
    clip = read_csv_or_empty(CLIP_CSV)
    clip = clip[clip["scorer"] == DEFAULT_SCORER].set_index("key")["clip_score"] if len(clip) else pd.Series(dtype=float)
    rat = read_csv_or_empty(RATINGS_CSV)
    hum = rat.groupby("key").agg(o=("overall", "mean"), p=("all_pass", "mean")) if len(rat) else pd.DataFrame()
    items = []
    for r in imgs.sort_values(["model", "seed"]).itertuples():
        cap = f"{style.label(r.model)} s{r.seed}"
        if r.key in clip.index:
            cap += f" · CLIP {clip[r.key]:.1f}"
        if len(hum) and r.key in hum.index:
            cap += f" · human {hum.loc[r.key, 'o']:.1f}/5 {'✓' if hum.loc[r.key, 'p'] >= 0.5 else '✗'}"
        items.append((r.path, cap))
    p = PROMPTS.set_index("id").loc[pid]
    return items, f"**{p['prompt']}**  \nChecklist: {p['checklist']}"


def figures():
    return [(str(f), f.stem) for f in sorted(RESULTS.glob("fig*.png"))]


with gr.Blocks(title="Mini-DrawBench") as demo:
    gr.Markdown("# Mini-DrawBench: who grades the imagination?")
    with gr.Tab("Live"):
        gr.Markdown(f"Models available here: **{', '.join(style.label(m) for m in available_models()) or 'none'}**")
        with gr.Row():
            preset = gr.Dropdown(CHOICES, label="Pick a benchmark prompt (optional)")
            seed = gr.Number(value=0, precision=0, label="Seed")
        prompt = gr.Textbox(label="Prompt", lines=2)
        go = gr.Button("Generate & score", variant="primary")
        gallery = gr.Gallery(label="Results", columns=2, height=560, object_fit="contain")
        preset.change(pick, preset, prompt)
        go.click(live, [prompt, seed], gallery)
    with gr.Tab("Browse benchmark"):
        sel = gr.Dropdown(CHOICES, label="Prompt")
        info = gr.Markdown()
        bgal = gr.Gallery(columns=4, height=640, object_fit="contain")
        sel.change(browse, sel, [bgal, info])
    with gr.Tab("Failures & charts"):
        refresh = gr.Button("Load charts from results/")
        fgal = gr.Gallery(columns=1, height=720, object_fit="contain")
        refresh.click(figures, None, fgal)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--port", type=int, default=7861)
    ap.add_argument("--no-warmup", action="store_true")
    a = ap.parse_args()
    if not a.no_warmup:
        warm_up()
    demo.queue().launch(share=a.share, server_port=a.port)
