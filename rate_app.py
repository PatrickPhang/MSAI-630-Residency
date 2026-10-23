"""Blind rating app. Serves the least-rated image the rater hasn't seen yet."""
import argparse
import json
import random
from datetime import datetime, timezone

import gradio as gr
from PIL import Image

from mdb.common import RATINGS_CSV, append_row, list_images, load_prompts, read_csv_or_empty

TARGET = 2  # ratings per image


def load_state():
    prompts = load_prompts().set_index("id")
    imgs = list_images()
    imgs = imgs[imgs["prompt_id"].isin(prompts.index)].set_index("key")
    ratings = read_csv_or_empty(RATINGS_CSV, ["key", "rater"])
    return prompts, imgs, ratings


def next_image(rater: str):
    prompts, imgs, ratings = load_state()
    counts = ratings.groupby("key").size().to_dict() if len(ratings) else {}
    mine = set(ratings.loc[ratings["rater"] == rater, "key"]) if len(ratings) else set()
    pool = [k for k in imgs.index if k not in mine]
    under = sum(1 for k in imgs.index if counts.get(k, 0) < TARGET)
    progress = f"**{rater}**: {len(mine)} rated · images still under {TARGET} ratings: {under} / {len(imgs)}"
    if not pool:
        return None, progress + "\n\nYou've rated every image. Thank you!"
    lowest = min(counts.get(k, 0) for k in pool)
    key = random.choice([k for k in pool if counts.get(k, 0) == lowest])
    row = imgs.loc[key]
    p = prompts.loc[row["prompt_id"]]
    return {"key": key, "model": row["model"], "prompt_id": row["prompt_id"], "seed": int(row["seed"]),
            "path": row["path"], "prompt": p["prompt"], "items": p["items"], "category": p["category"]}, progress


def show(cur, progress):
    if cur is None:
        return (cur, None, "", gr.CheckboxGroup(choices=[], value=[]), progress)
    return (cur, Image.open(cur["path"]), f"### {cur['prompt']}",
            gr.CheckboxGroup(choices=cur["items"], value=[]), progress)


def start(rater):
    rater = (rater or "").strip().lower()
    if not rater:
        raise gr.Error("Enter your name first.")
    cur, progress = next_image(rater)
    return (rater,) + show(cur, progress)


def submit(rater, cur, passed, overall, broken, notes):
    if not rater or cur is None:
        raise gr.Error("Press Start first.")
    items = cur["items"]
    passed = passed or []
    append_row(RATINGS_CSV, {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rater": rater, "key": cur["key"], "model": cur["model"], "prompt_id": cur["prompt_id"],
        "category": cur["category"], "seed": cur["seed"],
        "items_passed": json.dumps([it in passed for it in items]),
        "n_items": len(items), "n_passed": len(passed),
        "all_pass": int(len(passed) == len(items) and not broken),
        "overall": int(overall), "broken": int(bool(broken)), "notes": (notes or "").strip(),
    })
    nxt, progress = next_image(rater)
    return show(nxt, progress) + (3, False, "")


with gr.Blocks(title="Mini-DrawBench rater") as demo:
    gr.Markdown("# Mini-DrawBench · blind rating\nCheck every item that is **clearly** true in the image. "
                "Be strict: 'almost four apples' is not four. The overall score is your gut sense of how well "
                "the image matches the whole prompt.")
    rater_state, cur_state = gr.State(""), gr.State(None)
    with gr.Row():
        rater_box = gr.Textbox(label="Your name", scale=3)
        start_btn = gr.Button("Start", variant="primary", scale=1)
    progress_md = gr.Markdown()
    with gr.Row():
        img = gr.Image(type="pil", label="Image", height=512)
        with gr.Column():
            prompt_md = gr.Markdown()
            checks = gr.CheckboxGroup(label="Checklist: tick what is satisfied", choices=[])
            overall = gr.Slider(1, 5, value=3, step=1, label="Overall alignment (1 = unrelated, 5 = perfect)")
            broken = gr.Checkbox(label="Image is broken / blank / unreadable")
            notes = gr.Textbox(label="Notes (optional: describe a striking failure)", lines=2)
            submit_btn = gr.Button("Submit & next", variant="primary")

    start_btn.click(start, rater_box, [rater_state, cur_state, img, prompt_md, checks, progress_md])
    submit_btn.click(submit, [rater_state, cur_state, checks, overall, broken, notes],
                     [cur_state, img, prompt_md, checks, progress_md, overall, broken, notes])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    demo.queue(default_concurrency_limit=8).launch(share=a.share, server_port=a.port)
