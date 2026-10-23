"""Score each image against its prompt with CLIP. Already-scored images are skipped."""
import argparse

from PIL import Image

from mdb.clip_scorer import DEFAULT_SCORER, ClipScorer, clip_score
from mdb.common import CLIP_CSV, append_row, ensure_dirs, list_images, load_prompts, read_csv_or_empty


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scorer", default=DEFAULT_SCORER, help="open_clip 'arch:pretrained'")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    ensure_dirs()
    prompts = load_prompts().set_index("id")
    imgs = list_images()
    imgs = imgs[imgs["prompt_id"].isin(prompts.index)]
    prev = read_csv_or_empty(CLIP_CSV)
    if len(prev):
        done = set(prev.loc[prev["scorer"] == args.scorer, "key"])
        imgs = imgs[~imgs["key"].isin(done)]
    print(f"{len(imgs)} images to score with {args.scorer}")
    if imgs.empty:
        return

    scorer = ClipScorer(args.scorer)
    rows = imgs.to_dict("records")
    for i in range(0, len(rows), args.batch):
        batch = rows[i:i + args.batch]
        ims = [Image.open(r["path"]) for r in batch]
        texts = [prompts.loc[r["prompt_id"], "prompt"] for r in batch]
        for r, cos in zip(batch, scorer.score(ims, texts)):
            append_row(CLIP_CSV, {
                "scorer": args.scorer, "key": r["key"], "model": r["model"], "prompt_id": r["prompt_id"],
                "category": prompts.loc[r["prompt_id"], "category"], "seed": r["seed"],
                "clip_cos": round(cos, 5), "clip_score": round(clip_score(cos), 3),
            })
        print(f"  scored {min(i + args.batch, len(rows))}/{len(rows)}")


if __name__ == "__main__":
    main()
