"""Generate benchmark images. Resumable: images that already exist are skipped.

Examples:
  python generate.py --models sdxl_turbo sd15              # full benchmark (~15-20 min on a T4)
  python generate.py --models sdxl_turbo --ids C01 T01     # quick smoke test
"""
import argparse
import time
from datetime import datetime, timezone

import torch

from mdb.common import GENERATIONS_CSV, append_row, ensure_dirs, image_path, load_prompts, read_csv_or_empty
from mdb.generators import free_gpu, load_backend


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["sdxl_turbo", "sd15"], help="sdxl_turbo, sd15")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    ap.add_argument("--ids", nargs="*", help="Subset of prompt ids (default: all)")
    args = ap.parse_args()

    ensure_dirs()
    prompts = load_prompts(args.ids)
    done = read_csv_or_empty(GENERATIONS_CSV)
    done_refused = set()
    if len(done):
        r = done[done["status"] == "refused"]
        done_refused = set(zip(r["model"], r["prompt_id"], r["seed"].astype(int)))

    for model in args.models:
        backend = None
        todo = [(p, s) for _, p in prompts.iterrows() for s in args.seeds
                if not image_path(model, p["id"], s).exists() and (model, p["id"], s) not in done_refused]
        print(f"[{model}] {len(todo)} images to generate")
        if not todo:
            continue
        backend = load_backend(model)
        for i, (p, seed) in enumerate(todo, 1):
            t0 = time.time()
            try:
                img, status, info = backend(p["prompt"], seed)
            except torch.cuda.OutOfMemoryError:
                free_gpu()
                img, status, info = None, "error", "CUDA out of memory"
            except Exception as e:  # keep going; log the failure
                img, status, info = None, "error", repr(e)[:300]
            dt = time.time() - t0
            path = image_path(model, p["id"], seed)
            if img is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                img.save(path)
            # Errors are retried on the next run, so they go to a separate log instead of generations.csv.
            log = GENERATIONS_CSV if status != "error" else GENERATIONS_CSV.with_name("generation_errors.csv")
            append_row(log, {
                "model": model, "prompt_id": p["id"], "category": p["category"], "seed": seed,
                "status": status, "path": str(path.relative_to(path.parents[2])) if img is not None else "",
                "seconds": round(dt, 2), "info": info,
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            })
            print(f"  [{i}/{len(todo)}] {p['id']} seed={seed} {status} ({dt:.1f}s) {'' if status == 'ok' else info}")
        backend = None
        free_gpu()


if __name__ == "__main__":
    main()
