"""Shared paths, prompt loading, and file helpers for Mini-DrawBench."""
import csv
import os
import threading
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
PROMPTS_CSV = Path(os.environ.get("MDB_PROMPTS", REPO / "prompts.csv"))

# Everything generated lives under one root so it can sit on Google Drive in Colab.
DATA = Path(os.environ.get("MDB_DATA", REPO / "data"))
IMAGES = DATA / "images"
GENERATIONS_CSV = DATA / "generations.csv"
CLIP_CSV = DATA / "clip_scores.csv"
RATINGS_CSV = DATA / "ratings.csv"
RESULTS = DATA / "results"
DIVERSITY = DATA / "diversity"

CATEGORY_ORDER = ["counting", "spatial", "color_binding", "text_rendering", "rare_words", "conflicting"]
CATEGORY_LABELS = {
    "counting": "Counting",
    "spatial": "Spatial",
    "color_binding": "Color binding",
    "text_rendering": "Text rendering",
    "rare_words": "Rare words",
    "conflicting": "Conflicting",
}

_lock = threading.Lock()


def ensure_dirs():
    for p in (DATA, IMAGES, RESULTS, DIVERSITY):
        p.mkdir(parents=True, exist_ok=True)


def load_prompts(ids=None) -> pd.DataFrame:
    df = pd.read_csv(PROMPTS_CSV)
    df["items"] = df["checklist"].apply(lambda s: [x.strip() for x in str(s).split(";") if x.strip()])
    if ids:
        df = df[df["id"].isin(ids)]
    return df.reset_index(drop=True)


def image_key(model: str, prompt_id: str, seed: int) -> str:
    return f"{model}/{prompt_id}_s{seed}"


def image_path(model: str, prompt_id: str, seed: int, root: Path = IMAGES) -> Path:
    return root / model / f"{prompt_id}_s{seed}.png"


def list_images(root: Path = IMAGES) -> pd.DataFrame:
    """Scan the image folder (source of truth) -> model, prompt_id, seed, key, path."""
    rows = []
    if root.exists():
        for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            for f in sorted(model_dir.glob("*.png")):
                pid, s = f.stem.rsplit("_s", 1)
                rows.append({"model": model_dir.name, "prompt_id": pid, "seed": int(s),
                             "key": image_key(model_dir.name, pid, int(s)), "path": str(f)})
    return pd.DataFrame(rows, columns=["model", "prompt_id", "seed", "key", "path"])


def append_row(path: Path, row: dict):
    """Thread-safe CSV append; writes the header on first use."""
    with _lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        new = not path.exists() or path.stat().st_size == 0
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if new:
                w.writeheader()
            w.writerow(row)


def read_csv_or_empty(path: Path, columns=None) -> pd.DataFrame:
    if path.exists() and path.stat().st_size > 0:
        return pd.read_csv(path)
    return pd.DataFrame(columns=columns or [])
