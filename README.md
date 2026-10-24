# Mini-DrawBench

MSAI-630 residency hackathon, Project 2. A small text-to-image benchmark that compares SDXL-Turbo and Stable Diffusion 1.5 and grades every image twice: once automatically with CLIP, and once by people using a blind rating app. The question is how often the automatic grader agrees with people.

## Results

96 images (48 prompts × 2 models, one image each), each rated by 2 people. Full numbers are in `results/report.md`.

- Both graders prefer SDXL-Turbo overall. Under the strict checklist, people passed 30% of SDXL-Turbo's images and 18% of SD 1.5's.
- CLIP scored SDXL-Turbo higher in all 6 categories. People agreed in 4: in counting the two models tied, and in color binding SD 1.5 did slightly better.
- Given one image people passed and one they failed, CLIP scored the passing one higher 93% of the time for SD 1.5, but only 59% of the time for SDXL-Turbo. Chance is 50%.
- When CLIP and people disagreed, CLIP was usually too generous: 11 images in CLIP's top quarter were failed by people, against 3 the other way.
- Our two raters gave the same pass/fail verdict 79% of the time.

| Category | SD 1.5 pass rate | SDXL-Turbo pass rate | Extra CLIP points for SDXL-Turbo |
|---|---|---|---|
| Counting | 12% | 12% | +5.5 |
| Spatial | 19% | 44% | +3.0 |
| Color binding | 19% | 12% | +4.5 |
| Text rendering | 0% | 19% | +9.6 |
| Rare words | 38% | 56% | +2.3 |
| Conflicting | 19% | 38% | +3.1 |

## Running it

Open `notebooks/Mini_DrawBench_Colab.ipynb` in Google Colab with a T4 GPU and run the cells in order. Images, scores and ratings are saved to `MyDrive/mini-drawbench-data`, so any cell can be re-run after a disconnect.

1. **Setup:** clones this repo and installs `requirements.txt`.
2. **Generate:** `python generate.py --seeds 42` makes one image per prompt per model (96 images). Leave out `--seeds` to make 4 per prompt.
3. **CLIP scores:** `python score_clip.py`
4. **Human ratings:** `python rate_app.py --share` prints a public link. Each rater opens it and rates under their own name.
5. **Analysis:** `python analyze.py` writes the charts and `report.md` to `results/` in the Drive folder. Our copy is in `results/` here.
6. **Optional:** `python diversity.py` measures how varied repeated images of the same prompt are. We did not run it for these results.

## How it works

```
prompts.csv ──> generate.py ──> images ──┬──> score_clip.py (CLIP) ──┬──> analyze.py ──> results/
                SDXL-Turbo, SD 1.5       └──> rate_app.py (people) ──┘
```

| File | Purpose |
|---|---|
| `prompts.csv` | The 48 prompts and their checklists |
| `generate.py` | Generates images with SDXL-Turbo (4 steps) and SD 1.5 (25 steps) |
| `score_clip.py` | Scores each image against its prompt with CLIP (OpenCLIP ViT-H-14, which neither generator uses as its own text encoder) |
| `rate_app.py` | Blind rating app: shows one image, its prompt and checklist, never the model name, and serves the least-rated image next |
| `analyze.py` | Compares CLIP with the human ratings and makes the charts |
| `diversity.py` | Optional diversity test (not run) |
| `mdb/` | Shared code: file paths, model loading, the CLIP scorer, chart style |
| `results/` | Our charts, `report.md` and summary tables |

## Evaluation set

`prompts.csv` has 48 prompts, 8 in each of 6 categories: counting, spatial relations, color binding, text rendering, rare words, and conflicting descriptions. Each prompt has a checklist of things a rater can check by looking, for example "exactly 9 balloons; balloons tied to a mailbox". We wrote all prompts and checklists before generating any images.

## How agreement is measured

- **Human pass rate:** an image passes when a rater ticks every checklist item. Raters also give a 1 to 5 overall score.
- **CLIP score:** 100 × the similarity between the image and the prompt in CLIP's shared image-text space.
- **Pass/fail match:** given one image people passed and one they failed, how often CLIP scores the passing one higher (known as AUC; 50% is chance).
- **Correlation** between the CLIP score and the 1 to 5 human score (Spearman).
- **Rater agreement:** how often the two raters gave the same verdict (Cohen's kappa, Krippendorff's alpha).

## Limitations

- Small sample: 8 images per category per model, so one image moves a pass rate by about 12 points.
- Two raters, with moderate agreement on pass/fail and weaker agreement on the 1 to 5 score.
- Our CLIP grader and one of SDXL-Turbo's text encoders come from the same model family (OpenCLIP, trained on LAION data), which may favor SDXL-Turbo.

## Team

| Name | Role |
|---|---|
| Harshada Dhawade | Product Lead |
| Patrick Phang | Model/Prompt Engineer |
| Manoharsha Mallula | Integration Engineer |
| Ashraf Mohammad | Evaluation Lead |
| Muhammad Huzaifa | Ethics & Risk Lead |
