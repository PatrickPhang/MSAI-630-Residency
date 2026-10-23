# Mini-DrawBench

MSAI-630 residency hackathon, Project 2. A small text-to-image benchmark comparing SDXL-Turbo and SD 1.5, scored two ways: CLIP similarity and blind human ratings.

## Running it

Open `notebooks/Mini_DrawBench_Colab.ipynb` in Colab with a T4 GPU and run the cells in order. Outputs go to `MyDrive/mini-drawbench-data`.

Locally, with a CUDA GPU:

```bash
pip install -r requirements.txt
python generate.py
python score_clip.py
python rate_app.py --share
python analyze.py
python diversity.py
python demo_app.py --share
```

## Pipeline

1. `generate.py` creates 4 images per prompt per model (seeds 0 to 3).
2. `score_clip.py` computes CLIPScore with OpenCLIP ViT-H-14.
3. `rate_app.py` collects blind ratings: a pass/fail per checklist item and a 1 to 5 overall score.
4. `analyze.py` compares the two and writes charts and `report.md` to `results/` in the data folder.
5. `diversity.py` samples 10 images per prompt to measure output diversity, plus an SD 1.5 guidance sweep.

Model settings:

| Model | Steps | Guidance |
|---|---|---|
| SDXL-Turbo | 4 | 0 |
| SD 1.5 | 25 | 7.5 |

## Evaluation set

`prompts.csv` has 48 prompts, 8 in each of 6 categories: counting, spatial relations, color binding, text rendering, rare words, and conflicting descriptions. Each prompt has a checklist of things a rater can verify in the image.

## Metrics

- Human pass rate: share of ratings where every checklist item passed.
- CLIPScore: 100 × cosine similarity between image and prompt embeddings.
- Agreement between CLIP and humans:
  - Spearman correlation between CLIPScore and the human score.
  - AUC of CLIPScore for predicting a human pass.
  - How often CLIP ranks two images of the same prompt the same way humans do.
- Rater agreement: Krippendorff's alpha and pairwise Cohen's kappa.
- Diversity: mean pairwise distance between DINOv2 and CLIP embeddings of 10 samples of the same prompt.

## Team

| Name | Role |
|---|---|
| | Product Lead |
| | Model/Prompt Engineer |
| | Integration Engineer |
| | Evaluation Lead |
| | Ethics & Risk Lead |
