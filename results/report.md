# Mini-DrawBench results

CLIP scorer: `ViT-H-14:laion2b_s32b_b79k`

Images scored: 96 · images with human ratings: 96 · ratings: 192 · raters: 2

Refusals by model: {}


## Headline numbers

- Spearman ρ, CLIP vs human overall (all images): **0.53**
- AUC, CLIP predicting a human pass (0.5 = coin flip): **0.73**
- Within-prompt concordance (CLIP orders image pairs like humans): **77%** over 26 pairs
- CLIP picks the same winning model as humans: **75%** of 40 prompt-level comparisons
- 'CLIP fooled' images (top-quartile CLIP, failed by humans): **11**
- 'CLIP too harsh' images (bottom-quartile CLIP, passed by humans): **3**

## Human pass rate by model and category

| category       | sd15   | sdxl_turbo   |
|:---------------|:-------|:-------------|
| counting       | 12%    | 12%          |
| spatial        | 19%    | 44%          |
| color_binding  | 19%    | 12%          |
| text_rendering | 0%     | 19%          |
| rare_words     | 38%    | 56%          |
| conflicting    | 19%    | 38%          |

## CLIP vs human agreement

| scope    | group          |   n_images |   spearman_clip_vs_overall |   auc_clip_predicts_pass |   within_prompt_concordance |   n_pairs |
|:---------|:---------------|-----------:|---------------------------:|-------------------------:|----------------------------:|----------:|
| overall  | all            |         96 |                      0.529 |                    0.729 |                       0.769 |        26 |
| category | counting       |         16 |                      0.599 |                    0.833 |                       0.667 |         3 |
| category | spatial        |         16 |                      0.651 |                    0.77  |                       0.8   |         5 |
| category | color_binding  |         16 |                      0.591 |                    0.802 |                       0.75  |         4 |
| category | text_rendering |         16 |                      0.79  |                    0.786 |                       1     |         6 |
| category | rare_words     |         16 |                      0.202 |                    0.736 |                       0.5   |         4 |
| category | conflicting    |         16 |                      0.589 |                    0.917 |                       0.75  |         4 |
| model    | sd15           |         48 |                      0.68  |                    0.931 |                     nan     |         0 |
| model    | sdxl_turbo     |         48 |                      0.211 |                    0.593 |                     nan     |         0 |

## Rater reliability

| measure                     | raters              |   value |   n_images |   pct_agree |
|:----------------------------|:--------------------|--------:|-----------:|------------:|
| krippendorff_alpha_all_pass | all                 |   0.431 |         96 |     nan     |
| krippendorff_alpha_overall  | all                 |   0.308 |         96 |     nan     |
| cohen_kappa_all_pass        | harshada vs patrick |   0.444 |         96 |       0.792 |

## Mean CLIPScore by model and category

| category       |   sd15 |   sdxl_turbo |
|:---------------|-------:|-------------:|
| counting       |  32.14 |        37.68 |
| spatial        |  32.3  |        35.29 |
| color_binding  |  35.26 |        39.79 |
| text_rendering |  29.9  |        39.47 |
| rare_words     |  35.23 |        37.49 |
| conflicting    |  30.95 |        34.01 |
