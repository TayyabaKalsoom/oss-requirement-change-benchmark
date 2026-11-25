# OSS Requirement Change Benchmark (v1.0.0)

A multi-project benchmark of how software requirements change between releases, with human-validated Partial/Full labels and a forward-in-time test of whether requirement change can be predicted.

**Version:** 1.0.0 (2026-11-25) · **Author:** Tayyaba Kalsoom (PMAS-Arid Agriculture University / NUTECH) · **DOI:** _to be added after archiving on Zenodo_

## What is in this repository

| Folder | Content |
|---|---|
| `data/file_level/` | `requirement_change_699.csv`: 699 requirement-level records (iTrust2, GitLab, Diaspora, Sylius) across consecutive releases, labelled Unchanged / Partial / Full |
| `data/scenario_level/` | Per-scenario release histories (gzip-compressed CSV): Sylius (38,864 observations, 34 release pairs), Diaspora (10,828; 49 pairs), GitLab (23,424; 62 pairs) |
| `data/human_labels/` | Two independent raters' Partial/Full ratings on 150 pairs (agreement 87.2%, Cohen's kappa 0.745), the filled rater sheets, and the 20 pairs not yet adjudicated |
| `results/` | All result tables used in the paper |
| `code/` | Scripts that build the datasets and run every experiment |

See `DATA_DICTIONARY.md` for every column. Data licensing and upstream attribution are in `DATA_LICENSE.md`.

## Labels

Requirements are aligned between consecutive releases by file path or use-case ID.
- **Unchanged**: text identical.
- **Added / Deleted**: counted as **Full** change.
- **Modified**: **Partial** if character-level Ratcliff-Obershelp similarity (`difflib.SequenceMatcher`, `autojunk=False`) is at least **0.91**, otherwise **Full**.

The 0.91 cut-off was calibrated on the 130 pairs where both independent raters agreed (agreement with the humans 85.4%, kappa 0.71; held-out half 82.3%, kappa 0.65; bootstrap 95% CI for the cut-off 0.885-0.940). The remaining 85 Modified records are labelled by the rule only.

## Main findings (see `results/`)

1. The original 0.75 cut-off agreed with humans only 70.8% of the time; the earlier default of `autojunk=True` also flipped labels on long texts.
2. Labels are strongly tied to text length: a length-only Random Forest reaches macro-F1 0.960 versus 0.975 for all features (leave-one-project-out). The seven text-quality proxies alone are near chance (macro-F1 0.33).
3. Predicting change from old-release information only is weak. Forward in time: Sylius AUC 0.559 (95% CI 0.525-0.594), Diaspora 0.498, GitLab 0.521 (`results/`). The reported prediction runs are a negative or marginal result, not evidence of a usable predictor.

## Limitations

- Gold labels use only pairs where both raters agreed, so hard cases are under-represented. 20 pairs await adjudication.
- iTrust2 contributes only 7 agreed human-labelled pairs, all Full.
- Scenario matching across files is by title and text similarity within a file; moved files appear as Deleted. Release pairs where more than 50% of scenarios changed are mass restructures and should be filtered (the scripts do this).
- `results/scenario_prediction_3projects.csv` covers Diaspora and GitLab only (Sylius is in `scenario_prediction_forward_results.csv` and `scenario_prediction_edit_target.csv`).
- Requirement texts are not redistributed; regenerate them from the upstream tags using `code/` (see below).

## Reproducing

```
pip install -r requirements.txt
```
The scripts were run in a Linux workspace with the upstream repositories cloned under `/home/claude/repos/` and write to `/home/claude/work` and `/home/claude/out`. Change the path constants at the top of each script to your own folders. Order: `extract.py`, `build_sylius.py`, `build_all.py` (file level); `build_scenarios_generic.py <repo> <tag-regex> <out.pkl> [max-major]` then `scen_make_dataset.py <project>` (scenario level); `analyze_v4.py`, `predict_v2.py`, `scen_predict2.py`, `scen_predict3.py` (experiments).

## Citation

See `CITATION.cff`.
