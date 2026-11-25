# Data dictionary

## data/file_level/requirement_change_699.csv
| Column | Meaning |
|---|---|
| project | itrust2, gitlab, diaspora, sylius |
| version_pair | consecutive releases, `old->new` |
| req_id | use-case ID (iTrust2) or file path of the requirement document |
| status | Unchanged, Modified, Added, Deleted |
| similarity_corrected | Ratcliff-Obershelp character similarity, `autojunk=False`; meaningful for Modified |
| label | Unchanged, Partial or Full (rule in README, cut-off 0.91; Added and Deleted are Full) |
| old_len, new_len | text length in characters in the old and new release |
| Ambiguity, Completeness, Verification, Technological_Needs, Dependency, Specificity, Consistency | the seven text-quality proxies (lexicon-based; Completeness is type-token ratio), computed on the old text when present |
| group | project and requirement, used for grouped cross-validation |
| label_theta_0.75_superseded | label under the earlier 0.75 cut-off; kept only for comparison |

## data/scenario_level/{sylius,diaspora,gitlab}_scenarios.csv.gz (gzip-compressed CSV; `pandas.read_csv` reads it directly)
One row = one scenario present in the old release of a release pair.
| Column | Meaning |
|---|---|
| k | index of the release pair within the project |
| pair | `old->new` release tags |
| key | feature file path and lower-cased scenario title |
| file | feature file path |
| status | Unchanged, Modified (text edited, including matched renames) or Deleted |
| changed | 1 if Modified or Deleted, else 0 |
| prior_obs, prior_changes | earlier release pairs in which the scenario was seen, and how many of them changed it |
| since_change | release pairs since its last change (-1 if never) |
| age | release pairs since first seen |
| file_churn_prev, proj_churn_prev | fraction of changed scenarios in its file and in the project during the previous release pair |
| old_len, n_steps, n_ex | length, number of Given/When/Then/And/But steps, number of `|` characters (example-table cells) in the old scenario text |
| project | project name |

## data/human_labels/independent_ratings_150.csv
Two independent raters (rater_A, rater_B) rating Partial/Full/Unsure; `agreed` and `agreed_label` give the 130 agreed pairs used as gold labels; `similarity_corrected` is the automatic score.
