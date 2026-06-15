# Original task prompts

The two prompts that specified this project, kept verbatim for transparency and
to document the prompt-engineering behind the solution:

| File | Purpose |
|---|---|
| `eda_prompt.txt` | A 16-section senior-level Exploratory Data Analysis brief (data quality, univariate/target/correlation analysis, feature importance, text NLP, multicollinearity/VIF, outliers, leakage audit, final feature recommendations).
| `modeling_prompt.txt` | The full modelling specification: an importance prior over features, scale-aware normalisation, importance-weighted composite features, importance-guided interaction generation, text-feature strategies, the model zoo, Optuna HPO, and the blending/stacking ensemble. 

`eda.py` (repo root) is a condensed, reproducible implementation of the EDA
brief; the rest of the repo implements the modelling specification.
