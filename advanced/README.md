# Advanced techniques

Each module below is a focused, self-contained study of one technique that
improved the core pipeline. They reuse the core feature/model code and report a
**paired, year-weighted, out-of-fold** comparison so the gain is honest. Each
produces an out-of-fold + test prediction column that is then added to the
stacker in `train.py`.

| Module | What it demonstrates | Notes |
|---|---|---|
| `two_part_censored.py` | Hurdle model for the spike at 100: `(1−p)·E[y\|y<100] + p·100` with an isotonically-calibrated censoring classifier. | Pure scikit-learn/LightGBM, no extra downloads. Fast. |
| `feature_pruning.py` | Feature selection is *model-specific*: pruning to the top-K permutation-ranked features helps a neural net but not gradient-boosted trees. | Fast; uses TF-IDF text only (no model downloads). |
| `tabpfn_column.py` | TabPFN (a foundation model pre-trained on synthetic tabular data) as a decorrelated ensemble member — the single biggest gain. | Needs a free `TABPFN_TOKEN` and one-time gated-model acceptance; runs locally, data stays on the machine. |
| `finetune_bert.py` | Fine-tuning a Turkish BERT (`dbmdz/bert-base-turkish-cased`) regression head on the mentor text. | GPU/MPS recommended; downloads the pretrained encoder. |

## How the columns combine

The core pipeline (`train.py`) trains the six base models and a year-weighted
stacker. Each advanced module writes a new out-of-fold/test column; adding it to
the stacker's input matrix and re-running the honest referee is what produced
the final ensemble:

```
final stack ≈ { CatBoost(full) , CatBoost(uncensored) , ExtraTrees ,
                NN(feature-pruned) , LightGBM(uncensored) , XGBoost(uncensored) ,
                Ridge(lean text+tabular) , BERTurk(text) , TabPFN }
            → year-weighted Ridge meta-model
            → two-part hurdle blend with the calibrated censoring probability
```

The recurring lesson: **one strong column per model family**; additional
members of the same family are redundant once the first is in the stack.
