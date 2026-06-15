# Data

Place the competition files here:

```
data/
  train.csv      # 10,000 rows incl. career_success_score
  test_x.csv     # 10,000 rows, predictors only
```

The data is **not** committed (see `.gitignore`). `config.py` reads from this
folder. `sample_submission.csv` shows the expected output format
(`student_id,career_success_score`).
