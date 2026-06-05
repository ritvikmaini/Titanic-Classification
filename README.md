# Titanic — Machine Learning from Disaster

Kaggle competition: binary classification predicting which passengers survived the sinking of the RMS Titanic.

## Results

| Model | CV Accuracy | Std |
|---|---|---|
| XGBoost (Optuna-tuned) | 0.8469 | ±0.026 |
| LightGBM (Optuna-tuned) | 0.8467 | ±0.021 |
| CatBoost | 0.8449 | ±0.023 |
| ExtraTrees | 0.8409 | ±0.025 |
| GradientBoosting | 0.8402 | ±0.026 |
| RandomForest | 0.8375 | ±0.023 |
| **Stacking ensemble** | **0.8510** | **±0.024** |

Evaluated with `RepeatedStratifiedKFold(n_splits=5, n_repeats=5)` — 25 folds give stable estimates on the 891-sample training set. OOF accuracy with threshold optimisation: **0.8519**.

## Key Decisions

**Feature engineering**

- **Title extracted from Name** — carries social standing and a strong survival prior (Mrs: 79% survival, Mr: 16%). Rare titles are grouped to avoid sparse categories.
- **Age imputed by Title × Pclass group median** — more accurate than a global median because age distributions differ substantially by title and class.
- **Box-Cox transform on Age and Fare** — both features are right-skewed; normalising the distribution gives tree models cleaner split points.
- **K-means clustering for cabin letters and ticket prefixes** — groups categories by (survival rate, frequency) rather than a hand-coded mapping, so the grouping is learned from the data rather than assumed.
- **LOO group survival rates** — passengers sharing a ticket or surname+class had strongly correlated fates. Leave-one-out encoding lets each passenger's rate use only their group-mates' outcomes, preventing target leakage during cross-validation. Full training means are used for the test set (no leakage there).
- **FamGroup** — family size is split into three buckets (alone / small 2–4 / large 5+) rather than kept continuous, because the survival relationship is non-linear: small families survived best, large groups struggled to evacuate together.

**Modelling**

- **Stacking over soft voting** — the Logistic Regression meta-learner learns each base model's relative strength on different subgroups rather than weighting them equally.
- **Six diverse base learners** (RF, GB, ET, XGB, LGB, CatBoost) — diversity across model families reduces correlated errors, which is the primary benefit of ensembling.
- **Optuna Bayesian tuning** — 60 TPE trials per model. More efficient than grid search and avoids the combinatorial explosion of searching many hyperparameters jointly.
- **Threshold optimisation** — the default 0.5 decision boundary is not always optimal. Tuning on out-of-fold probabilities finds the threshold that maximises training CV accuracy without touching the test set.
- **Pseudo-labeling at ≥95% confidence** — adds high-confidence test predictions to training before the final fit. The conservative threshold avoids injecting noisy labels.

## Features

| Feature | Description |
|---|---|
| `Title` | Extracted from Name — Mr / Mrs / Miss / Master / Rare |
| `IsMarried` | Mrs flag |
| `NameLength` | Name character count (correlates with social standing) |
| `AgeGroup` | Six bins: baby / child / teen / adult / middle / senior |
| `FamilySize` | SibSp + Parch + 1 |
| `FamGroup` | Alone / Small (2–4) / Large (5+) |
| `IsAlone` | Binary family flag |
| `HasCabin` | Whether cabin was recorded |
| `DeckGroup` | Cabin letter K-means cluster (3 groups) |
| `CabinNumBin` | Cabin number quartile |
| `TicketGroup` | Ticket prefix K-means cluster (3 groups) |
| `TicketFreq` | Passengers sharing the same ticket |
| `FarePerPerson` | Fare / TicketFreq |
| `FareBin` | Fare quartile |
| `TicketSurvRate` | LOO survival rate of ticket-group members |
| `FamilySurvRate` | LOO survival rate of surname+class group |
| `GroupSurvRate` | Max of the two above |
| `GroupSurvBinary` | GroupSurvRate > 0.5 |
| `IsWoman` | Sex flag |
| `IsWomanOrChild` | Female or age < 12 |
| `PclassSex` | Pclass × Sex interaction |
| `AgeXPclass` | Age × Pclass interaction |
| `FareXPclass` | Fare × Pclass interaction |

## Setup

```bash
pip install -r requirements.txt
jupyter notebook Kaggle_Titanic.ipynb
```

Data files (`train.csv`, `test.csv`) are from the [Kaggle Titanic competition](https://www.kaggle.com/competitions/titanic). Place them in `data/`.

## Repository Structure

```
Titanic-Classification/
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── gender_submission.csv
├── Kaggle_Titanic.ipynb   # Full pipeline
├── submission.csv          # Predictions for 418 test passengers
├── shap_importance.png     # Top-20 features by mean |SHAP|
├── requirements.txt
└── README.md
```

## License

MIT
