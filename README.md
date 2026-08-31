# Predicting Smartphone Addiction — Kaggle Playground Series S6E8

End-to-end machine learning pipeline for the [Kaggle Playground Series S6E8](https://www.kaggle.com/competitions/playground-series-s6e8) competition — a binary classification task predicting whether a person is addicted to their smartphone, based on their daily habits (screen time, sleep, notifications, stress, etc.).

## 📊 Dataset

- **Train:** ~691K rows, 13 columns (12 features + target)
- **Test:** ~296K rows, 12 features
- **Target:** `addicted_label` (0 = not addicted, 1 = addicted)

**Features:**
| Column | Description |
|---|---|
| `age` | Age of the user |
| `daily_screen_time_hours` | Average daily screen time |
| `social_media_hours` | Time spent on social media |
| `gaming_hours` | Time spent gaming |
| `work_study_hours` | Time spent on work/study |
| `sleep_hours` | Hours of sleep |
| `notifications_per_day` | Number of daily notifications |
| `app_opens_per_day` | Number of daily app opens |
| `weekend_screen_time` | Screen time on weekends |
| `gender` | Male / Female / Other |
| `stress_level` | Low / Medium / High |
| `academic_work_impact` | Whether screen time impacts academic/work performance |

## 🧠 Approach

The notebook (`smartphone_addiction_pipeline.ipynb`) implements a full pipeline:

1. **Missing Value Handling**
   Missing-indicator flags are created *before* imputation (missingness itself can be predictive), then numeric columns are median-imputed and categorical columns get a `"Missing"` category.

2. **Feature Engineering**
   Ratio and interaction features built from existing columns to expose relationships the raw values don't show directly:
   - `screen_to_sleep_ratio`, `social_to_screen_ratio`, `gaming_to_screen_ratio`
   - `notif_per_open`, `free_time`, `weekend_vs_weekday_diff`, `total_active_hours`

3. **K-Fold Target Encoding**
   Categorical columns (`gender`, `stress_level`, `academic_work_impact`) are target-encoded using out-of-fold statistics with smoothing, to avoid leakage.

4. **Model Training (5-Fold Stratified CV)**
   Three gradient-boosting models are trained and compared:
   - **LightGBM** (hyperparameters tuned via Optuna)
   - **XGBoost**
   - **CatBoost**

5. **Ensembling**
   Out-of-fold (OOF) predictions from all three models are used to grid-search the optimal blend weights, which are then applied to the test-set predictions.

6. **Submission**
   Final blended predictions are written to `submission.csv` in the format required by Kaggle.

## 📈 Results (5-Fold CV Accuracy)

| Model | Mean Accuracy |
|---|---|
| Baseline LightGBM | 0.8975 |
| + Feature Engineering | 0.8984 |
| + Hyperparameter Tuning (Optuna) | 0.9025 |
| LightGBM (tuned, full pipeline) | ~0.9023 |
| XGBoost | ~0.9027 |
| CatBoost | ~0.8980 |
| **Best Blend (weighted ensemble)** | *see notebook output* |

## 🛠️ How to Run

1. Open `smartphone_addiction_pipeline.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Run the cells top to bottom.
3. When prompted, upload `train.csv` and `test.csv` from the competition's [data page](https://www.kaggle.com/competitions/playground-series-s6e8/data).
4. The final cell downloads `submission.csv`, ready to submit to Kaggle.

> ⏱️ Training all 3 models across 5 folds on the full dataset can take 40–70 minutes on Colab.

## 📦 Requirements

```
pandas
numpy
scikit-learn
lightgbm
xgboost
catboost
matplotlib
optuna   # only needed if re-running hyperparameter tuning
```

## 📁 Repo Structure

```
.
├── smartphone_addiction_pipeline.ipynb   # main end-to-end notebook
└── README.md
```

## 🔮 Possible Next Steps

- Stacking (meta-model on top of OOF predictions) instead of a linear blend
- Pseudo-labeling using high-confidence test predictions
- Additional feature interactions (e.g. `age × stress_level`)
- Neural network as an additional diverse model in the ensemble
