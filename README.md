# ML & DMV Practicals — BE AI & DS, Semester VII

Runnable lab code and exam notes for:

- **417521 — Machine Learning**
- **417522 — Data Modeling and Visualization**

SPPU 2020 pattern, AY 2026-27.

## What is inside

| Folder | Contents |
|---|---|
| `machine_learning/` | 6 runnable ML practical scripts |
| `dmv/` | 7 runnable DMV experiment scripts |
| `machine_learning/notes/` | Viva theory notes, one file per experiment |
| `dmv/notes/` | Viva theory notes, one file per experiment |
| `common/` | Shared syntax cheat sheet and evaluation-metrics guide |
| `data/` | Dataset instructions (the Uber CSV is not committed) |
| `outputs/` | Generated plots and CSVs (ignored by git) |

Every experiment has **two** notes files:

1. `notes/ExpN_*.md` — code, exact outputs, syllabus theory, and likely viva Q&A.
2. `notes/Plain_English/ExpN_*_PlainEnglish.md` — the same experiment explained without math or notation.

## Quick start

```bash
git clone <this-repo-url>
cd cl1-be

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt

python machine_learning/exp1_lda_iris.py
python dmv/exp1_data_loading_formats.py
```

All plots and generated CSVs are written to `outputs/`.

## Machine Learning experiments

| # | Script | Topic |
|---|---|---|
| 1 | `exp1_lda_iris.py` | Linear Discriminant Analysis on Iris |
| 2 | `exp2_uber_regression.py` | Linear, Ridge and Lasso regression on Uber fares |
| 3 | `exp3_svm_digits.py` | RBF-kernel SVM on 8×8 handwritten digits |
| 4 | `exp4_kmeans_elbow.py` | K-Means with elbow method and silhouette |
| 5 | `exp5_ensemble_iris.py` | Voting, AdaBoost, Gradient Boosting, XGBoost |
| 6 | `exp6_qlearning_maze.py` | Q-Learning in a grid maze |

These six match the ML practicals in scope: **1b, 2a, 3a, 4a, 5b, 6a**.

## Data Modeling & Visualization experiments

| # | Script | Topic |
|---|---|---|
| 1 | `exp1_data_loading_formats.py` | Loading CSV, Excel and JSON sales data |
| 2 | `exp2_web_api_weather.py` | OpenWeatherMap API and weather analysis |
| 3 | `exp3_data_cleaning_churn.py` | Cleaning telecom customer-churn data |
| 4 | `exp4_data_wrangling_real_estate.py` | Wrangling real-estate data |
| 5 | `exp5_matplotlib_aqi.py` | AQI visualization with matplotlib |
| 6 | `exp6_data_aggregation_retail.py` | Retail sales aggregation by region |
| 7 | `exp7_time_series.py` | Stock-price time series: moving averages, decomposition, ARIMA |

DMV experiments 1–7 correspond to syllabus **Labs 7–13**. The scripts generate their own synthetic datasets, so they run offline. `exp2_web_api_weather.py` uses a live API only if `OPENWEATHER_API_KEY` is set; otherwise it uses a clearly labelled synthetic fallback.

## Dataset note

The Uber fares CSV is about 23 MB and is **not** committed. See [`data/README.md`](data/README.md) for the download link and how to point `exp2_uber_regression.py` at it. All other scripts use built-in scikit-learn datasets or generate their own data.

## Requirements

Python 3.10+ and the packages in `requirements.txt`. XGBoost is optional; `exp5_ensemble_iris.py` skips that model with a printed note if the package is unavailable. `exp7_time_series.py` requires `statsmodels`.

## Notes and exam preparation

Start with the plain-English file to understand the experiment, then read the main note for the exact code, outputs, and likely viva questions. The `common/` folder has:

- `Common_Syntax_and_Viva_Questions.md` — pandas/sklearn syntax, common traps, and a cross-subject viva bank.
- `Evaluation_Metrics_Explained.md` — every metric with plain-text formulas and when to use it.

## Academic use

This repository is for study and revision. The code is written from the lab practical statements, and the DMV data is synthetic. No personal information or API keys are included.
