# Data Modeling & Visualization — 417522

Six runnable experiments. Run each from the repository root:

```bash
python dmv/exp1_data_loading_formats.py
python dmv/exp2_web_api_weather.py
python dmv/exp3_data_cleaning_churn.py
python dmv/exp4_data_wrangling_real_estate.py
python dmv/exp5_matplotlib_aqi.py
python dmv/exp6_data_aggregation_retail.py
```

| Script | What it demonstrates |
|---|---|
| `exp1_data_loading_formats.py` | Reading CSV, Excel and JSON; concat; cleaning; groupby summary |
| `exp2_web_api_weather.py` | HTTP request, JSON parsing, datetime, rolling average, plots |
| `exp3_data_cleaning_churn.py` | Duplicates, inconsistent text, types, imputation, IQR capping, encoding, scaling |
| `exp4_data_wrangling_real_estate.py` | Column-name cleaning, group imputation, merge, filter, outliers, pivot, encoding |
| `exp5_matplotlib_aqi.py` | Line, bar, stacked bar, box, violin, scatter and bubble charts |
| `exp6_data_aggregation_retail.py` | groupby/agg, share and rank, pivot table, grouped/stacked bars |

## Data

Every script generates its own synthetic data with a fixed seed, so no downloads are required. Generated CSVs and plots are written to `outputs/`.

`exp2_web_api_weather.py` uses a live OpenWeatherMap response only when the `OPENWEATHER_API_KEY` environment variable is set. Otherwise it uses a synthetic fallback with the same JSON structure and prints a clear notice.

## Notes

- `notes/ExpN_*.md` — code, exact outputs, syllabus theory, likely viva Q&A.
- `notes/Plain_English/ExpN_*_PlainEnglish.md` — the same experiment without math or jargon.
