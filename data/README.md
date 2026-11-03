# Data

## Uber fares dataset

`machine_learning/exp2_uber_regression.py` needs the Uber fares CSV (about 23 MB). It is not committed to this repository.

- Download: <https://www.kaggle.com/datasets/yasserh/uber-fares-dataset>
- Place the file at `data/uber.csv`, **or** set an environment variable:

```bash
# Windows PowerShell
$env:UBER_CSV = "C:\path\to\uber.csv"

# macOS / Linux
export UBER_CSV=/path/to/uber.csv
```

The script prints a clear message if the file is missing.

## Other datasets

- **ML exp1, exp3, exp4, exp5:** built-in scikit-learn datasets (`load_iris`, `load_digits`). No download needed.
- **ML exp6:** the maze is defined inside the script.
- **DMV exp1–exp6:** synthetic datasets are generated inside each script with a fixed random seed. Generated CSVs and plots go to `outputs/`.
- **DMV exp2:** uses a live OpenWeatherMap response only when `OPENWEATHER_API_KEY` is set; otherwise it uses a synthetic fallback with the same JSON shape.
