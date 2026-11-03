# Experiment 7 — Stock Market Time Series Analysis (Viva Theory)

> **30-second answer:** Analyse **420 business days** of daily stock prices with a datetime index: plot the closing-price trend, smooth it with **30-day and 90-day moving averages**, split it into **trend + seasonality + residual** with an additive decomposition (period 21), measure the **Close–Volume correlation (0.420)**, and forecast the last **30 days** with **ARIMA(5,1,0)** — MAE **6.96**, RMSE **8.87**.

## 1. MOST LIKELY — Core of the Experiment (High Yield)

**Aim:** Perform time-series analysis on stock-price data to identify trend, seasonality and relationships, and build a forecasting model.

**Dataset:** A deterministic synthetic daily series generated in the script — **420 business days**, columns `Date`, `Close`, `Volume`, from **2024-01-01 to 2025-08-08**. No missing dates.

**Pipeline in 8 bullets**
- Parse `Date` with `pd.to_datetime` and make it the **DatetimeIndex**; sort chronologically.
- Plot the closing price over time.
- Compute **MA30** and **MA90** with `rolling(window=30/90).mean()`.
- Use `seasonal_decompose(Close, model="additive", period=21)` to separate trend, seasonal cycle and residual.
- Compute `Close.corr(Volume)` → **0.420**.
- Split **chronologically**: first **390** days for training, last **30** for testing.
- Fit **ARIMA(5,1,0)** on the training series and forecast 30 steps.
- Evaluate with **MAE = 6.96** and **RMSE = 8.87**, then plot train/actual/forecast.

**Code skeleton**
```python
df = pd.read_csv("outputs/exp7_stock_prices.csv", parse_dates=["Date"], index_col="Date")
df = df.sort_index()

df["MA30"] = df["Close"].rolling(30).mean()
df["MA90"] = df["Close"].rolling(90).mean()
seasonal_decompose(df["Close"], model="additive", period=21).plot()

train, test = df["Close"].iloc[:-30], df["Close"].iloc[-30:]
forecast = ARIMA(train, order=(5, 1, 0)).fit().forecast(30)
mae = np.mean(np.abs(test - forecast))
rmse = np.sqrt(np.mean((test - forecast) ** 2))
```

**Expected output**
- Rows: **420**; date range **2024-01-01 to 2025-08-08**.
- Close: mean **139.96**, std **23.94**, min **91.63**, max **187.20**.
- Correlation Close vs Volume: **0.420** (moderate positive).
- Train **390**, test **30**; ARIMA forecast starts around **171.47**.
- ARIMA(5,1,0): **MAE 6.96**, **RMSE 8.87**.

**Viva Q&A**
1. **Why make `Date` the index?** It enables `.rolling`, correct time-axis plots, resampling and ARIMA to treat the data as ordered time.
2. **Why sort by date?** Every time-series operation assumes chronological order; unsorted rows would mix past and future.
3. **What does MA30 mean?** The average closing price over the last 30 days; it smooths short-term noise. MA90 shows the longer trend.
4. **Why two moving averages?** A short and a long window separate short-term movement from the underlying trend; crossovers can signal a change.
5. **What does `seasonal_decompose` do?** Splits the series into **trend**, **seasonal** and **residual** components.
6. **Additive vs multiplicative?** Additive when the seasonal swing stays roughly constant; multiplicative when the swing grows with the trend.
7. **Why `period=21`?** About 21 business days in a month, so it captures a monthly cycle.
8. **What does correlation 0.420 mean?** A moderate positive linear relationship between price and volume — not proof that one causes the other.
9. **Why not shuffle before train/test?** Shuffling lets future data leak into training (look-ahead bias). Time series must be split in time order.
10. **What do p, d, q mean in ARIMA(5,1,0)?** p = 5 autoregressive lags, d = 1 difference to remove trend, q = 0 moving-average error terms.
11. **Why d = 1?** One difference turns a trending series into a roughly stationary one.
12. **How did you evaluate the forecast?** MAE (average absolute error) and RMSE (penalises large misses) on the held-out last 30 days.
13. **Why is the ARIMA forecast fairly flat?** Long-horizon forecasts revert toward a stable level; ARIMA is not good at extrapolating a strong trend indefinitely.
14. **Moving average vs ARIMA?** A moving average describes/smooths the series; ARIMA is a forecasting model.

## 2. LIKELY — Syllabus Theory Around This Experiment

### Unit V: Time Series
- **Date/time tools:** `pd.to_datetime`, `pd.date_range`, `DatetimeIndex`, `.dt` accessors, `tz_localize`/`tz_convert`, `Period`.
- **Frequencies and shifting:** `asfreq`, `shift`, `resample`; daily ↔ monthly conversion.
- **Moving windows:** `rolling(window).mean()/sum()/std()` — smooths noise and creates features.
- **Components:** trend (long-term direction), seasonality (regular repeating pattern), cyclical movement (longer irregular cycles), residual/noise.
- **Stationarity:** constant mean, variance and autocovariance over time. ARIMA's `d` term handles non-stationarity by differencing.
- **ARIMA(p,d,q):** autoregression on past values (p), differencing (d), moving-average on past errors (q).
- **Alternatives:** exponential smoothing (ETS), SARIMA for seasonality, Prophet for business calendars.

**Likely Q&A**
- *What is a time series?* A sequence of observations ordered in time, where order carries information.
- *Why is random cross-validation wrong here?* It breaks temporal order and leaks future information.
- *How do you detect seasonality?* Plot the series, inspect the decomposition, or use ACF plots.
- *What is resampling?* Changing the time frequency and aggregating (for example, daily → monthly mean).
- *What is a rolling window?* A fixed-size window that slides one step at a time and applies a function.
- *What is exponential smoothing?* A weighted average where recent observations get more weight.
- *How do you handle time zones?* Store UTC, then convert for display with `tz_convert`.
- *Period vs Timestamp?* A `Period` is a span such as a month; a `Timestamp` is a single instant.

## 3. LEAST LIKELY — Stretch / Edge Cases

- **SARIMA** adds seasonal AR/MA terms `(P,D,Q,s)` when seasonality is strong.
- **ACF/PACF** help choose p and q; a slowly decaying ACF suggests differencing.
- **ADF/KPSS tests** check stationarity formally.
- **Walk-forward validation** retrains at each step; more realistic than one fixed split.
- **Look-ahead bias** occurs when future information leaks into features or scaling.
- **Missing timestamps** should be reindexed and then interpolated or filled explicitly.
- **Log transform** stabilises variance when the series grows multiplicatively.
- **Correlation is not causation** — volume and price can move together for many reasons.
- **Alternatives:** ETS, SARIMA, Prophet, gradient-boosted time-series models.

**If you remember only one thing:** A time series is analysed in time order — smooth it with rolling windows, split it chronologically, decompose trend/season/residual, and forecast with ARIMA while reporting MAE and RMSE on a held-out tail.
