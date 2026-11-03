# Experiment 2 — Interacting with Web APIs (Viva Theory)

> **30-second answer:** Call the **OpenWeatherMap 5-day/3-hour forecast** endpoint with `requests.get(params={q, appid, units:'metric'})`, parse the nested JSON into a flat DataFrame (40 records), clean timestamps and missing rain values, compute a **24-hour rolling mean** and linear trend, aggregate daily/time-of-day, and visualise line, bar, scatter and correlation-heatmap plots.

## 1. MOST LIKELY — Core of the Experiment (High Yield)

- **Aim:** Retrieve live weather data for a city from an API and analyse/visualise patterns over time.
- **Source:** `https://api.openweathermap.org/data/2.5/forecast` — 40 forecast entries (every 3 h for 5 days), 11 extracted fields.
- **Pipeline:**
  1. Register and obtain an API key; pass `q`, `appid`, `units='metric'`.
  2. `requests.get(url, params=..., timeout=10)`; check `status_code == 200`.
  3. `response.json()` → nested dict with `city` and `list`.
  4. Flatten `list[]`: `main.temp`, `main.humidity`, `wind.speed`, `clouds.all`, `rain.3h`, `weather[0].main`, `dt_txt`.
  5. `pd.to_datetime(Timestamp)`; fill missing `rain.3h` with **0** (API omits it when dry); interpolate other numeric gaps; sort and drop duplicate timestamps.
  6. Model: `rolling(window=8, min_periods=1).mean()` = 24 h (8 × 3 h); `np.polyfit` for a linear trend.
  7. Aggregate daily (`groupby('Date').agg`) and by time-of-day.
  8. Plot: line + rolling mean, bar (daily rain), bubble/scatter (temp vs humidity), heatmap of correlations.

**Code skeleton**
```python
params = {'q': 'Pune,IN', 'appid': API_KEY, 'units': 'metric'}
r = requests.get(URL, params=params, timeout=10)
raw = r.json()                      # r.status_code == 200
rec = [{'Timestamp': x['dt_txt'],
        'Temperature_C': x['main']['temp'],
        'Humidity_Pct': x['main']['humidity'],
        'Wind_Speed_ms': x['wind']['speed'],
        'Precipitation_mm': x.get('rain', {}).get('3h', np.nan),
        'Weather_Condition': x['weather'][0]['main']} for x in raw['list']]
df = pd.DataFrame(rec)
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['Precipitation_mm'] = df['Precipitation_mm'].fillna(0.0)
df['Temp_Rolling_24h_Avg'] = df['Temperature_C'].rolling(8, min_periods=1).mean()
```

**Expected output**
- Clean shape **40 rows × 11 columns**.
- Avg temp **28.66 °C**; max **34.91 °C** (2026-10-19 09:00); min **22.66 °C** (2026-10-23 00:00).
- Avg humidity **51.3 %**; total rain **0.12 mm**; max wind **3.58 m/s**.
- Daily table for 6 dates; diurnal means (Morning 32.03 °C → Night 26.62 °C).
- Four-panel figure: temperature trend, daily precipitation, temp–humidity bubble, correlation heatmap.

**Viva Q&A**
1. **What is an API?** An interface that lets one program request data/services from another over HTTP, usually returning JSON.
2. **Why `params=` instead of building the URL string?** `requests` URL-encodes and merges query parameters safely.
3. **What does `status_code == 200` mean?** Success; 401 = bad/missing key, 404 = wrong city/endpoint, 429 = rate limit.
4. **Why `timeout=10`?** Prevents the program hanging indefinitely on a slow/dead network.
5. **`response.json()` vs `response.text`?** `.json()` parses the body into Python dict/list; `.text` leaves it as a raw string.
6. **Why is the forecast data nested?** `main`, `wind`, `clouds`, `weather` are sub-objects; you must drill into keys.
7. **Why fill missing `rain.3h` with 0?** OpenWeatherMap omits the `rain` key when there is no rain; absence means zero, not unknown.
8. **Why 24 h = `window=8`?** Forecast points are 3 h apart; 8 × 3 h = 24 h.
9. **Why a rolling mean?** It smooths short-term noise so the underlying trend is visible.
10. **What does `np.polyfit(x, y, 1)` return?** Slope and intercept of the best-fit straight line (degree 1).
11. **Why sort by timestamp?** Rolling/trend calculations assume chronological order.
12. **What if the API fails?** The code falls back to a schema-identical simulated JSON so the practical still runs — a good demonstration of graceful degradation, but it must be labelled as synthetic.

## 2. LIKELY — Syllabus Theory Around This Experiment

### Unit II — stochastic processes & time series
- A **time series** is a sequence indexed by time; components are **trend, seasonality, cyclical and irregular**.
- **Moving average** smooths a series by averaging a sliding window; larger window = smoother but laggier.
- **Auto-regressive (AR)** models predict from past values; **moving-average (MA)** models use past errors; **ARMA/ARIMA** combine them.

### Unit IV — matplotlib/seaborn
- `plot` = line, `bar` = category totals, `scatter` = relationship between two numerics, `heatmap` = matrix of correlations.

| Plot | Question it answers |
|---|---|
| Line + rolling mean | How does temperature change over time? |
| Bar | Which day had most rain? |
| Scatter/bubble | Is temperature related to humidity? |
| Heatmap | Which variables move together? |

**Q&A**
1. **HTTP methods?** `GET` retrieves, `POST` sends/creates, `PUT` updates, `DELETE` removes. This practical uses `GET`.
2. **What is JSON?** JavaScript Object Notation — human-readable key–value text widely used by APIs.
3. **Why use `units='metric'`?** Returns Celsius and m/s instead of Kelvin and mph.
4. **What is interpolation?** Estimating missing points from neighbouring values (`method='linear'` joins them with a straight line).
5. **Correlation coefficient range?** −1 to +1; 0 = no linear relationship; sign = direction.
6. **Why `drop_duplicates(subset=['Timestamp'])`?** Forecast feeds can repeat timestamps; duplicates would distort rolling statistics.
7. **Difference between weather and climate?** Weather = short-term state; climate = long-term average/pattern.
8. **What is an API key?** A secret token that authenticates the caller and enforces quotas; never commit it to source control.
9. **Why store the response in a DataFrame?** DataFrames give vectorised cleaning, aggregation and plotting.

## 3. LEAST LIKELY — Stretch / Advanced / Edge Cases

1. **Secret handling:** hard-coding an API key in a notebook leaks it. Use an environment variable (`os.getenv('OWM_KEY')`) or a `.env` file that is git-ignored.
2. **Rate limits / retries:** production clients add exponential backoff and cache responses; a 429 should not be retried immediately.
3. **Fallback data trap:** a synthetic fallback keeps the demo alive, but conclusions from it are not real weather — always disclose it.
4. **Timezone handling:** `dt_txt` is UTC by default; convert with `.dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata')` for local analysis.
5. **Geospatial extension:** the syllabus mentions maps — `folium.Map` + `folium.Marker` can plot each forecast coordinate.
6. **Correlation ≠ causation:** temperature and humidity may correlate because both follow the diurnal cycle; a third variable (time) drives both.
7. **Overplotting:** 40 points are fine, but dense series need transparency (`alpha`) or aggregation before scatter plots.
8. **Cross-unit link (Unit V):** daily aggregation and rolling windows are the core of the time-series unit; resampling (`df.resample('D')`) is the idiomatic pandas alternative to `groupby('Date')`.

**If you remember only one thing:** request → parse nested JSON → flatten to DataFrame → datetime + fill missing → rolling/aggregate → plot.
