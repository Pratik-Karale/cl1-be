# Experiment 5 — Data Visualization using matplotlib (Viva Theory)

> **30-second answer:** Load 180 days of city air-quality data, convert dates, compute a **7-day moving average**, and build the full matplotlib gallery: line plots with AQI thresholds, individual pollutant lines, monthly **bar + stacked bar**, **box + violin** distributions, and **scatter + bubble** charts with a fitted trendline. Headline: mean AQI **146.62**, range **35–289**, PM2.5 is the dominant pollutant.

## 1. MOST LIKELY — Core of the Experiment (High Yield)

- **Aim:** Visualise AQI trends and pollutant relationships effectively using matplotlib.
- **Dataset:** `City_Air_Quality.csv` — **180 rows × 10 columns** (daily, 2026-05-01 onward): Date, Time, City, PM2.5, PM10, NO2, CO, AQI, Dominant_Pollutant, AQI_Category.
- **Pipeline:**
  1. `read_csv`; inspect shape, `info()`, `describe()`, `head()`.
  2. `pd.to_datetime(Date)`, sort chronologically, derive `Month` and `Month_Name`.
  3. Interpolate missing pollutant values; add `AQI_7Day_MA = AQI.rolling(7, min_periods=1).mean()`.
  4. **Line plot:** daily AQI + 7-day MA + dashed threshold lines at **100** and **200**.
  5. **Three stacked line plots:** PM2.5, PM10, CO with safe-standard references.
  6. **Monthly aggregation** then **bar** (avg AQI) and **stacked bar** (PM2.5 + PM10 + NO2).
  7. **Box plot** of AQI grouped by dominant pollutant; **violin plot** of pollutant concentrations.
  8. **Scatter** PM2.5 vs AQI with `np.polyfit` trendline; **bubble** chart PM10 vs AQI (size = CO, colour = PM2.5).
  9. Customise titles, axis labels, legends, grid, colour maps, `mdates.DateFormatter`, `plt.rcParams`.

**Code skeleton**
```python
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)
df['AQI_7Day_MA'] = df['AQI'].rolling(window=7, min_periods=1).mean()
fig, ax = plt.subplots(figsize=(13,5))
ax.plot(df.Date, df.AQI, label='Daily AQI', alpha=.6)
ax.plot(df.Date, df.AQI_7Day_MA, linewidth=2.5, label='7-Day MA')
ax.axhline(100, ls='--', label='Satisfactory (100)')
ax.axhline(200, ls='--', label='Moderate (200)')
ax.set_title('AQI Trend'); ax.set_xlabel('Date'); ax.set_ylabel('AQI')
ax.legend(); ax.grid(True, ls=':', alpha=.6); plt.show()
```

**Expected output**
- AQI: mean **146.62**, std **59.79**, min **35**, max **289**, median **146.50**.
- PM2.5 mean **65.12**; PM10 mean **123.11**; NO2 mean **31.78**; CO mean **1.10**.
- AQI categories: ≤50 Good, ≤100 Satisfactory, ≤200 Moderate, ≤300 Poor, >300 Very Poor.
- Figures produced: (1) trend + MA + thresholds, (2) three pollutant lines, (3) monthly bar + stacked bar, (4) box + violin, (5) scatter + trendline, (6) bubble chart.

**Viva Q&A**
1. **Why matplotlib?** Low-level, highly customisable plotting library; seaborn/pandas build on it.
2. **`plot` vs `scatter`?** `plot` draws connected lines (ordered/time data); `scatter` draws independent points to show relationships.
3. **Why a 7-day moving average?** Smooths day-to-day noise, revealing the underlying seasonal trend.
4. **What does `axhline` do?** Draws a horizontal reference line at a constant y (e.g. the 100 AQI limit).
5. **Why stacked bars?** Show both the total and the composition of that total across categories.
6. **Box plot anatomy?** Box = Q1–Q3 (IQR), line = median, whiskers = 1.5×IQR, dots = outliers.
7. **Violin vs box?** Violin adds a kernel-density shape, revealing multi-modality that a box plot hides.
8. **What does `np.polyfit(x,y,1)` give?** Slope/intercept of the least-squares line drawn over the scatter.
9. **Bubble chart encoding?** x = PM10, y = AQI, size = CO, colour = PM2.5 — four variables in one view.
10. **Why `mdates.DateFormatter`?** Formats date ticks as readable month/year labels instead of raw datetimes.
11. **Why `plt.tight_layout()`?** Prevents subplot titles/labels from overlapping.
12. **What if `plt.show()` is omitted?** In scripts the figure may not display; in notebooks it renders inline.
13. **Which pollutant dominates?** PM2.5 — it is the dominant pollutant for most high-AQI days.

## 2. LIKELY — Syllabus Theory Around This Experiment

### Unit III — basics of data visualisation
- **Exploratory graphics** help *you* find patterns; **presentation graphics** communicate a chosen message to others.
- **Design choices:** chart type, scale, colour, ordering, annotation, and avoiding chart junk.
- **Higher-dimensional displays:** bubble charts, colour/size encodings, small multiples.
- **3-D plots** exist but are often harder to read than well-designed 2-D views.

| Chart | Best used for |
|---|---|
| Line | trend over time |
| Bar | compare categories |
| Stacked bar | category composition |
| Box/violin | distribution + outliers |
| Scatter | relationship between two variables |
| Heatmap | matrix of correlations |

**Q&A**
1. **What is computational statistics?** Using computation (resampling, simulation) to support statistical inference and visual exploration.
2. **Presentation vs exploratory graphics?** Exploratory is for discovery, presentation is polished for an audience.
3. **Why choose a line chart for AQI?** Time is ordered and continuous; lines reveal trend and seasonality.
4. **Why not a pie chart for this?** Pie charts compare parts of a whole poorly when there are many slices; bars are more precise.
5. **What is overplotting?** Too many overlapping points hide density; fix with alpha, jitter, or aggregation.
6. **How can a chart mislead?** Truncated y-axis, inconsistent scales, 3-D distortion, cherry-picked ranges.
7. **What are output formats?** PNG (raster), SVG/PDF (vector, scalable) — vector is preferred for publication.
8. **What is a static graphic?** A fixed image (as opposed to interactive dashboards); matplotlib produces static graphics.
9. **Why customise labels/titles?** A chart must be self-explanatory without the accompanying text.

## 3. LEAST LIKELY — Stretch / Advanced / Edge Cases

1. **Misleading axes:** starting the y-axis at a non-zero value exaggerates differences; always label and choose scales honestly.
2. **Colour accessibility:** red–green palettes are problematic for colour-blind readers; use viridis/plasma or ordered palettes.
3. **Dual axes:** two y-axes can imply false correlations; use only when units genuinely differ and annotate clearly.
4. **Bubble-size scaling:** area (not radius) should encode magnitude; scaling raw values by 75 is a visual approximation, not a precise quantitative encoding.
5. **Small sample monthly averages:** monthly means from ~30 daily points are noisy; show confidence bands or raw points.
6. **Performance:** rendering millions of points is slow; aggregate, sample, or use hexbin/2-D histograms.
7. **Cross-unit link (Unit II):** correlation between pollutants and AQI should be tested, not just visualised — a heatmap suggests association, not significance.

**If you remember only one thing:** pick the chart by the question — line for time, bar for categories, box/violin for distributions, scatter/bubble for relationships.
