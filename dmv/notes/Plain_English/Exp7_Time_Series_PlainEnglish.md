# Experiment 7 — Stock Market Time Series in Plain English

> **In one sentence:** We look at 420 days of a stock price, smooth the jagged line to see the trend, split it into trend and repeating patterns, and use the recent pattern to forecast the next 30 days.

## The problem in everyday words

You have a diary with one stock price per day for about two years. The line jumps up and down every day. You want to answer three questions:

1. Is the price generally going up or down?
2. Is there a repeating pattern, such as a monthly cycle?
3. What might the next month look like?

The order of the days matters. Yesterday comes before today, and today comes before tomorrow.

## What time-series analysis is, without math

A time series is just a column of numbers with a date attached to each row. Because the rows are in time order, special tools become possible.

- **Moving average:** instead of looking at one noisy day, average the last 30 days. The line becomes smoother and the trend becomes easier to see. A 90-day average is even smoother and shows the long-term direction.
- **Decomposition:** imagine a song. It has a slow melody (the trend), a repeating beat (the season), and background static (the leftover noise). Decomposition separates the stock price into those three parts.
- **ARIMA:** a forecasting method that says, "look at the recent pattern and the recent mistakes, then project the next values forward." It first removes the trend by looking at day-to-day changes instead of raw prices.

## What the code does, step by step

1. Generate 420 business days of synthetic prices with a gentle upward trend, a monthly cycle, and random noise.
2. Turn the `Date` column into real dates and make it the index, so pandas knows the row order is time.
3. Sort by date to guarantee chronological order.
4. Draw the closing price.
5. Add a 30-day moving average and a 90-day moving average to the plot.
6. Decompose the price into trend, seasonal and residual parts with a 21-business-day period (about one month).
7. Measure the correlation between closing price and trading volume.
8. Keep the first 390 days for training and the last 30 for testing.
9. Fit ARIMA(5,1,0) on the training part and forecast the 30 test days.
10. Compare the forecast with the actual values using MAE and RMSE.

## What the result means

- The series has **420 rows** from **2024-01-01 to 2025-08-08**.
- The average closing price is about **139.96**, with a range from **91.63 to 187.20**.
- The correlation between price and volume is **0.420**. That is a moderate positive relationship: higher-price days tend to have somewhat higher volume, but the link is not strong and does not prove cause.
- The 30-day forecast has **MAE 6.96** and **RMSE 8.87**. In plain words, a typical forecast is off by about 7 price units, and the RMSE is larger because big misses are punished more.
- The forecast is fairly flat compared with the strong upward trend. That is normal for ARIMA over a long horizon: it becomes cautious and reverts toward a stable level.

## Why these choices?

- A datetime index makes rolling averages, plots and ARIMA time-aware.
- Two moving averages show both the short-term and the long-term direction.
- A 21-day seasonal period matches roughly one month of business days.
- The split is chronological (first 390 days, last 30) because shuffling would let the model peek at the future.
- ARIMA(5,1,0) is a simple, explainable baseline: five past values, one difference to remove trend, and no extra error terms.

## If the examiner asks (plain answers)

**Why can't we shuffle time-series data?** Because the future would leak into training, making the test score unrealistically good.

**What does a moving average do?** It averages a sliding window of recent values to smooth noise and reveal the trend.

**What is seasonality?** A pattern that repeats at a regular interval, such as monthly or weekly.

**What is stationarity?** A series whose average, spread and pattern stay roughly stable over time; ARIMA's differencing step aims for this.

**What does ARIMA(5,1,0) mean?** Use five past prices, take one difference to remove the trend, and do not use past error terms.

**How do you know the forecast is good?** Compare it with the actual last 30 days using MAE and RMSE; lower is better.

**Does correlation mean volume causes price?** No. It only says they tend to move together.

**What would you try next?** SARIMA or exponential smoothing if seasonality is strong, and walk-forward validation for a more realistic score.

**One-line takeaway:** Time series work is about order — smooth it, split it, decompose it, forecast it, and score the forecast on the most recent unseen days.
