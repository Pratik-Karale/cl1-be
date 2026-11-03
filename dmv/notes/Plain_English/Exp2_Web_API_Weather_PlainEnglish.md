# Experiment 2 — Interacting with Web APIs in Plain English

> **In one sentence:** We ask a weather service for a five-day forecast through its API, turn the returned JSON into a table, and then study temperature, humidity, wind and rain over time.

## The problem in everyday words

You want weather data for a city, but you do not have a file. The weather service publishes the data on its website through an API. You need to request it, understand the reply, and convert that reply into rows and columns you can analyse and plot.

## What the main idea is, without jargon

An API is like a waiter in a restaurant. You give an order (which city, which units, your API key), the kitchen prepares it, and the waiter brings back a tray. Here the tray is a JSON response — text organised as nested key-value pairs.

The forecast reply has a list of 40 time slots, each three hours apart. Each slot has sections: a `main` section (temperature, humidity, pressure), a `wind` section, a `clouds` section, and a `weather` section describing conditions. You drill into those sections to pull out flat columns for a table.

## What the code does, step by step

1. Get an API key from OpenWeatherMap and choose the city and metric units.
2. Call `requests.get(url, params=..., timeout=10)` so the program does not hang on a slow network.
3. Check that the status code is 200, which means success.
4. Parse the reply with `response.json()` to get a Python dictionary.
5. Walk through the 40 entries and extract timestamp, temperature, feels-like, minimum, maximum, humidity, pressure, wind speed, cloud cover, rain and weather condition into a list of records.
6. Build a DataFrame from that list.
7. Convert the timestamp text to a real date-time, fill missing rain with zero (the API simply leaves out the rain section when it is dry), interpolate other small gaps, sort by time, and remove duplicate timestamps.
8. Compute a 24-hour rolling average using `rolling(window=8)`, because eight forecast points times three hours equals 24 hours.
9. Fit a straight trend line to the temperature.
10. Group by date and by time of day, then draw a temperature line with its rolling average, a daily rain bar chart, a temperature-humidity bubble chart, and a correlation heatmap.

## What the result means

- The clean table has **40 rows and 11 columns**.
- Average temperature is **28.66 °C**; the maximum is **34.91 °C** on 2026-10-19 at 09:00, and the minimum is **22.66 °C** on 2026-10-23 at 00:00.
- Average humidity is **51.3%**, total rain is **0.12 mm**, and maximum wind speed is **3.58 m/s**.
- The daily table covers six dates. Morning temperatures average **32.03 °C** and night temperatures **26.62 °C**, showing a clear daily cycle.
- Four plots summarise the forecast: temperature trend, daily rain, temperature against humidity, and a heatmap of how the variables move together.
- If the API cannot be reached, the code uses a simulated reply with the same structure so the practical still runs. This must be labelled as synthetic data, not real weather.

## Why these choices?

- Pass query values through `params` instead of gluing them into the URL, so `requests` encodes them safely.
- Use a timeout so a dead network does not freeze the program.
- Fill missing rain with zero because "no rain section" means no rain, not unknown rain.
- Use a window of eight because the forecast points are three hours apart.
- Smooth with a rolling average so short-term noise does not hide the overall trend.
- Sort by timestamp because rolling averages and trend lines assume chronological order.

## If the examiner asks (plain answers)

1. **What is an API?** A way for one program to request data or a service from another program over the web.
2. **What does status code 200 mean?** Success. 401 means a bad or missing key, 404 means a wrong address, and 429 means too many requests.
3. **`response.json()` versus `response.text`?** The first gives a parsed dictionary; the second leaves the reply as raw text.
4. **Why is the data nested?** The API groups related values into sections such as main, wind and clouds.
5. **Why fill missing rain with zero?** The rain section is absent when there is no rain.
6. **Why does window eight mean 24 hours?** Each point is three hours apart, so eight points cover 24 hours.
7. **Why use a rolling average?** It smooths day-to-day noise and makes the trend easier to see.
8. **What if the API fails?** The practical falls back to simulated data with the same shape, clearly labelled as synthetic.
9. **What is the difference between weather and climate?** Weather is the short-term state; climate is the long-term pattern.

**One-line takeaway:** Request the data, parse the nested JSON, flatten it into a table, fix dates and missing values, smooth and aggregate, then plot.
