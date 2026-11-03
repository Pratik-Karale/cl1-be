# Experiment 5 — Data Visualization using matplotlib in Plain English

> **In one sentence:** We take 180 days of air-quality readings, smooth them with a seven-day average, and draw the right chart for each question: trend over time, monthly comparison, distribution, and relationships between pollutants.

## The problem in everyday words

A city records daily levels of PM2.5, PM10, nitrogen dioxide and carbon monoxide, plus an overall Air Quality Index. The data is a long list of numbers. We need to show clearly whether air quality is getting better or worse, which pollutant drives the bad days, and whether pollutants rise and fall together.

## What the main idea is, without jargon

A chart is a sentence made of shapes. A line chart says "this is what happened over time". A bar chart says "compare these categories". A stacked bar says "here is the total and what it is made of". A box or violin plot says "here is the typical range and the unusual values". A scatter plot says "do these two numbers move together?". A bubble chart adds a third or fourth number through size and colour. A heatmap says "which columns rise and fall together".

The rule is simple: choose the chart from the question you are answering, not from habit.

## What the code does, step by step

1. Read the CSV and inspect its shape, types, summary statistics and first rows.
2. Convert the date column to a real date, sort chronologically, and derive month and month name.
3. Interpolate small gaps in pollutant readings, then add a seven-day moving average of the AQI.
4. Draw a line chart of daily AQI with the seven-day average and dashed reference lines at 100 and 200.
5. Draw three separate line charts for PM2.5, PM10 and carbon monoxide, each with a safe-level reference.
6. Group by month and draw a bar chart of average AQI plus a stacked bar showing the pollutant mix.
7. Draw a box plot of AQI grouped by the dominant pollutant, and a violin plot of pollutant concentrations.
8. Draw a scatter plot of PM2.5 against AQI with a fitted straight trend line, and a bubble chart of PM10 against AQI where bubble size is carbon monoxide and colour is PM2.5.
9. Add clear titles, axis labels, legends, a grid, readable date labels and a tight layout.

## What the result means

- The dataset has **180 rows and 10 columns**.
- AQI averages **146.62**, with a standard deviation of **59.79**, a minimum of **35**, a maximum of **289** and a median of **146.50**.
- Average PM2.5 is **65.12**, PM10 is **123.11**, nitrogen dioxide is **31.78** and carbon monoxide is **1.10**.
- The AQI categories are: up to 50 Good, up to 100 Satisfactory, up to 200 Moderate, up to 300 Poor, and above 300 Very Poor.
- PM2.5 is the dominant pollutant on most high-AQI days.
- Six kinds of figure are produced: trend with moving average, pollutant lines, monthly bar and stacked bar, box and violin, scatter with trend line, and bubble chart.

## Why these choices?

- Use a seven-day moving average because daily values jump around and hide the underlying trend.
- Use a line chart for time because time is ordered and continuous.
- Add threshold lines so the reader can see when the AQI crosses a health category.
- Use a stacked bar to show both the total and what makes up that total.
- Use box and violin plots because they show spread and unusual values, not just an average.
- Use a scatter plot with a trend line to show the direction of a relationship.
- Use a bubble chart to show more than two variables in one view.
- Format the dates and tighten the layout so the chart explains itself without extra text.

## If the examiner asks (plain answers)

1. **Why matplotlib?** It is a low-level, highly customisable plotting library, and seaborn builds on top of it.
2. **`plot` versus `scatter`?** A line plot connects ordered points; a scatter plot shows independent points to reveal a relationship.
3. **Why a seven-day moving average?** It smooths day-to-day noise so the trend is easier to see.
4. **What is `axhline`?** It draws a horizontal reference line at a fixed y-value, such as the 100 AQI limit.
5. **Why stacked bars?** They show a total and its composition at the same time.
6. **What are the parts of a box plot?** The box covers the middle 50%, the line is the median, the whiskers extend one and a half times the box range, and dots are outliers.
7. **Violin versus box?** A violin also shows the shape of the distribution, including multiple peaks.
8. **What does a degree-one polynomial fit give?** The slope and intercept of the best straight line through the points.
9. **How does the bubble chart encode four things?** x is PM10, y is AQI, bubble size is carbon monoxide, and colour is PM2.5.
10. **Why format the date axis?** Raw date-times are hard to read; month and year labels are clearer.
11. **What if `plt.show()` is omitted?** In a script the figure may never appear on screen.
12. **Which pollutant dominates?** PM2.5, especially on the worst days.

**One-line takeaway:** Pick the chart by the question: line for time, bar for categories, box or violin for distributions, scatter or bubble for relationships.
