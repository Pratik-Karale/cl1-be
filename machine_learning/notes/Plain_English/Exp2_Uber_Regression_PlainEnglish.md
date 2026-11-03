# Experiment 2 — Uber Fare Prediction in Plain English

> **In one sentence:** We clean a large taxi-trip table and compare three ways of predicting the fare, then discover that the available clues are too weak to predict it well.

## The problem in everyday words

A ride-hailing company records every trip: when it started, where it started, where it ended, how many passengers, and how much the trip cost. We want to predict the fare before the trip happens.

The file has 200,000 rows and 9 columns. Like most real data, it is messy: some fares are negative or missing, some coordinates are outside the valid range, and some passenger counts are impossible. So the job has two halves: clean the data, then fit a prediction model.

## What regression is, without math

Regression means drawing a line or curve through past examples so that, given new inputs, the computer can output a number. Here the number is the fare.

**Linear regression** finds the straight-line combination of inputs that is closest to the fares. **Ridge** is a cautious version: it stops the model from relying too heavily on any one input by gently shrinking all the weights. **Lasso** is a stricter version: it can shrink useless inputs all the way to zero, effectively switching them off.

A useful picture: you are trying to guess a taxi fare, and you have several noisy hints. Ridge says "trust every hint a little." Lasso says "if a hint is useless, ignore it completely."

## What the code does, step by step

1. Read `uber.csv` into a table with `pandas`.
2. Remove infinite values, drop rows where the fare is missing, and keep only fares that are zero or positive.
3. Sanity-check the coordinates (longitude between -180 and 180, latitude between -90 and 90) and keep passenger counts between 1 and 6.
4. Convert the pickup time into useful numbers: hour of day, day, month, and weekday. Then drop the original timestamp.
5. Find fare outliers with the interquartile-range rule. The rule flags values that sit far outside the middle 50 percent of fares.
6. Show a box plot and a correlation heatmap so we can see which inputs move with the fare.
7. Build a pipeline: fill missing numbers with the median, put every input on the same scale, then apply Linear Regression, Ridge, or Lasso.
8. Split the cleaned data into training and test sets.
9. Compare the three models using R-squared, RMSE, and MAE.

## What the result means

After basic cleaning, **199,261 rows** remain, and the IQR rule flags **17,120** fare outliers.

The correlations with fare are all tiny. The largest is about **0.0235** for pickup month. That means no single input moves strongly with the fare.

All three models score an R-squared of about **0.0013**. In plain words, the model explains only **0.13 percent** of the variation in fares. It is barely better than always guessing the average fare. RMSE is about **10.07** fare units and MAE about **6.09**.

This is not a coding failure. It is a missing-information problem. The table does not contain trip distance, traffic, or route, which are the real drivers of a fare. The models can only use what they are given.

## Why these choices?

- Linear Regression is the simple baseline; Ridge and Lasso show what regularization does.
- Scaling is needed because Ridge and Lasso punish large weights, and unscaled features would be treated unfairly.
- Median imputation is safer than mean imputation when the data has extreme fares.
- The IQR rule is a robust way to spot outliers without assuming a bell-shaped distribution.
- R-squared, RMSE, and MAE together show both the share of variation explained and the size of typical errors.

## If the examiner asks (plain answers)

**What is regression?** Predicting a continuous number, such as a fare, rather than a category.

**What does R-squared mean?** The fraction of the fare variation explained by the model. Here it is 0.0013, or 0.13 percent.

**Why is the score so low?** The table lacks trip distance and traffic, which are the main fare drivers.

**What is an outlier?** A value far away from the rest of the data, often caused by errors or rare events.

**Why use the median instead of the mean for missing values?** The median is not dragged around by extreme fares.

**Ridge or Lasso?** Ridge shrinks all weights and keeps every feature. Lasso can set useless weights to zero and select features.

**Why scale the inputs?** So that a large-range feature does not dominate the penalty or the model.

**RMSE or MAE?** RMSE punishes big errors more because it squares them. MAE is the average size of the error.

**How could we improve the model?** Add trip distance, route information, traffic, time-of-day demand, and weather.

**One-line takeaway:** Cleaning matters, but no model can predict well from weak clues; the low score correctly points to missing features, not a broken algorithm.
