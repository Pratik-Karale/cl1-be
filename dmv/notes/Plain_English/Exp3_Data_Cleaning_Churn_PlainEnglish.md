# Experiment 3 — Data Cleaning and Preparation in Plain English

> **In one sentence:** We take a messy telecom customer file, remove duplicate customers, fix inconsistent text and wrong data types, fill missing values, tame extreme outliers, create useful new columns, and split the data into training and testing sets.

## The problem in everyday words

A telecom company wants to understand why customers leave, which is called churn. The raw file is messy: some customers appear twice, gender is written as "Male", "male" and "M", some numbers are stored as text, several cells are blank, and a few charges are absurdly high. If a model learns from this mess, its conclusions will be unreliable.

## What the main idea is, without jargon

Data cleaning is like preparing vegetables before cooking. You wash them (fix text), remove spoiled pieces (duplicates), fill gaps sensibly (imputation), trim extreme bits (outliers), chop them into useful pieces (feature engineering), and season them consistently (encoding and scaling). The old saying applies: garbage in, garbage out.

Two ideas matter most. First, missing values should be filled with something sensible rather than guessed randomly. Second, extreme values should be tamed, not silently deleted, so real customers are not lost.

## What the code does, step by step

1. Read the CSV and inspect its shape, column types, summary statistics and first rows.
2. Remove full duplicate rows, then remove duplicates by customer ID, keeping the first record. This takes 205 rows down to 200.
3. Strip extra spaces from text columns and map inconsistent spellings to one version: `M` and `male` become `Male`, and `month to month` becomes `Month-to-month`.
4. Convert `TotalCharges` from text to a number. Blank cells become missing values instead of causing an error.
5. Fill missing numbers with the median and missing categories with the most common category. For a customer with zero tenure, set total charges to zero. Missing values go from 4, 6 and 4 down to zero.
6. Find outliers with the IQR rule and cap them at the fence values instead of deleting the rows.
7. Create new columns: a tenure group, a count of total services, a charge-increase ratio, and a binary churn column.
8. Convert categories to numbers with one-hot encoding, then standardise the numeric columns.
9. Split the data 80/20 with stratification so the churn ratio stays the same in both parts. This gives 160 training rows and 40 test rows with 23 features.
10. Export one cleaned readable file and one model-ready file.

## What the result means

- Five duplicate rows are removed, leaving **200 customers**.
- Missing values before cleaning were tenure 4, monthly charges 6 and total charges 4. After cleaning there are **zero missing values**.
- The IQR rule flags and caps one outlier in each of tenure, monthly charges and total charges.
- After scaling, numeric features have mean roughly zero and standard deviation roughly one.
- The split is **160 training rows and 40 test rows**, with **23 features**.
- Two files are exported: a cleaned data file and a fully encoded model-ready file.

## Why these choices?

- Remove duplicates so one customer is not counted twice.
- Use the median rather than the mean because the data is skewed and contains outliers.
- Cap outliers rather than delete them, so the number of customers stays the same.
- Use one-hot encoding because most models need numbers, and nominal categories have no natural order.
- Drop the first dummy column to avoid redundant duplicate columns.
- Stratify the split so the churn and non-churn proportions are preserved.
- Scale features so distance-based and gradient-based models are not dominated by large-value columns.
- Split before scaling to avoid leaking test information into training.

## If the examiner asks (plain answers)

1. **Why clean before modelling?** Duplicates distort counts, blanks break many estimators, and outliers distort averages and scaling.
2. **`drop_duplicates()` versus `subset=`?** The first checks the whole row; the second checks only the listed key columns.
3. **Why is `TotalCharges` text?** New customers have blank values, so the column arrives as text.
4. **Mean or median imputation?** Median, because it is robust to skew and extreme values.
5. **When use the mode?** For categorical columns; fill with the most frequent category.
6. **What is the IQR rule?** Find the middle 50% of values, then flag anything more than one and a half times that range beyond the edges.
7. **Why cap instead of drop?** Capping keeps every customer and the sample size.
8. **Why one-hot encode?** Models need numbers, and nominal categories must not imply an order.
9. **Why `drop_first=True`?** It removes one redundant dummy column and avoids the dummy-variable trap.
10. **Why stratify?** It keeps the churn/non-churn ratio the same in training and testing.
11. **What if you scale before splitting?** Test information leaks into training, so the reported score is too optimistic.

**One-line takeaway:** Remove duplicates, fix text and types, impute median or mode, cap outliers, engineer features, encode and scale, then split with stratification.
