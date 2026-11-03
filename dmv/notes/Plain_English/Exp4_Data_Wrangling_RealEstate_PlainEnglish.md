# Experiment 4 — Data Wrangling in Plain English

> **In one sentence:** We take a messy real-estate file with ugly column names, missing values and extreme prices, clean it, merge in neighbourhood information, filter to recent residential sales, and summarise prices by neighbourhood and property type.

## The problem in everyday words

A property website exports its listings with column names like "Property ID #", "Area (Sq.Ft.)" and "Sale Price (INR Lakhs)". Some areas are blank, and a few prices and areas are wildly unrealistic. A separate small table lists neighbourhood amenities such as metro distance and livability score. The goal is one clean table that answers: which neighbourhood and property type is the most expensive?

## What the main idea is, without jargon

Data wrangling means cleaning, joining and reshaping data until it is ready for analysis. Cleaning fixes names and gaps. Joining is like matching two address books by area name: one book has properties, the other has area information, and the neighbourhood name is the common key. Filtering keeps only the rows relevant to the question. Reshaping, such as a pivot table, turns a long list into a neat grid.

## What the code does, step by step

1. Read the real-estate CSV.
2. Clean the column names with a pattern: remove special characters, replace spaces with underscores, and collapse repeated underscores.
3. Rename long names to short ones, such as `Neighborhood_Locality` to `Neighborhood` and `Sale_Price_INR_Lakhs` to `Sale_Price_Lakhs`.
4. Drop rows where the sale price is missing, because a price analysis cannot use a property with no price.
5. Fill missing area with the median area of that property type, missing bedroom count with the median, and missing furnishing status with the most common value.
6. Merge the amenities table on neighbourhood using a left join, which keeps every property and attaches amenities where they exist.
7. Filter to residential property types, sales from 2024 onward, and area of at least 600 square feet.
8. Cap extreme area and price values using the IQR rule, then derive price per square foot.
9. Group by neighbourhood to get average price, and build a pivot table of neighbourhood against property type.
10. Encode categories: keep furnishing order with an ordinal map, use label encoding for property type, and one-hot encode the remaining nominal columns.

## What the result means

- Missing values before cleaning: area 8, bedrooms 6, furnishing 4 and price 3. After cleaning there are **zero missing values**.
- The merge grows the table from **147 rows and 9 columns** to **147 rows and 12 columns**.
- Filtering keeps **83 residential rows**.
- After filtering, no area or price values fall outside the IQR fences.
- Average price by neighbourhood, in lakhs: Baner **161.70**, Hinjewadi **154.43**, Wakad **153.66**, Kothrud **151.60**, Koregaon Park **148.95**, Viman Nagar **147.29**.
- The final encoded table has **20 columns**, including location and property-type indicators.

## Why these choices?

- Clean column names because spaces and symbols break code and make merges unreliable.
- Drop rows with a missing target because inventing a sale price would fabricate the outcome we are studying.
- Use the median area per property type because villas and apartments have very different size ranges.
- Use a left join so a property is never silently lost just because its area is missing from the amenities table.
- Filter before aggregating because the business question is about recent residential sales, not commercial listings.
- Cap outliers instead of deleting them so the sample size is preserved.
- Use different encoding methods because ordered categories, tree models and nominal categories have different needs.

## If the examiner asks (plain answers)

1. **What is data wrangling?** Cleaning, joining and reshaping raw data into an analysis-ready table.
2. **Why clean column names?** Spaces and symbols cause syntax errors and make joins unreliable.
3. **Why drop rows with a missing price?** A missing target cannot be summarised, and imputing it would invent an outcome.
4. **Why use the median area per property type?** Villas and apartments have different typical sizes.
5. **What does `how='left'` do?** It keeps every left-hand row and attaches matching right-hand values, leaving blanks where there is no match.
6. **What if a neighbourhood is missing from the amenities table?** The left join keeps the property, with blank amenity fields.
7. **Why filter before aggregating?** It answers a specific business question and removes irrelevant commercial records.
8. **Why cap outliers rather than delete?** It keeps all properties and avoids selection bias.
9. **`pivot_table` versus `groupby`?** A pivot table reshapes results into a grid; a groupby returns a long table.
10. **Why three encoding methods?** Ordinal data keeps its order, labels suit tree models, and one-hot encoding avoids false order for nominal data.
11. **What does `drop_first=True` prevent?** Redundant dummy columns that cause multicollinearity.

**One-line takeaway:** Clean the names, handle gaps by group, merge the extra table, filter, cap outliers, summarise with a pivot, then encode.
