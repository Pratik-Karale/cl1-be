# Experiment 6 — Data Aggregation in Plain English

> **In one sentence:** We take 300 retail transactions, fix three missing sales amounts, group the rows by region to see which region earns the most, then cross region with product category to see what each region sells.

## The problem in everyday words

A retail chain records every sale as one row: date, region, product category, quantity, unit price, sales amount and payment mode. Head office asks two questions: which region performs best, and which product categories drive each region? A list of 300 individual rows cannot answer that. Summaries can.

## What the main idea is, without jargon

Aggregation means turning many rows into a few summary rows. Grouping is like sorting laundry into piles by colour and then counting or weighing each pile. The formal name is split, apply and combine: split the rows into groups, apply a calculation such as sum, count or average, then combine the answers into one table.

A pivot table is the same idea in a grid. Regions go down the side, product categories go across the top, and sales amounts fill the cells. It lets you compare every region-category pair at once.

## What the code does, step by step

1. Read the CSV and inspect its shape of 300 rows and 8 columns, its types, summary statistics and first rows.
2. Check missing values in the key columns. Sales amount has three missing values.
3. Fill each missing sales amount with quantity sold multiplied by unit price. This uses the business rule for how sales are calculated instead of guessing an average.
4. Standardise the region text by stripping spaces and applying title case, so " west", "West" and "WEST" become one group.
5. Group by region and calculate the number of transactions, total units sold, total sales and average order value.
6. Sort by total sales from highest to lowest and add a revenue share percentage, calculated as the region total divided by the company total.
7. Identify the top region and the top three regions.
8. Group by region and product category, then build a pivot table with regions as rows, categories as columns, and summed sales as values. Fill missing combinations with zero.
9. Draw a bar chart and pie chart for regional share, plus grouped and stacked bar charts for region against category.

## What the result means

- Missing sales amounts go from **3 to 0**.
- West is the top region: **95 transactions, 406 units, ₹39,33,559.22 in sales, average order value ₹41,405.89, and 33.91% of revenue**.
- South: 70 transactions, 277 units, ₹24,69,182.68, average order ₹35,274.04, and 21.29%.
- North: 59 transactions, 229 units, ₹18,92,976.70, average order ₹32,084.35, and 16.32%.
- East: 47 transactions, 202 units, ₹17,33,804.53, average order ₹36,889.46, and 14.95%.
- Central: 29 transactions, 113 units, ₹15,69,625.60, average order ₹54,125.02, and 13.53%.
- Total company sales are about **₹1,15,99,148.73**, roughly ₹1.16 crore.
- West earns the most overall, but Central has the highest average order value: it has fewer orders, but each order is larger.

## Why these choices?

- Fill missing sales from quantity times price because that recovers the true amount instead of using an unrelated average.
- Standardise region text so the same region is not split into several groups.
- Keep the group key as a normal column so it is easy to sort and plot.
- Use named aggregations so the output columns are readable.
- Add revenue share because absolute revenue hides relative importance.
- Fill missing pivot combinations with zero because no sale in that region-category pair means zero sales, not unknown sales.
- Aggregate before plotting because 300 raw rows are too noisy to compare.

## If the examiner asks (plain answers)

1. **What is aggregation?** Reducing many rows to summary statistics for each group.
2. **Why group by region?** It answers the business question at the level the manager cares about.
3. **What does `as_index=False` do?** It keeps the group key as a normal column instead of the index.
4. **Why fill missing sales from quantity times price?** It recomputes the correct value instead of guessing.
5. **What is average order value?** Total sales divided by the number of orders.
6. **Why add revenue share?** It shows each region's importance relative to the whole company.
7. **What is a pivot table?** A two-dimensional summary that crosses one variable against another with an aggregation.
8. **`pivot_table` versus `groupby`?** Both summarise, but a pivot table gives a wide grid while groupby gives a long table.
9. **Why fill missing pivot cells with zero?** A missing region-category combination means no sale occurred.
10. **Grouped versus stacked bars?** Grouped bars compare side by side; stacked bars show the composition of each total.
11. **Why standardise region text?** Otherwise spelling and spacing differences create duplicate groups.
12. **Which region has the highest average order value?** Central, even though it has the lowest total sales.

**One-line takeaway:** Repair missing values, standardise the group keys, group and aggregate, add share and rank, then pivot for a two-way view.
