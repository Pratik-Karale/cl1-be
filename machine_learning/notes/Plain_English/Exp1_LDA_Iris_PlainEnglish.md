# Experiment 1 — LDA on Iris in Plain English

> **In one sentence:** We teach the computer to tell three iris species apart by finding the viewing angle that makes the three groups look as separated as possible.

## The problem in everyday words

You have 150 iris flowers. For each flower you measured four things: sepal length, sepal width, petal length, and petal width. You also know which of three species each flower belongs to: setosa, versicolor, or virginica.

If you look at only one measurement, the species overlap. A long sepal does not automatically mean setosa. The question is: can we combine the four measurements into a simpler picture where the three species clearly sit apart? And once we have that picture, can we guess the species of a new flower?

## What LDA is, without math

Imagine three groups of students standing in a room. You are the photographer. From some camera angles the groups look jumbled together; from one special angle the groups look far apart and each group looks tightly packed. LDA searches for that best camera angle.

The "angle" is called a **projection**. It means we squash the four measurements into one or two new numbers by looking at the data from a chosen direction. LDA picks the direction where the species are as far apart as possible while each species stays as compact as possible.

Because LDA uses the species labels to choose that direction, it is **supervised**. It is also a classifier: once the flowers are projected, a new flower is assigned to the species it lands closest to.

Three words an examiner may use:

- **Class mean:** the average flower of each species.
- **Within-class spread:** how messy or spread out each species is around its own average. We want this small.
- **Between-class distance:** how far the species averages are from each other. We want this large.

LDA simply looks for the direction with the best ratio of between-class distance to within-class spread.

## What the code does, step by step

1. `load_iris()` loads the 150 flowers, their four measurements, and the three species labels.
2. `train_test_split` hides one quarter of the flowers as a test set, so we can check the model on flowers it has never seen. The `stratify` option keeps the same species mix in both sets.
3. `LinearDiscriminantAnalysis()` creates the LDA model.
4. `model.fit(...)` lets LDA study the training flowers and find the separating directions.
5. `model.predict(...)` guesses the species of each test flower.
6. `accuracy_score`, `confusion_matrix`, and `classification_report` compare the guesses with the true species.
7. The code also asks the model to predict one new flower with measurements 5.1, 3.5, 1.4, and 0.2.

## What the result means

The test accuracy is **1.00**. That means all 38 test flowers were classified correctly. The confusion matrix is a 3-by-3 table; all of its numbers sit on the diagonal, which means no flower was put in the wrong species.

The new flower was predicted as **setosa**. That makes sense because its measurements are typical of the small-petal setosa flowers.

A perfect score here does not mean LDA is always perfect. Iris is a small, clean, well-separated dataset, and one species is very easy to separate. On messy real-world data, LDA would make mistakes.

## Why these choices?

- LDA uses the labels, so it looks for group separation rather than just overall spread.
- Iris is a classic teaching dataset: small, clean, and only three species.
- A 25 percent test set is enough to check the model without wasting too much data.
- Stratified splitting keeps the 50-50-50 species balance in both train and test.
- LDA can produce at most one fewer direction than the number of classes, so three species give at most two directions.

## If the examiner asks (plain answers)

**What does LDA do?** It finds the direction that best separates known groups, then classifies new points in that direction.

**Is LDA supervised or unsupervised?** Supervised, because it uses the species labels.

**How is it different from PCA?** PCA looks for the direction with the most spread and ignores labels. LDA looks for the direction that best separates the labelled groups.

**Why did it get 100 percent?** The Iris data is small and the species are easy to separate; the test flowers were not tricky.

**What is a projection?** Squashing several measurements into fewer new numbers by viewing the data from a chosen direction.

**Why only two directions for three species?** There are only two independent ways to separate three group averages; a third adds nothing.

**Does LDA need scaling?** It is not strictly required, but scaling is good practice and helps in general.

**What does the confusion matrix show?** Rows are the true species and columns are the predicted species; the diagonal holds the correct guesses.

**One-line takeaway:** LDA is supervised group separation: find the viewing angle where the species are farthest apart and each species is tightest, then classify there.
