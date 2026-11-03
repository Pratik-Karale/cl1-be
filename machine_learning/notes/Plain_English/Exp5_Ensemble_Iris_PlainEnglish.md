# Experiment 5 — Ensemble Learning on Iris in Plain English

> **In one sentence:** Instead of trusting one classifier, we ask a panel of classifiers and compare four ways of combining their opinions.

## The problem in everyday words

We return to the Iris flowers: 150 flowers, four measurements, three species. A single classifier can do well, but different classifiers make different mistakes. Sometimes a group of decent classifiers can beat the best single one.

This is **ensemble learning**: combining several models into one stronger decision. The experiment trains four ensemble-style models and compares their accuracy.

## What the four methods are, without math

Think of a panel of experts deciding a difficult case.

- **Hard voting** is a simple majority vote. Three different experts each give an answer, and the answer with the most votes wins. The experts are a logistic regression model, a small decision tree, and a random forest.
- **AdaBoost** is a panel that learns from its own mistakes. The first expert tries the problem. The next expert is told to focus on the cases the first one got wrong. This repeats, and the final decision weights the experts by how well they did.
- **Gradient Boosting** is similar but smoother. Each new expert is trained to correct the remaining error of the panel so far, like a team gradually polishing a rough draft.
- **XGBoost** is an improved, regularized version of gradient boosting. It is designed to be fast and to avoid overfitting, and it adds penalties for overly complicated trees.

Two ideas appear often:

- **Bagging** means training several models independently on random samples and averaging them. It mainly reduces **variance**, meaning the model becomes less jumpy.
- **Boosting** means training models one after another, each fixing the previous errors. It mainly reduces **bias**, meaning the model becomes less systematically wrong.

**Random Forest**, used inside the voting panel, is a bagging method built from many decision trees.

## What the code does, step by step

1. Load Iris and split it into 75 percent training and 25 percent testing, keeping the species balance.
2. Build a **hard voting** model from three different classifiers: logistic regression, a shallow decision tree, and a random forest.
3. Build an **AdaBoost** model with 100 small weak learners.
4. Build a **Gradient Boosting** model.
5. Build an **XGBoost** model, if the library is available.
6. Train each model on the same training data.
7. Predict the species for the test flowers.
8. Compare the models using accuracy, macro precision, macro recall, and macro F1.

## What the result means

**Gradient Boosting wins** with an accuracy of **0.9737**, or about 97.4 percent. AdaBoost and XGBoost both reach **0.9211**. Hard voting reaches **0.8947**.

In plain words: the boosting methods made better use of the training data than the simple majority vote. Gradient Boosting was the best at correcting its own errors on this small, clean dataset.

The macro averages treat all three species equally, so they are a fair comparison even if one species appeared more often in the test set. Here the test set is balanced, so accuracy and macro averages tell a similar story.

## Why these choices?

- Voting combines genuinely different models, which is useful when their mistakes differ.
- AdaBoost and Gradient Boosting are the standard boosting families, so comparing them shows how the idea evolves.
- XGBoost is included because it is widely used in practice and often strong on tabular data.
- Macro averages are used because this is a multiclass problem and every species matters equally.
- Iris is small, so the experiment runs quickly and the differences between methods are easy to see.

## If the examiner asks (plain answers)

**What is ensemble learning?** Combining several models so their combined decision is better than any single model.

**What is a weak learner?** A simple model that is only slightly better than guessing.

**Hard voting versus soft voting?** Hard voting counts labels; soft voting averages predicted probabilities and usually gives a more confident answer.

**Bagging versus boosting?** Bagging trains models independently and averages them to reduce variance. Boosting trains them in sequence to reduce bias.

**How does AdaBoost work?** It gives more attention to the examples that were previously misclassified.

**How does Gradient Boosting work?** Each new tree corrects the remaining error of the current ensemble.

**What is special about XGBoost?** It adds regularization, handles missing values, and is optimized for speed.

**Why did Gradient Boosting win?** On this small, clean dataset it corrected errors efficiently without becoming too complex.

**What is a random forest?** Many decision trees trained on random samples and random feature subsets, then averaged.

**What is stacking?** Training a final model to combine the predictions of several base models.

**One-line takeaway:** An ensemble is a panel of experts; voting combines opinions, while boosting builds a sequence of experts where each one fixes the previous one's mistakes, and here Gradient Boosting was the best panel.
