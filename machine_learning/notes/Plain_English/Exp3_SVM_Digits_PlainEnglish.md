# Experiment 3 — SVM for Handwritten Digits in Plain English

> **In one sentence:** We teach a computer to read small handwritten digits by drawing the widest possible boundary between the numbers.

## The problem in everyday words

You have 1,797 tiny pictures of handwritten digits, each only 8 by 8 pixels. Every pixel is a shade of gray, so each picture is really a row of 64 numbers. The computer must look at those 64 numbers and say whether the picture is a 0, a 1, a 2, and so on up to 9.

This is a classification problem: the answer is one of ten categories. It is also a hard one because different people write the same digit differently, and at 8 by 8 pixels an 8 can look a lot like a 1 or a 9.

## What SVM is, without math

Imagine two groups of houses on a map. You want to build a road between them. Many roads could separate the groups, but the best road is the one with the widest empty space on both sides. SVM finds that widest road. The empty space is called the **margin**, and the houses closest to the road are called **support vectors** because they decide where the road goes.

Real digit pictures are not neatly separable by a straight road. The **RBF kernel** solves this by bending the road into curves. A useful way to think about the RBF kernel is that it measures how similar two pictures are: pictures with similar pixel patterns pull the boundary one way, very different pictures pull it another.

Two settings control the road:

- **C** says how strictly to avoid mistakes. A large C means "almost no mistakes allowed," which can make the road twist too tightly around the training pictures. A small C allows a few mistakes and gives a smoother, wider road.
- **gamma** says how far each picture's influence reaches. A large gamma means each picture only affects its immediate neighbors, which can overfit. A small gamma gives a smoother, broader boundary.

## What the code does, step by step

1. `load_digits()` loads the 1,797 pictures and their correct digit labels.
2. `train_test_split` keeps one quarter of the pictures for testing, so we check the model on unseen handwriting.
3. `StandardScaler` puts all 64 pixel values on the same scale, because SVM is sensitive to distance.
4. `SVC(kernel="rbf", C=5, gamma="scale")` creates the SVM with a curved boundary. The `scale` option sets gamma automatically from the number of features and their spread.
5. `model.fit(...)` lets the SVM learn from the training pictures.
6. `model.predict(...)` guesses the digit for each test picture.
7. `accuracy_score`, `confusion_matrix`, and `classification_report` measure how many guesses were right.

## What the result means

The test accuracy is **0.98**, which means about **441 out of 450** test pictures were classified correctly.

The confusion matrix is a 10-by-10 table. Most numbers sit on the diagonal, meaning the correct digit was chosen. The few off-diagonal numbers show confusions such as an 8 being read as a 1 twice, a 5 being read as a 9, and a 9 being read as 6 or 7.

The weakest class is **8**, with a recall of about **0.93**. The overall macro average is about **0.98**. In plain words: the model is very reliable, but low-resolution 8s are the hardest digit because their loops can resemble other digits.

## Why these choices?

- SVM works well when there are many input features, and 64 pixels is a lot of features.
- The RBF kernel handles boundaries that are not straight lines, which digit shapes require.
- Scaling is essential because raw pixel brightness values would otherwise dominate the distance calculation.
- C equal to 5 is a moderate setting: strict enough to learn, not so strict that it memorizes noise.
- One-quarter test data gives 450 unseen pictures, enough to measure performance honestly.

## If the examiner asks (plain answers)

**What is SVM?** A classifier that separates classes with the widest possible boundary.

**What is the margin?** The empty space between the boundary and the nearest training points.

**What are support vectors?** The closest training points; they alone determine the boundary.

**What does C do?** It controls how many mistakes are tolerated. Large C means fewer mistakes allowed and a tighter boundary.

**What does gamma do?** It controls how far each training point's influence reaches. Large gamma means local, wiggly boundaries.

**Why scale the pixels?** SVM uses distances, so features on different scales would distort the boundary.

**How does SVM handle ten classes?** It builds one classifier for each pair of digits and lets them vote.

**Why is 8 the hardest digit?** At 8 by 8 pixels, the loops of an 8 can look like a 1, a 9, or a 0.

**Would a straight-line SVM work?** It would be faster but usually less accurate because digit boundaries are curved.

**One-line takeaway:** SVM draws the widest possible road between classes; the RBF kernel bends that road so it can separate messy handwritten digits, reaching 98 percent accuracy.
