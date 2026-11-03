# Experiment 4 — K-Means Clustering on Iris in Plain English

> **In one sentence:** We let the computer discover groups in unlabeled flower data, then use a graph to decide that three groups is the natural choice.

## The problem in everyday words

This time nobody tells the computer which flower is which species. We hand over the 150 flowers and their four measurements and ask a simpler question: can you find natural groups of similar flowers?

This is clustering. It is **unsupervised**, because there are no labels to learn from. The computer must decide on its own which flowers belong together. The true species count is known to us, but the algorithm does not use it; we only use that knowledge afterward to check whether the groups make sense.

## What K-Means is, without math

Imagine a class of students and a game where you pick a few team captains. Every student joins the nearest captain. Then each captain walks to the average position of their team. Students rejoin the nearest captain, the captains move again, and this repeats until nobody changes teams.

That is K-Means. The captains are called **centroids**. The number of captains is **k**, the number of clusters you want. The algorithm alternates two steps: assign each point to the nearest centroid, then move each centroid to the middle of its assigned points.

Two measuring sticks help us judge the result:

- **Inertia** is the total squared distance between every point and its own centroid. It measures how tightly packed the clusters are. Smaller is tighter.
- **Silhouette score** measures how well separated the clusters are from each other. It ranges from -1 to 1; closer to 1 means clean, well-separated groups.

Inertia always decreases when you add more clusters, because more captains means smaller teams. So we cannot simply pick the k with the lowest inertia. Instead we look for the **elbow**: the point where adding another cluster stops giving a big improvement. On a graph of inertia versus k, the curve often bends sharply and then flattens, like an elbow.

## What the code does, step by step

1. Load the Iris measurements. The labels are ignored during clustering.
2. `StandardScaler` puts all four measurements on the same scale, because K-Means uses distance.
3. Loop over k from 1 to 10. For each k, fit K-Means and record its inertia.
4. Plot inertia against k and look for the elbow.
5. Choose k equal to 3, because the curve bends there and Iris is known to have three species.
6. Fit the final K-Means model and get a cluster number for each flower.
7. Count how many flowers landed in each cluster.
8. Compute the silhouette score to judge cluster quality.

## What the result means

The inertia values fall from **600.0** at k equal to 1 down to **47.39** at k equal to 10. The big drops happen early; after three clusters the improvement becomes much smaller. That is the elbow.

The final clusters contain **53, 50, and 47** flowers. The true species groups have 50 flowers each, so the sizes are very close.

The silhouette score is **0.4599**. In plain words, that is a moderate result: the clusters are real and reasonably separated, but they are not perfectly clean. Some flowers sit near the boundary between groups.

The cluster numbers are not species names. Cluster 0 does not automatically mean setosa. K-Means only groups by similarity; the cluster numbers are arbitrary.

## Why these choices?

- K-Means is fast and works well when clusters are roughly round and similar in size, which suits Iris.
- Scaling prevents the measurement with the largest range from dominating the distance.
- The elbow method gives an objective way to choose k instead of guessing.
- Running the algorithm ten times (`n_init=10`) and keeping the best result protects against unlucky random starting positions.
- The silhouette score adds a second opinion about whether the chosen k is sensible.

## If the examiner asks (plain answers)

**What is clustering?** Grouping similar points without using labels.

**How is it different from classification?** Classification learns from known labels; clustering discovers groups without labels.

**What is a centroid?** The center of a cluster, usually the average of its points.

**What is inertia?** The total squared distance from points to their own cluster center; lower means tighter clusters.

**Why not always choose the smallest inertia?** Inertia always falls as k grows, so it cannot decide k by itself.

**What is the elbow method?** Plot inertia against k and pick the bend where extra clusters stop helping much.

**What does a silhouette score of 0.46 mean?** The clusters are moderately separated, not perfect.

**Why scale the data first?** K-Means uses distances, so unscaled features would dominate unfairly.

**Are the clusters the same as the species?** Not necessarily; clustering is unsupervised, so clusters may cut across true species.

**One-line takeaway:** K-Means is a repeated game of join-the-nearest-captain-and-move-the-captain; the elbow picks the team count and the silhouette score checks how clean the teams are.
