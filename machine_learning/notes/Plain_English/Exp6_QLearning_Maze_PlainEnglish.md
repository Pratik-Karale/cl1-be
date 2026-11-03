# Experiment 6 — Q-Learning in a Maze in Plain English

> **In one sentence:** An agent learns the best moves in a small maze by trying things repeatedly and remembering which moves led to the goal.

## The problem in everyday words

A robot is placed at the top-left corner of a 5 by 5 grid. The goal is the bottom-right corner. Some squares are walls, so the robot cannot walk through them. The robot is not told the route. It must learn the route by trial and error.

This is **reinforcement learning**. There are no labelled examples saying "from this square, move right." Instead, the agent acts, receives a reward, and gradually learns which actions are good in which situations.

## What Q-Learning is, without math

Imagine playing a video game and keeping a personal cheat sheet. For every square you stand on and every move you could make, you write down a score: "how promising is this move from here?" At first every score is zero because you know nothing.

You play many games. Most moves give a small penalty, so the agent prefers short routes. Reaching the goal gives a big reward, so the moves that lead there get their scores raised.

The cheat sheet is the **Q-table**. It has one row per square and one column per action. The score in a cell is the **Q-value**: the expected future reward for taking that action in that square.

The agent balances two behaviours:

- **Exploration** means trying random moves to discover new routes.
- **Exploitation** means using the cheat sheet and taking the move with the best known score.

The **epsilon-greedy** rule does both: with a small probability epsilon, pick a random move; otherwise pick the best-known move. Epsilon starts at 0.25 and slowly shrinks, so the agent explores early and trusts its cheat sheet more later.

Three settings shape learning:

- **Alpha** is the learning rate: how strongly a new experience overwrites the old score.
- **Gamma** is the discount factor: how much future rewards matter compared with immediate ones.
- **Epsilon** is the exploration probability.

## What the code does, step by step

1. Represent the maze as five strings. A dot is open space, a hash is a wall, S is the start, and G is the goal.
2. List every non-wall square as a state, and give each one a row number in the Q-table.
3. Define four actions: up, down, left, and right.
4. Create a Q-table full of zeros, with one row per state and one column per action.
5. Write a `step` function that moves the agent, keeps it in place if it hits a wall or the edge, gives a reward of +100 at the goal and -1 otherwise, and says whether the episode is finished.
6. Run 5,000 episodes. In each episode, choose an action by epsilon-greedy, move, and update the Q-value for the chosen action.
7. Slowly reduce epsilon after each episode so exploration fades.
8. After training, start at S and repeatedly take the action with the highest Q-value until the goal is reached.

## What the result means

The Q-table has shape **20 by 4**, because there are 20 open squares and 4 possible moves.

The learned path is **(0,0) then (0,1) then (0,2) then (0,3) then (1,3) then (1,4) then (2,4) then (3,4) then (4,4)**. It reaches the goal in **8 steps**, which is the shortest possible route on this grid.

In plain words: after enough practice, the agent's cheat sheet points consistently toward the goal, and following it produces a clean shortest path. The agent was never told the route; it discovered it from rewards.

## Why these choices?

- The maze is small enough for a Q-table, so no neural network is needed.
- A reward of +100 at the goal and -1 per step encourages the agent to finish quickly.
- Epsilon decay balances early exploration with later exploitation.
- Alpha equal to 0.2 lets the agent learn steadily without forgetting old lessons too quickly.
- Gamma equal to 0.95 makes the agent care about future rewards, which is necessary for multi-step planning.

## If the examiner asks (plain answers)

**What is reinforcement learning?** Learning by interacting with an environment and using rewards to improve behavior.

**What is the Q-table?** A table of scores for every state and action, estimating how good each move is.

**What does alpha do?** It sets how much new experience changes an old estimate.

**What does gamma do?** It sets how much the agent cares about future rewards versus immediate ones.

**What does epsilon do?** It sets how often the agent tries a random move instead of the best-known move.

**Why reduce epsilon over time?** To explore early and exploit learned knowledge later.

**Why is Q-Learning called off-policy?** It updates using the best possible next move, even if the agent did not actually take that move.

**Why is there a penalty for every step?** It pushes the agent toward the shortest route instead of wandering.

**What happens if the agent hits a wall?** It stays in the same square and pays the step penalty, so it learns to avoid walls.

**Is the learned path guaranteed optimal?** Not always in general, but in this simple deterministic maze it finds the 8-step shortest path.

**One-line takeaway:** Q-Learning is a cheat sheet built from trial and error: try moves, reward the good ones, penalize the slow ones, and gradually follow the best-known move to the goal.
