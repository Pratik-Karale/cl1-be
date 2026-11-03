# Experiment 6 — Q-Learning for Path Finding in a Maze (Viva Theory)

> **30-second answer:** An agent learns to navigate a **5×5 grid maze** from start **S (0,0)** to goal **G (4,4)** using **Q-Learning**. It trains for **5,000 episodes** with α = 0.2, γ = 0.95, ε = 0.25, then follows the learned Q-table to reach the goal in **8 steps**.

## 1. MOST LIKELY — Core of the Experiment (High Yield)

**Aim:** Implement Q-Learning so an agent learns action values from rewards in a grid maze, then follows the learned Q-values to find a path from start S to goal G.

**Environment:** Maze rows `["S....", ".##..", "...#.", "##...", "....G"]` — **5×5**, `#` = wall, **20 non-wall states**, **4 actions** (Up, Down, Left, Right). Start **(0,0)**, goal **(4,4)**.

**Algorithm in 5 bullets**
- Build a **Q-table** of shape `(states, actions)` — initialised to zeros; here **(20, 4)**.
- **ε-greedy action:** with probability ε explore a random action, else take `argmax Q[s]`.
- Execute action; invalid moves (wall/out of bounds) leave the state unchanged.
- Reward: **+100** on reaching the goal, otherwise **−1** per step.
- **Update:** `Q(s,a) = Q(s,a) + α × [reward + γ × best future Q(s′,a′) − Q(s,a)]` — nudge the current estimate toward "this reward plus the best value available from the next state."
- Decay ε: `epsilon = max(0.02, epsilon * 0.999)`; run 5,000 episodes, max 100 steps each.

**Code skeleton**
```python
Q = np.zeros((len(states), len(actions)))
alpha, gamma, epsilon = 0.2, 0.95, 0.25

for episode in range(5000):
    s = start
    for _ in range(100):
        if random.random() < epsilon:
            a = random.randrange(4)                 # explore
        else:
            a = int(np.argmax(Q[state_id[s]]))      # exploit
        ns, reward, done = step(s, a)
        target = reward + gamma * (0 if done else np.max(Q[state_id[ns]]))
        Q[state_id[s], a] += alpha * (target - Q[state_id[s], a])
        s = ns
        if done: break
    epsilon = max(0.02, epsilon * 0.999)
```

**Expected output**
- Training episodes: **5000**; Q-table shape: **(20, 4)**.
- Learned path: **(0,0) → (0,1) → (0,2) → (0,3) → (1,3) → (1,4) → (2,4) → (3,4) → (4,4)**.
- Reached goal: **True**; steps: **8** (the shortest grid path).

**Viva Q&A**
1. **What is Q-Learning?** A **model-free, off-policy** reinforcement-learning algorithm that learns the value of state–action pairs from experience.
2. **What is the Q-table?** A lookup table storing expected cumulative reward `Q(s,a)` for each state–action pair.
3. **What does α (alpha) control?** The learning rate — how much new information overrides old Q-values (0 = no learning, 1 = ignore history).
4. **What does γ (gamma) control?** The discount factor — how much future rewards matter (0 = only immediate reward, near 1 = long-term planning).
5. **What is ε (epsilon)?** Exploration probability in ε-greedy; high ε = more random exploration.
6. **Why decay ε?** Start by exploring, then exploit learned values; decaying to 0.02 balances both.
7. **What is the Bellman update here?** `target = r + γ · max Q(s', a')`; the temporal-difference error is `target − Q(s,a)`.
8. **Why is Q-Learning off-policy?** It updates using the **maximum** next-state Q-value, not necessarily the action the behaviour policy took.
9. **Why reward −1 per step?** Encourages the shortest path; +100 at the goal provides the terminal incentive.
10. **What happens on an invalid move?** The agent stays in the same state and gets −1, teaching it to avoid walls.
11. **Is the learned path optimal?** In this maze yes — 8 steps is the Manhattan distance from (0,0) to (4,4).
12. **What if ε stayed 1.0?** The agent would keep moving randomly and never exploit its Q-table.

## 2. LIKELY — Syllabus Theory Around This Experiment

### Unit VI: Reinforcement Learning
- **RL:** an agent learns by interacting with an environment, receiving rewards, to maximise cumulative return. No labelled examples.
- **RL vs supervised/unsupervised:** supervised has correct answers; unsupervised finds structure; RL learns from delayed reward via trial and error.
- **Elements:** agent, environment, state `s`, action `a`, reward `r`, policy `π` (the strategy), value function `V` or `Q` (how good a state/action is).
- **Markov property:** the future depends only on the current state, not the full history.
- **MDP (Markov Decision Process):** defined by states `S`, actions `A`, transition probabilities `P`, rewards `R`, and discount `γ`.
- **Return:** the discounted sum of future rewards — `reward₁ + γ × reward₂ + γ² × reward₃ + ...` (near rewards count more).
- **Bellman equation (in words):** the value of a state = the expected reward now + the discounted value of the next state. You do not need to memorise the summation version.
- **Q-function:** `Q(s,a)` = expected return after taking action `a` in state `s`, then following the policy.
- **Exploration vs exploitation:** explore unknown actions vs exploit known good ones (ε-greedy, softmax).
- **Q-Learning update:** `Q(s,a) = Q(s,a) + α × [reward + γ × best future Q(s′,a′) − Q(s,a)]`.

| Aspect | Q-Learning | SARSA |
|---|---|---|
| Policy | Off-policy (max next Q) | On-policy (actual next action) |
| Behaviour | Learns optimal policy regardless of exploration | Learns policy being followed |
| Risk | More aggressive | Safer near cliffs |

**Likely Q&A**
- *What is the difference between a Markov chain, MRP and MDP?* Chain = states/transitions; MRP adds rewards; MDP adds actions.
- *What is a policy?* A mapping from states to actions (deterministic or stochastic).
- *What is the discount factor's role?* Controls how far-sighted the agent is; γ = 1 values all future rewards equally.
- *What are real-time RL applications?* Game playing, robotics, recommendation, traffic control, resource scheduling.
- *What is the exploration–exploitation dilemma?* Balancing trying new actions against using known-good ones.

## 3. LEAST LIKELY — Stretch / Edge Cases

- **State-space explosion:** Q-tables grow with states × actions; large problems need function approximation (DQN).
- **Reward shaping:** poorly designed rewards cause unintended behaviour (e.g., looping to farm reward).
- **Learning stability:** high α or γ can diverge; decaying α (learning-rate schedule) helps.
- **Sparse rewards:** if only the goal gives reward, learning is slow; intermediate shaping or curiosity helps.
- **Deterministic vs stochastic environments:** this maze is deterministic; real environments need transition probabilities.
- **Cross-unit link:** clustering (Unit IV) can discretise continuous states; ensembles (Unit V) can approximate Q-functions.
- **Trap question:** "Does more training always improve the policy?" No — overtraining with a fixed ε can still be noisy; convergence depends on schedules and environment.

**If you remember only one thing:** Q-Learning repeatedly applies the Bellman update `Q ← Q + α[r + γ max Q' − Q]` with ε-greedy exploration, turning trial-and-error into a shortest path (8 steps here).
