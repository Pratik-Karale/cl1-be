"""Experiment 6 - Q-Learning for path finding in a maze.

Aim:
    Train a Q-Learning agent on a 5x5 grid maze and use the learned Q-table
    to walk from the start S to the goal G.

Environment:
    S....
    .##..
    ...#.
    ##...
    ....G

Run:
    python exp6_qlearning_maze.py

Outputs:
    Console: training episodes, Q-table shape, maze, learned path, steps.
"""

import random

import numpy as np

MAZE = ["S....", ".##..", "...#.", "##...", "....G"]
ROWS, COLS = len(MAZE), len(MAZE[0])
START = (0, 0)
GOAL = (4, 4)
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
ALPHA, GAMMA, EPSILON = 0.2, 0.95, 0.25
EPISODES = 5000


def step(state: tuple, action: int) -> tuple:
    """Take one action; walls and out-of-bounds moves leave the state unchanged."""
    dr, dc = ACTIONS[action]
    next_state = (state[0] + dr, state[1] + dc)

    if not (0 <= next_state[0] < ROWS and 0 <= next_state[1] < COLS):
        next_state = state
    elif MAZE[next_state[0]][next_state[1]] == "#":
        next_state = state

    done = next_state == GOAL
    reward = 100 if done else -1
    return next_state, reward, done


def train(state_id: dict, q_table: np.ndarray) -> np.ndarray:
    epsilon = EPSILON
    for _ in range(EPISODES):
        state = START
        for _ in range(100):
            if random.random() < epsilon:
                action = random.randrange(4)
            else:
                action = int(np.argmax(q_table[state_id[state]]))

            next_state, reward, done = step(state, action)
            best_next = 0 if done else np.max(q_table[state_id[next_state]])
            target = reward + GAMMA * best_next
            q_table[state_id[state], action] += ALPHA * (
                target - q_table[state_id[state], action]
            )
            state = next_state
            if done:
                break
        epsilon = max(0.02, epsilon * 0.999)
    return q_table


def follow_path(state_id: dict, q_table: np.ndarray) -> list:
    state = START
    path = [state]
    for _ in range(40):
        if state == GOAL:
            break

        candidates = []
        for action in range(4):
            next_state, _, _ = step(state, action)
            if next_state != state:
                candidates.append((q_table[state_id[state], action], next_state))

        if not candidates:
            break
        state = max(candidates, key=lambda item: item[0])[1]
        path.append(state)

    return path


def main() -> None:
    random.seed(42)
    np.random.seed(42)

    states = [
        (r, c)
        for r in range(ROWS)
        for c in range(COLS)
        if MAZE[r][c] != "#"
    ]
    state_id = {state: i for i, state in enumerate(states)}
    q_table = np.zeros((len(states), len(ACTIONS)))

    q_table = train(state_id, q_table)
    print("Training episodes:", EPISODES)
    print("Q-table shape:", q_table.shape)

    path = follow_path(state_id, q_table)
    print("\nMaze:")
    for row in MAZE:
        print(row)
    print("Learned path:", " -> ".join(map(str, path)))
    print("Reached goal:", path[-1] == GOAL, "| Steps:", len(path) - 1)


if __name__ == "__main__":
    main()
