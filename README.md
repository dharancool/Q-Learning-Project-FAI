# Q-Learning Reproduction & Benchmarking
### CS5100 Foundations of AI — Capstone Project, Phase 1

> A faithful reproduction of Watkins & Dayan (1992) *"Q-Learning"*, validated across four OpenAI Gymnasium environments with Bellman convergence verification.

📹 **[Video Demonstration](#)** · 📄 **[Full Report](#)** · 🧠 **[Phase 2: SARSA & Expected SARSA →](#)**

---

## Overview

This project reproduces the Q-Learning algorithm from the landmark 1992 paper by Watkins and Dayan and evaluates its convergence behavior across four reinforcement learning environments of increasing complexity. Each environment is trained with multi-seed averaging (10 seeds) to produce statistically robust learning curves, and convergence is verified through Bellman residual analysis.

**Environments tested:**

| Environment | States | Actions | Episodes | Final Avg Reward |
|---|---|---|---|---|
| GridWorld 4×4 (custom) | 16 | 4 | 2,000 | ~0.75 |
| FrozenLake-v1 | 16 | 4 | 5,000 | ~0.74 |
| CliffWalking-v1 | 48 | 4 | 3,000 | ~−13 |
| Taxi-v3 | 500 | 6 | 10,000 | ~7.5 |

---

## Background

Q-Learning is a **model-free, off-policy** temporal difference algorithm. The agent learns an action-value function Q(s, a) — the expected cumulative reward of taking action *a* in state *s* and following the optimal policy thereafter — purely from interaction with the environment.

**Core update rule (Bellman equation):**

```
Q(s, a) ← Q(s, a) + α [ r + γ · max_a' Q(s', a') − Q(s, a) ]
```

| Symbol | Meaning |
|---|---|
| `α` | Learning rate — controls update step size |
| `γ` | Discount factor — weight of future rewards |
| `r` | Immediate reward from transition (s, a) → s' |
| `max Q(s', a')` | Best estimated future value (off-policy target) |
| TD error | `r + γ · max Q(s', a') − Q(s, a)` |

**Off-policy** means the target uses the greedy max, regardless of the agent's actual behavior policy (ε-greedy). This is the defining characteristic that separates Q-Learning from SARSA.

---

## Repository Structure

```
q-learning-reproduction/
│
├── Q_Learning_Implementation.ipynb   # Main notebook — all code and results
├── README.md                         # This file
│
├── results/
│   ├── gridworld_learning_curve.png
│   ├── gridworld_heatmap.png
│   ├── gridworld_policy_arrows.png
│   ├── frozenlake_learning_curve.png
│   ├── frozenlake_heatmap.png
│   ├── cliffwalking_learning_curve.png
│   └── taxi_learning_curve.png
│
└── report/
    └── CS5100_Capstone_Report.pdf
```

---

## Implementation Details

### Agent

```python
class QLearningAgent:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.Q = np.zeros((n_states, n_actions))   # Initialize Q-table to zero
```

The Q-table is initialized to **all zeros**, consistent with the original paper. This gives the agent no prior knowledge; all estimates are built entirely from experience.

**Action selection — ε-greedy policy:**
```python
def select_action(self, state):
    if np.random.rand() < self.epsilon:
        return np.random.randint(self.n_actions)  # Explore
    return np.argmax(self.Q[state])               # Exploit
```

With probability ε the agent explores randomly; otherwise it exploits its current best estimate. Exploration is essential to ensure all state-action pairs are visited during training.

**Bellman update:**
```python
def update(self, s, a, r, s_next, done):
    best_next = 0 if done else np.max(self.Q[s_next])
    td_target = r + self.gamma * best_next
    td_error  = td_target - self.Q[s, a]
    self.Q[s, a] += self.alpha * td_error
```

### Hyperparameters

| Parameter | GridWorld | FrozenLake | CliffWalking | Taxi-v3 |
|---|---|---|---|---|
| α (learning rate) | 0.1 | 0.1 | 0.1 | 0.1 |
| γ (discount) | 0.99 | 0.99 | 0.99 | 0.99 |
| ε (exploration) | 0.1 | 0.2 | 0.1 | 0.2 |
| Episodes | 2,000 | 5,000 | 3,000 | 10,000 |
| Max steps/episode | 200 | 200 | 500 | 300 |

Higher ε is used for stochastic (FrozenLake) and large state-space (Taxi) environments to ensure sufficient exploration.

### Multi-Seed Evaluation

All experiments are averaged over **10 independent seeds**. This produces learning curves with mean ± standard deviation bands, making the convergence behavior statistically meaningful rather than anecdotally lucky.

```python
def run_multi_seed(env_name, env_params, train_params, seeds=range(10)):
    all_rewards = []
    for seed in seeds:
        set_seed(seed)
        agent = QLearningAgent(**env_params)
        rewards, _ = train(env_name, agent, seed=seed, **train_params)
        all_rewards.append(rewards)
    arr = np.array(all_rewards)
    return arr.mean(axis=0), arr.std(axis=0)
```

### Bellman Convergence Verification

Beyond reward curves, convergence is validated by computing the mean TD error over 500 greedy-policy transitions on the learned Q-table. A well-converged Q-table should have near-zero Bellman residual.

```python
error = abs(Q[state, action] - (r + gamma * np.max(Q[next_s])))
```

| Environment | Bellman Tolerance | Result |
|---|---|---|
| FrozenLake | 0.1 | ✅ PASS |
| CliffWalking | 1.0 | ✅ PASS |
| Taxi-v3 | 2.0 | ✅ PASS |

*(GridWorld uses a custom env; Bellman check is embedded in the training loop.)*

---

## Environments

### GridWorld 4×4 (Custom)
A deterministic 4×4 grid. The agent starts at (0,0) and must reach (3,3). Three hole states terminate the episode with reward −1. Each step incurs a small penalty of −0.01 to encourage efficiency. This custom environment allows full control over transitions and rewards.

### FrozenLake-v1
A stochastic grid environment — actions are slippery, so the agent moves in the intended direction only a fraction of the time. This introduces irreducible uncertainty and tests the algorithm's robustness to stochasticity.

### CliffWalking-v1
A 4×12 grid from Sutton & Barto (Example 6.6). The agent must walk from start to goal without falling off a cliff running along the bottom edge (reward −100). This environment highlights the behavioral difference between Q-Learning (optimal but risky path) and SARSA (safer detour) — a key comparison in Phase 2.

### Taxi-v3
A 5×5 grid with 500 discrete states encoding taxi position, passenger location, and destination. The agent must navigate to a passenger, pick them up, and deliver them to the correct destination. Wrong pickups/dropoffs incur penalties. At 500 states and 6 actions, this is the most complex tabular environment tested.

---

## Results Summary

**GridWorld** — Converges cleanly within 1,500 episodes. Policy arrows form a coherent path navigating around all three holes. Q-value heatmap confirms higher state values near the goal.

**FrozenLake** — Noisy convergence due to stochasticity, but learning curve climbs reliably over 5,000 episodes. Bellman check passes, confirming internal Q-table consistency despite environment noise.

**CliffWalking** — Final reward of ~−13 matches the theoretical optimal for Q-Learning on this environment (Sutton & Barto, 2018). The agent learns to hug the cliff edge — optimal in expectation but risky in practice.

**Taxi-v3** — Reward starts near −200 (random flailing) and climbs to ~+7.5, matching known benchmarks. The learning curve shows a sharp inflection around episode 2,000 as the agent begins successfully completing pickups and dropoffs.

---

## Installation & Usage

**Requirements:**
```bash
pip install gymnasium[toy_text] matplotlib seaborn numpy pandas
```

**Run the notebook:**
```bash
jupyter notebook Q_Learning_Implementation.ipynb
```

Or open directly in **Google Colab** — the first cell installs all dependencies automatically.

**Python version:** 3.8+

---

## What's Next — Phase 2

Phase 2 extends this work by implementing **SARSA** and **Expected SARSA** and comparing all three algorithms head-to-head across the same four environments.

The central hypothesis: on-policy algorithms (SARSA, Expected SARSA) will learn safer but sub-optimal policies in risky environments like CliffWalking, while off-policy Q-Learning will converge to the theoretically optimal but riskier path. Expected SARSA is predicted to be the most stable of the three due to its variance-reduced update.

---

## References

- Watkins, C. J. C. H., & Dayan, P. (1992). *Q-learning*. Machine Learning, 8(3–4), 279–292.
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
- Gymnasium Documentation: https://gymnasium.farama.org

---

## Author

**Gangatharan Suresh**
M.S. Computer Science — AI/ML Specialization
Northeastern University | Spring 2026

> *CS5100 Foundations of AI — Capstone Project*
