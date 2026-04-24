# Q-Learning Reproduction & Benchmarking
### CS5100 Foundations of AI — Capstone Project, Phase 1

> A reproduction of Watkins & Dayan (1992) *"Q-Learning"*, validated across four OpenAI Gymnasium environments with Bellman convergence verification.



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


---

## References

- Watkins, C. J. C. H., & Dayan, P. (1992). *Q-learning*. Machine Learning, 8(3–4), 279–292.
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
- Gymnasium Documentation: https://gymnasium.farama.org

---

## Author

**Gangatharan Idayachandiran**
M.S. Computer Science
Northeastern University | Spring 2026

> *CS5100 Foundations of AI — Capstone Project*
