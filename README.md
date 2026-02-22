# RL Differential Drive Navigation with LiDAR and EKF

## 1. Project Overview

This project investigates reinforcement learning (RL) methods for autonomous navigation of a differential-drive ground robot operating in a stochastic and partially observable environment. 

The robot is equipped with:
- Wheel encoders for odometry
- A 2D LiDAR sensor for obstacle detection

To simulate realistic robotics conditions:
- Wheel slip is introduced as stochastic process noise
- LiDAR measurements are corrupted with Gaussian noise
- An Extended Kalman Filter (EKF) is used for state estimation

The objective is to learn a navigation policy that safely reaches a randomly generated goal while avoiding randomly generated circular obstacles.

---

## 2. Markov Decision Process (MDP) Formulation

The navigation task is modeled as a Markov Decision Process (MDP):

\[
\mathcal{M} = (S, A, P, R, \gamma)
\]

### 2.1 State Space \( S \)

The true robot state is:

\[
x_t = (x, y, \theta)
\]

However, due to sensor noise and wheel slip, the agent does not have direct access to the true state.

Instead, the agent observes an estimated belief state from the EKF:

- Estimated pose: \( (\hat{x}, \hat{y}, \hat{\theta}) \)
- LiDAR beam measurements
- Relative goal position in robot frame

Thus, the RL state is defined as:

\[
s_t = [\hat{x}, \hat{y}, \hat{\theta}, \text{LiDAR}, \text{Goal}_{relative}]
\]

---

### 2.2 Action Space \( A \)

For the simplified tabular subtask (Grid World):

- Up
- Down
- Left
- Right

For the full robot environment:

- Forward
- Turn Left
- Turn Right
- Stop

Later versions may use continuous actions \((v, \omega)\).

---

### 2.3 Transition Function \( P(s'|s,a) \)

State transitions are governed by differential-drive kinematics:

\[
d = \frac{\Delta s_r + \Delta s_l}{2}
\]

\[
\Delta \theta = \frac{\Delta s_r - \Delta s_l}{b}
\]

Wheel slip is modeled as stochastic process noise:

\[
\Delta s_l' = \Delta s_l + \epsilon_l
\]

\[
\Delta s_r' = \Delta s_r + \epsilon_r
\]

where:

\[
\epsilon \sim \mathcal{N}(0, \sigma^2)
\]

Thus, transitions are stochastic.

---

### 2.4 Reward Function \( R(s,a) \)

The reward function is defined as:

- +100 for reaching the goal
- -100 for collision
- -1 per time step (encourages shorter paths)
- Optional shaping reward based on distance reduction

This shaping encourages efficient and safe navigation.

---

### 2.5 Terminal States

Episodes terminate when:

- Goal is reached
- Collision occurs
- Maximum step limit is reached

---

## 3. Extended Kalman Filter (EKF)

The EKF estimates robot pose using:

### Prediction:
- Differential-drive motion model
- Wheel encoder inputs

### Correction:
- LiDAR-based geometric observations

The RL agent receives the EKF estimated pose rather than ground-truth state, making the environment partially observable.

---

## 4. Simplified Grid World Subtask (For Dynamic Programming)

Since the full navigation problem has a continuous state space, a simplified discrete Grid World environment is implemented to demonstrate tabular Dynamic Programming methods.

Grid World characteristics:

- Finite state space
- Deterministic transitions
- Obstacles
- Single goal state

This subtask allows implementation of:

- Policy Iteration
- Value Iteration
- Q-value improvements

---

## 5. Implemented Algorithms (Version 1)

### Dynamic Programming
- Policy Iteration
- Value Iteration

### Tabular Reinforcement Learning
- Monte Carlo Prediction
- TD(0)
- Q-learning
- Epsilon-Greedy exploration

Future versions will include:
- TD(n)
- TD(λ)
- Sarsa(n)
- Sarsa(λ)
- Function approximation
- Deep Q Networks (DQN)

---

## 6. Agent Framework

All agents inherit from a base class:

```python
class BaseAgent:
    def train(self, env):
        pass

    def evaluate(self, env):
        pass

    def select_action(self, state):
        pass
