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

The objective is to learn a navigation policy that safely reaches a goal while avoiding obstacles.

---

## 2. Markov Decision Process (MDP) Formulation


The navigation task is modeled as a Markov Decision Process (MDP):

M = (S, A, P, R, γ)

---

### 2.1 State Space (S)

The true robot state is:

x_t = (x, y, θ)

However, due to sensor noise and wheel slip, the agent does not have direct access to the true state.

Instead, the agent observes an estimated belief state from the EKF:

- Estimated pose: (x̂, ŷ, θ̂)
- LiDAR beam measurements
- Relative goal position in the robot frame

Thus, the RL state is defined as:

s_t = [x̂, ŷ, θ̂, LiDAR, Goal_relative]

---

### 2.2 Action Space (A)

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

Later versions may use continuous actions (v, ω).

---

### 2.3 Transition Function P(s' | s, a)

State transitions are governed by differential-drive kinematics:

d = (Δs_r + Δs_l) / 2  
Δθ = (Δs_r − Δs_l) / b  

Wheel slip is modeled as stochastic process noise:

Δs_l' = Δs_l + ε_l  
Δs_r' = Δs_r + ε_r  

where:

ε ~ N(0, σ²)

Thus, transitions are stochastic.

---

### 2.4 Reward Function R(s, a)

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

### 2.6 Stochasticity and Model Learning Motivation

The transition dynamics are stochastic due to wheel slip in the motion model, and observations are corrupted by LiDAR measurement noise. Although the environment is fully implemented, the agent does not have direct access to the true transition probabilities or reward structure. Instead, it must learn these properties through interaction.

This formulation enables future extensions toward model-based reinforcement learning, where the agent may explicitly estimate or approximate the transition function P(s' | s, a) and reward function R(s, a) from experience.

---

## 3. Extended Kalman Filter (EKF)

Because the robot operates under motion and sensing uncertainty, state estimation is performed using an Extended Kalman Filter (EKF).

The EKF maintains a Gaussian belief over the robot pose:

x̂_t = (x̂, ŷ, θ̂)

Prediction step:
- Uses nonlinear differential-drive kinematics  
- Propagates covariance using Jacobians of the motion model  
- Incorporates wheel slip as process noise  

Correction step:
- Incorporates LiDAR-derived geometric measurements  
- Updates pose estimate using measurement residuals and Kalman gain  

The reinforcement learning agent receives the EKF-estimated pose rather than ground-truth state, making the task partially observable and more representative of real robotic systems.

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

## Running (Version 1)
```bash
pip install -r requirements.txt
python -m src.main --algo value_iteration
python -m src.main --algo policy_iteration
