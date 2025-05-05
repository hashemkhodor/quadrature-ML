# Adaptive Time Step Integrators

A machine learning-based framework for training and testing adaptive time step integrators for solving ordinary differential equations (ODEs), using methods like Runge–Kutta–Dormand–Prince (RKDP), Cash–Karp, Fehlberg78, and Gauss–Legendre.

This project uses reinforcement learning (RL) to learn optimal step size selection dynamically during integration, balancing accuracy and computational efficiency.

---

## 📄 Paper

This implementation builds upon ideas and base code presented in the paper:

> **"Efficient time stepping for numerical integration using reinforcement learning"**  
> [arXiv:2104.03562](https://arxiv.org/abs/2104.03562)
---

## 🧠 Project Highlights

* RL-based adaptive integration for ODE solvers.
* Plug-and-play support for multiple classic numerical integration methods.
* Modular and configurable design using JSON-based configuration files.
* Supports multiple reward strategies to tune learning behavior.

---

## 📚 Table of Contents

* [Installation](#installation)
* [Usage](#usage)

  * [Training](#training)
  * [Testing](#testing)
* [Configuration](#configuration)
* [Features](#features)
* [Reward Functions](#reward-functions)
* [Dependencies](#dependencies)
* [Examples](#examples)
* [Notebook: LSTM Lorenz Forecaster](#notebook-lstm-lorenz-forecaster)
* [Troubleshooting](#troubleshooting)
* [Contributors](#contributors)
* [License](#license)

---

## 📦 Installation

```bash
git clone https://github.com/hashemkhodor/quadrature-ML.git
cd quadrature-ML
pip install -r requirements.txt
```

Python 3.8+ is required.

---

## 🚀 Usage

### ✅ Training

```bash
py -m train_ode_solver --config ".\config\train\lorenz_config.json"
```

### 🧪 Testing

```bash
python -m test_ode_solver --cfg_path config/train/lorenz_config.json --save_path results/custom_eval.json
```

---

## ⚙️ Configuration

All configuration files must follow the structure in `config/config_template.json`. Each block corresponds to a specific ODE integrator (e.g., `"rkdp"` or `"cash_karp"`).

### Key Parameters

| Parameter      | Description                                                              |
| -------------- | ------------------------------------------------------------------------ |
| `x0`           | Initial conditions for the Lorenz system (list of floats)                |
| `t0`           | Start time of integration (float)                                        |
| `t1`           | End time of integration (float)                                          |
| `d`            | Dimensionality of the system (usually 3 for Lorenz)                      |
| `step_sizes`   | Finite set of candidate time steps to pick from during RL                |
| `tol`          | Tolerance level for error at each step (how much deviation is tolerated) |
| `eps_start`    | Initial exploration rate for RL                                          |
| `reward_fn`    | Name of reward function used during training (see below)                 |
| `batch_size`   | Number of samples per training batch                                     |
| `num_episodes` | Number of training episodes                                              |
| `gamma`        | Discount factor in RL                                                    |
| `scaler_path`  | Path to input scaler (used for normalization)                            |
| `save_path`    | Where to save the trained model weights                                  |

---

## ✨ Features

* **Adaptive RL Control**: Uses reinforcement learning to choose optimal time steps.
* **Integrator Agnostic**: Switch between RKDP and Cash–Karp.
* **Extensible Rewards**: Easily plug in different reward logic.
* **Structured Configs**: Modular and reproducible experimentation setup.

---

## 🏆 Reward Functions

Reward functions influence how the RL agent values step size decisions.

### Supported Reward Functions

| Function Name | Description                                                            |
| ------------- | ---------------------------------------------------------------------- |
| `log`         | Logarithmic reward scaling (penalizes large errors logarithmically)    |
| `exp`         | Exponential reward based on error                                      |
| `linear`      | Linearly maps the step reward based on accuracy and step size          |
| `sigmoid`     | S-shaped response to step decisions (smoothed transitions)             |
| `inverse`     | Inverse relationship between error and reward                          |
| `quadratic`   | Penalizes large deviations more harshly (squared error emphasis)       |
| `asym_exp`    | Asymmetric exponential for better sensitivity to under- vs. over-steps |

Set your desired reward using the `"reward_fn"` key in your configuration file.

---

## 📦 Dependencies

Install required packages via:

```bash
pip install -r requirements.txt
```
---

## 📈 Examples

Train using RKDP integrator on the Lorenz system:

```bash
py -m train_ode_solver --config ".\config\train\lorenz_config.json"
```

Test your model:

```bash
python -m test_ode_solver --cfg_path config/train/lorenz_config.json --save_path results/custom_eval.json
```


---

## 📓 Notebook: LSTM Lorenz Forecaster

The notebook [`LSTM_lorenz.ipynb`](./notebooks/lstm_lorenz.ipynb) demonstrates how to use the precomputed adaptive time steps (from trained RL agents) to train and evaluate a deep learning model (model A) on the Lorenz system.

This model is based on an LSTM architecture and is designed to predict the next state \([x_{i+1}, y_{i+1}, z_{i+1}]\) from the current state and time step \([x_i, y_i, z_i, \Delta t_i]\).

### 🔍 What it Shows

- How to use adaptive RL-generated time steps with neural forecasting models
- How delta t is embedded into the model input for temporal awareness
- Comparison of adaptive vs. fixed-step prediction quality
- Training pipeline, evaluation plots, and visualized Lorenz trajectories

### 🛠️ Modify Easily

You can change:
- The integrator or reward shaping strategy by modifying which `dt_values` array you load
- The initial conditions and sequence length
- The model architecture (swap out LSTM for GRU, Transformer, etc.)

This notebook serves as a complete, standalone tutorial for forecasting chaotic systems using machine learning with RL-driven temporal discretization.
