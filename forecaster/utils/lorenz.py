import numpy as np
import pandas as pd
import os
from scipy.integrate import solve_ivp

def lorenz(t, state, sigma=10, rho=28, beta=8/3):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

def generate_lorenz_data(t_span=(0, 40), t_eval_step=0.01, init_state=[1, 1, 1]):
    t_eval = np.arange(t_span[0], t_span[1], t_eval_step)
    sol = solve_ivp(lorenz, t_span, init_state, t_eval=t_eval)
    return sol.t, sol.y.T, t_eval_step

def save_data_to_csv(data, filename="forecaster/data/lorenz_data.csv"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    pd.DataFrame(data, columns=["x", "y", "z"]).to_csv(filename, index=False)
