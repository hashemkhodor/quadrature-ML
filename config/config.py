from typing import List, Literal
from pydantic import BaseModel, Field


class RKDPConfig(BaseModel):
    nodes_per_step: int = 6
    memory: int = 0


class QODEConfig(BaseModel):
    integrator: Literal['rkdp', 'cash_karp', 'fehlberg78', 'gauss_legendre_2'] = 'rkdp'
    x0: List[float] = Field(default_factory=lambda: [10, 10, 10])
    t0: float = 0.0
    t1: float = 10.0
    d: int = 3
    eps_start: float = 0.5
    step_sizes: List[float] = Field(default_factory=lambda: [0.02, 0.022, 0.025, 0.029, 0.033, 0.039, 0.045, 0.052, 0.060, 0.070])
    tol: float = 1e-4
    rkdp: RKDPConfig = Field(default_factory=RKDPConfig, alias="RKDP")
    scaler_path: str = "adaptive/model_ode/Lorenz/scaler.pkl"
    reward_range: List[float] = Field(default_factory=lambda: [0.1, 2])
    reward_fn: str = "asym_exp"
    batch_size: int = 64
    num_episodes: int = 100
    gamma: float = 0.0
    save_path: str = "adaptive/model_ode/Lorenz/qmodel.pkl"
