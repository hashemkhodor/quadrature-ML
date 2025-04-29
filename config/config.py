from typing import List
from pydantic import BaseModel, Field

class RKDPConfig(BaseModel):
    nodes_per_step: int
    memory: int

class QODEConfig(BaseModel):
    x0: List[float]
    t0: float
    t1: float
    d: int
    eps_start: float
    step_sizes: List[float]
    tol: float
    rkdp: RKDPConfig = Field(..., alias="RKDP")
    scaler_path: str
    reward_range: List[float]
    reward_fn: str
    batch_size: int
    num_episodes: int
    gamma: float
    save_path: str