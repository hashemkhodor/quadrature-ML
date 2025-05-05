from typing import List, Literal
from pydantic import BaseModel, Field
import os
import json


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
    step_sizes: List[float] = Field(
        default_factory=lambda: [0.02, 0.022, 0.025, 0.029, 0.033, 0.039, 0.045, 0.052, 0.060, 0.070])
    tol: float = 1e-4
    rkdp: RKDPConfig = Field(default_factory=RKDPConfig, alias="RKDP")
    scaler_path: str = "adaptive/model_ode/Lorenz/scaler.pkl"
    reward_range: List[float] = Field(default_factory=lambda: [0.1, 2])
    reward_fn: str = "asym_exp"
    batch_size: int = 64
    num_episodes: int = 100
    gamma: float = 0.0
    save_path: str = "adaptive/model_ode/Lorenz/qmodel.pkl"


def load_config(config_path: str) -> dict[str, list[QODEConfig]]:
    assert os.path.exists(config_path), f"Config file {config_path} does not exist"
    cfgs: dict[str, list[QODEConfig]] = {}
    with open(config_path, 'r') as f:
        cfgs_raw: dict = json.loads(f.read())
        for key, cfg_arr in cfgs_raw.items():
            cfgs[key] = [QODEConfig.model_validate({**cfg, "integrator": key}) for cfg in cfg_arr]

    return cfgs


def load_test_config(config_path: str) -> dict:
    pass


if __name__ == "__main__":
    cfg = load_config("config/train/lorenz_config_full.json")
    print(cfg)
