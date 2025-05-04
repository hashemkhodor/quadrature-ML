import json
import os.path
from argparse import ArgumentParser
from loguru import logger
from config.config import QODEConfig
from adaptive.integrator import RKDP, CashKarp45, Fehlberg78, GaussLegendre2Stage
from adaptive.reward_functions import (
    RewardLog10, RewardExp,
    RewardLinear, RewardSigmoid, RewardInverse,
    RewardQuadratic, RewardAsymmetricExp
)

from sklearn.preprocessing import StandardScaler
from joblib import load
from adaptive.predictor import PredictorQODE
from adaptive.build_models import build_value_modelODE
from adaptive.environments import ODEEnv
from functions import LorenzSystem
from adaptive.experience import ExperienceODE
import numpy as np

logger.remove()  # remove default handler
logger.add(
    sink=lambda msg: print(msg, end=""),  # stdout
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
           "<level>{level: <8}</level> | "
           "{message}",
    level="INFO"
)
logger.add("logs/debug.log", format="{time} | {level} | {message}", level="DEBUG", rotation="10 MB")


def choose_action(actions, eps, dim_action):
    fav = np.argmax(actions)
    r = np.random.rand()
    if r < 0.5 * eps:
        return min(fav + 1, dim_action - 1)
    if r < eps:
        return max(fav - 1, 0)
    return fav


def train(predictor, env, experience, integrator, num_episodes: int, gamma: float, eps_start: float, step_sizes):
    logger.info(f"Starting training: episodes={num_episodes}, gamma={gamma}, eps_start={eps_start}")
    for episode in range(num_episodes):
        state = env.reset(integrator=integrator)
        done = False
        eps = eps_start
        logger.info(f"→ Episode {episode + 1}/{num_episodes} begins")

        while not done:
            actions = predictor.get_actions(state)
            action = choose_action(actions, eps, len(step_sizes))
            step_size = predictor.action_to_stepsize(action)

            next_state, reward, done, _ = env.iterate(step_size, integrator)
            logger.debug(f"    step_size={step_size:.5f}, reward={reward:.3e}, done={done}")

            future_q = predictor.get_actions(next_state)
            target = reward + gamma * np.max(future_q)
            targets = actions.squeeze()
            targets[action] = target
            experience.append(state=state, target=targets)

            if experience.is_full() or done:
                states, targets = experience.get_samples()
                predictor.train_on_batch(states, targets)
                experience.reset()
                logger.debug("    Trained on batch and cleared buffer")

            state = next_state.copy()

    logger.info("Training complete")


def start_training(cfg: QODEConfig):
    # Build components
    logger.info("Initializing integrator and predictor")
    factory_integrator: dict = {
        "rkdp": RKDP,
        "cash_karp": CashKarp45,
        "fehlberg78": Fehlberg78,
        "gauss_legendre_2": GaussLegendre2Stage
    }
    # integrator = RKDP()
    integrator = CashKarp45()
    scaler = load(cfg.scaler_path)
    predictor = PredictorQODE(
        step_sizes=cfg.step_sizes,
        model=build_value_modelODE(
            dim_state=cfg.rkdp.nodes_per_step * cfg.d + 1,
            dim_action=len(cfg.step_sizes),
            memory=cfg.rkdp.memory
        ),
        scaler=scaler
    )

    logger.info("Configuring reward function and environment")
    reward_range = tuple(cfg.reward_range)
    step_range = (cfg.step_sizes[0], cfg.step_sizes[-1])

    step_mid = (step_range[0] + step_range[1]) * 0.5
    factory = {
        "log": lambda: RewardLog10(cfg.tol, step_range, reward_range),
        "exp": lambda: RewardExp(cfg.tol, step_range, reward_range),
        "linear": lambda: RewardLinear(cfg.tol, step_range, reward_range),
        "sigmoid": lambda: RewardSigmoid(cfg.tol, step_mid),
        "inverse": lambda: RewardInverse(cfg.tol, step_mid),
        "quadratic": lambda: RewardQuadratic(cfg.tol, step_mid),
        "asym_exp": lambda: RewardAsymmetricExp(cfg.tol, step_range)
    }

    if cfg.reward_fn not in factory:
        raise ValueError(f"Unknown reward_fn '{cfg.reward_fn}'")

    reward_fun = factory[cfg.reward_fn]()

    env = ODEEnv(
        fun=LorenzSystem(),
        max_iterations=10000,
        initial_step_size=cfg.step_sizes[0],
        step_size_range=step_range,
        reward_fun=reward_fun,
        error_tol=cfg.tol,
        nodes_per_integ=cfg.rkdp.nodes_per_step,
        memory=cfg.rkdp.memory,
        x0=np.array(cfg.x0),
        t0=cfg.t0,
        max_dist=cfg.t1 - cfg.t0
    )

    logger.info("Creating experience replay buffer")
    experience = ExperienceODE(batch_size=cfg.batch_size)

    # Run training
    train(
        predictor=predictor,
        env=env,
        experience=experience,
        integrator=integrator,
        num_episodes=cfg.num_episodes,
        gamma=cfg.gamma,
        eps_start=cfg.eps_start,
        step_sizes=cfg.step_sizes
    )

    logger.info(f"Saving weights to {cfg.save_path}")
    if os.path.dirname(cfg.save_path):
        os.makedirs(os.path.dirname(cfg.save_path), exist_ok=True)
        predictor.model.save_weights(cfg.save_path)
    logger.success("All done!")


if __name__ == "__main__":
    parser = ArgumentParser(description="Run QODE with a JSON config")
    parser.add_argument(
        "-c", "--config",
        type=str,
        required=True,
        help="Path to your JSON config file"
    )
    args = parser.parse_args()

    logger.info(f"Loading config from {args.config}")
    with open(args.config, 'r') as f:
        config = json.loads(f.read())
        logger.info(f"Loaded config from {args.config}")
        if isinstance(config, list):
            cfgs: list[QODEConfig] = [QODEConfig.model_validate(cfg) for cfg in config]
        elif isinstance(config,dict):
            cfgs: dict[str, list[QODEConfig]] = {}
            for k, v in config.items():
                cfgs[k] = [QODEConfig.model_validate(cfg) for cfg in v]

    for cfg in cfgs:
        if isinstance(cfg, str):
            for _cfg in cfgs[cfg]:
                logger.info(f"Training on {_cfg.save_path}")
                start_training(_cfg)

        else:
            logger.info(f"Training on {cfg.save_path}")
            start_training(cfg)
