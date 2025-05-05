import os.path
from argparse import ArgumentParser
from loguru import logger
from config.config import QODEConfig
from adaptive.integrator import RKDP, CashKarp45, Fehlberg78, GaussLegendre2Stage
from adaptive.predictor import PredictorQODE
from adaptive.build_models import build_value_modelODE
from adaptive.reward_functions import RewardLog10, RewardExp, RewardLinear, RewardSigmoid, RewardInverse, \
    RewardQuadratic, RewardAsymmetricExp
from adaptive.environments import ODEEnv
from adaptive.comparison_ode import integrate_env
from functions import LorenzSystem
from joblib import load
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
import json
from config.config import load_config

def plot_evaluation_results(results, t0, t1, cfg, save_dir="results/plots"):
    import os
    import matplotlib.pyplot as plt
    from functions import LorenzSystem
    from scipy.integrate import solve_ivp
    import numpy as np

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"{save_dir}/metric_bars", exist_ok=True)
    os.makedirs(f"{save_dir}/metric_grids", exist_ok=True)
    os.makedirs(f"{save_dir}/error_vs_nfev", exist_ok=True)

    plt.style.use("ggplot")
    metrics = ["reward", "nfev", "mean_error", "min_error", "max_error", "min_stepsize", "max_stepsize"]
    model_names = list(results.keys())
    y_pos = np.arange(len(model_names))

    # Bar chart per metric
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(8, 3 + 0.4 * len(model_names)))
        values = [results[m][metric] for m in model_names]
        ax.barh(y_pos, values, align="center", color="#1f77b4")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(model_names, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel(metric.replace("_", " "))
        ax.set_title(metric.replace("_", " ").title(), fontsize=11, pad=6)
        ax.margins(x=0.05)
        for y, val in zip(y_pos, values):
            fmt = f"{val:.2e}" if abs(val) < 1e-2 or abs(val) > 1e3 else f"{val:.5g}"
            ax.text(val, y, f" {fmt}", va="center", fontsize=7)
        fig.tight_layout()
        fig_path = os.path.join(save_dir, "metric_bars", f"{metric}.png")
        fig.savefig(fig_path)
        logger.info(f"Saved bar chart → {fig_path}")
        plt.close(fig)

    # Combined grid
    n_cols = 2
    n_rows = int(np.ceil(len(metrics) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(17, 3.2 * n_rows))
    axes = axes.flatten()
    for ax, metric in zip(axes, metrics):
        values = [results[m][metric] for m in model_names]
        ax.barh(y_pos, values, align="center")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(model_names, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel(metric.replace("_", " "))
        ax.set_title(metric.replace("_", " ").title(), fontsize=11, pad=6)
        for y, val in zip(y_pos, values):
            ax.text(val, y, f" {val:.2e}" if abs(val) < 1e-2 or abs(val) > 1e3 else f" {val:.3g}",
                    va="center", fontsize=8, color="black")
    for ax in axes[len(metrics):]:
        fig.delaxes(ax)

    fig.tight_layout()
    fig_path = os.path.join(save_dir, "metric_grids", "metrics_grid.png")
    fig.savefig(fig_path)
    logger.info(f"Saved grid chart → {fig_path}")
    plt.close(fig)

    # Error vs NFEV plot
    logger.info("Plotting Error vs NFEV for all models …")
    fig = plt.figure(figsize=(7.5, 6))
    markers = ["o", "s", "^", "D", "v", "<", ">", "P", "X"]
    for i, (name, data) in enumerate(results.items()):
        err = data["mean_error"]
        rate = data["nfev"] / (t1 - t0)
        plt.loglog(err, rate, marker=markers[i % len(markers)], linestyle="none", label=name)

    # Add RK45 baseline
    f = LorenzSystem()
    rk45 = []
    for tol in [5e-6, 1e-5, 2.5e-5, 5e-5]:
        f.reset()
        sol = solve_ivp(f, (t0, t1), cfg.x0, atol=tol, rtol=tol)
        step_errs = [np.linalg.norm(sol.y[:, j + 1] - f.solve(sol.t[j], sol.y[:, j], sol.t[j + 1]))
                     for j in range(sol.t.size - 1)]
        rk45.append((np.mean(step_errs), sol.nfev / (t1 - t0)))
    rk45 = np.array(rk45)
    plt.loglog(rk45[:, 0], rk45[:, 1], "k-", marker="x", label="RK45 (var tol)")

    plt.xlabel("Mean Error per RK Step")
    plt.ylabel("Function Evaluations / unit time")
    plt.grid(which="both", linestyle=":")
    plt.legend()
    plt.tight_layout()
    fig_path = os.path.join(save_dir, "error_vs_nfev", "error_vs_nfev.png")
    fig.savefig(fig_path)
    logger.info(f"Saved error vs NFEV plot → {fig_path}")
    plt.close(fig)

def save_json(filename: str, data) -> bool:
    try:
        if os.path.dirname(filename):
            os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            f.write(json.dumps(data, indent=4))

        return True
    except Exception as e:
        logger.error(e)
        return False


def load_environment(cfg: QODEConfig):
    step_range = (cfg.step_sizes[0], cfg.step_sizes[-1])
    reward_range = tuple(cfg.reward_range)
    step_mid = (step_range[0] + step_range[1]) * 0.5

    factory: dict = {
        "log": RewardLog10(cfg.tol, step_range, reward_range),
        "exp": RewardExp(cfg.tol, step_range, reward_range),
        "linear": RewardLinear(cfg.tol, step_range, reward_range),
        "sigmoid": RewardSigmoid(cfg.tol, step_mid),
        "inverse": RewardInverse(cfg.tol, step_mid),
        "quadratic": RewardQuadratic(cfg.tol, step_mid),
        "asym_exp": RewardAsymmetricExp(cfg.tol, step_range)
    }

    reward_fun = factory[cfg.reward_fn]

    return ODEEnv(
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
        max_dist=cfg.t1 - cfg.t0,
    )


def evaluate(cfg_path: str, save_path: str = "results/eval_all.json") -> None:
    factory_integrator: dict = {
        "rkdp": RKDP,
        "cash_karp": CashKarp45,
        "fehlberg78": Fehlberg78,
        "gauss_legendre_2": GaussLegendre2Stage
    }
    cfgs_dict: dict[str, list[QODEConfig]] = load_config(cfg_path)
    results: dict[str, dict[str, dict[str, float]]] = {}
    t0, t1 = 0, 20
    for integrator_name, cfgs in cfgs_dict.items():
        results[integrator_name] = {}
        integrator = factory_integrator[integrator_name]()

        for cfg in cfgs:
            scaler = load(cfg.scaler_path)
            predictor = PredictorQODE(
                step_sizes=cfg.step_sizes,
                model=build_value_modelODE(
                    dim_state=cfg.rkdp.nodes_per_step * cfg.d + 1,
                    dim_action=len(cfg.step_sizes),
                    memory=cfg.rkdp.memory,
                ),
                scaler=scaler,
            )
            env = load_environment(cfg)

            logger.info(f"Evaluating {integrator_name}/{cfg.reward_fn}")

            time_steps: list = []
            predictor.model.load_weights(cfg.save_path)
            env.reset(integrator=integrator)
            reward, nfev = integrate_env(predictor, integrator, env, t0=t0, t1=t1, time_steps=time_steps)
            results[integrator_name][cfg.reward_fn] = {
                "reward": reward,
                "nfev": nfev,
                "mean_error": np.mean(env.errors),
                "min_error": np.min(env.errors),
                "max_error": np.max(env.errors),
                "min_stepsize": np.min(env.deltas),
                "max_stepsize": np.max(env.deltas),
                "time_steps": time_steps.copy()
            }
    save_json(filename=save_path, data=results)
    logger.info(f"Saved results to {save_path}")
    for method in results:
        plot_evaluation_results(results[method], t0, t1, cfgs[0], save_dir=f"results/{method}/plots")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate QODE configurations.")
    parser.add_argument(
        "--cfg_path", type=str, required=True,
        help="Path to the configuration JSON file."
    )
    parser.add_argument(
        "--save_path", type=str, default="results/eval_all.json",
        help="Path to save evaluation results (default: results/eval_all.json)."
    )
    args = parser.parse_args()

    evaluate(cfg_path=args.cfg_path, save_path=args.save_path)
    # evaluate(cfg_path="config/train/lorenz_config.json")