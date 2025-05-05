import os.path
from argparse import ArgumentParser
from loguru import logger
from config.config import QODEConfig
from adaptive.integrator import RKDP, CashKarp45, Fehlberg78, GaussLegendre2Stage
from adaptive.predictor import PredictorQODE
from adaptive.build_models import build_value_modelODE
from adaptive.reward_functions import RewardLog10, RewardExp
from adaptive.environments import ODEEnv
from adaptive.comparison_ode import integrate_env
from functions import LorenzSystem
from joblib import load
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
import json


def load_environment(cfg: QODEConfig):
    step_range = (cfg.step_sizes[0], cfg.step_sizes[-1])
    reward_range = tuple(cfg.reward_range)

    reward_fun_cls = RewardLog10 if cfg.reward_fn == "log" else RewardExp
    reward_fun = reward_fun_cls(
        error_tol=cfg.tol,
        step_size_range=step_range,
        reward_range=reward_range,
    )

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


def evaluate(cfg_path: str, models_cfg_path: str, save_results_path: str | None = None):
    logger.info("Loading config & model list ...")
    print(cfg_path)
    with open(cfg_path, 'r') as f:
        config = json.loads(f.read())[0]

    cfg = QODEConfig.model_validate(config)
    with open(models_cfg_path) as fp:
        models_dict: dict[str, str] = json.load(fp)

    factory_integrator: dict = {
        "rkdp": RKDP,
        "cash_karp": CashKarp45,
        "fehlberg78": Fehlberg78,
        "gauss_legendre_2": GaussLegendre2Stage
    }
    integrator = RKDP()
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

    t0, t1 = 0, 20
    results: dict[str, dict[str, float]] = {}

    for name, weights in models_dict.items():
        logger.info(f"Evaluating {name}")
        time_steps: list = []
        predictor.model.load_weights(weights)
        env.reset(integrator=integrator)
        reward, nfev = integrate_env(predictor, integrator, env, t0=t0, t1=t1,time_steps=time_steps)
        results[name] = {
            "reward": reward,
            "nfev": nfev,
            "mean_error": np.mean(env.errors),
            "min_error": np.min(env.errors),
            "max_error": np.max(env.errors),
            "min_stepsize": np.min(env.deltas),
            "max_stepsize": np.max(env.deltas),
            "time_steps": time_steps.copy()
        }

    if save_results_path:
        logger.info(f"Writing metrics → {save_results_path}")
        if os.path.dirname(save_results_path):
            os.makedirs(os.path.dirname(save_results_path), exist_ok=True)

        with open(save_results_path, "w") as fp:
            json.dump(results, fp, indent=4)

    plt.style.use("ggplot")
    metrics = [
        "reward", "nfev", "mean_error", "min_error",
        "max_error", "min_stepsize", "max_stepsize",
    ]
    model_names = list(results.keys())
    y_pos = np.arange(len(model_names))

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
        plt.show(block=False)

    # plt.style.use("ggplot")
    metrics = [
        "reward", "nfev", "mean_error", "min_error",
        "max_error", "min_stepsize", "max_stepsize",
    ]
    n_cols = 2
    n_rows = int(np.ceil(len(metrics) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(17, 3.2 * n_rows))
    axes = axes.flatten()

    model_names = list(results.keys())
    y_pos = np.arange(len(model_names))

    for ax, metric in zip(axes, metrics):
        values = [results[m][metric] for m in model_names]
        ax.barh(y_pos, values, align="center")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(model_names, fontsize=9)
        ax.invert_yaxis()  # highest bar at top
        ax.set_xlabel(metric.replace("_", " "))
        ax.set_title(metric.replace("_", " ").title(), fontsize=11, pad=6)
        for y, val in zip(y_pos, values):
            ax.text(val, y, f" {val:.2e}" if abs(val) < 1e-2 or abs(val) > 1e3 else f" {val:.3g}",
                    va="center", fontsize=8, color="black")
    for ax in axes[len(metrics):]:
        fig.delaxes(ax)

    fig.tight_layout()
    plt.show()

    logger.info("Plotting Error vs NFEV for all models …")
    plt.figure(figsize=(7.5, 6))
    markers = ["o", "s", "^", "D", "v", "<", ">", "P", "X"]
    for i, (name, data) in enumerate(results.items()):
        err = data["mean_error"]
        rate = data["nfev"] / (t1 - t0)
        plt.loglog(err, rate, marker=markers[i % len(markers)], linestyle="none", label=name)

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
    plt.show()



if __name__ == "__main__":
    # p = ArgumentParser(description="Evaluate multiple QODE models and plot results nicely.")
    # p.add_argument("-c", "--config", required=True, help="Path to QODE config JSON")
    # p.add_argument("-m", "--models", required=True, help="Path to models JSON (name:path)")
    # p.add_argument("-s", "--save", help="Optional path to save metrics JSON")
    # args = p.parse_args()

    # evaluate(args.config, args.models, args.save)
    evaluate(cfg_path="config/config_template.json", models_cfg_path="config/eval/models_path.json", save_results_path="results/cashkarp45/eval_results.json")