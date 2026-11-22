import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import yaml

from algorithms.dp import GridDP
from env.inventory_env import InventoryConfig, InventoryEnv

CHECKPOINT = Path("checkpoints/relative_price_available_only/dp_grid_0p0078125.npz")
CONFIG = Path("configs/base_config.yaml")
PPO_CONFIG = Path("configs/ppo_config.yaml")
INITIALS = {"empty": [0, 0, 0], "nonempty": [3, 2.3, 1.7]}
TEST_SEEDS = list(range(42, 142))
TRAIN_SEEDS = list(range(142, 147))
OLDEST_BINS = ["zero", "(0,0.5]", "(0.5,1]", "(1,2]", ">2"]


def write_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    tmp.replace(path)


def train(seed, season_path=None):
    import stable_baselines3
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    config = InventoryConfig.from_yaml(CONFIG)
    params = yaml.safe_load(PPO_CONFIG.read_text(encoding="utf-8"))
    policy = params.pop("policy")
    net_arch = params.pop("net_arch")
    total_steps = params.pop("total_timesteps")
    assert params["n_steps"] % params["batch_size"] == 0
    root = Path("results/seasonal" if season_path else "results/ppo")
    root.mkdir(parents=True, exist_ok=True)
    if season_path:
        from env.seasonal import SeasonalInventoryEnv
        environment = SeasonalInventoryEnv.from_yaml(config, season_path)
    else:
        environment = InventoryEnv(config)
    env = Monitor(environment, filename=str(root / f"monitor_{seed}"))
    model = PPO(policy, env, seed=seed, device="cpu", verbose=1,
                policy_kwargs={"net_arch": net_arch}, **params)
    started = time.perf_counter()
    model.learn(total_timesteps=total_steps, log_interval=40)
    elapsed = time.perf_counter()-started
    assert model.num_timesteps == total_steps
    model.save(root / f"ppo_{seed}")
    env.close()
    write_json(root / f"train_{seed}.json", {
        "seed": seed, "device": "cpu", "steps": model.num_timesteps,
        "seconds": elapsed, "environment": asdict(config),
        "seasonal_schedule": [list(environment.seasons), list(environment.multipliers)] if season_path else None,
        "ppo": {**params, "policy": policy, "net_arch": net_arch},
        "stable_baselines3": stable_baselines3.__version__, "torch": torch.__version__})


def evaluate(config, actions, initial):
    envs = [InventoryEnv(config) for _ in TEST_SEEDS]
    states = np.array([env.reset(seed=seed, options={"initial_state": initial})[0]
                       for env, seed in zip(envs, TEST_SEEDS)])
    profits = np.zeros(len(envs))
    counts = np.zeros((3, 3), dtype=int)
    by_oldest = np.zeros((len(OLDEST_BINS), 3, 3), dtype=int)
    oldest_visits = np.zeros(len(OLDEST_BINS), dtype=int)
    for step in range(config.episode_steps):
        batch = actions(states)
        oldest = np.searchsorted([0., .5, 1., 2.], states[:, 2], side="left")
        age_rows, ages = np.nonzero(states[:, :3] > 0)
        np.add.at(oldest_visits, oldest, 1)
        np.add.at(counts, (ages, batch[age_rows, ages+1]), 1)
        np.add.at(by_oldest, (oldest[age_rows], ages, batch[age_rows, ages+1]), 1)
        for i, (env, action) in enumerate(zip(envs, batch)):
            states[i], reward, terminated, truncated, info = env.step(action)
            assert not terminated and truncated == (step == config.episode_steps-1)
            assert np.isclose(reward, info["revenue"]-sum(info["costs"].values()))
            profits[i] += reward
    for env in envs:
        env.close()
    return {"profits": profits.tolist(), "annual_mean": float(profits.mean()),
            "available_price_counts": counts.tolist(),
            "available_price_counts_by_oldest_stock": by_oldest.tolist(),
            "oldest_stock_visits": oldest_visits.tolist()}


def evaluate_dp():
    Path("results/ppo").mkdir(parents=True, exist_ok=True)
    dp = GridDP.load(CHECKPOINT)
    config = InventoryConfig.from_yaml(CONFIG)
    assert asdict(dp.config) == asdict(config)
    result = {name: evaluate(config, lambda states: np.array([dp.action(s) for s in states]), initial)
              for name, initial in INITIALS.items()}
    write_json(Path("results/ppo/dp.json"), result)
    print({name: row["annual_mean"] for name, row in result.items()}, flush=True)


def evaluate_model(seed):
    import torch
    from stable_baselines3 import PPO

    torch.set_num_threads(1)
    config = InventoryConfig.from_yaml(CONFIG)
    model = PPO.load(Path("results/ppo") / f"ppo_{seed}", device="cpu")
    result = {}
    for mode in ("deterministic", "stochastic"):
        result[mode] = {}
        for index, (name, initial) in enumerate(INITIALS.items()):
            torch.manual_seed(100_000 + 10*seed + index)
            result[mode][name] = evaluate(
                config, lambda states: model.predict(states, deterministic=mode == "deterministic")[0],
                initial)
            print(seed, mode, name, result[mode][name]["annual_mean"], flush=True)
    write_json(Path("results/ppo") / f"eval_{seed}.json", result)


def summarize():
    directory = Path("results/ppo")
    dp = json.loads((directory / "dp.json").read_text())
    ppo = {seed: json.loads((directory / f"eval_{seed}.json").read_text()) for seed in TRAIN_SEEDS}
    training = {seed: json.loads((directory / f"train_{seed}.json").read_text()) for seed in TRAIN_SEEDS}
    result = {"training_seeds": TRAIN_SEEDS, "test_seeds": TEST_SEEDS, "initials": INITIALS,
              "oldest_stock_bins": OLDEST_BINS, "prices": [10, 7, 5],
              "dp": dp, "training": training, "ppo": {}}
    for mode in ("deterministic", "stochastic"):
        result["ppo"][mode] = {}
        for name in INITIALS:
            rows = [ppo[s][mode][name] for s in TRAIN_SEEDS]
            profits = np.asarray([row["profits"] for row in rows])
            dp_profits = np.asarray(dp[name]["profits"])
            paired = dp_profits[None, :] - profits
            ppo_mean = float(profits.mean())
            result["ppo"][mode][name] = {
                "annual_mean": ppo_mean, "seed_means": profits.mean(axis=1).tolist(),
                "paired_dp_minus_ppo_mean": float(paired.mean()),
                "paired_dp_minus_ppo_by_training_seed": paired.mean(axis=1).tolist(),
                "gap_pct_of_dp": float(100*paired.mean()/dp_profits.mean()),
                **{key: np.sum([row[key] for row in rows], axis=0).tolist()
                   for key in ("available_price_counts", "available_price_counts_by_oldest_stock",
                               "oldest_stock_visits")}}
            print(mode, name, json.dumps(result["ppo"][mode][name]), flush=True)
    write_json(directory / "comparison.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=("train", "dp", "eval", "summary"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--seasonal-schedule", type=Path, help="optional schedule for the train command")
    args = parser.parse_args()
    if args.seasonal_schedule and args.task != "train":
        parser.error("--seasonal-schedule only applies to training")
    if args.task in ("train", "eval") and args.seed not in TRAIN_SEEDS:
        parser.error("use a training seed from 142 through 146")
    if args.task == "train":
        train(args.seed, args.seasonal_schedule)
    elif args.task == "eval":
        evaluate_model(args.seed)
    elif args.task == "dp":
        evaluate_dp()
    else:
        summarize()
