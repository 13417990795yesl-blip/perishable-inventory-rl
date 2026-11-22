import copy
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
from scipy.stats import t

from algorithms.dp import GridDP
from env.inventory_env import InventoryConfig
from experiments.ppo_study import CHECKPOINT, CONFIG, INITIALS, evaluate, write_json



def main():
    dynamic = GridDP.load(CHECKPOINT)
    config = InventoryConfig.from_yaml(CONFIG)
    assert asdict(config) == asdict(dynamic.config)
    assert config.exclude_empty_from_mnl
    start = time.perf_counter()
    full = GridDP(config, dynamic.options)
    print(f"Built shared transition kernels in {time.perf_counter() - start:.1f}s", flush=True)

    ranked = []
    best = None
    best_value = -np.inf
    for i, price in enumerate(full.prices):
        # Each price vector owns two kernels, for absent and available oldest stock.
        fixed = copy.copy(full)
        fixed.price_indices = full.price_indices[i:i+1]
        fixed.prices = full.prices[i:i+1]
        fixed.shares = full.shares[i:i+1]
        fixed.kernels = full.kernels[2*i:2*i+2]
        fixed.rewards = full.rewards[2*i:2*i+2]
        fixed.action_count = len(full.orders)
        summary, _ = fixed.solve()
        score = summary["empty_state_discounted_value"]
        assert summary["bellman_residual"] <= fixed.options.bellman_tolerance
        assert score <= dynamic.values[0, 0, 0] + 2e-5
        ranked.append({"prices": price.tolist(), "price_indices": fixed.price_indices[0].tolist(),
                       "empty_state_discounted_value": score, "iterations": summary["iterations"],
                       "bellman_residual": summary["bellman_residual"],
                       "empty_state_order": summary["empty_state_action"][0]})
        print(json.dumps(ranked[-1]), flush=True)
        if score > best_value:
            best, best_value = fixed, score

    ranked.sort(key=lambda row: row["empty_state_discounted_value"], reverse=True)
    results = Path("results/fixed_price")
    results.mkdir(parents=True, exist_ok=True)
    write_json(results / "fixed_price_ranking.json", ranked)
    print("Best fixed schedule:", ranked[0], flush=True)

    prior = json.loads(Path("results/ppo/comparison.json").read_text())
    report = {"best": ranked[0], "runner_up": ranked[1],
              "dynamic_discounted_value": float(dynamic.values[0, 0, 0]),
              "selection": "maximum discounted value from empty stock over all 27 stationary age-price schedules",
              "evaluation": {}}
    for name, initial in INITIALS.items():
        evaluation = evaluate(config, lambda states: np.array([best.action(s) for s in states]), initial)
        fixed_profits = np.array(evaluation["profits"])
        dynamic_profits = np.array(prior["dp"][name]["profits"])
        difference = dynamic_profits - fixed_profits
        assert np.all(np.isfinite(fixed_profits)) and fixed_profits.shape == dynamic_profits.shape
        standard_error = difference.std(ddof=1)/np.sqrt(len(difference))
        width = t.ppf(.975, len(difference)-1)*standard_error
        delta = difference.mean()
        entry = {"fixed": evaluation, "dynamic_mean": float(dynamic_profits.mean()),
                 "ppo_deterministic_mean": prior["ppo"]["deterministic"][name]["annual_mean"],
                 "ppo_stochastic_mean": prior["ppo"]["stochastic"][name]["annual_mean"],
                 "fixed_standard_error": float(fixed_profits.std(ddof=1)/np.sqrt(len(fixed_profits))),
                 "paired_dynamic_minus_fixed": float(delta),
                 "paired_gain_95pct_ci": [float(delta-width), float(delta+width)],
                 "gain_percent_of_fixed": float(100*delta/fixed_profits.mean())}
        report["evaluation"][name] = entry
        print(name, {**{k: v for k, v in entry.items() if k != "fixed"},
                     "fixed_mean": evaluation["annual_mean"]}, flush=True)
    write_json(results / "fixed_vs_dynamic.json", report)
    np.savez_compressed(results / "best_fixed_policy.npz", prices=best.prices[0],
                        price_indices=best.price_indices[0], values=best.values,
                        orders=best.policy[..., 0], grid=best.grid, continuations=best.continuations,
                        config_json=json.dumps(asdict(config)), options_json=json.dumps(asdict(best.options)))


if __name__ == '__main__':
    main()
