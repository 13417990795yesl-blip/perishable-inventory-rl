from dataclasses import asdict
import json
from pathlib import Path
from statistics import mean, stdev
import time

import numpy as np
from scipy.stats import t

from algorithms.dp import GridDP
from env.inventory_env import InventoryConfig
from experiments.ppo_study import CHECKPOINT, CONFIG, INITIALS, evaluate, write_json


def main():
    config = InventoryConfig.from_yaml(CONFIG)
    full = GridDP.load(CHECKPOINT)
    assert asdict(config) == asdict(full.config)
    output = Path("results/ar_dp")
    output.mkdir(parents=True, exist_ok=True)
    print("Running ar-DP on the original grid", flush=True)
    reduced = GridDP(config, full.options, ordered_prices_only=True)
    assert len(full.prices) == 27 and len(reduced.prices) == 10
    assert np.all(np.diff(reduced.prices, axis=1) <= 0)
    print(f"Transition and reward construction: {reduced.build_seconds:.2f}s", flush=True)
    summary, history = reduced.solve()
    summary["method"] = "ar-dp"
    assert summary["bellman_residual"] <= reduced.options.bellman_tolerance
    print(json.dumps(summary), flush=True)
    reduced.save(output / "ar_dp_0p0078125.npz")
    restored = GridDP.load(output / "ar_dp_0p0078125.npz")
    np.testing.assert_array_equal(reduced.price_indices, restored.price_indices)
    np.testing.assert_array_equal(reduced.action([3, 2.3, 1.7]), restored.action([3, 2.3, 1.7]))
    np.savetxt(output / "ar_dp_convergence.csv", np.column_stack((np.arange(1, len(history)+1), history)),
               delimiter=",", header="iteration,bellman_change_before_shift", comments="")

    gap = full.values - reduced.values
    tolerance = full.summary["projected_value_error_bound"] + summary["projected_value_error_bound"]
    assert float(gap.min()) >= -tolerance-1e-10
    action_match = np.all(full.policy == reduced.policy, axis=-1)
    pricing_match = (full.policy[..., 1:] == reduced.policy[..., 1:])
    available = np.broadcast_arrays(full.orders[:, None, None] > 0,
                                    full.grid[None, :, None] > 0,
                                    full.grid[None, None, :] > 0)
    effective_match = (full.policy[..., 0] == reduced.policy[..., 0]) & np.all(
        pricing_match | ~np.stack(available, axis=-1), axis=-1)
    prior = json.loads(Path("results/ppo/comparison.json").read_text())
    ppo_times = [row["seconds"] for row in prior["training"].values()]
    ar_seconds = summary["build_seconds"] + summary["solve_seconds"]
    full_seconds = full.summary["build_seconds"] + full.summary["solve_seconds"]
    report = {"ar_dp": summary, "full_dp": full.summary,
              "price_rule": "posted prices must satisfy fresh >= middle >= oldest; equality allowed",
              "full_prices": len(full.prices), "ar_dp_prices": reduced.prices.tolist(),
              "value_comparison": {"empty_state_gap": float(gap[0, 0, 0]),
                                   "max_abs_gap": float(np.abs(gap).max()),
                                   "mean_gap": float(gap.mean()),
                                   "grid_states_above_residual_error_bound": int(np.count_nonzero(gap > tolerance)),
                                   "combined_value_error_bound": tolerance,
                                   "full_action_match_fraction": float(action_match.mean()),
                                   "order_and_available_price_match_fraction": float(effective_match.mean())},
              "timing": {"dp_total": full_seconds, "ar_dp_total": ar_seconds,
                         "ar_dp_total_reduction_pct": 100*(1-ar_seconds/full_seconds),
                         "ar_dp_solve_reduction_pct": 100*(1-summary["solve_seconds"]/full.summary["solve_seconds"]),
                         "ppo_training_seconds_by_seed": ppo_times,
                         "ppo_train_seconds_mean": mean(ppo_times), "ppo_train_seconds_sd": stdev(ppo_times),
                         "scope": "DP build and solve only. PPO model.learn only, five parallel CPU processes."},
              "evaluation": {}}
    print(json.dumps(report["value_comparison"]), flush=True)
    start = time.perf_counter()
    for name, initial in INITIALS.items():
        result = evaluate(config, lambda states: np.array([reduced.action(s) for s in states]), initial)
        full_profits = np.array(prior["dp"][name]["profits"])
        delta = full_profits - np.array(result["profits"])
        se = delta.std(ddof=1)/np.sqrt(len(delta))
        width = t.ppf(.975, len(delta)-1)*se
        report["evaluation"][name] = {**result, "full_mean": float(full_profits.mean()),
                                      "paired_full_minus_ar_dp": float(delta.mean()),
                                      "paired_gap_95_ci": [float(delta.mean()-width), float(delta.mean()+width)]}
        print(name, result["annual_mean"], "paired full-minus-ar", float(delta.mean()), "CI", [float(delta.mean()-width), float(delta.mean()+width)], flush=True)
    report["timing"]["ar_dp_evaluation_seconds"] = time.perf_counter()-start
    write_json(output / "comparison.json", report)
    print(json.dumps(report["timing"]), flush=True)


if __name__ == '__main__':
    main()
