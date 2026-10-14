"""Environment rollout and DP refinement audits through one command line."""

import argparse
import csv
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import t as student_t

from algorithms.dp import DPOptions, GridDP, output_paths
from env import InventoryConfig, InventoryEnv


def audit_environment(config):
    """Audit 365-step episodes across shelf lives and positive lead times."""
    rows = []
    paths=output_paths(config)
    for shelf, lead in [(3, 1), (3, 2), (3, 3), (1, 1), (5, 2)]:
        for seed in [42, 123, 2026]:
            case_config = replace(config, shelf_life=shelf, lead_time=lead,
                             price_sensitivity=tuple(np.linspace(.3, .7, shelf)))
            env = InventoryEnv(case_config)
            before, _ = env.reset(seed=seed)
            profit = 0.
            waste = 0.
            for step in range(config.episode_steps):
                action = env.action_space.sample()
                after, reward, terminated, truncated, info = env.step(action)
                # Across on-shelf stock plus pipeline: arrivals are internal moves.
                balance = before.sum()+action[0]-info["sales_by_age"].sum()-info["expired"]
                assert np.isclose(after.sum(), balance, rtol=0, atol=1e-10)
                assert env.observation_space.contains(after)
                assert np.isfinite(reward)
                assert np.isclose(reward, info["revenue"]-sum(info["costs"].values()))
                assert not terminated and truncated == (step == config.episode_steps-1)
                profit += reward
                waste += info["expired"]
                before = after
                if shelf == 3 and lead == 1 and seed == 42 and step == 29:
                    env.save_checkpoint(paths["checkpoints"]/"phase3_environment_step30.json")
            rows.append({"shelf_life": shelf, "lead_time": lead, "seed": seed,
                         "steps": config.episode_steps, "random_policy_profit": profit, "expired": waste,
                         "status": "passed"})
    report = {"phase": 3, "status": "passed", "episodes": len(rows),
              "steps": sum(x["steps"] for x in rows), "policy": "random validation only",
              "rollouts": rows}
    (paths["logs"]/"phase3_rollout_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Rollout audit passed: {report['episodes']} episodes, {report['steps']} steps")


def audit_dp(config, dp_config, *, reuse_solved=False):
    """Solve configured grids, compare common states and evaluate real rollouts."""
    paths=output_paths(config)
    settings = yaml.safe_load(Path(dp_config).read_text(encoding="utf-8"))
    solvers, summaries = [], []
    for step in settings["validation_grid_steps"]:
        print(f"Building/solving grid h={step}",flush=True)
        option_values = {k:v for k,v in settings.items() if k in DPOptions.__dataclass_fields__}
        option_values["grid_step"] = step
        options = DPOptions(**option_values)
        tag = str(step).replace(".","p")
        if reuse_solved:
            dp = GridDP.load(paths["checkpoints"]/f"dp_grid_{tag}.npz")
            if dp.options != options or asdict(dp.config) != asdict(config):
                raise ValueError("checkpoint does not match current validation settings")
            summary=dp.summary
        else:
            dp = GridDP(config,options)
            summary, history = dp.solve()
            dp.save(paths["checkpoints"]/f"dp_grid_{tag}.npz")
            (paths["logs"]/f"dp_grid_{tag}.json").write_text(json.dumps(summary,indent=2))
            np.savetxt(paths["tables"]/f"dp_convergence_{tag}.csv", np.column_stack((np.arange(1,len(history)+1),history)),
                       delimiter=",",header="iteration,bellman_residual_before_update",comments="")
        solvers.append(dp)
        summaries.append(summary)
        print(json.dumps(summary),flush=True)
    refinements = []
    for coarse,fine in zip(solvers[:-1],solvers[1:]):
        indices = np.searchsorted(fine.grid,coarse.grid)
        np.testing.assert_allclose(fine.grid[indices],coarse.grid,rtol=0,atol=1e-12)
        fine_values = fine.values[:,indices,:][:,:,indices]
        fine_policy = fine.policy[:,indices,:,:][:,:,indices,:]
        changes = np.abs(coarse.values-fine_values)
        refinements.append({"coarse_h":coarse.options.grid_step,"fine_h":fine.options.grid_step,
                            "common_states":coarse.state_count,"max_common_state_value_change":float(changes.max()),
                            "mean_common_state_value_change":float(changes.mean()),
                            "max_change_relative_to_fine_value_scale":float(changes.max()/max(1.,np.abs(fine_values).max())),
                            "common_state_action_agreement":float(np.mean(np.all(coarse.policy==fine_policy,axis=-1)))})
    # Independently refine the outside region, preserving the dense grid.
    final = solvers[-1]
    outer_check = None
    if final.options.grid_kind == "near_zero_refined":
        print("Checking outside grid with half coarse_step",flush=True)
        outer = GridDP(config,replace(final.options,coarse_step=final.options.coarse_step/2))
        initial = np.empty(outer.shape)
        for fresh in final.orders:
            intermediate = np.array([np.interp(outer.grid,final.grid,final.values[fresh,:,j]) for j in range(final.levels)]).T
            initial[fresh] = np.array([np.interp(outer.grid,final.grid,row) for row in intermediate])
        summary, _ = outer.solve(initial_values=initial)
        outer.save(paths["checkpoints"]/"dp_outer_grid_check.npz")
        indices = np.searchsorted(outer.grid,final.grid)
        changes = np.abs(final.values-outer.values[:,indices,:][:,:,indices])
        outer_check = {"coarse_step":final.options.coarse_step,"refined_coarse_step":outer.options.coarse_step,
                       "max_common_state_value_change":float(changes.max()),
                       "max_change_relative_to_refined_value_scale":float(changes.max()/max(1.,np.abs(outer.values).max())),
                       "empty_state_value_change":float(abs(final.values[0,0,0]-outer.values[0,0,0])),
                       "solution":summary}
        print(json.dumps(outer_check),flush=True)
    # Common random seeds compare policies on the actual unrounded Environment.
    # These are Phase 4 diagnostics, not comparisons to paper DP/PPO results.
    rollouts = []
    seeds = list(range(42,42+settings.get("validation_seed_count",100)))
    off_grid_initial = np.minimum([3., 2.3, 1.7], config.qmax).tolist()
    for dp in solvers:
        started = time.perf_counter()
        for name,initial in [("empty",[0.,0.,0.]),("off_grid",off_grid_initial)]:
            profits, discounted_profits, orders, ledgers = [],[],[],[]
            for seed in seeds:
                env = InventoryEnv(config)
                state,_ = env.reset(seed=seed,options={"initial_state":initial})
                total, discounted, ordered = 0.,0.,0.
                ledger={key:0. for key in ["revenue","ordering","holding","lost_sales","perishable","sold_units","expired_units"]}
                for t in range(config.episode_steps):
                    action = dp.action(state)
                    state,reward,terminated,truncated,info = env.step(action)
                    assert not terminated and truncated == (t==config.episode_steps-1)
                    assert env.observation_space.contains(state)
                    total += reward
                    discounted += dp.options.gamma**t*reward
                    ordered += action[0]
                    ledger["revenue"]+=info["revenue"]
                    for key in ["ordering","holding","lost_sales","perishable"]:
                        ledger[key]+=info["costs"][key]
                    ledger["sold_units"]+=float(info["sales_by_age"].sum())
                    ledger["expired_units"]+=info["expired"]
                assert np.isclose(total,ledger["revenue"]-sum(ledger[k] for k in ["ordering","holding","lost_sales","perishable"]))
                ledgers.append(ledger)
                profits.append(total)
                discounted_profits.append(discounted)
                orders.append(ordered)
            rollouts.append({"h":dp.options.grid_step,"initial_state":name,"seeds":seeds,
                             "mean_365_step_profit":float(np.mean(profits)),
                             "profit_standard_error":float(np.std(profits,ddof=1)/np.sqrt(len(seeds))),
                             "mean_365_step_discounted_profit":float(np.mean(discounted_profits)),
                             "mean_total_order_quantity":float(np.mean(orders)),
                             "mean_ledger":{k:float(np.mean([x[k] for x in ledgers])) for k in ledgers[0]},
                             "profits":profits,"discounted_profits":discounted_profits})
        print(f"Continuous policy audits for h={dp.options.grid_step}: {time.perf_counter()-started:.2f}s",flush=True)
    no_order_share = float(np.mean(final.policy[...,0]==0))
    last_refinement = refinements[-1]
    # Engineering sensitivity criterion, not a rigorous continuum error bound.
    sensitivity_pass = last_refinement["max_change_relative_to_fine_value_scale"] <= .001
    if outer_check is not None:
        sensitivity_pass &= outer_check["max_change_relative_to_refined_value_scale"] <= .001
    paired_changes = []
    for name in ["empty","off_grid"]:
        coarser = next(x for x in rollouts if x["h"]==solvers[-2].options.grid_step and x["initial_state"]==name)
        finer = next(x for x in rollouts if x["h"]==final.options.grid_step and x["initial_state"]==name)
        difference = np.array(finer["profits"])-np.array(coarser["profits"])
        se=float(difference.std(ddof=1)/np.sqrt(len(seeds)))
        ci_half_width=float(student_t.ppf(.975,len(seeds)-1)*se)
        paired_changes.append({"initial_state":name,"fine_minus_coarse_mean_365_profit":float(difference.mean()),
                               "paired_difference_standard_error":se,
                               "paired_95_percent_ci":[float(difference.mean()-ci_half_width),float(difference.mean()+ci_half_width)],
                               "absolute_mean_plus_ci_half_width":float(abs(difference.mean())+ci_half_width),
                               "max_absolute_episode_profit_change":float(np.abs(difference).max())})
    profit_limit=settings.get("paired_profit_change_limit",.5)
    profit_sensitivity_pass=all(x["absolute_mean_plus_ci_half_width"] <= profit_limit for x in paired_changes)
    sensitivity_pass &= profit_sensitivity_pass
    report = {"phase":4,"status":"passed" if sensitivity_pass else "requires_finer_grid",
              "config":asdict(config),"solutions":summaries,"grid_refinements":refinements,
              "outside_grid_check":outer_check,"paired_continuous_profit_changes":paired_changes,
              "engineering_relative_grid_change_threshold":.001,"grid_sensitivity_pass":sensitivity_pass,
              "paired_profit_change_limit":profit_limit,"profit_sensitivity_pass":profit_sensitivity_pass,
              "rollouts":rollouts,"policy_diagnostic_no_order_state_fraction":no_order_share,
              "notes":["DP residual bounds apply only to the projected grid problem.",
                       "No demand/inventory rounding occurs in the Environment.",
                       "365-step undiscounted profit is distinct from infinite discounted V.",
                       "Empty-state no-order outcome, if present, is a model diagnostic, not confirmation of paper results."]}
    report["author_numerical_reproduction_confirmed"]=False
    report["price_interpretation_source"]="user-selected; author implementation not verified"
    (paths["logs"]/"phase4_validation.json").write_text(json.dumps(report,indent=2))
    with (paths["tables"]/"dp_grid_sensitivity.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(refinements[0]))
        writer.writeheader()
        writer.writerows(refinements)
    with (paths["tables"]/"dp_grid_summary.csv").open("w",newline="") as handle:
        keys=["grid_step","states","actions","iterations","bellman_residual","projected_value_error_bound","empty_state_discounted_value","build_seconds","solve_seconds"]
        writer=csv.DictWriter(handle,fieldnames=keys,extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)
    print(json.dumps({"grid_refinements":refinements,"no_order_state_fraction":no_order_share,"status":report["status"]},indent=2),flush=True)
    if not sensitivity_pass:
        raise RuntimeError("Grid sensitivity threshold was not met; refine before calling Phase 4 complete")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit", choices=["environment", "dp", "all"])
    parser.add_argument("--config", default="configs/base_config.yaml")
    parser.add_argument("--dp-config", default="configs/dp_config.yaml")
    parser.add_argument("--reuse-solved", action="store_true",
                        help="Reuse matching DP checkpoints; still run refinement and policy audits")
    args = parser.parse_args()
    if args.audit == "environment" and args.reuse_solved:
        parser.error("--reuse-solved requires dp or all")
    config = InventoryConfig.from_yaml(args.config)
    if args.audit in {"environment", "all"}:
        audit_environment(config)
    if args.audit in {"dp", "all"}:
        audit_dp(config, args.dp_config, reuse_solved=args.reuse_solved)


if __name__ == "__main__":
    main()
