from dataclasses import replace

import numpy as np
import pytest

from algorithms.dp import DPOptions, GridDP
from env.demand_model import demand_shares
from env.inventory_env import InventoryConfig, InventoryEnv
from env.seasonal import SeasonalInventoryEnv


def test_masked_demand_excludes_empty_age_and_keeps_outside_option():
    config = InventoryConfig.from_yaml("configs/base_config.yaml")
    p = [10, 7, 5]
    shares = demand_shares(p, [3, 5, 7], 10, [True, False, True], config.purchase_utility)
    weights = np.exp(2 - np.array([3, 5, 7])*np.array(p)/10)
    weights[1] = 0
    np.testing.assert_allclose(shares, weights/(1+weights.sum()))
    np.testing.assert_array_equal(shares, demand_shares(
        [10, 10, 5], [3, 5, 7], 10, [True, False, True], config.purchase_utility))


def test_reduced_dp_checkpoint_retains_restricted_action_set(tmp_path):
    config = replace(InventoryConfig.from_yaml("configs/base_config.yaml"), qmax=2)
    options = DPOptions(grid_kind="uniform", grid_step=0.5)
    dp = GridDP(config, options, ordered_prices_only=True)
    assert dp.action_count == 30
    assert np.all(np.diff(dp.prices, axis=1) <= 0)
    dp.solve()
    file = tmp_path / "reduced.npz"
    dp.save(file)
    restored = GridDP.load(file)
    np.testing.assert_array_equal(restored.price_indices, dp.price_indices)
    np.testing.assert_array_equal(restored.action([1, .3, .6]), dp.action([1, .3, .6]))


def test_season_changes_gamma_mean_and_exposes_next_phase():
    config = InventoryConfig.from_yaml("configs/base_config.yaml")
    env = SeasonalInventoryEnv(config, [2, 1], [.7, 1.6])
    stationary = InventoryEnv(config)
    env.reset(seed=15)
    stationary.reset(seed=15)
    for multiplier, next_phase in ((.7, .5), (.7, 1), (1.6, 0), (.7, .5)):
        obs, _, _, _, info = env.step([1, 0, 0, 0])
        _, _, _, _, base_info = stationary.step([1, 0, 0, 0])
        assert info["total_demand"] == pytest.approx(base_info["total_demand"]*multiplier)
        assert obs[-1] == next_phase
        assert env.observation_space.contains(obs)
