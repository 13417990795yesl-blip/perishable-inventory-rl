"""Optional periodic demand scenario with the calendar phase observable."""

from pathlib import Path

import gymnasium as gym
import numpy as np
import yaml

from .inventory_env import InventoryEnv


class _SeasonalDemandEnv(InventoryEnv):
    def __init__(self, config, day_multipliers):
        super().__init__(config)
        self.day_multipliers = day_multipliers

    def _sample_total_demand(self):
        day = self._elapsed % len(self.day_multipliers)
        return super()._sample_total_demand() * self.day_multipliers[day]


class SeasonalInventoryEnv(gym.Wrapper):
    """Append a normalized cycle day and scale the Gamma mean by season."""

    def __init__(self, config, season_lengths, demand_multipliers):
        if (not season_lengths or len(season_lengths) != len(demand_multipliers)
                or any(type(days) is not int or days < 1 for days in season_lengths)):
            raise ValueError("use matching, nonempty seasons of positive integer lengths")
        if any(not np.isfinite(value) or value <= 0 for value in demand_multipliers):
            raise ValueError("demand multipliers must be positive and finite")
        self.seasons = tuple(season_lengths)
        self.multipliers = tuple(demand_multipliers)
        day_multipliers = np.repeat(self.multipliers, self.seasons)
        super().__init__(_SeasonalDemandEnv(config, day_multipliers))
        self.cycle_days = len(day_multipliers)
        self._ends = np.cumsum(self.seasons)
        self._day = 0
        # The cycle day encodes time remaining in the season as well as its identity.
        self.observation_space = gym.spaces.Box(
            np.r_[self.env.observation_space.low, 0.0],
            np.r_[self.env.observation_space.high, 1.0], dtype=np.float64)

    @classmethod
    def from_yaml(cls, config, path):
        schedule = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls(config, schedule["season_lengths"], schedule["demand_multipliers"])

    def _observation(self, stock):
        return np.r_[stock, self._day / max(1, self.cycle_days - 1)]

    def reset(self, *, seed=None, options=None):
        stock, info = self.env.reset(seed=seed, options=options)
        self._day = 0
        return self._observation(stock), info

    def step(self, action):
        stock, reward, terminated, truncated, info = self.env.step(action)
        season = int(np.searchsorted(self._ends, self._day, side="right"))
        info.update(season_index=season, demand_multiplier=self.multipliers[season])
        self._day = (self._day + 1) % self.cycle_days
        return self._observation(stock), reward, terminated, truncated, info
