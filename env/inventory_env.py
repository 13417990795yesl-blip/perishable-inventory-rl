"""Gymnasium environment with explicitly recorded reproduction conventions."""

from dataclasses import asdict, dataclass, field
import copy
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import yaml

from .demand_model import demand_shares, sample_demand
from .state_transition import transition


@dataclass(frozen=True)
class InventoryConfig:
    """Baseline settings; continuous demand and conventional delivery timing."""

    shelf_life: int = 3
    lead_time: int = 1
    qmax: int = 10
    regular_price: float = 10.0
    discount_percent: tuple = (0, 30, 50)
    price_sensitivity: tuple = (0.3, 0.5, 0.7)
    demand_mean: float = 4.0
    demand_cv: float = 0.5
    demand_price_basis: str = "relative"
    costs: dict = field(default_factory=lambda: {"holding": 1., "ordering": 3., "lost_sales": 5., "perishable": 7.})
    episode_steps: int = 365
    seed: int = 42

    def __post_init__(self):
        """Reject invalid parameters instead of silently changing the model."""
        for key in ["shelf_life", "lead_time", "episode_steps", "qmax", "seed"]:
            value = getattr(self, key)
            minimum = 0 if key in ["qmax", "seed"] else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"invalid {key}")
        if self.demand_price_basis not in ("absolute", "relative"):
            raise ValueError("demand_price_basis must be absolute or relative")
        for key in ["regular_price", "demand_mean", "demand_cv"]:
            value = getattr(self, key)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"invalid {key}")
        if len(self.price_sensitivity) != self.shelf_life:
            raise ValueError("one sensitivity is required per inventory age")
        if not self.discount_percent or any(not np.isfinite(x) or not 0 <= x <= 100 for x in self.discount_percent):
            raise ValueError("invalid discount options")
        if any(not np.isfinite(x) or x < 0 for x in self.price_sensitivity):
            raise ValueError("invalid price sensitivity")
        if set(self.costs) != {"holding", "ordering", "lost_sales", "perishable"}:
            raise ValueError("four cost components are required")
        if any(not np.isfinite(x) or x < 0 for x in self.costs.values()):
            raise ValueError("costs must be finite and nonnegative")

    @classmethod
    def from_yaml(cls, path):
        """Read project baseline YAML, validating documented modeling choices."""
        values = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if values.pop("arrival_timing", None) != "before_next_period_sales":
            raise ValueError("unsupported or unresolved arrival timing")
        if values.pop("demand_discretization", None) != "continuous":
            raise ValueError("unsupported or unresolved demand discretization")
        values.pop("case", None)
        values.pop("gamma", None)  # Algorithm discount is not an environment parameter.
        # Missing field denotes historical absolute-price settings.
        values.setdefault("demand_price_basis", "absolute")
        return cls(**values)

    @property
    def demand_price_scale(self):
        """Divisor for demand utility; cash revenue always uses actual prices."""
        return self.regular_price if self.demand_price_basis == "relative" else 1.0


class InventoryEnv(gym.Env):
    """Discrete ordering/pricing with continuous age-specific Gamma demand.

    Actions are [order, discount_index_age1,...,discount_index_ageM].
    State has M on-shelf groups and L-1 outstanding delivery groups.
    No FIFO substitution, rounding, availability renormalization or shaping.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, config=None, render_mode=None):
        """Construct spaces; reset is required before the first action."""
        super().__init__()
        self.config = copy.deepcopy(config or InventoryConfig())
        if render_mode not in (None, "ansi"):
            raise ValueError("unsupported render mode")
        self.render_mode = render_mode
        c = self.config
        self.action_space = gym.spaces.MultiDiscrete([c.qmax + 1] + [len(c.discount_percent)] * c.shelf_life)
        self.observation_space = gym.spaces.Box(0., float(c.qmax), shape=(c.shelf_life + c.lead_time - 1,), dtype=np.float64)
        self._state = None
        self._elapsed = 0
        self._seed = None

    def reset(self, *, seed=None, options=None):
        """Reset empty inventory by default; allow a validated initial state."""
        if seed is None and self._state is None:
            seed = self.config.seed
        super().reset(seed=seed)
        if seed is not None:
            self.action_space.seed(seed)
            self._seed = seed
        options = options or {}
        if set(options) - {"initial_state"}:
            raise ValueError("unknown reset option")
        state = np.asarray(options.get("initial_state", np.zeros(self.observation_space.shape)), dtype=np.float64)
        if not self.observation_space.contains(state) or not np.all(np.isfinite(state)):
            raise ValueError("initial state is outside observation space")
        self._state = state.copy()
        self._elapsed = 0
        return self._state.copy(), {}

    def step(self, action):
        """Return next observation, profit, termination, truncation and ledger."""
        if self._state is None or self._elapsed >= self.config.episode_steps:
            raise RuntimeError("reset is required before stepping")
        raw = np.asarray(action)
        if not np.issubdtype(raw.dtype, np.integer) or not self.action_space.contains(raw):
            raise ValueError("action requires valid integer order and discount indices")
        c = self.config
        prices = c.regular_price * (1.0 - np.asarray(c.discount_percent)[raw[1:]] / 100.0)
        shares = demand_shares(prices, c.price_sensitivity, c.demand_price_scale)
        total_demand = sample_demand(self.np_random, c.demand_mean, c.demand_cv)
        self._state, reward, info = transition(self._state, int(raw[0]), prices, total_demand * shares, c.shelf_life, c.lead_time, c.costs)
        self._elapsed += 1
        info.update(total_demand=total_demand, demand_shares=shares.copy(), outside_demand=total_demand * (1. - shares.sum()),
                    demand_prices=prices/c.demand_price_scale, demand_price_basis=c.demand_price_basis)
        return self._state.copy(), float(reward), False, self._elapsed >= c.episode_steps, info

    def render(self):
        """Return an inventory summary when ANSI rendering is selected."""
        if self.render_mode == "ansi":
            return f"period={self._elapsed}, inventory={self._state}"
        return None

    def save_checkpoint(self, path):
        """Save configuration, state and both RNGs for exact simulation resume."""
        if self._state is None:
            raise RuntimeError("reset before saving")
        data = {"schema": 1, "config": asdict(self.config), "state": self._state.tolist(),
                "elapsed": self._elapsed, "seed": self._seed,
                "rng": self.np_random.bit_generator.state,
                "action_rng": self.action_space.np_random.bit_generator.state}
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def from_checkpoint(cls, path):
        """Restore a saved simulation, validating configuration and state."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data["schema"] != 1:
            raise ValueError("unsupported checkpoint schema")
        data["config"].setdefault("demand_price_basis", "absolute")
        env = cls(InventoryConfig(**data["config"]))
        env.reset(seed=data["seed"], options={"initial_state": data["state"]})
        if not isinstance(data["elapsed"], int) or not 0 <= data["elapsed"] <= env.config.episode_steps:
            raise ValueError("invalid checkpoint time")
        env._elapsed = data["elapsed"]
        env.np_random.bit_generator.state = data["rng"]
        env.action_space.np_random.bit_generator.state = data["action_rng"]
        return env
