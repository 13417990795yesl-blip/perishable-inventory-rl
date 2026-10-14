"""Grid value iteration with analytic Gamma integrals of bilinear values.

The projected finite MDP approximates the continuous Environment. All baseline
actions remain available; this is not action-reduction DP. Scope: M=3,L=1.
"""

import argparse
from dataclasses import asdict, dataclass
from itertools import product
import json
from pathlib import Path
import time

import numpy as np
from scipy import sparse
from scipy.special import gammainc, gammaincc, poch
import yaml

from env import InventoryConfig
from env.demand_model import demand_shares


@dataclass(frozen=True)
class DPOptions:
    """Explicit numerical settings for the projected Bellman equation."""

    grid_step: float = 0.0078125
    gamma: float = 0.99
    bellman_tolerance: float = 1e-7
    max_iterations: int = 5000
    grid_kind: str = "near_zero_refined"
    dense_end: float = 2.0
    coarse_step: float = 0.25
    value_shift_acceleration: bool = True

    def __post_init__(self):
        """Validate finite grid, contraction and convergence settings."""
        if not np.isfinite(self.grid_step) or self.grid_step <= 0:
            raise ValueError("grid_step must be positive")
        if not np.isclose(1/self.grid_step, round(1/self.grid_step), rtol=0, atol=1e-10):
            raise ValueError("grid_step must divide one unit")
        if not np.isfinite(self.gamma) or not 0 <= self.gamma < 1:
            raise ValueError("gamma must lie in [0,1)")
        if not np.isfinite(self.bellman_tolerance) or self.bellman_tolerance <= 0:
            raise ValueError("bellman_tolerance must be positive")
        if isinstance(self.max_iterations, bool) or not isinstance(self.max_iterations, int) or self.max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer")
        if not isinstance(self.value_shift_acceleration,bool):
            raise ValueError("value_shift_acceleration must be boolean")
        if self.grid_kind not in ("uniform", "near_zero_refined"):
            raise ValueError("unsupported grid_kind")
        if self.grid_kind == "near_zero_refined":
            for value in (self.coarse_step, self.dense_end):
                if not np.isfinite(value) or value <= 0:
                    raise ValueError("grid geometry must be positive")
            if not np.isclose(1/self.coarse_step,round(1/self.coarse_step)):
                raise ValueError("coarse_step must divide one unit")
            if self.grid_step > self.coarse_step:
                raise ValueError("dense grid must not be coarser")
            if not float(self.dense_end).is_integer():
                raise ValueError("dense_end must be an integer inventory level")


def inventory_grid(qmax, options):
    """Generate nested grids, retaining every integer arrival quantity."""
    if options.grid_kind == "uniform":
        return np.linspace(0,qmax,round(qmax/options.grid_step)+1)
    split = min(qmax,options.dense_end)
    dense = np.linspace(0,split,round(split/options.grid_step)+1)
    coarse = np.linspace(split,qmax,round((qmax-split)/options.coarse_step)+1)
    return np.unique(np.concatenate((dense,coarse)))


def gamma_interval_moments(lower, upper, shape, scale, degree=2):
    """Compute E[D^r 1(lower<=D<upper)] including the infinite upper tail.

    This is scale^r*(shape)_r times a Gamma(shape+r) CDF difference.
    Survival differences avoid loss of precision from subtracting 1-1.
    """
    if not (0 <= lower < upper and shape > 0 and scale > 0):
        raise ValueError("invalid Gamma interval")
    orders = np.arange(degree+1)
    shapes = shape+orders
    lo, hi = lower/scale, upper/scale
    probability = np.where(lo > shapes, gammaincc(shapes, lo)-gammaincc(shapes, hi),
                           gammainc(shapes, hi)-gammainc(shapes, lo))
    return scale**orders * poch(shape, orders) * probability


def expected_sales(stock, share, mean, cv):
    """Analytically compute E[min(stock, share*D)] for Gamma D."""
    stock = np.asarray(stock, dtype=np.float64)
    if share == 0:
        return np.zeros_like(stock)
    shape, scale = 1/cv**2, mean*cv**2
    threshold = stock/(share*scale)
    return share*mean*gammainc(shape+1, threshold)+stock*gammaincc(shape, threshold)


def expected_reward(stock, prices, shares, costs, mean, cv):
    """Expected reward before ordering cost, without rounding any sales."""
    stock = np.asarray(stock, dtype=np.float64)
    sales = np.array([expected_sales(stock[j], shares[j], mean, cv) for j in range(3)])
    residual = stock-sales
    lost = mean*np.sum(shares)-sales.sum()
    return float(np.dot(prices, sales)-costs["holding"]*residual[:2].sum()
                 -costs["perishable"]*residual[2]-costs["lost_sales"]*lost)


def residual_kernel(stock, shares, grid_step, levels, mean, cv, grid=None):
    """Integrate bilinear weights for the two surviving inventory ages.

    P_h(z|x,p)=E[w_z((x_0-lambda_0 D)^+,(x_1-lambda_1 D)^+)].
    These are value interpolation weights, not physically rounded inventories.
    """
    stock, shares = np.asarray(stock, dtype=float), np.asarray(shares, dtype=float)
    if stock.shape != (2,) or shares.shape != (2,) or np.any(stock < 0) or np.any(shares < 0):
        raise ValueError("two nonnegative stocks and shares are required")
    grid = np.arange(levels)*grid_step if grid is None else np.asarray(grid)
    if levels == 1:
        if np.any(stock != 0):
            raise ValueError("one-point grid only supports empty stock")
        return np.array([0],dtype=np.int32),np.array([1.]),{"mass_error_before_cleanup":0.,"minimum_before_cleanup":0.}
    boundaries = [0.]
    for x, lam in zip(stock, shares):
        if lam > 0 and x > 0:
            boundaries.extend(((x-grid[grid < x])/lam).tolist())
    boundaries = np.append(np.unique(boundaries), np.inf)
    shape, scale = 1/cv**2, mean*cv**2
    # Batch the special functions across all intervals for efficient refinement.
    lower,upper = boundaries[:-1],boundaries[1:]
    orders = np.arange(3)
    shapes = shape+orders
    lo,hi = lower[:,None]/scale,upper[:,None]/scale
    probability = np.where(lo > shapes,gammaincc(shapes,lo)-gammaincc(shapes,hi),
                           gammainc(shapes,hi)-gammainc(shapes,lo))
    moments = scale**orders*poch(shape,orders)*probability
    m0,m1,m2 = moments.T
    j1,j2 = m1-lower*m0,m2-2*lower*m1+lower**2*m0
    midpoint = np.where(np.isinf(upper),lower+1,(lower+upper)/2)
    residual_mid = np.maximum(stock-shares*midpoint[:,None],0)
    low = np.clip(np.searchsorted(grid,residual_mid,side="right")-1,0,levels-2)
    widths = grid[low+1]-grid[low]
    residual_lower = np.maximum(stock-shares*lower[:,None],0)
    fraction = np.clip((residual_lower-grid[low])/widths,0,1)
    active = residual_mid > 0
    fraction = np.where(active,fraction,0)
    slope = np.where(active,shares/widths,0)
    columns,masses = [],[]
    for i,j in product(range(2),repeat=2):
        a = fraction[:,0] if i else 1-fraction[:,0]
        b = -slope[:,0] if i else slope[:,0]
        c = fraction[:,1] if j else 1-fraction[:,1]
        d = -slope[:,1] if j else slope[:,1]
        columns.append((low[:,0]+i)*levels+low[:,1]+j)
        masses.append(a*c*m0+(a*d+b*c)*j1+b*d*j2)
    indices,inverse = np.unique(np.concatenate(columns),return_inverse=True)
    probabilities = np.bincount(inverse,weights=np.concatenate(masses))
    minimum = float(probabilities.min())
    error = float(abs(probabilities.sum()-1))
    if minimum < -1e-10 or error > 1e-9:
        raise ArithmeticError(f"invalid integrated row: min={minimum}, mass_error={error}")
    # Clean roundoff only; no demand cutoff or selected tail omission.
    probabilities = np.maximum(probabilities, 0)
    probabilities /= probabilities.sum()
    nonzero = probabilities > 0
    indices,probabilities = indices[nonzero].astype(np.int32),probabilities[nonzero]
    return indices, probabilities, {"mass_error_before_cleanup": error, "minimum_before_cleanup": minimum}


class GridDP:
    """Full-action baseline DP using sparse transition kernels."""

    def __init__(self, config=None, options=None):
        """Enumerate states/actions and build expected rewards and P_h."""
        started = time.perf_counter()
        self.config, self.options = config or InventoryConfig(), options or DPOptions()
        c, o = self.config, self.options
        if c.shelf_life != 3 or c.lead_time != 1:
            raise ValueError("Phase 4 DP supports M=3,L=1 only")
        self.grid = inventory_grid(c.qmax,o)
        self.levels = len(self.grid)
        self.orders = np.arange(c.qmax+1)
        self.price_indices = np.array(list(product(range(len(c.discount_percent)), repeat=3)), dtype=np.int64)
        self.prices = c.regular_price*(1-np.asarray(c.discount_percent)[self.price_indices]/100)
        self.shares = np.array([demand_shares(p, c.price_sensitivity, c.demand_price_scale) for p in self.prices])
        self.shape = (c.qmax+1, self.levels, self.levels)
        self.state_count, self.action_count = int(np.prod(self.shape)), len(self.orders)*len(self.prices)
        self.kernels, self.rewards = [], []
        self.kernel_diagnostics = {"max_mass_error_before_cleanup": 0., "minimum_before_cleanup": 0.}
        for price, shares in zip(self.prices, self.shares):
            rows, columns, probabilities = [], [], []
            for fresh in self.orders:
                for age2, stock2 in enumerate(self.grid):
                    indices, masses, audit = residual_kernel([fresh, stock2], shares[:2], o.grid_step, self.levels, c.demand_mean, c.demand_cv, self.grid)
                    rows.extend([fresh*self.levels+age2]*len(indices))
                    columns.extend(indices)
                    probabilities.extend(masses)
                    key = "max_mass_error_before_cleanup"
                    self.kernel_diagnostics[key] = max(self.kernel_diagnostics[key], audit["mass_error_before_cleanup"])
                    key = "minimum_before_cleanup"
                    self.kernel_diagnostics[key] = min(self.kernel_diagnostics[key], audit[key])
            self.kernels.append(sparse.csr_matrix((probabilities, (rows, columns)), shape=((c.qmax+1)*self.levels, self.levels**2)))
            s0 = expected_sales(self.orders, shares[0], c.demand_mean, c.demand_cv)[:,None,None]
            s1 = expected_sales(self.grid, shares[1], c.demand_mean, c.demand_cv)[None,:,None]
            s2 = expected_sales(self.grid, shares[2], c.demand_mean, c.demand_cv)[None,None,:]
            r0, r1, r2 = self.orders[:,None,None]-s0, self.grid[None,:,None]-s1, self.grid[None,None,:]-s2
            lost = c.demand_mean*shares.sum()-s0-s1-s2
            self.rewards.append(price[0]*s0+price[1]*s1+price[2]*s2-c.costs["holding"]*(r0+r1)-c.costs["perishable"]*r2-c.costs["lost_sales"]*lost)
        self.rewards = np.array(self.rewards)
        self.build_seconds = time.perf_counter()-started
        self.values = self.policy = self.continuations = self.summary = None

    def bellman(self, values, return_policy=False):
        """Apply T_h to all actions, factoring repeated order calculations."""
        values = np.asarray(values, dtype=float).reshape(self.shape)
        result = np.full(self.shape, -np.inf)
        policy = np.zeros(self.shape+(4,), dtype=np.int64) if return_policy else None
        for p, kernel in enumerate(self.kernels):
            continuation = (self.options.gamma*(kernel @ values.reshape(len(self.orders), -1).T)).reshape(len(self.orders), self.levels, len(self.orders))
            order_values = continuation-self.config.costs["ordering"]*self.orders
            best_orders = order_values.argmax(axis=2)
            candidate = self.rewards[p]+order_values.max(axis=2)[:,:,None]
            better = candidate > result
            result = np.maximum(result, candidate)
            if return_policy:
                policy[...,0][better] = np.broadcast_to(best_orders[:,:,None], self.shape)[better]
                policy[...,1:][better] = self.price_indices[p]
        return (result, policy) if return_policy else result

    def solve(self, initial_values=None):
        """Value-iterate and certify a residual for the projected MDP only."""
        started = time.perf_counter()
        values = np.zeros(self.shape) if initial_values is None else np.asarray(initial_values,dtype=float).copy()
        if values.shape != self.shape or not np.all(np.isfinite(values)):
            raise ValueError("invalid initial value array")
        history = []
        for iteration in range(1, self.options.max_iterations+1):
            next_values = self.bellman(values)
            delta = next_values-values
            bellman_change = float(np.max(np.abs(delta)))
            history.append(bellman_change)
            if self.options.value_shift_acceleration:
                # T(V+c)=T(V)+gamma*c. Center the next residual interval;
                # its norm is bounded by gamma*(max(delta)-min(delta))/2.
                gamma = self.options.gamma
                shift = gamma*(float(delta.max())+float(delta.min()))/(2*(1-gamma))
                stopping_metric = gamma*float(delta.max()-delta.min())/2
                next_values = next_values+shift
            else:
                stopping_metric = bellman_change
            change = float(np.max(np.abs(next_values-values)))
            values = next_values
            if stopping_metric <= self.options.bellman_tolerance:
                break
        else:
            raise RuntimeError("DP failed to converge within max_iterations")
        backed_up, self.policy = self.bellman(values, return_policy=True)
        residual = float(np.max(np.abs(backed_up-values)))
        self.values = values
        self.continuations = np.array([(self.options.gamma*(k @ values.reshape(len(self.orders), -1).T)).reshape(len(self.orders), self.levels, len(self.orders)) for k in self.kernels])
        self.summary = {"method": "grid_dp", "grid_step": self.options.grid_step,
                        "demand_price_basis":self.config.demand_price_basis,"demand_price_scale":self.config.demand_price_scale,
                        "grid_kind":self.options.grid_kind,"dense_end":self.options.dense_end,"coarse_step":self.options.coarse_step,
                        "value_shift_acceleration":self.options.value_shift_acceleration,
                        "states": self.state_count, "actions": self.action_count,
                        "iterations": iteration, "bellman_residual": residual,
                        "projected_value_error_bound": residual/(1-self.options.gamma),
                        "empty_state_discounted_value": float(values[0,0,0]),
                        "empty_state_action": self.policy[0,0,0].tolist(),
                        "build_seconds": self.build_seconds,
                        "solve_seconds": time.perf_counter()-started,
                        "kernel_diagnostics": self.kernel_diagnostics,
                        "last_iteration_change": change,
                        "last_bellman_change_before_shift":bellman_change,
                        "fidelity": "projected continuous model; author numerical reproduction unconfirmed"}
        return self.summary, history

    def action(self, state):
        """Choose an action on continuous stock using interpolated continuation.

        Reward expectations use the actual observed stock. Fresh arrivals are
        integer; only age2 continuation needs interpolation between grid rows.
        """
        if self.values is None:
            raise RuntimeError("solve before requesting actions")
        x = np.asarray(state, dtype=float)
        if x.shape != (3,) or not np.all(np.isfinite(x)) or np.any(x < 0) or np.any(x > self.config.qmax):
            raise ValueError("state is outside baseline domain")
        if not np.isclose(x[0], round(x[0]), rtol=0, atol=1e-9):
            raise ValueError("fresh stock must be integer at post-arrival decisions")
        fresh = round(x[0])
        low = min(max(int(np.searchsorted(self.grid,x[1],side="right"))-1,0),self.levels-1)
        high = min(low+1,self.levels-1)
        fraction = (x[1]-self.grid[low])/(self.grid[high]-self.grid[low]) if high > low else 0.
        continuation = (1-fraction)*self.continuations[:,fresh,low,:]+fraction*self.continuations[:,fresh,high,:]
        immediate = np.array([expected_reward(x, p, s, self.config.costs, self.config.demand_mean, self.config.demand_cv) for p, s in zip(self.prices, self.shares)])
        q_values = immediate[:,None]+continuation-self.config.costs["ordering"]*self.orders
        price_index, order = np.unravel_index(q_values.argmax(), q_values.shape)
        return np.concatenate(([order], self.price_indices[price_index])).astype(np.int64)

    def save(self, path):
        """Save values, policy, interpolation table and all settings."""
        if self.values is None:
            raise RuntimeError("solve before saving")
        np.savez_compressed(path, values=self.values, policy=self.policy,
                            continuations=self.continuations,
                            config_json=json.dumps(asdict(self.config)),
                            options_json=json.dumps(asdict(self.options)),
                            summary_json=json.dumps(self.summary))

    @classmethod
    def load(cls, path):
        """Load a solved policy without rebuilding the transition kernels."""
        with np.load(path, allow_pickle=False) as data:
            obj = cls.__new__(cls)
            config_values=json.loads(str(data["config_json"]))
            config_values.setdefault("demand_price_basis","absolute")
            obj.config = InventoryConfig(**config_values)
            option_values=json.loads(str(data["options_json"]))
            option_values.setdefault("grid_kind","uniform")
            option_values.setdefault("value_shift_acceleration",False)
            obj.options = DPOptions(**option_values)
            c, o = obj.config, obj.options
            obj.orders = np.arange(c.qmax+1)
            obj.grid = inventory_grid(c.qmax,o)
            obj.levels = len(obj.grid)
            obj.price_indices = np.array(list(product(range(len(c.discount_percent)), repeat=3)), dtype=np.int64)
            obj.prices = c.regular_price*(1-np.asarray(c.discount_percent)[obj.price_indices]/100)
            obj.shares = np.array([demand_shares(p, c.price_sensitivity, c.demand_price_scale) for p in obj.prices])
            obj.values, obj.policy = data["values"].copy(), data["policy"].copy()
            obj.continuations = data["continuations"].copy()
            obj.summary = json.loads(str(data["summary_json"]))
            obj.shape = obj.values.shape
            obj.state_count, obj.action_count = obj.summary["states"], obj.summary["actions"]
        return obj


def output_paths(config):
    """Keep results and checkpoints separated by the demand price convention."""
    label=f"{config.demand_price_basis}_price"
    paths={"checkpoints":Path("checkpoints")/label,
           "logs":Path("results/logs")/label,"tables":Path("results/tables")/label}
    for path in paths.values():
        path.mkdir(parents=True,exist_ok=True)
    return paths


def main():
    """Solve grid DP and save value/policy checkpoint, timing and residuals."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base_config.yaml")
    parser.add_argument("--dp-config", default="configs/dp_config.yaml")
    parser.add_argument("--grid-step", type=float)
    args = parser.parse_args()
    settings = yaml.safe_load(Path(args.dp_config).read_text(encoding="utf-8"))
    options_dict = {k:v for k,v in settings.items() if k in DPOptions.__dataclass_fields__}
    if args.grid_step is not None:
        options_dict["grid_step"] = args.grid_step
    options = DPOptions(**options_dict)
    dp = GridDP(InventoryConfig.from_yaml(args.config), options)
    paths=output_paths(dp.config)
    summary, history = dp.solve()
    tag = str(options.grid_step).replace(".", "p")
    dp.save(paths["checkpoints"]/f"dp_grid_{tag}.npz")
    (paths["logs"]/f"dp_grid_{tag}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savetxt(paths["tables"]/f"dp_convergence_{tag}.csv", np.column_stack((np.arange(1,len(history)+1),history)), delimiter=",", header="iteration,bellman_residual_before_update", comments="")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
