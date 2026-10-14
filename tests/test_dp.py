"""Independent integration, exhaustive-action and solver convergence checks."""

from dataclasses import replace
from itertools import product

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import gamma as gamma_distribution

from algorithms.dp import (DPOptions, GridDP, expected_reward, expected_sales,
                           gamma_interval_moments, residual_kernel, inventory_grid)
from env import InventoryConfig, InventoryEnv
from env.demand_model import demand_shares
from env.state_transition import transition


@pytest.mark.parametrize("interval", [(0., 1.), (1., 7.), (7., np.inf), (35., np.inf)])
def test_gamma_moments_independent_quad(interval):
    """Compare analytic moments, including a tiny tail, to quadrature."""
    distribution = gamma_distribution(a=4., scale=1.)
    actual = gamma_interval_moments(*interval, 4., 1.)
    expected = [quad(lambda d: d**r*distribution.pdf(d), *interval,
                     epsabs=1e-14, epsrel=1e-10)[0] for r in range(3)]
    np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-13)


@pytest.mark.parametrize("stock,share", [(0., .2), (.15, .2), (2., .2), (2., 0.)])
def test_expected_sales_independent_quad(stock, share):
    """Check E[min(x,lambda D)] through independent numerical integration."""
    distribution = gamma_distribution(a=4., scale=1.)
    points = [0., stock/share, np.inf] if stock > 0 and share > 0 else [0., np.inf]
    expected = sum(quad(lambda d: min(stock,share*d)*distribution.pdf(d), lo, hi,
                        epsabs=1e-12)[0] for lo,hi in zip(points[:-1],points[1:]))
    assert expected_sales(stock,share,4.,.5) == pytest.approx(expected, abs=1e-10)


def test_expected_reward_independent_transition_quadrature():
    """Integrate the Environment's physical ledger against Gamma density."""
    c = InventoryConfig()
    stock, prices = np.array([.1,.2,.05]), np.array([10.,7.,5.])
    shares = demand_shares(prices,c.price_sensitivity,c.demand_price_scale)
    distribution = gamma_distribution(a=4.,scale=1.)
    points = sorted([0., *list(stock/shares), np.inf])
    expected = sum(quad(lambda d: transition(stock,0,prices,d*shares,3,1,c.costs)[1]*distribution.pdf(d), lo,hi,
                        epsabs=1e-11)[0] for lo,hi in zip(points[:-1],points[1:]))
    assert expected_reward(stock,prices,shares,c.costs,4.,.5) == pytest.approx(expected,abs=1e-9)


@pytest.mark.parametrize("stock,shares", [([0.,0.],[.3,.2]), ([1.,.5],[.1,.05]),
                                         ([1.,.5],[.3,.2]), ([.8,.3],[.2,.1])])
def test_kernel_mass_means_and_joint_moment(stock,shares):
    """Check row sums and first/cross moments against direct integration."""
    h, levels = .25, 5
    indices, probabilities, _ = residual_kernel(stock,shares,h,levels,4.,.5)
    assert np.all(probabilities >= 0)
    assert probabilities.sum() == pytest.approx(1.,abs=1e-14)
    states = np.column_stack((indices//levels,indices%levels))*h
    expected_means = [stock[j]-expected_sales(stock[j],shares[j],4.,.5) for j in range(2)]
    np.testing.assert_allclose(probabilities @ states, expected_means, atol=1e-11)
    distribution = gamma_distribution(a=4.,scale=1.)
    limit = min(x/lam for x,lam in zip(stock,shares))
    cross = quad(lambda d: max(stock[0]-shares[0]*d,0)*max(stock[1]-shares[1]*d,0)*distribution.pdf(d),0,limit,
                 epsabs=1e-12)[0] if limit > 0 else 0.
    assert probabilities @ (states[:,0]*states[:,1]) == pytest.approx(cross,abs=1e-11)


def test_kernel_arbitrary_values_against_piecewise_quadrature():
    """Compare an arbitrary bilinear value surface using unrelated quadrature."""
    h, levels, stock, shares = .5, 5, [2.,1.5], [.3,.2]
    values = np.random.default_rng(123).normal(size=(levels,levels))
    ids, probs, _ = residual_kernel(stock,shares,h,levels,4.,.5)
    distribution = gamma_distribution(a=4.,scale=1.)
    def integrand(d):
        """Evaluate interpolation directly at a sampled demand."""
        point = np.maximum(np.array(stock)-np.array(shares)*d,0)/h
        low = np.minimum(np.floor(point).astype(int),levels-2)
        fraction = point-low
        total = 0.
        for i,j in product(range(2),repeat=2):
            wi = fraction[0] if i else 1-fraction[0]
            wj = fraction[1] if j else 1-fraction[1]
            total += wi*wj*values[low[0]+i,low[1]+j]
        return total*distribution.pdf(d)
    bounds = sorted(set([0., *[(x-k*h)/lam for x,lam in zip(stock,shares) for k in range(round(x/h)+1)], np.inf]))
    expected = sum(quad(integrand,a,b,epsabs=1e-11)[0] for a,b in zip(bounds[:-1],bounds[1:]))
    assert probs @ values.ravel()[ids] == pytest.approx(expected,abs=1e-9)


@pytest.fixture(scope="module")
def tiny_solver():
    """A small enough model for fully independent exhaustive checks."""
    return GridDP(replace(InventoryConfig(),qmax=2), DPOptions(grid_step=1., grid_kind="uniform", value_shift_acceleration=False, gamma=.7, bellman_tolerance=1e-9))


def test_factored_bellman_equals_all_state_action_enumeration(tiny_solver):
    """Compare optimized backup with explicit enumeration of every action."""
    dp = tiny_solver
    values = np.random.default_rng(42).normal(size=dp.shape)
    expected = np.full(dp.shape,-np.inf)
    for i,j,k in product(range(3),repeat=3):
        for p in range(len(dp.prices)):
            row = dp.kernels[p].getrow(i*dp.levels+j)
            for q in dp.orders:
                candidate = dp.rewards[p,i,j,k]-dp.config.costs["ordering"]*q+dp.options.gamma*(row @ values[q].ravel()).item()
                expected[i,j,k] = max(expected[i,j,k],candidate)
    np.testing.assert_allclose(dp.bellman(values),expected,atol=1e-12)


def test_value_solution_vs_direct_policy_linear_system(tiny_solver,tmp_path):
    """Validate convergence against an independently solved policy equation."""
    dp = tiny_solver
    summary,_ = dp.solve()
    count = dp.state_count
    matrix, rewards = np.zeros((count,count)),np.zeros(count)
    for i,j,k in product(range(3),repeat=3):
        row = np.ravel_multi_index((i,j,k),dp.shape)
        action = dp.policy[i,j,k]
        q = action[0]
        p = np.where(np.all(dp.price_indices==action[1:],axis=1))[0][0]
        kernel = dp.kernels[p].getrow(i*dp.levels+j)
        matrix[row,q*dp.levels**2+kernel.indices] = kernel.data
        rewards[row] = dp.rewards[p,i,j,k]-dp.config.costs["ordering"]*q
    direct = np.linalg.solve(np.eye(count)-dp.options.gamma*matrix,rewards)
    np.testing.assert_allclose(dp.values.ravel(),direct,atol=summary["projected_value_error_bound"]+1e-10,rtol=0)
    assert summary["bellman_residual"] <= dp.options.bellman_tolerance
    warm_summary,_ = dp.solve(initial_values=direct.reshape(dp.shape))
    assert warm_summary["iterations"] == 1
    checkpoint = tmp_path/"dp.npz"
    dp.save(checkpoint)
    loaded = GridDP.load(checkpoint)
    for state in [[0,0,0],[1,.3,.7],[2,2,2]]:
        np.testing.assert_array_equal(dp.action(state),loaded.action(state))
        assert InventoryEnv(dp.config).action_space.contains(loaded.action(state))
    for i,j,k in product(range(3),repeat=3):
        np.testing.assert_array_equal(dp.action([i,j,k]),dp.policy[i,j,k])


def test_zero_order_capacity_closed_form():
    """For qmax=0, compare to the exact infinite discounted shortage cost."""
    dp = GridDP(replace(InventoryConfig(),qmax=0),DPOptions(grid_step=1.,grid_kind="uniform",gamma=.5,bellman_tolerance=1e-10))
    summary,_ = dp.solve()
    exact = -dp.config.costs["lost_sales"]*dp.config.demand_mean*dp.shares.sum(axis=1).min()/(1-dp.options.gamma)
    assert summary["empty_state_discounted_value"] == pytest.approx(exact,abs=1e-9)


@pytest.mark.parametrize("options", [{"grid_step":0}, {"grid_step":.3}, {"gamma":1.},
                                   {"bellman_tolerance":0}, {"max_iterations":0}])
def test_invalid_dp_options(options):
    """Reject invalid numerical settings before building large kernels."""
    with pytest.raises(ValueError):
        DPOptions(**options)


def test_unimplemented_dimensions_fail_explicitly():
    """Prevent silently applying the baseline DP to a different pipeline."""
    with pytest.raises(ValueError):
        GridDP(replace(InventoryConfig(),lead_time=2))


def test_nonuniform_kernel_preserves_moments():
    """Check the refined grid covers the domain and preserves means/correlation."""
    options = DPOptions(grid_step=.0625,grid_kind="near_zero_refined",dense_end=2.,coarse_step=.25)
    grid = inventory_grid(3,options)
    assert grid[0]==0 and grid[-1]==3
    assert all(x in grid for x in range(4))
    x,shares = [3.,2.125],[.3,.15]
    ids,probs,_ = residual_kernel(x,shares,options.grid_step,len(grid),4.,.5,grid)
    points = np.column_stack((grid[ids//len(grid)],grid[ids%len(grid)]))
    means = np.array(x)-[expected_sales(x[j],shares[j],4.,.5) for j in range(2)]
    np.testing.assert_allclose(probs @ points,means,atol=1e-11)
    distribution=gamma_distribution(a=4,scale=1)
    limit=min(x[j]/shares[j] for j in range(2))
    cross=quad(lambda d: (x[0]-shares[0]*d)*(x[1]-shares[1]*d)*distribution.pdf(d),0,limit,epsabs=1e-11)[0]
    assert probs @ (points[:,0]*points[:,1]) == pytest.approx(cross,abs=1e-10)


def test_shift_acceleration_preserves_bellman_solution(tiny_solver):
    """Compare accelerated VI with ordinary VI for the same finite MDP."""
    plain = tiny_solver
    plain.solve()
    accelerated = GridDP(plain.config,replace(plain.options,value_shift_acceleration=True))
    summary,_ = accelerated.solve()
    bound = summary["projected_value_error_bound"]+plain.summary["projected_value_error_bound"]+1e-10
    np.testing.assert_allclose(accelerated.values,plain.values,atol=bound,rtol=0)
    residual = np.max(np.abs(accelerated.bellman(accelerated.values)-accelerated.values))
    assert residual <= accelerated.options.bellman_tolerance


@pytest.mark.parametrize("stock", [[1.,.015625],[10.,10.],[0.,.0078125]])
@pytest.mark.parametrize("prices", [[10.,7.,5.],[5.,5.,5.]])
def test_final_mesh_with_baseline_shares(stock,prices):
    """Check final-mesh integrated means/correlation at baseline MNL scales."""
    c=InventoryConfig()
    options=DPOptions(grid_step=1/128,grid_kind="near_zero_refined")
    grid=inventory_grid(c.qmax,options)
    shares=demand_shares(prices,c.price_sensitivity,c.demand_price_scale)[:2]
    ids,probs,_=residual_kernel(stock,shares,options.grid_step,len(grid),4.,.5,grid)
    points=np.column_stack((grid[ids//len(grid)],grid[ids%len(grid)]))
    means=np.array(stock)-[expected_sales(stock[j],shares[j],4.,.5) for j in range(2)]
    np.testing.assert_allclose(probs @ points,means,rtol=0,atol=1e-10)
    limit=min(stock[j]/shares[j] for j in range(2))
    distribution=gamma_distribution(a=4,scale=1)
    expected=quad(lambda d: (stock[0]-shares[0]*d)*(stock[1]-shares[1]*d)*distribution.pdf(d),
                  0,limit,epsabs=1e-10,epsrel=1e-11)[0] if limit else 0.
    assert probs @ (points[:,0]*points[:,1]) == pytest.approx(expected,abs=1e-9)


def test_dp_relative_price_matches_environment(tiny_solver):
    """DP and Environment use the same normalized utility, without double scaling."""
    dp=tiny_solver
    env=InventoryEnv(dp.config)
    env.reset(seed=42)
    _,_,_,_,info=env.step(np.array([0,0,0,0]))
    np.testing.assert_allclose(dp.shares[0],info["demand_shares"])
    assert dp.shares[0].sum()==pytest.approx(.6483744224600235)


def test_legacy_dp_checkpoint_keeps_absolute_price(tmp_path):
    """A historical checkpoint without the basis field is not silently converted."""
    import json
    old=GridDP(replace(InventoryConfig(),qmax=0,demand_price_basis="absolute"),
               DPOptions(grid_step=1.,grid_kind="uniform",gamma=.5))
    old.solve()
    path=tmp_path/"old.npz"
    old.save(path)
    with np.load(path,allow_pickle=False) as saved:
        payload={key:saved[key].copy() for key in saved.files}
    config=json.loads(str(payload["config_json"]))
    del config["demand_price_basis"]
    payload["config_json"]=json.dumps(config)
    np.savez_compressed(path,**payload)
    restored=GridDP.load(path)
    assert restored.config.demand_price_basis=="absolute"
    np.testing.assert_allclose(restored.shares,old.shares)
    np.testing.assert_array_equal(restored.action([0.,0.,0.]),old.action([0.,0.,0.]))
