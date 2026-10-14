"""Model-accounting, delivery timing and reproducibility acceptance tests."""

from dataclasses import replace
from itertools import product

from gymnasium.utils.env_checker import check_env
import numpy as np
import pytest

from env import InventoryConfig, InventoryEnv
from env.demand_model import demand_shares, sample_demand
from env.state_transition import transition


def test_gymnasium_contract():
    """Check reset/step API, observation spaces and seeded determinism."""
    check_env(InventoryEnv(), skip_render_check=True)


def test_hand_calculated_ledger():
    """Check sales, waste, aging and all four costs against a hand calculation."""
    state, reward, info = transition([4, 3, 2], 5, [10, 7, 5], [2, 4, 1], 3, 1, InventoryConfig().costs)
    # Sales [2,3,1] => revenue 46. Carry 2, waste 1, shortage 1.
    # Costs: order 15 + hold 2 + lost 5 + waste 7 = 29; profit 17.
    np.testing.assert_array_equal(state, [5, 2, 0])
    np.testing.assert_array_equal(info["sales_by_age"], [2, 3, 1])
    assert info["costs"] == {"ordering": 15, "holding": 2, "lost_sales": 5, "perishable": 7}
    assert info["revenue"] == 46 and reward == 17


@pytest.mark.parametrize("lead", [1, 2, 3])
def test_delivery_not_early_or_late(lead):
    """An order at t=0 first becomes sellable at t=lead, never earlier."""
    state = np.zeros(3 + lead - 1)
    for t in range(lead):
        state, _, info = transition(state, 6 if t == 0 else 0, [10]*3, [0]*3, 3, lead, InventoryConfig().costs)
        assert info["sales_by_age"].sum() == 0
        assert state[0] == (6 if t == lead - 1 else 0)
        assert state.sum() == 6
    _, _, info = transition(state, 0, [10]*3, [8, 0, 0], 3, lead, InventoryConfig().costs)
    assert info["sales_by_age"][0] == 6


def test_three_full_selling_periods_and_expiry():
    """A cohort remains sellable for M periods and expires only after the last."""
    s = np.array([6., 0, 0])
    for expected, waste in [([0, 6, 0], 0), ([0, 0, 6], 0), ([0, 0, 0], 6)]:
        s, _, info = transition(s, 0, [10]*3, [0]*3, 3, 1, InventoryConfig().costs)
        np.testing.assert_array_equal(s, expected)
        assert info["expired"] == waste
        assert info["held"] == (0 if waste else 6)


def test_stockouts_no_substitution():
    """Empty fresh stock does not consume old stock to fulfill fresh demand."""
    s, _, info = transition([0, 8, 4], 0, [10, 7, 5], [2, 0, 0], 3, 1, InventoryConfig().costs)
    assert info["lost_sales"] == 2 and info["sales_by_age"].sum() == 0
    np.testing.assert_array_equal(s, [0, 0, 8])
    assert info["expired"] == 4


def test_mnl_outside_option_and_prices():
    """Check equal-utility shares, price sensitivity and baseline discount mapping."""
    np.testing.assert_array_equal(demand_shares([0]*3, [.3, .5, .7]), [.25]*3)
    w = np.exp(-np.array([.3, .5, .7])*[10, 7, 5])
    np.testing.assert_allclose(demand_shares([10, 7, 5], [.3, .5, .7]), w/(1+w.sum()))
    assert demand_shares([5]*3, [.3, .5, .7])[0] > demand_shares([10]*3, [.3, .5, .7])[0]
    e = InventoryEnv()
    e.reset(seed=42)
    _, _, _, _, info = e.step(np.array([0, 0, 1, 2]))
    np.testing.assert_array_equal(info["prices"], [10, 7, 5])
    np.testing.assert_allclose(info["demand_prices"],[1,.7,.5])
    normalized_weights=np.exp(-np.array([.3,.5,.7])*[1.,.7,.5])
    np.testing.assert_allclose(info["demand_shares"],normalized_weights/(1+normalized_weights.sum()))
    assert info["demand_by_age"].sum()+info["outside_demand"] == pytest.approx(info["total_demand"])


def test_gamma_moments():
    """Validate Gamma mean and CV empirically with a fixed large sample."""
    rng = np.random.default_rng(42)
    samples = np.array([sample_demand(rng, 4, .5) for _ in range(100000)])
    assert samples.min() >= 0
    assert abs(samples.mean()-4) < .03
    assert abs(samples.std()/samples.mean()-.5) < .01


def test_all_baseline_actions_conservation_and_rewards():
    """Exercise every baseline action against accounting invariants."""
    e = InventoryEnv()
    for values in product(range(11), range(3), range(3), range(3)):
        before, _ = e.reset(seed=42, options={"initial_state": [4., 3., 2.]})
        after, reward, _, _, info = e.step(np.array(values))
        assert e.observation_space.contains(after)
        assert after.sum() == pytest.approx(before.sum()+values[0]-info["sales_by_age"].sum()-info["expired"])
        assert reward == pytest.approx(info["revenue"]-sum(info["costs"].values()))
        assert np.all(info["sales_by_age"] <= before)


def test_seed_and_checkpoint_resume(tmp_path):
    """A restored checkpoint reproduces actions, demand, states and profit."""
    a = InventoryEnv()
    a.reset(seed=42)
    for _ in range(30):
        a.step(a.action_space.sample())
    path = tmp_path/"simulation.json"
    a.save_checkpoint(path)
    b = InventoryEnv.from_checkpoint(path)
    for _ in range(20):
        aa, ab = a.action_space.sample(), b.action_space.sample()
        np.testing.assert_array_equal(aa, ab)
        sa, ra, ta, xa, ia = a.step(aa)
        sb, rb, tb, xb, ib = b.step(ab)
        np.testing.assert_array_equal(sa, sb)
        assert (ra, ta, xa, ia["total_demand"]) == (rb, tb, xb, ib["total_demand"])


def test_horizon_and_observation_copy():
    """Time limits truncate and caller edits cannot mutate internal inventory."""
    e = InventoryEnv(replace(InventoryConfig(), episode_steps=2))
    with pytest.raises(RuntimeError):
        e.step(np.array([0, 0, 0, 0]))
    s, _ = e.reset(seed=42, options={"initial_state": [2, 0, 0]})
    s[:] = 0
    _, _, terminated, truncated, info = e.step(np.array([0, 0, 0, 0]))
    assert info["sales_by_age"][0] > 0 and not terminated and not truncated
    _, _, terminated, truncated, _ = e.step(np.array([0, 0, 0, 0]))
    assert not terminated and truncated
    with pytest.raises(RuntimeError):
        e.step(np.array([0, 0, 0, 0]))


@pytest.mark.parametrize("action", [[-1,0,0,0], [11,0,0,0], [0,3,0,0], [1.5,0,0,0], [0,0,0]])
def test_invalid_actions(action):
    """Reject invalid quantities, indices, nonintegers and wrong dimensions."""
    e = InventoryEnv()
    e.reset(seed=42)
    with pytest.raises(ValueError):
        e.step(action)


@pytest.mark.parametrize("state", [[-1,0,0], [11,0,0], [np.nan,0,0], [0,0]])
def test_invalid_initial_state(state):
    """Reject negative, excessive, nonfinite or misshaped initial inventory."""
    with pytest.raises(ValueError):
        InventoryEnv().reset(options={"initial_state": state})


@pytest.mark.parametrize("values", [{"lead_time":0}, {"shelf_life":2}, {"demand_cv":0}, {"qmax":2.5}, {"costs":{"holding":1}}, {"demand_price_basis":"unknown"}])
def test_invalid_config(values):
    """Reject invalid lead time, dimensions and economic parameters."""
    with pytest.raises(ValueError):
        InventoryConfig(**values)


def test_baseline_yaml():
    """Project YAML loads the chosen timing and parameter settings."""
    c = InventoryConfig.from_yaml("configs/base_config.yaml")
    e = InventoryEnv(c)
    assert e.observation_space.shape == (3,)
    assert np.prod(e.action_space.nvec) == 297
    assert c.demand_price_basis == "relative"


def test_relative_price_scaling_and_actual_cash_revenue():
    """Only demand uses p/list_price; cash receipts keep actual monetary units."""
    config=InventoryConfig(costs={"holding":0.,"ordering":0.,"lost_sales":0.,"perishable":0.})
    env=InventoryEnv(config)
    env.reset(seed=42,options={"initial_state":[10.,10.,10.]})
    _,reward,_,_,info=env.step(np.array([0,0,1,2]))
    np.testing.assert_allclose(info["demand_prices"],[1.,.7,.5])
    assert reward==pytest.approx(np.dot([10.,7.,5.],info["sales_by_age"]))
    assert reward != pytest.approx(np.dot([1.,.7,.5],info["sales_by_age"]))


def test_absolute_price_mode_and_legacy_environment_checkpoint(tmp_path):
    """Historical checkpoints remain absolute-price models after migration."""
    import json
    env=InventoryEnv(replace(InventoryConfig(),demand_price_basis="absolute"))
    env.reset(seed=42)
    _,_,_,_,info=env.step(np.array([1,0,0,0]))
    assert info["demand_shares"].sum()==pytest.approx(.05431709209069076)
    path=tmp_path/"legacy.json"
    env.save_checkpoint(path)
    data=json.loads(path.read_text())
    del data["config"]["demand_price_basis"]
    path.write_text(json.dumps(data))
    restored=InventoryEnv.from_checkpoint(path)
    assert restored.config.demand_price_basis=="absolute"
    a,b=env.step(np.array([0,0,0,0])),restored.step(np.array([0,0,0,0]))
    np.testing.assert_array_equal(a[0],b[0])
    assert a[1]==b[1]


def test_price_scale_and_sensitivity_unit_equivalence():
    """Changing price units alone requires inverse adjustment of sensitivity."""
    b=np.array([.3,.5,.7])
    np.testing.assert_allclose(demand_shares([10.,7.,5.],b,10.),demand_shares([1.,.7,.5],b))
    np.testing.assert_allclose(demand_shares([10.,7.,5.],b),demand_shares([1.,.7,.5],10*b))
    assert demand_shares([10.]*3,b,10.).sum()==pytest.approx(.6483744224600235)


@pytest.mark.parametrize("scale",[0.,-1.,np.inf,np.nan])
def test_invalid_price_scale(scale):
    """Reject scales that would make demand utility undefined."""
    with pytest.raises(ValueError):
        demand_shares([10.]*3,[.3,.5,.7],scale)
