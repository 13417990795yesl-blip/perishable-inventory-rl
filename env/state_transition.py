"""Pure inventory dynamics, independently testable without random sampling."""

import numpy as np


def transition(state, order, prices, age_demand, shelf_life, lead_time, costs):
    """Sell, charge costs, expire, age, then receive orders before next sales.

    State is [fresh,...,oldest, transit_due_in_1,...,transit_due_in_L-1].
    Decision observations already include arrivals for the current period.
    For L=1, the current order becomes next period's fresh inventory.
    """
    state = np.asarray(state, dtype=np.float64)
    prices = np.asarray(prices, dtype=np.float64)
    age_demand = np.asarray(age_demand, dtype=np.float64)
    if state.shape != (shelf_life + lead_time - 1,):
        raise ValueError("incorrect state shape")
    if prices.shape != (shelf_life,) or age_demand.shape != prices.shape:
        raise ValueError("incorrect demand or price shape")
    if any(not np.all(np.isfinite(x)) or np.any(x < 0) for x in [state, prices, age_demand]):
        raise ValueError("state, demand and prices must be finite and nonnegative")
    if not np.isfinite(order) or order < 0:
        raise ValueError("order must be finite and nonnegative")
    sales = np.minimum(state[:shelf_life], age_demand)
    residual = state[:shelf_life] - sales
    lost_sales = float(np.maximum(age_demand - state[:shelf_life], 0).sum())
    expired = float(residual[-1])
    held = float(residual[:-1].sum())
    revenue = float(np.dot(prices, sales))
    # Eqs. (9–10): holding excludes expired stock, arrivals, and the pipeline.
    cost_terms = {"ordering": float(costs["ordering"] * order),
                  "holding": float(costs["holding"] * held),
                  "lost_sales": float(costs["lost_sales"] * lost_sales),
                  "perishable": float(costs["perishable"] * expired)}
    # Queue advances at period boundary: order_t is first sellable at t+L.
    pipeline = np.concatenate((state[shelf_life:], [order]))
    arrived = float(pipeline[0])
    next_state = np.concatenate(([arrived], residual[:-1], pipeline[1:]))
    info = {"sales_by_age": sales.copy(), "demand_by_age": age_demand.copy(),
            "revenue": revenue, "costs": cost_terms, "total_cost": sum(cost_terms.values()),
            "lost_sales": lost_sales, "expired": expired, "held": held, "arrived": arrived,
            "order": float(order), "prices": prices.copy()}
    return next_state, revenue - info["total_cost"], info
