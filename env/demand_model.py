"""Gamma aggregate demand and age-dependent MNL shares (paper Eqs. 1–2)."""

import numpy as np


def demand_shares(prices, sensitivity, price_scale=1.0, available=None, purchase_utility=0.0):
    """Return MNL shares from actual prices divided by an explicit unit scale.

    price_scale=regular_price means relative price (1,.7,.5). The numeric
    sensitivities remain fixed under the user-selected interpretation.
    price_scale=1 retains the archived absolute-price interpretation.
    Outside-option utility remains zero, including stockouts.
    """
    prices = np.asarray(prices, dtype=np.float64)
    sensitivity = np.asarray(sensitivity, dtype=np.float64)
    if prices.ndim != 1 or prices.shape != sensitivity.shape:
        raise ValueError("prices and sensitivity must be matching vectors")
    if not np.all(np.isfinite(prices)) or not np.all(np.isfinite(sensitivity)):
        raise ValueError("inputs must be finite")
    if np.any(prices < 0) or np.any(sensitivity < 0):
        raise ValueError("prices and sensitivity must be nonnegative")
    if not np.isscalar(price_scale) or not np.isfinite(price_scale) or price_scale <= 0:
        raise ValueError("price_scale must be finite and positive")
    # Eq. (1): lambda_j = exp(-b_j*p_j)/(1 + sum(exp(-b_k*p_k))).
    if not np.isfinite(purchase_utility):
        raise ValueError("purchase utility must be finite")
    utility = purchase_utility - sensitivity * (prices / price_scale)
    if available is not None:
        available = np.asarray(available, dtype=bool)
        if available.shape != prices.shape:
            raise ValueError("availability must match prices")
        utility = np.where(available, utility, -np.inf)
    shift = max(0.0, np.max(utility, initial=-np.inf))
    weights = np.exp(utility - shift)
    return weights / (np.exp(-shift) + weights.sum())


def sample_demand(rng, mean, cv):
    """Sample Gamma demand with shape=CV^-2 and scale=mean*CV^2."""
    if not np.isfinite(mean) or not np.isfinite(cv) or mean <= 0 or cv <= 0:
        raise ValueError("Gamma mean and CV must be finite and positive")
    return float(rng.gamma(shape=1.0 / cv**2, scale=mean * cv**2))
