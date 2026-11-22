"""Return pre-processing and (weighted) correlation estimation.

Ports of EWMA_Wts.m, covMatWtd.m, QIS_Wtd.m, preProcessRets.m and
correlToDistMetric.m, plus the distance/similarity block that is repeated in
getFilteredNetwork.m and getClusters.m.

Conventions: returns are ``(n_dates, p_assets)`` arrays, time weights are
``(n_dates,)`` arrays.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist, squareform


def _check_weights(weights, n_dates):
    """Return weights rescaled to sum to one (equal weights if ``None``)."""
    if weights is None or np.size(weights) == 0:
        return np.full(n_dates, 1.0 / n_dates)
    w = np.asarray(weights, dtype=float).ravel()
    if w.size != n_dates:
        raise ValueError(f"Expected {n_dates} time weights, got {w.size}.")
    return w / w.sum()


def is_equal_weighted(weights) -> bool:
    w = np.asarray(weights, dtype=float).ravel()
    return bool(np.allclose(w, 1.0 / w.size))


def ewma_weights(n_dates: int, alpha: float):
    """Exponentially weighted time weights (Pozzi, Di Matteo & Aste, 2012).

    ``w_t ∝ exp(alpha * (t - T))``; ``alpha == 0`` gives equal weights.

    Returns
    -------
    weights : (n_dates,) array summing to one
    effective_obs : float, effective sample size ``1 / sum(w**2)``
    """
    if alpha == 0:
        w = np.full(n_dates, 1.0 / n_dates)
    else:
        w = np.exp(alpha * (np.arange(1, n_dates + 1) - n_dates))
    w = w / w.sum()
    return w, 1.0 / np.sum(w**2)


def cov_weighted(returns, weights=None):
    """Weighted covariance matrix.

    Returns
    -------
    cov : (p, p) array
    effective_obs : float
    """
    x = np.asarray(returns, dtype=float)
    n = x.shape[0]
    w = _check_weights(weights, n)
    demeaned = x - w @ x
    cov = demeaned.T @ (demeaned * w[:, None])
    cov = 0.5 * (cov + cov.T)
    return cov, 1.0 / np.sum(w**2)


def qis_weighted(returns, weights=None):
    """Weighted covariance with Ledoit–Wolf Quadratic-Inverse Shrinkage (QIS).

    Adapted from Ledoit & Wolf's ``QIS.m`` (covShrinkage), using the
    effective number of observations implied by the time weights.
    """
    x = np.asarray(returns, dtype=float)
    n_dates, p = x.shape
    w = _check_weights(weights, n_dates)
    sample, n_eff = cov_weighted(x, w)

    k = 1  # data are demeaned in the covariance calculation
    n = n_eff - k
    c = p / n
    lam, u = np.linalg.eigh(sample)
    order = np.argsort(lam)
    lam, u = lam[order], u[:, order]

    n_int = int(np.floor(n))  # integer version for index arithmetic
    h = min(c**2, 1.0 / c**2) ** 0.35 / p**0.35
    m = min(p, n_int)
    invlam = 1.0 / lam[max(0, p - n_int):p]
    Lj = np.tile(invlam, (m, 1))          # Lj[i, j] = 1/lambda_j
    Lj_i = Lj - Lj.T                       # 1/lambda_j - 1/lambda_i
    denom = Lj_i**2 + h**2 * Lj**2
    theta = np.mean(Lj * Lj_i / denom, axis=1)
    Htheta = np.mean(Lj * (h * Lj) / denom, axis=1)
    Atheta2 = theta**2 + Htheta**2

    if p <= n:
        delta = 1.0 / ((1 - c) ** 2 * invlam
                       + 2 * c * (1 - c) * invlam * theta
                       + c**2 * invlam * Atheta2)
    else:
        delta0 = 1.0 / ((c - 1) * np.mean(invlam))
        delta = np.concatenate([np.full(p - m, delta0), 1.0 / (invlam * Atheta2)])

    delta_qis = delta * (lam.sum() / delta.sum())
    sigma = (u * delta_qis) @ u.T
    return 0.5 * (sigma + sigma.T)


def cov_to_corr(cov):
    """Convert a covariance matrix to a correlation matrix (like ``corrcov``)."""
    sd = np.sqrt(np.diag(cov))
    corr = cov / np.outer(sd, sd)
    corr = 0.5 * (corr + corr.T)
    np.fill_diagonal(corr, 1.0)
    return corr


def corr_to_dist(corr):
    """Mantegna (1999) distance ``d_ij = sqrt(2 (1 - rho_ij))``."""
    return np.sqrt(np.clip(2.0 * (1.0 - np.asarray(corr)), 0.0, None))


def _weighted_std(x, w):
    mu = w @ x
    return np.sqrt(w @ (x - mu) ** 2)


def _weighted_pca(x, w):
    """Weighted PCA, matching MATLAB ``pca(X, 'Weights', w)``.

    Returns coefficients (columns = components, MATLAB sign convention),
    scores and percentage of variance explained.
    """
    w = w / w.sum()
    mu = w @ x
    xc = x - mu
    cov = xc.T @ (xc * w[:, None])
    lam, vec = np.linalg.eigh(cov)
    order = np.argsort(lam)[::-1]
    lam, vec = lam[order], vec[:, order]
    # MATLAB convention: largest-magnitude element of each coefficient is positive
    idx = np.argmax(np.abs(vec), axis=0)
    signs = np.sign(vec[idx, np.arange(vec.shape[1])])
    signs[signs == 0] = 1
    vec = vec * signs
    scores = xc @ vec
    explained = 100.0 * lam / lam.sum()
    return vec, scores, explained


def preprocess_returns(returns, weights=None, standardise=False, remove_market_mode=False):
    """Standardise returns and/or remove the market mode (1st principal component).

    Returns
    -------
    final_rets : (n, p) processed returns
    mkt_mode_rets : (n,) market-mode factor returns (``None`` if not removed)
    check_mkt_mode : share of assets loading positively on PC1 (``None`` if not removed)
    perc_expl_mkt_mode : % variance explained by PC1 (``None`` if not removed)
    """
    x = np.asarray(returns, dtype=float)
    n, p = x.shape
    w = _check_weights(weights, n)

    if standardise:
        rets1 = (x - w @ x) / _weighted_std(x, w)
    else:
        rets1 = x

    mkt, check, perc = None, None, None
    if remove_market_mode:
        coeff, scores, explained = _weighted_pca(rets1, w)
        check = np.sum(coeff[:, 0] > 0) / p
        perc = explained[0]
        mkt = -scores[:, 0] if check < 0.5 else scores[:, 0]
        sw = np.sqrt(w)
        X = np.column_stack([sw, sw * mkt])
        beta, *_ = np.linalg.lstsq(X, sw[:, None] * rets1, rcond=None)
        rets2 = rets1 - (beta[0] + np.outer(mkt, beta[1]))
    else:
        rets2 = rets1

    if standardise:
        final = (rets2 - w @ rets2) / _weighted_std(rets2, w)
    else:
        final = rets2
    return final, mkt, check, perc


def similarity_and_distance(processed_rets, weights, dist_method):
    """Correlation, distance and similarity matrices for a distance method.

    ``dist_method``:
      * ``'correlDistMetric'``      – weighted sample correlation → Mantegna distance
      * ``'QIS_correlDistMetric'``  – weighted QIS correlation → Mantegna distance
      * any ``scipy.spatial.distance.pdist`` metric (e.g. ``'euclidean'``,
        ``'cityblock'``) computed on the asset return series

    For the correlation distances ``S = 2 - D**2 / 2 = 1 + rho``. For other
    metrics ``S = 1 / (1 + D)``.

    Returns
    -------
    corr, D, S : (p, p) arrays
    """
    if dist_method == "QIS_correlDistMetric":
        cov = qis_weighted(processed_rets, weights)
    else:
        cov, _ = cov_weighted(processed_rets, weights)
    corr = cov_to_corr(cov)

    if dist_method in ("correlDistMetric", "QIS_correlDistMetric"):
        D = corr_to_dist(corr)
        S = 2.0 - 0.5 * D**2
    else:
        D = squareform(pdist(np.asarray(processed_rets).T, dist_method))
        S = 1.0 / (1.0 + D)
    S = 0.5 * (S + S.T)
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)
    return corr, D, S


def rolling_end_indices(n_dates: int, window: int, step: int):
    """End positions (exclusive) of rolling windows, oldest first.

    Matches the MATLAB loop: windows end at the last date and step back by
    ``step`` while a full window fits.
    """
    ends = []
    k = 0
    while n_dates - k * step >= window:
        ends.append(n_dates - k * step)
        k += 1
    return ends[::-1]
