"""Distribution of pairwise correlations. Ports of correlationAnalysis.m and
correlationAnalysis_Dynamic.m."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .estimation import (_check_weights, cov_to_corr, cov_weighted, preprocess_returns,
                         qis_weighted, rolling_end_indices)
from .network import _progress, _subtitle
from . import plotting


def ksdensity(x, xi=None, n_points=100):
    """Gaussian kernel density estimate, following MATLAB ``ksdensity`` defaults.

    Bandwidth: ``sigma * (4 / (3 n))**(1/5)`` with ``sigma = MAD / 0.6745``.
    Default evaluation points: 100 points spanning the data range ± 3 bandwidths.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    sig = np.median(np.abs(x - np.median(x))) / 0.6745
    if sig <= 0:
        sig = (x.max() - x.min()) or 1.0
    bw = sig * (4.0 / (3.0 * n)) ** 0.2
    if xi is None or np.size(xi) == 0:
        xi = np.linspace(x.min() - 3 * bw, x.max() + 3 * bw, n_points)
    xi = np.asarray(xi, dtype=float).ravel()
    z = (xi[:, None] - x[None, :]) / bw
    f = np.exp(-0.5 * z**2).sum(axis=1) / (n * bw * np.sqrt(2 * np.pi))
    return f, xi


def correlation_analysis(returns, weights, names, standardise=True, remove_market_mode=False,
                         method="Normal", dist_xi=None, plot=False):
    """Summary of the cross-correlations in the (weighted) correlation matrix.

    Parameters
    ----------
    method : ``'Normal'`` (sample) or ``'QIS'`` (Ledoit–Wolf shrinkage)
    dist_xi : points at which to evaluate the kernel density (default automatic)

    Returns
    -------
    dict with ``covMat``, ``corrMat``, ``corrValues`` (lower-triangular entries),
    ``corrFi``/``corrXi`` (density), ``corrDistSummary`` (min, quartiles, max),
    ``corrAve`` and ``inputs``.
    """
    returns = np.asarray(returns, dtype=float)
    n_dates, p = returns.shape
    w = _check_weights(weights, n_dates)
    processed, *_ = preprocess_returns(returns, w, standardise, remove_market_mode)
    cov = qis_weighted(processed, w) if method == "QIS" else cov_weighted(processed, w)[0]
    corr = cov_to_corr(cov)
    # MATLAB column-major order of the strictly lower triangle
    cols, rows = np.nonzero(np.tril(np.ones((p, p), dtype=bool), -1).T)
    vals = corr[rows, cols]
    fi, xi = ksdensity(vals, dist_xi)
    summary = np.percentile(vals, [0, 25, 50, 75, 100], method="hazen")
    ave = vals.mean()

    if plot:
        plotting.plot_density(xi, fi, ave, "Distribution of Correlations",
                              _subtitle(method, w, standardise, remove_market_mode))
    return {"covMat": cov, "corrMat": corr, "corrValues": vals, "corrFi": fi, "corrXi": xi,
            "corrDistSummary": summary, "corrAve": ave,
            "inputs": {"inReturns": returns, "processedRets": processed, "corrMat": corr,
                       "inNames": list(names), "method": method, "inTimeWts": w,
                       "standardise": standardise, "remove_market_mode": remove_market_mode}}


def correlation_analysis_dynamic(returns, dates, weights, window, step, names, standardise=True,
                                 remove_market_mode=False, method="Normal", plot=False,
                                 progress=True):
    """Run :func:`correlation_analysis` on rolling windows.

    Densities are evaluated at the 41 points ``-1, -0.95, ..., 1``.
    """
    returns = np.asarray(returns, dtype=float)
    dates = pd.DatetimeIndex(pd.to_datetime(dates))
    n_dates, p = returns.shape
    w = _check_weights(weights, window)
    ends = rolling_end_indices(n_dates, window, step)
    T = len(ends)
    xi = np.round(np.arange(-1, 1.00001, 0.05), 10)
    res = {"dates": [], "corrValues": np.zeros((p * (p - 1) // 2, T)),
           "corrDistFi": np.zeros((len(xi), T)), "corrAve": np.zeros(T),
           "corrDistSummary": np.zeros((5, T)), "inReturns": [], "processedRets": [],
           "corrDistXi": xi}
    for t, end in enumerate(_progress(ends, progress, "Running dynamic correlation analysis")):
        tmp = correlation_analysis(returns[end - window:end], w, names, standardise,
                                   remove_market_mode, method, xi, False)
        res["inReturns"].append(tmp["inputs"]["inReturns"])
        res["processedRets"].append(tmp["inputs"]["processedRets"])
        res["dates"].append(dates[end - 1])
        res["corrValues"][:, t] = tmp["corrValues"]
        res["corrDistFi"][:, t] = tmp["corrFi"]
        res["corrAve"][t] = tmp["corrAve"]
        res["corrDistSummary"][:, t] = tmp["corrDistSummary"]

    res["dates"] = pd.DatetimeIndex(res["dates"])
    if plot:
        plotting.plot_density_contour(res["dates"], xi, res["corrDistFi"],
                                      "Dynamic Correlations: Distribution",
                                      _subtitle(method, w, standardise, remove_market_mode,
                                                "Distance"))
    return {"dynamicCorrelations": res,
            "inputs": {"inReturns": returns, "window": window, "step": step, "inTimeWts": w,
                       "standardise": standardise, "remove_market_mode": remove_market_mode,
                       "method": method}}
