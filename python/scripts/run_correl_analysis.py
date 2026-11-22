"""Distribution of correlations over the full period and dynamically, with and
without the market mode (scr_RunCorrelAnalysis.m)."""
import _common  # noqa: F401
import matplotlib.pyplot as plt

import netfilter as nf

data = nf.load_data()
rets, names, dates = data["rets"], data["names"], data["dates"]

inputs = dict(
    standardise=True,
    ewma_alpha=0.005,                        # {0, 0.005}
    window=52 * 3,                           # 3 years lookback
    step=4,                                  # one month step
)
roll_wts, _ = nf.ewma_weights(inputs["window"], inputs["ewma_alpha"])

# ---------------- Static analysis (equal time weights) ----------------
static = {
    "normal":         nf.correlation_analysis(rets, None, names, inputs["standardise"], False, "Normal", None, True),
    "normal_QIS":     nf.correlation_analysis(rets, None, names, inputs["standardise"], False, "QIS", None, True),
    "mktModeRem":     nf.correlation_analysis(rets, None, names, inputs["standardise"], True, "Normal", None, True),
    "mktModeRem_QIS": nf.correlation_analysis(rets, None, names, inputs["standardise"], True, "QIS", None, True),
}
for k, v in static.items():
    print(f"{k:15s} average correlation {v['corrAve']:.3f}  quartiles {v['corrDistSummary'].round(3)}")

# ---------------- Dynamic analysis (EWMA weights if required) ----------------
dynamic = {
    "normal":     nf.correlation_analysis_dynamic(rets, dates, roll_wts, inputs["window"], inputs["step"],
                                                  names, inputs["standardise"], False, "Normal", True),
    "mktModeRem": nf.correlation_analysis_dynamic(rets, dates, roll_wts, inputs["window"], inputs["step"],
                                                  names, inputs["standardise"], True, "Normal", True),
}
plt.show()
