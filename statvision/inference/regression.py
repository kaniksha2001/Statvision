"""Linear regression (simple & multiple) backed by statsmodels OLS.

Statsmodels is used rather than scikit-learn because it exposes the full
inferential apparatus — standard errors, t-statistics, p-values, R² and
adjusted R² — that an SPSS-style toolkit is expected to report.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm


def fit_ols(df: pd.DataFrame, target: str, predictors: list[str]) -> dict:
    """Fit an ordinary-least-squares model and return a rich result bundle.

    Works for both simple (one predictor) and multiple regression. Rows with
    missing values in the modelled columns are dropped listwise.
    """
    cols = [target] + predictors
    clean = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    y = clean[target]
    X = sm.add_constant(clean[predictors], has_constant="add")
    model = sm.OLS(y, X).fit()

    coef_table = pd.DataFrame(
        {
            "Coefficient": model.params,
            "Std Error": model.bse,
            "t": model.tvalues,
            "P>|t|": model.pvalues,
            "CI Lower": model.conf_int()[0],
            "CI Upper": model.conf_int()[1],
        }
    )

    fitted = model.fittedvalues
    residuals = model.resid
    return {
        "model": model,
        "target": target,
        "predictors": predictors,
        "r_squared": float(model.rsquared),
        "adj_r_squared": float(model.rsquared_adj),
        "f_statistic": float(model.fvalue),
        "f_pvalue": float(model.f_pvalue),
        "aic": float(model.aic),
        "bic": float(model.bic),
        "n_obs": int(model.nobs),
        "coefficients": coef_table,
        "fitted": np.asarray(fitted),
        "actual": np.asarray(y),
        "residuals": np.asarray(residuals),
        "equation": _equation(model, target, predictors),
        "summary_text": model.summary().as_text(),
    }


def predict(result: dict, new_data: dict[str, float]) -> float:
    """Predict the target for a single observation given predictor values."""
    model = result["model"]
    row = {"const": 1.0}
    row.update({p: new_data[p] for p in result["predictors"]})
    X = pd.DataFrame([row])[["const"] + result["predictors"]]
    return float(model.predict(X)[0])


def _equation(model, target: str, predictors: list[str]) -> str:
    """Render a human readable regression equation string."""
    parts = [f"{model.params['const']:.4f}"]
    for p in predictors:
        coef = model.params[p]
        sign = "+" if coef >= 0 else "-"
        parts.append(f" {sign} {abs(coef):.4f}·{p}")
    return f"{target} = " + "".join(parts)
