"""Probability distribution registry.

Each distribution is described by a :class:`DistSpec` that exposes its
parameters (for auto-generated sliders), a domain generator, and the PMF/PDF and
CDF callables backed by ``scipy.stats``. The UI iterates over this registry so
adding a distribution requires no UI changes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class Param:
    """A single tunable distribution parameter (drives a slider)."""

    name: str
    label: str
    min: float
    max: float
    default: float
    step: float
    integer: bool = False


@dataclass
class DistSpec:
    """Full description of a distribution used for plotting and exploration."""

    key: str
    name: str
    kind: str  # "discrete" or "continuous"
    params: list[Param]
    _dist: Callable[..., object]
    summary: str = ""

    # -- construction --------------------------------------------------------
    def frozen(self, values: dict[str, float]):
        """Return a frozen scipy distribution from a dict of parameter values."""
        return self._dist(values)

    def support(self, values: dict[str, float], n: int = 200) -> np.ndarray:
        """Return sensible x / k values spanning the distribution's mass."""
        dist = self.frozen(values)
        if self.kind == "discrete":
            lo = int(np.floor(dist.ppf(0.001)))
            hi = int(np.ceil(dist.ppf(0.999)))
            lo = max(lo, 0)
            hi = max(hi, lo + 1)
            return np.arange(lo, hi + 1)
        lo, hi = dist.ppf(0.001), dist.ppf(0.999)
        if not np.isfinite(lo):
            lo = dist.ppf(0.01)
        if not np.isfinite(hi):
            hi = dist.ppf(0.99)
        return np.linspace(lo, hi, n)

    def pmf_pdf(self, x: np.ndarray, values: dict[str, float]) -> np.ndarray:
        """Probability mass (discrete) or density (continuous)."""
        dist = self.frozen(values)
        return dist.pmf(x) if self.kind == "discrete" else dist.pdf(x)

    def cdf(self, x: np.ndarray, values: dict[str, float]) -> np.ndarray:
        """Cumulative distribution function."""
        return self.frozen(values).cdf(x)

    def moments(self, values: dict[str, float]) -> dict[str, float]:
        """Mean, variance, skewness and (excess) kurtosis."""
        m, v, s, k = self.frozen(values).stats(moments="mvsk")
        return {"Mean": float(m), "Variance": float(v),
                "Skewness": float(s), "Kurtosis": float(k)}


# --- Registry --------------------------------------------------------------
# Discrete -------------------------------------------------------------------
DISCRETE: dict[str, DistSpec] = {
    "bernoulli": DistSpec(
        "bernoulli", "Bernoulli", "discrete",
        [Param("p", "Success probability p", 0.0, 1.0, 0.5, 0.01)],
        lambda v: stats.bernoulli(v["p"]),
        "Single trial with success probability p.",
    ),
    "binomial": DistSpec(
        "binomial", "Binomial", "discrete",
        [Param("n", "Trials n", 1, 100, 20, 1, integer=True),
         Param("p", "Success probability p", 0.0, 1.0, 0.5, 0.01)],
        lambda v: stats.binom(int(v["n"]), v["p"]),
        "Number of successes in n independent Bernoulli trials.",
    ),
    "geometric": DistSpec(
        "geometric", "Geometric", "discrete",
        [Param("p", "Success probability p", 0.01, 1.0, 0.3, 0.01)],
        lambda v: stats.geom(v["p"]),
        "Number of trials until the first success.",
    ),
    "negbinom": DistSpec(
        "negbinom", "Negative Binomial", "discrete",
        [Param("n", "Successes r", 1, 50, 5, 1, integer=True),
         Param("p", "Success probability p", 0.01, 1.0, 0.4, 0.01)],
        lambda v: stats.nbinom(int(v["n"]), v["p"]),
        "Number of failures before the r-th success.",
    ),
    "hypergeom": DistSpec(
        "hypergeom", "Hypergeometric", "discrete",
        [Param("M", "Population size M", 2, 200, 50, 1, integer=True),
         Param("n", "Successes in population n", 1, 100, 20, 1, integer=True),
         Param("N", "Sample size N", 1, 100, 10, 1, integer=True)],
        lambda v: stats.hypergeom(int(v["M"]), int(min(v["n"], v["M"])),
                                  int(min(v["N"], v["M"]))),
        "Successes in N draws without replacement from a finite population.",
    ),
    "poisson": DistSpec(
        "poisson", "Poisson", "discrete",
        [Param("mu", "Rate λ", 0.1, 50.0, 5.0, 0.1)],
        lambda v: stats.poisson(v["mu"]),
        "Count of events in a fixed interval at constant average rate λ.",
    ),
}

# Continuous -----------------------------------------------------------------
CONTINUOUS: dict[str, DistSpec] = {
    "uniform": DistSpec(
        "uniform", "Uniform", "continuous",
        [Param("loc", "Lower a", -10.0, 10.0, 0.0, 0.1),
         Param("scale", "Width (b - a)", 0.1, 20.0, 1.0, 0.1)],
        lambda v: stats.uniform(v["loc"], v["scale"]),
        "Constant density on the interval [a, a + width].",
    ),
    "normal": DistSpec(
        "normal", "Normal", "continuous",
        [Param("loc", "Mean μ", -20.0, 20.0, 0.0, 0.1),
         Param("scale", "Std dev σ", 0.1, 20.0, 1.0, 0.1)],
        lambda v: stats.norm(v["loc"], v["scale"]),
        "The classic bell curve parameterised by mean and standard deviation.",
    ),
    "exponential": DistSpec(
        "exponential", "Exponential", "continuous",
        [Param("scale", "Scale (1/λ)", 0.1, 20.0, 1.0, 0.1)],
        lambda v: stats.expon(scale=v["scale"]),
        "Waiting time between events in a Poisson process.",
    ),
    "gamma": DistSpec(
        "gamma", "Gamma", "continuous",
        [Param("a", "Shape k", 0.1, 20.0, 2.0, 0.1),
         Param("scale", "Scale θ", 0.1, 20.0, 1.0, 0.1)],
        lambda v: stats.gamma(v["a"], scale=v["scale"]),
        "Sum of k exponential variables; flexible right-skewed family.",
    ),
    "beta": DistSpec(
        "beta", "Beta", "continuous",
        [Param("a", "Alpha α", 0.1, 20.0, 2.0, 0.1),
         Param("b", "Beta β", 0.1, 20.0, 2.0, 0.1)],
        lambda v: stats.beta(v["a"], v["b"]),
        "Distribution on [0, 1]; conjugate prior for proportions.",
    ),
    "lognormal": DistSpec(
        "lognormal", "Lognormal", "continuous",
        [Param("s", "Shape σ", 0.05, 3.0, 0.5, 0.05),
         Param("scale", "Scale exp(μ)", 0.1, 20.0, 1.0, 0.1)],
        lambda v: stats.lognorm(v["s"], scale=v["scale"]),
        "Variable whose logarithm is normally distributed.",
    ),
    "weibull": DistSpec(
        "weibull", "Weibull", "continuous",
        [Param("c", "Shape k", 0.1, 10.0, 1.5, 0.1),
         Param("scale", "Scale λ", 0.1, 20.0, 1.0, 0.1)],
        lambda v: stats.weibull_min(v["c"], scale=v["scale"]),
        "Reliability / survival modelling with tunable hazard shape.",
    ),
}

ALL_DISTRIBUTIONS: dict[str, DistSpec] = {**DISCRETE, **CONTINUOUS}


def get_distribution(key: str) -> DistSpec:
    """Look up a distribution spec by key."""
    return ALL_DISTRIBUTIONS[key]
