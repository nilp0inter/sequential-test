from dataclasses import dataclass
from typing import Iterable

from scipy.stats import norm
import numpy as np


@dataclass
class SequentialTestResult:
    distribution: str
    number_of_observations: int
    likelihood_ratios: list[float]
    stopping_time: int
    decision: str
    text: str
    alpha: float

    def __str__(self) -> str:
        return (
            f"Distribution: {self.distribution!r},\n"
            f"number of observations: {self.number_of_observations!r},\n"
            f"stopping time: {self.stopping_time!r},\n"
            f"decision: {self.decision!r},\n"
            f"text: {self.text!r},\n"
            f"alpha: {self.alpha!r}"
        )


def calculate_mixture_variance(alpha: float, sigma: float, truncation: float) -> float:
    if not isinstance(alpha, float) and not (alpha > 0 and alpha < 1):
        raise ValueError("Alpha must be between 0 and 1")

    b = (2 * np.log(1 / alpha)) / np.sqrt(truncation * sigma**2)

    return round(
        sigma**2 * (norm.cdf(-b) / ((1 / b) * norm.pdf(b) - norm.cdf(-b))), 2
    )


def _validate_inputs(
    sigma: float,
    tau: float,
    theta: float,
    distribution: str,
    alpha: float,
) -> None:
    if distribution == "normal" and not isinstance(sigma, float):
        raise TypeError("sigma must be numeric")
    if not isinstance(theta, float):
        raise TypeError("theta must be numeric")
    if not isinstance(tau, float):
        raise TypeError("tau must be numeric")
    if not tau > 0:
        raise ValueError("tau must be positive")
    if not isinstance(alpha, float):
        raise TypeError("alpha must be numeric")
    if not (alpha > 0 and alpha < 1):
        raise ValueError("alpha value has to be between 0 and 1")
    if distribution not in ("normal", "bernoulli"):
        raise ValueError("Distribution should be either 'normal' or 'bernoulli'")


def calculate_test_statistics(
    x: Iterable,
    y: Iterable,
    sigma: float,
    tau: float,
    theta: float,
    distribution: str,
    alpha: float,
    warmup_observations: int,
    lower_bound: float | None = None,
) -> SequentialTestResult:
    upper_bound = 1.0 / alpha
    likelihood_ratios = []
    sum_x = 0.0
    sum_y = 0.0
    sum_z = 0.0
    stopping_time = None

    for i_zero, (xi, yi) in enumerate(zip(x, y)):
        i = i_zero + 1
        sum_x += xi
        sum_y += yi
        sum_z += (xi - yi)

        if i_zero < warmup_observations:
            likelihood_ratios.append(0)
            continue

        mean_x = sum_x / i
        mean_y = sum_y / i
        mean_z = sum_z / i

        if distribution == "bernoulli":
            Vn = mean_x * (1 - mean_x) + mean_y * (1 - mean_y)
            if Vn == 0:
                likelihood_ratios.append(0)
                continue
            lr = np.sqrt(Vn / (Vn + i * tau**2)) * np.exp(
                (i**2 * tau**2 * (mean_z - theta) ** 2)
                / (2 * Vn * (Vn + i * tau**2))
            )
        elif distribution == "normal":
            double_variance = 2 * sigma**2
            lr = np.sqrt(double_variance / (double_variance + i * tau**2)) * np.exp(
                (i**2 * tau**2 * (mean_x - mean_y - theta) ** 2)
                / (4 * sigma**2 * (2 * sigma**2 + i * tau**2))
            )

        likelihood_ratios.append(lr)

        if lr > upper_bound:
            stopping_time = i
            decision = "Accept H1"
            break

        if lower_bound is not None and lr < lower_bound:
            stopping_time = i
            decision = "Accept H0"
            break

    number_of_observations = len(likelihood_ratios)

    if stopping_time is None:
        stopping_time = number_of_observations
        decision = "Accept H0"

    text = f"Decision made after {stopping_time} observations were collected"

    return SequentialTestResult(
        distribution=distribution,
        number_of_observations=number_of_observations,
        likelihood_ratios=likelihood_ratios,
        stopping_time=stopping_time,
        decision=decision,
        text=text,
        alpha=alpha,
    )


def sequential_test(
    x: Iterable,
    y: Iterable,
    sigma: float = 0.0,
    theta: float = 0.0,
    truncation: float = 200,
    distribution: str = "normal",
    alpha: float = 0.05,
    warmup_observations: int = 100,
    lower_bound: float | None = None,
) -> SequentialTestResult:
    mixture_variance = calculate_mixture_variance(
        alpha=alpha, sigma=sigma, truncation=truncation
    )
    _validate_inputs(
        sigma=sigma,
        tau=mixture_variance,
        theta=theta,
        distribution=distribution,
        alpha=alpha,
    )
    return calculate_test_statistics(
        x=x,
        y=y,
        sigma=sigma,
        tau=mixture_variance,
        theta=theta,
        distribution=distribution,
        alpha=alpha,
        warmup_observations=warmup_observations,
        lower_bound=lower_bound,
    )
