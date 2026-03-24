# Sequential Test

A Python library for sequential hypothesis testing. Useful for A/B testing and other scenarios where you want to reach a decision as early as the data allows, without waiting for a fixed sample size.

## Installation

```bash
pip install sequential-test
```

Requires Python >=3.10, <3.13.

## Dependencies

- NumPy
- SciPy

## Usage

```python
from sequential_test import sequential_test
```

### Normal distribution (default)

```python
result = sequential_test(x=x, y=y, sigma=1.0)
```

### Bernoulli distribution

```python
result = sequential_test(
    x=x,
    y=y,
    theta=0.0,
    sigma=0.5,
    distribution="bernoulli",
    alpha=0.01,
    lower_bound=0.05,
)
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `x` | Iterable | required | Treatment observations |
| `y` | Iterable | required | Control observations |
| `sigma` | float | `0.0` | Known standard deviation (used for normal distribution) |
| `theta` | float | `0.0` | Hypothesized difference under H0 |
| `truncation` | float | `200` | Maximum number of observations for mixture variance calculation |
| `distribution` | str | `"normal"` | `"normal"` or `"bernoulli"` |
| `alpha` | float | `0.05` | Significance level |
| `warmup_observations` | int | `100` | Observations to collect before testing begins |
| `lower_bound` | float \| None | `None` | Lower decision boundary for early H0 acceptance. When set, the test stops early if the likelihood ratio drops below this value. |

### Result

`sequential_test()` returns a `SequentialTestResult` with the following fields:

| Field | Type | Description |
|---|---|---|
| `distribution` | str | Distribution used |
| `number_of_observations` | int | Total observations processed |
| `likelihood_ratios` | list[float] | Likelihood ratio at each step |
| `stopping_time` | int | Observation at which a decision was reached |
| `decision` | str | `"Accept H1"` or `"Accept H0"` |
| `text` | str | Human-readable summary |
| `alpha` | float | Significance level used |

## How it works

### Mixture variance

$$
\tau^2 = \sigma^2 \frac{\Phi(-b)}{\frac{1}{b}\phi(b)-\Phi(-b)}
$$

### Test statistic (normal)

$$
\tilde{\Lambda}_n = \sqrt{\frac{2\sigma^2}{2\sigma^2 + n\tau^2}}\exp\left(\frac{n^2\tau^2(\bar{X}_n - \bar{Y}_n-\theta_0)^2}{4\sigma^2(2\sigma^2+n\tau^2)}\right)
$$

### Test statistic (Bernoulli)

$$
\tilde{\Lambda}_n = \sqrt{\frac{V_n}{V_n + n\tau^2}}\exp{\left(\frac{n^2\tau^2(\bar{X}_n - \bar{Y}_n-\theta_0)^2}{2V_n(V_n+n\tau^2)}\right)}
$$

The test rejects H0 when the likelihood ratio exceeds $1/\alpha$. If `lower_bound` is set, it accepts H0 early when the ratio drops below that threshold.

### Differences from standard mSPRT

This library extends the classic mSPRT in two ways:

- **Lower decision boundary** — standard mSPRT only tests against an upper threshold ($1/\alpha$) and can only reject H0. The optional `lower_bound` parameter adds a lower threshold for early H0 acceptance, closer to a two-sided SPRT.
- **Warmup period** — standard mSPRT evaluates from the first observation. The `warmup_observations` parameter skips a configurable number of initial observations to avoid noisy early decisions.

## References

1. Wald, A. (1945). Sequential Tests of Statistical Hypotheses. *The Annals of Mathematical Statistics*, 16(2), 117–186. [DOI: 10.1214/aoms/1177731118](https://doi.org/10.1214/aoms/1177731118)
2. Robbins, H. (1970). Statistical Methods Related to the Law of the Iterated Logarithm. *The Annals of Mathematical Statistics*, 41(5), 1397–1409. [DOI: 10.1214/aoms/1177696786](https://doi.org/10.1214/aoms/1177696786)
3. Johari, R., Koomen, P., Pekelis, L., & Walsh, D. (2022). Always Valid Inference: Continuous Monitoring of A/B Tests. *Operations Research*, 70(3), 1806–1821. [DOI: 10.1287/opre.2021.2135](https://doi.org/10.1287/opre.2021.2135)
4. Stenberg, E. mixtureSPRT — R and C++ implementation. [GitHub](https://github.com/erik-stenberg/mixtureSPRT)
5. Kuzminas, O. msprt — original Python implementation. [GitHub](https://github.com/ovidijusku/msprt/tree/main)

## License

This project is licensed under the GPL-3.0-or-later license.
