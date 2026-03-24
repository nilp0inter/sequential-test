import warnings
import numpy as np
from sequential_test import sequential_test


def coin_generator(coin_fn, n):
    for _ in range(n):
        yield coin_fn()


def baseline_generator(p, n):
    for _ in range(n):
        yield np.random.binomial(n=1, p=p)


def test_is_coin_fair(coin_function, max_samples, alpha, sigma, lower_bound):
    y = coin_generator(coin_function, max_samples)
    x = baseline_generator(0.50, max_samples)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = sequential_test(
            x=y,
            y=x,
            theta=0.0,
            sigma=sigma,
            alpha=alpha,
            distribution='bernoulli',
            lower_bound=lower_bound,
        )

    passed = result.decision != 'Accept H1'
    return passed, result.number_of_observations


if __name__ == "__main__":
    import argparse
    import random

    parser = argparse.ArgumentParser(description="Test coin fairness using sequential testing")
    parser.add_argument("-n", "--trials", type=int, default=100, help="number of trials (default: 100)")
    parser.add_argument("-s", "--samples", type=int, default=10000, help="max samples per trial (default: 10000)")
    parser.add_argument("-a", "--alpha", type=float, default=0.01, help="significance level (default: 0.01)")
    parser.add_argument("--sigma", type=float, default=0.5, help="standard deviation (default: 0.5)")
    parser.add_argument("--lower-bound", type=float, default=0.05, help="lower decision boundary (default: 0.05)")
    parser.add_argument("--rigged-prob", type=float, default=0.58, help="rigged coin heads probability (default: 0.58)")
    args = parser.parse_args()

    make_fair = lambda: 1 if random.random() < 0.50 else 0
    make_rigged = lambda: 1 if random.random() < args.rigged_prob else 0

    fair_results = [test_is_coin_fair(make_fair, args.samples, args.alpha, args.sigma, args.lower_bound) for _ in range(args.trials)]
    fair_pass = sum(p for p, _ in fair_results)
    fair_obs = [n for _, n in fair_results]

    rigged_results = [test_is_coin_fair(make_rigged, args.samples, args.alpha, args.sigma, args.lower_bound) for _ in range(args.trials)]
    rigged_fail = sum(not p for p, _ in rigged_results)
    rigged_obs = [n for _, n in rigged_results]

    print(f"Fair coin:   {fair_pass}/{args.trials} passed   (samples: {np.mean(fair_obs):.1f} +/- {np.std(fair_obs):.1f})")
    print(f"Rigged coin: {rigged_fail}/{args.trials} detected (samples: {np.mean(rigged_obs):.1f} +/- {np.std(rigged_obs):.1f})")
