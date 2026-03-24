import warnings
import numpy as np
from sequential_test import sequential_test


def coin_generator(coin_fn, n):
    for _ in range(n):
        yield coin_fn()


def baseline_generator(p, n):
    for _ in range(n):
        yield np.random.binomial(n=1, p=p)


def test_is_coin_fair(coin_function, max_samples=10000):
    """
    TDD Test to verify if a stochastic coin function is "Fair".

    :param coin_function: A callable function that returns 1 (Heads) or 0 (Tails).
    :param max_samples: The maximum iterations the CI/CD pipeline will wait before timing out.
    """

    # ---------------------------------------------------------
    # 1. Product Owner Requirements
    # ---------------------------------------------------------
    target_probability = 0.50
    tolerance_threshold = 0.05
    target_accuracy = 0.99

    # ---------------------------------------------------------
    # 2. Data Generation (as generators)
    # ---------------------------------------------------------
    y = coin_generator(coin_function, max_samples)

    np.random.seed(42)
    x = baseline_generator(target_probability, max_samples)

    # ---------------------------------------------------------
    # 3. Execute the Sequential Test
    # ---------------------------------------------------------
    alpha_value = 1.0 - target_accuracy

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # SWAPPED X AND Y:
        # x = live code (what we are testing)
        # y = perfect baseline (the target)
        # This inverts the logic so "Accept H1" means "Code is Different (Broken)"

        result = sequential_test(
            x=y,       # Live code (treatment)
            y=x,       # Perfect target (control)
            theta=0.0, # We expect ZERO difference between them
            sigma=0.5, # StdDev for Bernoulli distribution
            alpha=alpha_value,
            distribution='bernoulli',
            lower_bound=0.05,  # Lower decision boundary for early H0 acceptance
        )

    # ---------------------------------------------------------
    # 4. TDD Assertions (Green / Red Outcomes)
    # ---------------------------------------------------------

    # The 'decision' attribute tells us the final result
    # 'Accept H1' means: "I found evidence that X and Y are DIFFERENT" (BROKEN)
    # Anything else means: "I found no evidence they are different" (PASS)

    decision = getattr(result, 'decision', None)

    if decision == 'Accept H1':
        # The math found proof that the code deviates from the target
        print(f"❌ FAIL: The coin is rigged! (Rejected H0).")
        print(f"   The code deviated past the {tolerance_threshold*100}% boundaries.")
        print(f"   Stopped early after {result.number_of_observations} of {max_samples} samples.")
        print(result)
        return False, "Coin function is outside acceptable probability bounds."

    else:
        # We reached max_samples without finding evidence of a bug
        print(f"✅ PASS: No statistical evidence of bias found after {result.number_of_observations} samples.")
        print(result)
        return True, "Coin function is within acceptable probability bounds."


# =====================================================================
# EXAMPLE USAGE:
# =====================================================================
if __name__ == "__main__":
    import random

    # Scenario 1: Perfect Coin
    def perfect_coin():
        return 1 if random.random() < 0.50 else 0

    print("Testing Perfect Coin...")
    test_is_coin_fair(perfect_coin)
    print("-" * 40)

    # Scenario 2: Broken Coin (70% heads)
    def broken_coin():
        return 1 if random.random() < 0.58 else 0

    print("Testing Broken Coin...")
    test_is_coin_fair(broken_coin)
