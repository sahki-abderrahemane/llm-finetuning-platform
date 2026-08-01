"""
Hyperparameter optimization.

Grid search over training hyperparameters. Each candidate is scored
by an objective function supplied by the caller.
"""

from __future__ import annotations

from collections.abc import Callable
from itertools import product
from typing import Any

Objective = Callable[[dict[str, Any]], float]


class HyperparameterOptimizer:
    """
    Grid search over a hyperparameter space.

    Example
    -------
    >>> optimizer = HyperparameterOptimizer()
    >>> best = optimizer.optimize(
    ...     search_space={"learning_rate": [1e-5, 2e-5], "rank": [4, 8]},
    ...     objective=lambda params: params["rank"],
    ... )
    """

    def __init__(self) -> None:
        pass

    def optimize(
        self,
        search_space: dict[str, list[Any]],
        objective: Objective,
        n_trials: int | None = None,
        maximize: bool = True,
    ) -> dict[str, Any]:
        """
        Run a grid search and return the best hyperparameter set.

        Parameters
        ----------
        search_space:
            Mapping of parameter name to candidate values.
        objective:
            Callable that scores a hyperparameter set (higher is
            better when ``maximize`` is True).
        n_trials:
            Optional cap on the number of evaluated combinations.
        maximize:
            Whether to maximize (True) or minimize (False) the score.
        """

        if not search_space:
            raise ValueError(
                "search_space must contain at least one parameter."
            )

        combinations = [
            dict(zip(search_space, values, strict=True))
            for values in product(*search_space.values())
        ]

        if n_trials is not None:
            combinations = combinations[:n_trials]

        if not combinations:
            raise ValueError(
                "search_space produced no combinations."
            )

        best_combo: dict[str, Any] | None = None
        best_score: float | None = None

        for combo in combinations:

            score = objective(combo)

            if best_score is None or (
                maximize and score > best_score
            ) or (
                not maximize and score < best_score
            ):
                best_score = score
                best_combo = combo

        assert best_combo is not None

        return best_combo
