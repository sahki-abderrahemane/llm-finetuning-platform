"""
Tests for the hyperparameter optimizer.
"""

from __future__ import annotations

from mentorai_finetuning.experiments.hpo import (
    HyperparameterOptimizer,
)


def test_grid_search_finds_maximum():
    """The best combination must be selected when maximizing."""

    optimizer = HyperparameterOptimizer()

    best = optimizer.optimize(
        search_space={
            "learning_rate": [1e-5, 2e-5],
            "rank": [4, 8],
        },
        objective=lambda params: params["rank"] * params["learning_rate"],
        maximize=True,
    )

    assert best == {
        "learning_rate": 2e-5,
        "rank": 8,
    }


def test_grid_search_finds_minimum():
    """The best combination must be selected when minimizing."""

    optimizer = HyperparameterOptimizer()

    best = optimizer.optimize(
        search_space={
            "learning_rate": [1e-5, 2e-5],
        },
        objective=lambda params: params["learning_rate"],
        maximize=False,
    )

    assert best == {"learning_rate": 1e-5}


def test_n_trials_caps_combinations():
    """n_trials must limit the number of evaluated combinations."""

    optimizer = HyperparameterOptimizer()

    evaluated: list[dict] = []

    def objective(params):
        evaluated.append(params)
        return 1.0

    optimizer.optimize(
        search_space={
            "a": [1, 2, 3],
            "b": [1, 2, 3],
        },
        objective=objective,
        n_trials=3,
    )

    assert len(evaluated) == 3


def test_empty_search_space_raises():
    """An empty search space must raise a ValueError."""

    optimizer = HyperparameterOptimizer()

    try:
        optimizer.optimize(
            search_space={},
            objective=lambda params: 0.0,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError.")
