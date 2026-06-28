"""
Cross-validation utilities for hyperparameter tuning of kernel-based indicators.

This module provides tools for performing k-fold and stratified cross-validation
on kernel indicators (DoubleKernelIndicator and SingleKernelIndicator) to find
optimal hyperparameters.
"""

from __future__ import annotations

import gc
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from itertools import product
from typing import Any

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

from maxcorr import indicator
from maxcorr.cuda_path_utils import setup_cuda_paths
from maxcorr.typing import BackendType, SemanticsType


@dataclass
class CVResult:
    """Result of a single cross-validation fold."""

    fold: int
    """The fold number (0-indexed)."""

    train_score: float
    """The score computed on the training set."""

    test_score: float
    """The score computed on the test set."""

    params: dict[str, Any]
    """The hyperparameters used for this fold."""


@dataclass
class CVTrialResult:
    """Result of a complete cross-validation trial for a given hyperparameter set."""

    params: dict[str, Any] = field()
    """The hyperparameters tested."""

    fold_results: list[CVResult] = field()
    """Results from each fold."""

    mean_test_score: float = field()
    """Mean test score across all folds."""

    std_test_score: float = field()
    """Standard deviation of test scores across folds."""

    mean_train_score: float = field()
    """Mean training score across all folds."""

    std_train_score: float = field()
    """Standard deviation of training scores across folds."""

    def __repr__(self) -> str:
        return (
            f"CVTrialResult(params={self.params}, "
            f"mean_test={self.mean_test_score:.6f}±{self.std_test_score:.6f})"
        )


class BaseCrossValidator(ABC):
    """Abstract base class for cross-validators."""

    def __init__(
        self,
        n_splits: int = 5,
        semantics: SemanticsType = "hgr",
        backend: BackendType = "numpy",
        verbose: int = 0,
    ):
        """
        Initialize the cross-validator.

        :param n_splits:
            Number of folds for cross-validation.

        :param semantics:
            Semantics of the indicator ('hgr', 'gedi', or 'nlc').

        :param backend:
            Backend to use ('numpy', 'torch', or 'tensorflow').

        :param verbose:
            Verbosity level (0=silent, 1=minimal, 2=detailed).
        """
        self.n_splits = n_splits
        self.semantics = semantics
        self.backend = backend
        self.verbose = verbose

    @abstractmethod
    def split(self, a: np.ndarray, b: np.ndarray):
        """
        Generate train/test indices for cross-validation.

        :param a:
            First data vector.

        :param b:
            Second data vector.

        :yields:
            Tuple of (train_indices, test_indices).
        """
        pass

    def validate(
        self,
        a: np.ndarray,
        b: np.ndarray,
        algorithm: str,
        param_grid: dict[str, list[Any]],
        n_iter: int | None = None,
    ) -> list[CVTrialResult]:
        """
        Perform cross-validation with hyperparameter grid search.

        :param a:
            First data vector.

        :param b:
            Second data vector.

        :param algorithm:
            Algorithm to use ('dk' or 'sk').

        :param param_grid:
            Dictionary mapping parameter names to lists of values to try.

        :param n_iter:
            Number of iterations to run.

        :return:
            List of CVTrialResult objects, sorted by mean_test_score (descending).
        """
        results = []

        # Generate all parameter combinations
        param_combinations = self._generate_param_combinations(param_grid)

        # Prevent combinatorial explosion by sampling if n_iter is provided
        if n_iter is not None and n_iter < len(param_combinations):
            # Sort keys to ensure deterministic random seeding if needed
            random.seed(42)
            param_combinations = random.sample(param_combinations, n_iter)

        total_combinations = len(param_combinations)

        if self.verbose >= 1:
            print(
                f"Testing {total_combinations} parameter combinations with {self.n_splits}-fold CV..."
            )

        for combo_idx, params in enumerate(param_combinations):
            if self.verbose >= 1:
                print(f"\n[{combo_idx + 1}/{total_combinations}] Testing {params}")

            fold_results = []
            train_scores = []
            test_scores = []

            for fold_idx, (train_idx, test_idx) in enumerate(self.split(a, b)):
                a_train, a_test = a[train_idx], a[test_idx]
                b_train, b_test = b[train_idx], b[test_idx]

                # Create and compute indicator on training set
                ind = indicator(
                    algorithm=algorithm,
                    semantics=self.semantics,
                    backend=self.backend,
                    **params,
                )

                # Get scores
                train_score = ind.compute(a_train, b_train)
                test_score = ind.value(a_test, b_test)

                train_scores.append(train_score)
                test_scores.append(test_score)

                fold_results.append(
                    CVResult(
                        fold=fold_idx,
                        train_score=train_score,
                        test_score=test_score,
                        params=params.copy(),
                    )
                )

                if self.verbose >= 2:
                    print(
                        f"  Fold {fold_idx + 1}/{self.n_splits}: "
                        f"train={train_score:.6f}, test={test_score:.6f}"
                    )

                del ind
                gc.collect()
                if self.backend == "torch":
                    try:
                        import torch

                        setup_cuda_paths()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except ImportError:
                        pass
                elif self.backend == "tensorflow":
                    try:
                        import tensorflow as tf

                        setup_cuda_paths()
                        # Destroy the current TF graph and clear internal state
                        tf.keras.backend.clear_session(free_memory=True)
                    except ImportError:
                        pass

            # Compute summary statistics
            trial_result = CVTrialResult(
                params=params.copy(),
                fold_results=fold_results,
                mean_test_score=float(np.mean(test_scores)),
                std_test_score=float(np.std(test_scores)),
                mean_train_score=float(np.mean(train_scores)),
                std_train_score=float(np.std(train_scores)),
            )
            results.append(trial_result)

            if self.verbose >= 1:
                print(
                    f"  Result: test={trial_result.mean_test_score:.6f}±"
                    f"{trial_result.std_test_score:.6f}"
                )

        # Sort by mean test score (descending)
        results.sort(key=lambda x: x.mean_test_score, reverse=True)

        if self.verbose >= 1:
            print("\n" + "=" * 80)
            print("Cross-validation complete!")
            print("=" * 80)

        return results

    @staticmethod
    def _generate_param_combinations(
        param_grid: dict[str, list[Any]],
    ) -> list[dict[str, Any]]:
        """Generate all parameter combinations from a grid."""
        if not param_grid:
            return [{}]

        keys = param_grid.keys()
        values = param_grid.values()
        combinations = []

        for value_combo in product(*values):
            combinations.append(dict(zip(keys, value_combo)))

        return combinations


class KFoldCrossValidator(BaseCrossValidator):
    """Standard k-fold cross-validator."""

    def split(self, a: np.ndarray, b: np.ndarray):
        """Generate k-fold train/test splits."""
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        # for train_idx, test_idx in kf.split(a):
        #     yield train_idx, test_idx
        yield from kf.split(a)


class StratifiedCrossValidator(BaseCrossValidator):
    """Stratified k-fold cross-validator for categorical groups."""

    def __init__(
        self,
        n_splits: int = 5,
        semantics: SemanticsType = "hgr",
        backend: BackendType = "numpy",
        verbose: int = 0,
    ):
        """
        Initialize the stratified cross-validator.

        :param n_splits:
            Number of folds for cross-validation.

        :param semantics:
            Semantics of the indicator.

        :param backend:
            Backend to use.

        :param verbose:
            Verbosity level.
        """
        super().__init__(n_splits, semantics, backend, verbose)
        self._groups = None

    def set_groups(self, groups: np.ndarray):
        """Set the groups for stratification."""
        self._groups = groups

    def split(self, a: np.ndarray, b: np.ndarray):
        """Generate stratified k-fold train/test splits."""
        if self._groups is None:
            raise ValueError("Groups must be set before splitting. Use set_groups().")

        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        yield from skf.split(a, self._groups)


class TimeSeriesCrossValidator(BaseCrossValidator):
    """Time-series aware cross-validator using an expanding window."""

    def __init__(
        self,
        n_splits: int = 5,
        semantics: SemanticsType = "hgr",
        backend: BackendType = "numpy",
        verbose: int = 0,
    ):
        """
        Initialize the time-series cross-validator.

        Uses expanding window strategy: training set grows while the test set is fixed size.

        :param n_splits:
            Number of folds for cross-validation.

        :param semantics:
            Semantics of the indicator.

        :param backend:
            Backend to use.

        :param verbose:
            Verbosity level.
        """
        super().__init__(n_splits, semantics, backend, verbose)

    def split(self, a: np.ndarray, b: np.ndarray):
        """Generate time-series aware train/test splits."""
        n = len(a)
        test_len = max(1, n // (self.n_splits + 1))

        for fold_idx in range(self.n_splits):
            split_point = n - (self.n_splits - fold_idx) * test_len
            test_end = split_point + test_len

            train_idx = np.arange(split_point)
            test_idx = np.arange(split_point, test_end)

            yield train_idx, test_idx


def find_best_params(
    a: np.ndarray,
    b: np.ndarray,
    algorithm: str = "dk",
    param_grid: dict[str, list[Any]] | None = None,
    n_splits: int = 5,
    n_iter: int | None = None,
    semantics: SemanticsType = "hgr",
    backend: BackendType = "numpy",
    cv_type: str = "kfold",
    verbose: int = 1,
) -> tuple[dict[str, Any], CVTrialResult]:
    """
    Find the best hyperparameters for kernel indicators using cross-validation.

    :param a:
        First data vector.

    :param b:
        Second data vector.

    :param algorithm:
        Algorithm to tune ('dk' for DoubleKernelIndicator or 'sk' for SingleKernelIndicator).

    :param param_grid:
        Dictionary of parameter names and lists of values to try.
        For the 'dk' algorithm, common parameters are 'kernel_a' and 'kernel_b'.
        For the 'sk' algorithm, the common parameter is 'kernel'.
        Additional parameters can include 'maxiter', 'tol', 'method', etc.

    :param n_splits:
        Number of cross-validation folds.

    :param n_iter:
        Number of iterations to run.

    :param semantics:
        Semantics of the indicator ('hgr', 'gedi', or 'nlc').

    :param backend:
        Backend to use ('numpy', 'torch', or 'tensorflow').

    :param cv_type:
        Type of cross-validator ('kfold', 'stratified', or 'timeseries').

    :param verbose:
        Verbosity level (0=silent, 1=minimal, 2=detailed).

    :return:
        Tuple of (best_params, best_result).
    """
    if param_grid is None:
        # Default parameter grid
        if algorithm == "dk":
            param_grid = {"kernel_a": [2, 3, 4], "kernel_b": [2, 3, 4]}
        else:  # 'sk'
            param_grid = {"kernel": [2, 3, 4]}

    # Select cross-validator
    if cv_type == "kfold":
        cv = KFoldCrossValidator(
            n_splits=n_splits,
            semantics=semantics,
            backend=backend,
            verbose=verbose,
        )
    elif cv_type == "stratified":
        cv = StratifiedCrossValidator(
            n_splits=n_splits,
            semantics=semantics,
            backend=backend,
            verbose=verbose,
        )
    elif cv_type == "timeseries":
        cv = TimeSeriesCrossValidator(
            n_splits=n_splits,
            semantics=semantics,
            backend=backend,
            verbose=verbose,
        )
    else:
        raise ValueError(f"Unknown cv_type: {cv_type}")

    # Perform cross-validation
    results = cv.validate(a, b, algorithm, param_grid, n_iter=n_iter)

    # Return best result
    best_result = results[0]
    return best_result.params, best_result


def compare_algorithms(
    a: np.ndarray,
    b: np.ndarray,
    groups: np.ndarray | None = None,
    algorithms: list[str] | None = None,
    param_grids: dict[str, dict[str, list[Any]]] | None = None,
    n_splits: int = 5,
    n_iter: int | None = None,
    semantics: SemanticsType = "hgr",
    backend: BackendType = "numpy",
    verbose: int = 1,
) -> dict[str, tuple[dict[str, Any], CVTrialResult]]:
    """
    Compare multiple algorithms and find the best parameters for each.

    :param a:
        First data vector.

    :param b:
        Second data vector.

    :param groups:
        Groups to use for stratified cross-validation.

    :param algorithms:
        List of algorithms to compare ('dk', 'sk', etc.).

    :param param_grids:
        Dictionary mapping algorithm names to their parameter grids.

    :param n_splits:
        Number of cross-validation folds.

    :param n_iter:
        Number of iterations to run.

    :param semantics:
        Semantics of the indicator.

    :param backend:
        Backend to use.

    :param verbose:
        Verbosity level.

    :return:
        Dictionary mapping algorithm names to (best_params, best_result) tuples.
    """
    if algorithms is None:
        algorithms = ["dk", "sk"]

    if param_grids is None:
        param_grids = {
            "dk": {"kernel_a": [2, 3, 4], "kernel_b": [2, 3, 4]},
            "sk": {"kernel": [2, 3, 4]},
        }

    results = {}

    if verbose >= 1:
        print(f"\nComparing {len(algorithms)} algorithms...")
        print("=" * 80)

    for algo in algorithms:
        if verbose >= 1:
            print(f"\nTuning {algo.upper()} algorithm...")

        param_grid = param_grids.get(algo)
        best_params, best_result = find_best_params(
            a,
            b,
            algorithm=algo,
            param_grid=param_grid,
            n_splits=n_splits,
            n_iter=n_iter,
            semantics=semantics,
            backend=backend,
            cv_type="kfold" if groups is None else "stratified",
            verbose=verbose,
        )
        results[algo] = (best_params, best_result)

    if verbose >= 1:
        print("\n" + "=" * 80)
        print("Algorithm Comparison Summary:")
        print("=" * 80)
        for algo, (params, result) in results.items():
            print(
                f"{algo.upper():4s}: test={result.mean_test_score:.6f}±"
                f"{result.std_test_score:.6f}  params={params}"
            )

    return results
