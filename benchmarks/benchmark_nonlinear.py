"""
Benchmarking framework for non-linear relationship detection.

This module provides tools to test maxcorr indicators against various non-linear
dataset patterns, interpret results, and visualize performance.

Datasets included:
- Interlocking Moons
- Concentric Circles (Pearson ≈ 0, but dependent)
- SwissRoll (3D manifold)
- Spiral
- Sine
- Exponential
- Poly2
- Poly3
- Quadratic
- StepFunction
- Linear
- Independent
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from sklearn.datasets import make_circles, make_moons, make_swiss_roll

from maxcorr import AlgorithmType, SemanticsType, indicator
from maxcorr.cross_validation import find_best_params

warnings.filterwarnings("ignore")


@dataclass
class TestResult:
    """Container for a single dataset test result."""

    dataset_name: str
    algorithm: str
    best_params: dict
    test_score: float
    train_score: float
    test_std: float
    train_std: float
    compute_time: float
    n_samples: int
    n_folds: int

    @property
    def overfitting_gap(self) -> float:
        """Calculate the gap between train and test scores."""
        return abs(self.train_score - self.test_score)

    @property
    def status(self) -> str:
        """Determine status based on scores and gap."""
        if self.test_score > 0.85 and self.overfitting_gap < 0.05:
            return "Excellent"
        elif self.test_score > 0.70 and self.overfitting_gap < 0.10:
            return "Good"
        elif self.test_score > 0.50 and self.overfitting_gap < 0.15:
            return "Moderate"
        elif self.overfitting_gap > 0.15:
            return "Overfitting"
        else:
            return "Poor"

    def __repr__(self) -> str:
        return (
            f"TestResult({self.dataset_name}, {self.algorithm}, "
            f"test={self.test_score:.4f}±{self.test_std:.4f}, "
            f"train={self.train_score:.4f}±{self.train_std:.4f}, "
            f"gap={self.overfitting_gap:.4f}, {self.status})"
        )


class NonLinearDatasetGenerator:
    """Generate various non-linear relationship datasets for testing."""

    @staticmethod
    def moons(
        n_samples: int = 500,
        noise_ratio: float = 0.10,
        random_state: int = 42,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        """Interlocking crescents with dynamically calibrated noise."""
        X, _ = make_moons(
            n_samples=n_samples,
            noise=0.0,
            random_state=random_state,
        )
        rng = np.random.default_rng(random_state)

        a = X[:, 0] + rng.normal(0, np.std(X[:, 0]) * noise_ratio, n_samples)
        b = X[:, 1] + rng.normal(0, np.std(X[:, 1]) * noise_ratio, n_samples)
        return a, b, "Moons"

    @staticmethod
    def circles(
        n_samples: int = 500,
        noise_ratio: float = 0.10,
        random_state: int = 42,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        """Concentric circles with dynamically calibrated noise."""
        X, _ = make_circles(
            n_samples=n_samples,
            noise=0.0,
            factor=0.5,
            random_state=random_state,
        )
        rng = np.random.default_rng(random_state)

        a = X[:, 0] + rng.normal(0, np.std(X[:, 0]) * noise_ratio, n_samples)
        b = X[:, 1] + rng.normal(0, np.std(X[:, 1]) * noise_ratio, n_samples)
        return a, b, "Circles"

    @staticmethod
    def swiss_roll(
        n_samples: int = 500,
        noise_ratio: float = 0.10,
        random_state: int = 42,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        """Swiss roll (3D to 2D projection) with dynamically calibrated noise."""
        X, _ = make_swiss_roll(
            n_samples=n_samples,
            noise=0.0,
            random_state=random_state,
        )
        rng = np.random.default_rng(random_state)

        # We extract X and Z (indices 0 and 2) for the 2D projection
        a = X[:, 0] + rng.normal(0, np.std(X[:, 0]) * noise_ratio, n_samples)
        b = X[:, 2] + rng.normal(0, np.std(X[:, 2]) * noise_ratio, n_samples)
        return a, b, "SwissRoll"

    @staticmethod
    def spiral(
        n_samples: int = 500,
        noise_ratio: float = 0.10,
        random_state: int = 42,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        """Logarithmic spiral with dynamically calibrated noise."""
        rng = np.random.default_rng(random_state)
        t = np.linspace(0, 4 * np.pi, n_samples)

        signal_a = t * np.cos(t)
        signal_b = t * np.sin(t)

        a = signal_a + rng.normal(0, np.std(signal_a) * noise_ratio, n_samples)
        b = signal_b + rng.normal(0, np.std(signal_b) * noise_ratio, n_samples)
        return a, b, "Spiral"

    @staticmethod
    def s_curve(
        n_samples: int = 500,
        noise_ratio: float = 0.10,
        random_state: int = 42,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        """S-shaped curve with dynamically calibrated noise."""
        rng = np.random.default_rng(random_state)
        t = np.linspace(0, 4 * np.pi, n_samples)

        signal_a = t
        signal_b = np.sin(t) + t * 0.1

        a = signal_a + rng.normal(0, np.std(signal_a) * noise_ratio, n_samples)
        b = signal_b + rng.normal(0, np.std(signal_b) * noise_ratio, n_samples)
        return a, b, "S-Curve"

    @staticmethod
    def sine_wave(
        n_samples: int = 500,
        noise_ratio: float = 0.10,
        random_state: int = 42,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        """Sine wave with dynamically calibrated noise."""
        rng = np.random.default_rng(random_state)
        a = rng.uniform(-5, 5, n_samples)

        signal_b = np.sin(a)
        b = signal_b + rng.normal(0, np.std(signal_b) * noise_ratio, n_samples)
        return a, b, "Sine"

    @staticmethod
    def exponential(
        n_samples: int = 500,
        noise_ratio: float = 0.10,
        random_state: int = 42,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        """True exponential relationship with mathematically calibrated SNR."""
        rng = np.random.default_rng(random_state)

        # Expand the domain so the curve actually bends aggressively
        a = rng.uniform(0, 4, n_samples)

        # The True Signal: e^0 to e^4 (1.0 to ~54.6)
        signal = np.exp(a)

        # Dynamic SNR Calibration:
        # Calculate the standard deviation of the signal, then scale the noise so it is exactly 'noise_ratio' of the signal's variance.
        signal_std = np.std(signal)
        calibrated_noise = rng.normal(0, signal_std * noise_ratio, n_samples)

        b = signal + calibrated_noise

        return a, b, "Exponential"

    @staticmethod
    def polynomial(
        n_samples: int = 500,
        degree: int = 3,
        noise_ratio: float = 0.10,
        random_state: int = 42,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        """Polynomial relationship with dynamically calibrated noise."""
        rng = np.random.default_rng(random_state)
        a = rng.uniform(-3, 3, n_samples)

        signal = a**degree
        signal_std = np.std(signal)
        calibrated_noise = rng.normal(0, signal_std * noise_ratio, n_samples)

        return a, signal + calibrated_noise, f"Poly{degree}"

    @staticmethod
    def step_function(
        n_samples: int = 500,
        noise_ratio: float = 0.10,
        random_state: int = 42,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        """Step function with dynamically calibrated noise."""
        rng = np.random.default_rng(random_state)
        a = rng.uniform(-5, 5, n_samples)

        signal = np.sign(a) * 2
        signal_std = np.std(signal)
        calibrated_noise = rng.normal(0, signal_std * noise_ratio, n_samples)

        return a, signal + calibrated_noise, "StepFunction"

    @staticmethod
    def linear(
        n_samples: int = 500,
        noise_ratio: float = 0.10,
        random_state: int = 42,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        """Linear baseline with dynamically calibrated noise."""
        rng = np.random.default_rng(random_state)
        a = rng.uniform(-5, 5, n_samples)

        signal = 2 * a + 1
        signal_std = np.std(signal)
        calibrated_noise = rng.normal(0, signal_std * noise_ratio, n_samples)

        return a, signal + calibrated_noise, "Linear"

    @staticmethod
    def independent(
        n_samples: int = 500,
        random_state: int = 42,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        """Independent variables - no relationship."""
        rng = np.random.default_rng(random_state)
        a = rng.normal(0, 1, n_samples)
        b = rng.normal(0, 1, n_samples)
        return a, b, "Independent"

    @staticmethod
    def quadratic(
        n_samples: int = 500,
        noise_ratio: float = 0.10,
        random_state: int = 42,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        """Quadratic relationship with dynamically calibrated noise."""
        rng = np.random.default_rng(random_state)
        a = rng.uniform(-5, 5, n_samples)

        signal = a**2
        signal_std = np.std(signal)
        calibrated_noise = rng.normal(0, signal_std * noise_ratio, n_samples)

        return a, signal + calibrated_noise, "Quadratic"

    def get_all_datasets(
        self,
        n_samples: int = 500,
        random_state: int = 42,
    ) -> list[tuple[np.ndarray, np.ndarray, str]]:
        """Get all available datasets."""
        return [
            self.moons(n_samples, random_state=random_state),
            self.circles(n_samples, random_state=random_state),
            self.swiss_roll(n_samples, random_state=random_state),
            self.spiral(n_samples, random_state=random_state),
            self.s_curve(n_samples, random_state=random_state),
            self.sine_wave(n_samples, random_state=random_state),
            self.exponential(n_samples, random_state=random_state),
            self.polynomial(n_samples, degree=2, random_state=random_state),
            self.polynomial(n_samples, degree=3, random_state=random_state),
            self.step_function(n_samples, random_state=random_state),
            self.quadratic(n_samples, random_state=random_state),
            self.linear(n_samples, random_state=random_state),
            self.independent(n_samples, random_state=random_state),
        ]


class NonLinearInterpreter:
    """Interpret maxcorr results for different non-linear patterns."""

    EXPECTED_RANGES = {  # noqa: RUF012
        "Circles": (0.80, 0.95),  # Radial dependency (noise capped)
        "Moons": (0.75, 0.85),  # Crescent pattern (Polynomial structural limit)
        "SwissRoll": (0.75, 0.85),  # 3D manifold (Euclidean slicing limit)
        "Spiral": (0.70, 0.90),  # Spiral pattern
        "S-Curve": (0.70, 0.90),  # S-shaped
        "Sine": (0.90, 0.99),  # Smooth periodic (The perfect baseline)
        "Exponential": (0.80, 0.95),  # Growth pattern
        "Poly2": (0.85, 0.98),  # Quadratic
        "Poly3": (0.80, 0.95),  # Cubic
        "Quadratic": (0.85, 0.98),  # Quadratic
        "StepFunction": (0.60, 0.80),  # Discontinuous
        "Linear": (0.90, 0.99),  # Baseline linear
        "Independent": (-0.1, 0.15),  # Strict false-positive boundary
    }

    INTERPRETATIONS = {  # noqa: RUF012
        "Circles": {
            "high": "✅ Exceptional! Detected radial dependency (Pearson would give ~0)",
            "medium": "⚠️ Partial detection. Try higher kernel degree",
            "low": "❌ Failed to detect radial dependency.",
        },
        "Moons": {
            "high": "✅ Excellent! Reached polynomial capacity limit",
            "medium": "⚠️ Moderate detection. Underfitting crescents. Try higher degree.",
            "low": "❌ Missing crescent pattern completely.",
        },
        "SwissRoll": {
            "high": "✅ Euclidean Limit Reached (~0.84). Polynomials slice rather than unroll 3D manifolds.",
            "medium": "⚠️ Sub-optimal Euclidean slicing. Topology is not resolved.",
            "low": "❌ Cannot resolve manifold.",
        },
        "Spiral": {
            "high": "✅ Excellent! Successfully unwrapped the logarithmic spiral",
            "medium": "⚠️ Partial spiral detection. Changing radius is hard for global polynomials.",
            "low": "❌ Failed to detect the spiral structure.",
        },
        "S-Curve": {
            "high": "✅ Perfect! Captured the continuous S-shaped manifold",
            "medium": "⚠️ Moderate detection of the S-curve bends.",
            "low": "❌ Failed to capture the non-linear bends.",
        },
        "Sine": {
            "high": "✅ Perfect Baseline! Polynomial successfully mimics Taylor expansion.",
            "medium": "⚠️ Good approximation. Degree 3-4 captures most oscillation.",
            "low": "❌ Poor approximation. Check algorithm capacity.",
        },
        "Exponential": {
            "high": "✅ Excellent! Exponential growth well-captured",
            "medium": "⚠️ Moderate capture. Try higher degree kernels",
            "low": "❌ Poor detection.",
        },
        "Poly2": {
            "high": "✅ Perfect! Quadratic relationship easily captured",
            "medium": "⚠️ Sub-optimal. A degree 2+ kernel should easily capture this",
            "low": "❌ Failed to detect basic polynomial relationship",
        },
        "Poly3": {
            "high": "✅ Perfect! Cubic relationship captured",
            "medium": "⚠️ Sub-optimal. Ensure kernel degree is ≥ 3",
            "low": "❌ Failed to detect cubic relationship",
        },
        "Quadratic": {
            "high": "✅ Perfect! Quadratic parabola perfectly mapped",
            "medium": "⚠️ Sub-optimal. Even low-degree kernels should catch this",
            "low": "❌ Failed to map the parabola",
        },
        "StepFunction": {
            "high": "✅ Excellent! Handled the discontinuous jumps well",
            "medium": "⚠️ Polynomials struggle with sharp discontinuities (Gibbs phenomenon)",
            "low": "❌ Failed completely.",
        },
        "Independent": {
            "high": "❌ FALSE POSITIVE! Detected spurious correlation in noise",
            "medium": "⚠️ Slight hallucination, but within acceptable noise variance.",
            "low": "✅ Safe! Cross-validation successfully rejected spurious correlation",
        },
        "Linear": {
            "high": "✅ Perfect! Linear relationship detected (baseline)",
            "medium": "⚠️ Good linear detection",
            "low": "❌ Unexpectedly low for linear data. Check for bugs.",
        },
    }

    def interpret_score(self, score: float, dataset_name: str) -> str:
        """Interpret a correlation score for a specific dataset."""
        if dataset_name not in self.EXPECTED_RANGES:
            return f"Unknown dataset: {dataset_name}"

        low, high = self.EXPECTED_RANGES[dataset_name]

        if score >= high * 0.9:
            level = "high"
        elif score >= low + (high - low) * 0.5:
            level = "medium"
        else:
            level = "low"

        interpretations = self.INTERPRETATIONS.get(dataset_name, {})
        return interpretations.get(level, "No interpretation available")

    def get_expected_range(self, dataset_name: str) -> tuple[float, float]:
        """Get the expected score range for a dataset."""
        return self.EXPECTED_RANGES.get(dataset_name, (0.0, 1.0))


class NonLinearIndicatorTester:
    """Systematic tester for maxcorr indicators on non-linear datasets."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results: list[TestResult] = []
        self.interpreter = NonLinearInterpreter()

    def test_dataset(
        self,
        a: np.ndarray,
        b: np.ndarray,
        dataset_name: str,
        algorithm: AlgorithmType = "sk",
        param_grid: dict | None = None,
        n_splits: int = 5,
        semantics: SemanticsType = "hgr",
        plot_transformations: bool = False,
        use_hard_constraint: bool = True,
        var_lower_bound: float = 1.0,
        var_upper_bound: float = 1.0,
    ) -> TestResult:
        """Test indicator on a single dataset."""

        if param_grid is None:
            param_grid = (
                {"kernel": [2, 3, 4, 5]}
                if algorithm == "sk"
                else {"kernel_a": [3, 4, 5], "kernel_b": [3, 4, 5]}
            )

        param_grid.update(
            **{
                "use_hard_constraint": [use_hard_constraint],
                "var_lower_bound": [var_lower_bound],
                "var_upper_bound": [var_upper_bound],
            }
        )

        if self.verbose:
            print(f"\n{'=' * 80}")
            print(f"Testing: {dataset_name}")
            print(f"Algorithm: {algorithm} | Semantics: {semantics}")
            print(f"Samples: {len(a)} | Folds: {n_splits}")
            print(f"{'=' * 80}")

        # Start the high-resolution timer right before optimization begins
        start_time = time.perf_counter()

        best_params, cv_result = find_best_params(
            a,
            b,
            algorithm=algorithm,
            param_grid=param_grid,
            n_splits=n_splits,
            semantics=semantics,
            verbose=0,  # Suppress internal verbosity
        )

        # Stop the timer and calculate the delta
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        result = TestResult(
            dataset_name=dataset_name,
            algorithm=algorithm,
            best_params=best_params,
            test_score=cv_result.mean_test_score,
            train_score=cv_result.mean_train_score,
            test_std=cv_result.std_test_score,
            train_std=cv_result.std_train_score,
            compute_time=elapsed_time,
            n_samples=len(a),
            n_folds=n_splits,
        )

        if self.verbose:
            print(f"\n✓ Best parameters: {best_params}")
            print(f"✓ Test score: {result.test_score:.6f} ± {result.test_std:.6f}")
            print(f"✓ Train score: {result.train_score:.6f} ± {result.train_std:.6f}")
            print(f"✓ Overfitting gap: {result.overfitting_gap:.6f}")
            print(f"✓ Computation Time: {result.compute_time:.2f} seconds")
            print(f"✓ Status: {result.status}")

            interpretation = self.interpreter.interpret_score(
                result.test_score,
                dataset_name,
            )
            print(f"✓ Interpretation: {interpretation}")

        self.results.append(result)

        if plot_transformations:
            # Fit a final indicator with the best parameters to extract the projections
            ind = indicator(
                algorithm=algorithm,
                semantics=semantics,
                backend="numpy",
                **best_params,
            )
            ind.compute(a, b)
            NonLinearVisualizer.plot_transformation(
                a,
                b,
                ind.f(a),
                ind.g(b),
                dataset_name=f"{dataset_name} ({algorithm.upper()})",
            )

        return result

    def test_all_datasets(
        self,
        algorithms: list[str] | None = None,
        n_splits: int = 5,
        semantics: str = "hgr",
        n_samples: int = 500,
        random_state: int = 42,
        datasets: list[tuple[np.ndarray, np.ndarray, str]] | None = None,
        plot_proofs: bool = False,
        use_hard_constraint: bool = True,
        var_lower_bound: float = 1.0,
        var_upper_bound: float = 1.0,
    ) -> pd.DataFrame:
        """Test indicator on all available datasets."""

        if algorithms is None:
            algorithms = ["sk", "dk"]

        # Use provided subset, or fetch all 13 by default
        if datasets is None:
            gen = NonLinearDatasetGenerator()
            datasets = gen.get_all_datasets(
                n_samples=n_samples,
                random_state=random_state,
            )

        print(f"\n{'=' * 80}")
        print(f"Testing {len(datasets)} Non-Linear Datasets")
        print(
            f"\n{'=' * 80}\nTesting {len(datasets)} Datasets across {len(algorithms)} Algorithms\n{'=' * 80}"
        )
        print(f"{'=' * 80}")

        for a, b, name in datasets:
            for algo in algorithms:
                self.test_dataset(
                    a,
                    b,
                    dataset_name=name,
                    algorithm=algo,
                    n_splits=n_splits,
                    semantics=semantics,
                    use_hard_constraint=use_hard_constraint,
                    var_lower_bound=var_lower_bound,
                    var_upper_bound=var_upper_bound,
                )

        # Trigger the 5-column proof using the CV results
        if plot_proofs:
            print("\nGenerating Visual Proofs from Cross-Validation Results...")
            NonLinearVisualizer.plot_topology_proofs(self, datasets)

        return self.get_results_dataframe()

    def get_results_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame."""
        data = []
        for r in self.results:
            data.append(
                {
                    "Dataset": r.dataset_name,
                    "Algorithm": r.algorithm,
                    "Test Score": r.test_score,
                    "Test Std": r.test_std,
                    "Train Score": r.train_score,
                    "Train Std": r.train_std,
                    "Compute Time (s)": r.compute_time,
                    "Overfitting Gap": r.overfitting_gap,
                    "Status": r.status,
                    "Best Params": str(r.best_params),
                }
            )
        return pd.DataFrame(data)

    def print_summary(self):
        """Print summary of all results."""
        df = self.get_results_dataframe()

        print(f"\n{'=' * 80}")
        print("SUMMARY OF RESULTS")
        print(f"{'=' * 80}\n")

        print(df.to_string(index=False))


class NonLinearVisualizer:
    """Visualization tools for non-linear relationship testing."""

    @staticmethod
    def plot_datasets(
        n_samples: int = 500,
        random_state: int = 42,
        save_path: str | None = None,
    ):
        """Plot all available datasets."""
        gen = NonLinearDatasetGenerator()
        datasets = gen.get_all_datasets(n_samples=n_samples, random_state=random_state)

        n_datasets = len(datasets)
        n_cols = 4
        n_rows = (n_datasets + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
        axes = axes.flatten()

        for idx, (a, b, name) in enumerate(datasets):
            ax = axes[idx]

            ax.scatter(a, b, c=a, cmap="viridis", alpha=0.6, s=20)
            ax.set_title(name, fontsize=12, fontweight="bold")
            ax.set_xlabel("a")
            ax.set_ylabel("b")
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(n_datasets, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"✓ Saved to {save_path}")

        plt.show()

    @staticmethod
    def plot_transformation(
        a_raw: np.ndarray,
        b_raw: np.ndarray,
        a_transformed: np.ndarray,
        b_transformed: np.ndarray,
        dataset_name: str,
        save_path: str | None = None,
    ):
        """Plot the raw non-linear data vs. the learned copulas vs. the linearized transformation."""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

        # Panel 1: The Raw Data
        ax1.scatter(a_raw, b_raw, c=a_raw, cmap="viridis", alpha=0.6, s=20)
        ax1.set_title(
            f"Raw Data: {dataset_name}\n(Original Geometry)",
            fontweight="bold",
        )
        ax1.set_xlabel("a (Original)")
        ax1.set_ylabel("b (Original)")
        ax1.grid(True, alpha=0.3)

        # Panel 2: The Learned Copulas
        # We must sort the raw data to draw continuous lines for the mathematical functions
        sort_a, sort_b = np.argsort(a_raw), np.argsort(b_raw)

        ax2.plot(
            a_raw[sort_a],
            a_transformed[sort_a],
            color="blue",
            label="f(a)",
            linewidth=2,
        )
        ax2.plot(
            b_raw[sort_b],
            b_transformed[sort_b],
            color="orange",
            label="g(b)",
            linewidth=2,
        )
        ax2.set_title("Learned Copula Transformations", fontweight="bold")
        ax2.set_xlabel("Original Input Value")
        ax2.set_ylabel("Transformed Output Value")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Panel 3: The Transformed Data
        ax3.scatter(
            a_transformed,
            b_transformed,
            c=a_raw,
            cmap="viridis",
            alpha=0.6,
            s=20,
        )
        ax3.set_title("Linearized Projection\n(Maximal Correlation)", fontweight="bold")
        ax3.set_xlabel("f(a)")
        ax3.set_ylabel("g(b)")
        ax3.grid(True, alpha=0.3)

        # Draw the perfect correlation line
        min_val = min(a_transformed.min(), b_transformed.min())
        max_val = max(a_transformed.max(), b_transformed.max())
        ax3.plot(
            [min_val, max_val],
            [min_val, max_val],
            "r--",
            alpha=0.5,
            label="Perfect Linear Correlation",
        )
        ax3.legend()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"✓ Saved to {save_path}")
        plt.show()

    @staticmethod
    def plot_results_comparison(df: pd.DataFrame, save_path: str | None = None):
        """Create a 4-panel comparison plot comparing algorithms side-by-side with full legends and data labels."""

        fig, axes = plt.subplots(3, 2, figsize=(16, 18))

        # Sort DataFrame alphabetically by Dataset so the Y-axis is perfectly aligned
        df_sorted = df.sort_values(by=["Dataset", "Algorithm"], ascending=[False, True])
        palette = {"sk": "#4C72B0", "dk": "#DD8452"}  # Classic Blue and Orange

        # Create universal Legend Patches for the Bar Charts
        sk_patch = mpatches.Patch(color=palette["sk"], label="HGR-SK")
        dk_patch = mpatches.Patch(color=palette["dk"], label="HGR-DK")

        # Panel 1: Test Scores by Dataset
        ax = axes[0, 0]
        sns.barplot(
            data=df_sorted,
            y="Dataset",
            x="Test Score",
            hue="Algorithm",
            palette=palette,
            ax=ax,
        )
        ax.set_xlabel("Test Score (Cross-Validated)", fontweight="bold")
        ax.set_ylabel("")
        ax.set_title(
            "Performance by Dataset (SK vs DK)",
            fontweight="bold",
            fontsize=12,
        )
        ax.set_xlim([0, 1.05])
        ax.grid(True, alpha=0.3, axis="x")

        # Universal Legend
        ax.legend(handles=[sk_patch, dk_patch], loc="lower right")

        # Panel 2: Computation Time
        ax = axes[0, 1]
        sns.barplot(
            data=df_sorted,
            y="Dataset",
            x="Compute Time (s)",
            hue="Algorithm",
            palette=palette,
            ax=ax,
        )
        ax.set_xlabel("Time in Seconds (Lower is better)", fontweight="bold")
        ax.set_ylabel("")
        ax.set_title("Computational Cost (SK vs DK)", fontweight="bold", fontsize=12)
        ax.grid(True, alpha=0.3, axis="x")

        # Unified Legend for Panel 2
        ax.legend(handles=[sk_patch, dk_patch], loc="lower right")

        # Panel 3: Overfitting Gap
        ax = axes[1, 0]
        sns.barplot(
            data=df_sorted,
            y="Dataset",
            x="Overfitting Gap",
            hue="Algorithm",
            palette=palette,
            ax=ax,
        )
        ax.set_xlabel("Train-Test Gap", fontweight="bold")
        ax.set_ylabel("")
        ax.set_title(
            "Generalization Quality (Lower is better)",
            fontweight="bold",
            fontsize=12,
        )
        ax.axvline(
            x=0.05,
            color="green",
            linestyle="--",
            alpha=0.6,
            label="Excellent (<0.05)",
        )
        ax.axvline(
            x=0.10,
            color="orange",
            linestyle="--",
            alpha=0.6,
            label="Good (<0.10)",
        )

        # Unified Legend: Combine SK/DK patches with the Threshold Lines
        handles, labels = ax.get_legend_handles_labels()
        line_handles = [
            h for h, l in zip(handles, labels) if "Excellent" in l or "Good" in l
        ]
        line_labels = [l for l in labels if "Excellent" in l or "Good" in l]
        ax.legend(
            handles=[sk_patch, dk_patch] + line_handles,
            labels=["HGR-SK", "HGR-DK"] + line_labels,
            loc="lower right",
        )
        ax.grid(True, alpha=0.3, axis="x")

        # Panel 4: Train vs. Test Scatter
        ax = axes[1, 1]
        sns.scatterplot(
            data=df_sorted,
            x="Train Score",
            y="Test Score",
            hue="Algorithm",
            style="Algorithm",
            markers={"sk": "o", "dk": "s"},
            palette=palette,
            s=120,
            alpha=0.8,
            ax=ax,
            legend=False,
        )
        ax.plot([0, 1.05], [0, 1.05], "k--", alpha=0.5, label="Perfect Fit")
        ax.set_xlabel("Train Score", fontweight="bold")
        ax.set_ylabel("Test Score", fontweight="bold")
        ax.set_title("Train vs Test Validation", fontweight="bold", fontsize=12)
        ax.set_xlim([0, 1.05])
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3)

        # Custom Legend for Scatter
        custom_legend = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=palette["sk"],
                markersize=10,
                label="HGR-SK",
            ),
            Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                markerfacecolor=palette["dk"],
                markersize=10,
                label="HGR-DK",
            ),
            Line2D(
                [0],
                [0],
                color="k",
                linestyle="--",
                alpha=0.5,
                label="Perfect Generalization",
            ),
        ]
        ax.legend(handles=custom_legend, loc="lower right")

        # Only label outliers if there are many datasets
        total_datasets = len(df["Dataset"].unique())

        # Directly label every dot with its dataset name and the (Train, Test) numbers
        for _, row in df_sorted.iterrows():
            # Condition: Is it an anomaly? (Overfitting > 0.05 OR struggling to hit 0.75 score)
            is_anomaly = row["Overfitting Gap"] > 0.05 or row["Test Score"] < 0.75

            # If it's a small defense subset, label everything. If it's all 13, only label anomalies.
            if total_datasets <= 4 or is_anomaly:
                x_offset = -0.015 if row["Algorithm"] == "sk" else 0.015
                y_offset = 0.015 if row["Algorithm"] == "sk" else -0.015
                ha = "right" if row["Algorithm"] == "sk" else "left"
                va = "bottom" if row["Algorithm"] == "sk" else "top"

                # Keep the text minimal
                label_text = f"{row['Dataset']}\n({row['Train Score']:.2f}, {row['Test Score']:.2f})"

                ax.text(
                    row["Train Score"] + x_offset,
                    row["Test Score"] + y_offset,
                    label_text,
                    fontsize=7,
                    color=palette[row["Algorithm"]],
                    ha=ha,
                    va=va,
                    alpha=0.9,
                    fontweight="medium",
                )

        # Panel 5: Stability (Std Deviation)
        ax = axes[2, 0]
        sns.barplot(
            data=df_sorted,
            y="Dataset",
            x="Test Std",
            hue="Algorithm",
            palette=palette,
            ax=ax,
        )
        ax.set_xlabel("Standard Deviation across Folds", fontweight="bold")
        ax.set_ylabel("")
        ax.set_title(
            "Cross-Fold Stability (Lower is better)",
            fontweight="bold",
            fontsize=12,
        )
        ax.axvline(
            x=0.05,
            color="red",
            linestyle="--",
            alpha=0.5,
            label="Unstable (>0.05)",
        )

        # Unified Legend: Combine SK/DK patches with the Unstable Line
        handles, labels = ax.get_legend_handles_labels()
        line_handles = [h for h, l in zip(handles, labels) if "Unstable" in l]
        line_labels = [l for l in labels if "Unstable" in l]
        ax.legend(
            handles=[sk_patch, dk_patch] + line_handles,
            labels=["HGR-SK", "HGR-DK"] + line_labels,
            loc="lower right",
        )
        ax.grid(True, alpha=0.3, axis="x")

        # Hide the 6th unused panel to keep the layout clean
        axes[2, 1].axis("off")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"✓ Saved to {save_path}")

        plt.show()

    @staticmethod
    def plot_topology_proofs(
        tester: NonLinearIndicatorTester,
        datasets: list[tuple[np.ndarray, np.ndarray, str]],
    ):
        """
        Generates the 5-column head-to-head proof using the cross-validated results from the NonLinearIndicatorTester.
        """
        # Group tester results by dataset name for easy lookup
        results_by_dataset = {}
        for r in tester.results:
            if r.dataset_name not in results_by_dataset:
                results_by_dataset[r.dataset_name] = {}
            results_by_dataset[r.dataset_name][r.algorithm] = r

        fig, axes = plt.subplots(len(datasets), 5, figsize=(24, 4 * len(datasets)))

        for i, (a, b, name) in enumerate(datasets):
            res_sk = results_by_dataset.get(name, {}).get("sk")
            res_dk = results_by_dataset.get(name, {}).get("dk")

            # PLOT 1: Colored Original Geometry
            axes[i, 0].scatter(a, b, c=a, cmap="viridis", alpha=0.6, s=20)
            axes[i, 0].set_title(f"{name}\nOriginal Data", fontweight="bold")
            axes[i, 0].set_xlabel("Variable A")
            axes[i, 0].set_ylabel("Variable B")

            sort_a, sort_b = np.argsort(a), np.argsort(b)

            # Iterate through both algorithms to plot Copulas and Projections
            configs = [(1, res_sk, "sk"), (3, res_dk, "dk")]

            for col_offset, res, algo in configs:
                if res:
                    # Re-instantiate the model using the optimal CV parameters
                    model = indicator(
                        **res.best_params,
                        algorithm=algo,
                        semantics="hgr",
                    )
                    model.compute(a, b)
                    fa, gb = model.f(a), model.g(b)

                    # PLOT 2 & 4: The Learned Copulas
                    axes[i, col_offset].plot(
                        a[sort_a],
                        fa[sort_a],
                        color="blue",
                        label="f(a)",
                        linewidth=2,
                    )
                    axes[i, col_offset].plot(
                        b[sort_b],
                        gb[sort_b],
                        color="orange",
                        label="g(b)",
                        linewidth=2,
                    )
                    best_params = res.best_params.copy()
                    del best_params["var_lower_bound"]
                    del best_params["var_upper_bound"]
                    del best_params["use_hard_constraint"]
                    axes[i, col_offset].set_title(
                        f"HGR-{algo.upper()} Copulas\nBest Params: {best_params}",
                        fontweight="bold",
                    )
                    axes[i, col_offset].legend()

                    # PLOT 3 & 5: The Linearized Projections
                    axes[i, col_offset + 1].scatter(
                        fa,
                        gb,
                        c=a,
                        cmap="viridis",
                        alpha=0.6,
                        s=20,
                    )
                    axes[i, col_offset + 1].set_title(
                        f"HGR-{algo.upper()} Projection\nCV Score: {res.test_score:.3f}",
                        fontweight="bold",
                    )

                    # Draw diagonal
                    min_v, max_v = (
                        min(np.min(fa), np.min(gb)),
                        max(np.max(fa), np.max(gb)),
                    )
                    axes[i, col_offset + 1].plot(
                        [min_v, max_v], [min_v, max_v], "r--", alpha=0.5
                    )
                    axes[i, col_offset + 1].set_xlabel("f(a)")
                    axes[i, col_offset + 1].set_ylabel("g(b)")

        plt.tight_layout()
        plt.show()


def main():
    """Run comprehensive non-linear relationship testing."""

    use_hard_constraint = True
    var_lower_bound = 1.0 if use_hard_constraint else 0.01
    var_upper_bound = 1.0 if use_hard_constraint else np.inf

    print("\n" + "=" * 80)
    print("Non-Linear Benchmark Suite for MaxCorr")
    print("=" * 80)

    tester = NonLinearIndicatorTester(verbose=True)

    # Run Head-to-Head Comparison for SK and DK
    df_results = tester.test_all_datasets(
        algorithms=["sk", "dk"],
        n_splits=5,
        random_state=42,
        use_hard_constraint=use_hard_constraint,
        var_lower_bound=var_lower_bound,
        var_upper_bound=var_upper_bound,
    )
    tester.print_summary()

    # Visualize a specific transformation to prove it works
    print("\nGenerating Transformation Visualization for Moons...")
    a_moons, b_moons, name = NonLinearDatasetGenerator.moons(
        n_samples=500,
        random_state=42,
    )
    # expanded_grid = {"kernel_a": [5, 7, 9, 11], "kernel_b": [5, 7, 9, 11]}
    tester.test_dataset(
        a_moons,
        b_moons,
        dataset_name=name,
        # param_grid=expanded_grid,
        algorithm="dk",
        plot_transformations=True,
        use_hard_constraint=use_hard_constraint,
        var_lower_bound=var_lower_bound,
        var_upper_bound=var_upper_bound,
    )


if __name__ == "__main__":
    main()
