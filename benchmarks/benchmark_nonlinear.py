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

import warnings
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_circles, make_moons, make_swiss_roll

from maxcorr import indicator
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
            return "✅ Excellent"
        elif self.test_score > 0.70 and self.overfitting_gap < 0.10:
            return "✅ Good"
        elif self.test_score > 0.50 and self.overfitting_gap < 0.15:
            return "⚠️  Moderate"
        elif self.overfitting_gap > 0.15:
            return "⚠️  Overfitting"
        else:
            return "❌ Poor"

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
        noise: float = 0.05,
        random_state: int = 42,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        """Interlocking crescents - highly non-linear pattern."""
        X, _ = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
        return X[:, 0], X[:, 1], "Moons"

    @staticmethod
    def circles(
        n_samples: int = 500,
        noise: float = 0.05,
        random_state: int = 42,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        """Concentric circles - Pearson correlation ≈ 0 but highly dependent."""
        X, _ = make_circles(
            n_samples=n_samples,
            noise=noise,
            factor=0.5,
            random_state=random_state,
        )
        return X[:, 0], X[:, 1], "Circles"

    @staticmethod
    def swiss_roll(
        n_samples: int = 500,
        noise: float = 0.1,
        random_state: int = 42,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        """Swiss roll - 3D manifold projected to 2D."""
        X, _ = make_swiss_roll(
            n_samples=n_samples,
            noise=noise,
            random_state=random_state,
        )
        return X[:, 0], X[:, 2], "SwissRoll"

    @staticmethod
    def spiral(
        n_samples: int = 500,
        noise: float = 0.1,
        random_state: int = 42,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        """Logarithmic spiral pattern."""
        rng = np.random.default_rng(random_state)
        t = np.linspace(0, 4 * np.pi, n_samples)
        a = t * np.cos(t) + rng.normal(0, noise, n_samples)
        b = t * np.sin(t) + rng.normal(0, noise, n_samples)
        return a, b, "Spiral"

    @staticmethod
    def s_curve(
        n_samples: int = 500,
        noise: float = 0.1,
        random_state: int = 42,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        """S-shaped curve."""
        rng = np.random.default_rng(random_state)
        t = np.linspace(0, 4 * np.pi, n_samples)
        a = t + rng.normal(0, noise, n_samples)
        b = np.sin(t) + t * 0.1 + rng.normal(0, noise, n_samples)
        return a, b, "S-Curve"

    @staticmethod
    def sine_wave(
        n_samples: int = 500,
        noise: float = 0.1,
        random_state: int = 42,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        """Simple sine wave: b = sin(a)."""
        rng = np.random.default_rng(random_state)
        a = rng.uniform(-5, 5, n_samples)
        b = np.sin(a) + rng.normal(0, noise, n_samples)
        return a, b, "Sine"

    @staticmethod
    def exponential(
        n_samples: int = 500,
        noise: float = 0.1,
        random_state: int = 42,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        """Exponential relationship: b = e^(a/5)."""
        rng = np.random.default_rng(random_state)
        a = rng.uniform(0, 3, n_samples)
        b = np.exp(a) + rng.normal(0, noise * 100, n_samples)
        return a, b, "Exponential"

    @staticmethod
    def polynomial(
        n_samples: int = 500,
        degree: int = 3,
        noise: float = 0.1,
        random_state: int = 42,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        """Polynomial relationship: b = a^degree."""
        rng = np.random.default_rng(random_state)
        a = rng.uniform(-3, 3, n_samples)
        b = a**degree + rng.normal(0, noise, n_samples)
        return a, b, f"Poly{degree}"

    @staticmethod
    def step_function(
        n_samples: int = 500,
        noise: float = 0.1,
        random_state: int = 42,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        """Step function - discontinuous non-linear relationship."""
        rng = np.random.default_rng(random_state)
        a = rng.uniform(-5, 5, n_samples)
        b = np.sign(a) * 2 + rng.normal(0, noise, n_samples)
        return a, b, "StepFunction"

    @staticmethod
    def linear(
        n_samples: int = 500,
        noise: float = 0.1,
        random_state: int = 42,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        """Linear baseline: b = 2*a + 1."""
        rng = np.random.default_rng(random_state)
        a = rng.uniform(-5, 5, n_samples)
        b = 2 * a + 1 + rng.normal(0, noise, n_samples)
        return a, b, "Linear"

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
        noise: float = 0.1,
        random_state: int = 42,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        """Quadratic relationship: b = a^2."""
        rng = np.random.default_rng(random_state)
        a = rng.uniform(-5, 5, n_samples)
        b = a**2 + rng.normal(0, noise, n_samples)
        return a, b, "Quadratic"

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
            "medium": "⚠️  Partial detection. Try higher kernel degree",
            "low": "❌ Failed to detect radial dependency.",
        },
        "Moons": {
            "high": "✅ Excellent! Reached polynomial capacity limit (Avoids Runge's Phenomenon)",
            "medium": "⚠️  Moderate detection. Underfitting crescents. Try higher degree.",
            "low": "❌ Missing crescent pattern completely.",
        },
        "SwissRoll": {
            "high": "✅ Euclidean Limit Reached (~0.84). Polynomials slice rather than unroll 3D manifolds.",
            "medium": "⚠️  Sub-optimal Euclidean slicing. Topology is not resolved.",
            "low": "❌ Cannot resolve manifold.",
        },
        "Spiral": {
            "high": "✅ Excellent! Successfully unwrapped the logarithmic spiral",
            "medium": "⚠️  Partial spiral detection. Changing radius is hard for global polynomials.",
            "low": "❌ Failed to detect the spiral structure.",
        },
        "S-Curve": {
            "high": "✅ Perfect! Captured the continuous S-shaped manifold",
            "medium": "⚠️  Moderate detection of the S-curve bends.",
            "low": "❌ Failed to capture the non-linear bends.",
        },
        "Sine": {
            "high": "✅ Perfect Baseline! Polynomial successfully mimics Taylor expansion.",
            "medium": "⚠️  Good approximation. Degree 3-4 captures most oscillation.",
            "low": "❌ Poor approximation. Check algorithm capacity.",
        },
        "Exponential": {
            "high": "✅ Excellent! Exponential growth well-captured",
            "medium": "⚠️  Moderate capture. Try higher degree kernels",
            "low": "❌ Poor detection.",
        },
        "Poly2": {
            "high": "✅ Perfect! Quadratic relationship easily captured",
            "medium": "⚠️  Sub-optimal. A degree 2+ kernel should easily capture this",
            "low": "❌ Failed to detect basic polynomial relationship",
        },
        "Poly3": {
            "high": "✅ Perfect! Cubic relationship captured",
            "medium": "⚠️  Sub-optimal. Ensure kernel degree is ≥ 3",
            "low": "❌ Failed to detect cubic relationship",
        },
        "Quadratic": {
            "high": "✅ Perfect! Quadratic parabola perfectly mapped",
            "medium": "⚠️  Sub-optimal. Even low-degree kernels should catch this",
            "low": "❌ Failed to map the parabola",
        },
        "StepFunction": {
            "high": "✅ Excellent! Handled the discontinuous jumps well",
            "medium": "⚠️  Polynomials struggle with sharp discontinuities (Gibbs phenomenon)",
            "low": "❌ Failed completely.",
        },
        "Independent": {
            "high": "❌ FALSE POSITIVE! Detected spurious correlation in noise",
            "medium": "⚠️  Slight hallucination, but within acceptable noise variance.",
            "low": "✅ Safe! Cross-validation successfully rejected spurious correlation",
        },
        "Linear": {
            "high": "✅ Perfect! Linear relationship detected (baseline)",
            "medium": "⚠️  Good linear detection",
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
        algorithm: str = "sk",
        param_grid: dict | None = None,
        n_splits: int = 5,
        semantics: str = "hgr",
        plot_transformations: bool = False,
    ) -> TestResult:
        """Test indicator on a single dataset."""

        if param_grid is None:
            param_grid = (
                {"kernel": [2, 3, 4, 5]}
                if algorithm == "sk"
                else {"kernel_a": [3, 4, 5], "kernel_b": [3, 4, 5]}
            )

        if self.verbose:
            print(f"\n{'=' * 80}")
            print(f"Testing: {dataset_name}")
            print(f"Algorithm: {algorithm} | Semantics: {semantics}")
            print(f"Samples: {len(a)} | Folds: {n_splits}")
            print(f"{'=' * 80}")

        best_params, cv_result = find_best_params(
            a,
            b,
            algorithm=algorithm,
            param_grid=param_grid,
            n_splits=n_splits,
            semantics=semantics,
            verbose=0,  # Suppress internal verbosity
        )

        result = TestResult(
            dataset_name=dataset_name,
            algorithm=algorithm,
            best_params=best_params,
            test_score=cv_result.mean_test_score,
            train_score=cv_result.mean_train_score,
            test_std=cv_result.std_test_score,
            train_std=cv_result.std_train_score,
            n_samples=len(a),
            n_folds=n_splits,
        )

        if self.verbose:
            print(f"\n✓ Best parameters: {best_params}")
            print(f"✓ Test score: {result.test_score:.6f} ± {result.test_std:.6f}")
            print(f"✓ Train score: {result.train_score:.6f} ± {result.train_std:.6f}")
            print(f"✓ Overfitting gap: {result.overfitting_gap:.6f}")
            print(f"✓ Status: {result.status}")

            interpretation = self.interpreter.interpret_score(
                result.test_score, dataset_name
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
    ) -> pd.DataFrame:
        """Test indicator on all available datasets."""

        if algorithms is None:
            algorithms = ["sk", "dk"]

        gen = NonLinearDatasetGenerator()
        datasets = gen.get_all_datasets(n_samples=n_samples, random_state=random_state)

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
                )

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
        n_samples: int = 500, random_state: int = 42, save_path: str | None = None
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
            ax.scatter(a, b, alpha=0.5, s=20)
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
        """Plot the raw non-linear data vs. the linearized transformation."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Panel 1: The Raw Data (Before)
        ax1.scatter(a_raw, b_raw, alpha=0.6, color="blue")
        ax1.set_title(f"Raw Data: {dataset_name}\n(Pearson ≈ 0)", fontweight="bold")
        ax1.set_xlabel("a (Original)")
        ax1.set_ylabel("b (Original)")
        ax1.grid(True, alpha=0.3)

        # Panel 2: The Transformed Data (After)
        ax2.scatter(a_transformed, b_transformed, alpha=0.6, color="green")
        ax2.set_title(
            f"Transformed Projections\n(Maximal Correlation)", fontweight="bold"
        )
        ax2.set_xlabel("f(a)")
        ax2.set_ylabel("g(b)")
        ax2.grid(True, alpha=0.3)

        # Draw the perfect correlation line
        min_val = min(a_transformed.min(), b_transformed.min())
        max_val = max(a_transformed.max(), b_transformed.max())
        ax2.plot(
            [min_val, max_val],
            [min_val, max_val],
            "k--",
            alpha=0.5,
            label="Perfect Linear Correlation",
        )
        ax2.legend()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

    @staticmethod
    def plot_results_comparison(df: pd.DataFrame, save_path: str | None = None):
        """Create 4-panel comparison plot."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Panel 1: Test Scores by Dataset
        ax = axes[0, 0]
        colors = [
            "green" if "✅" in s else "orange" if "⚠️" in s else "red"
            for s in df["Status"]
        ]
        ax.barh(df["Dataset"], df["Test Score"], color=colors, alpha=0.7)
        ax.set_xlabel("Test Score", fontweight="bold")
        ax.set_title("Performance by Dataset", fontweight="bold", fontsize=12)
        ax.set_xlim([0, 1])
        ax.grid(True, alpha=0.3, axis="x")

        # Panel 2: Overfitting Gap
        ax = axes[0, 1]
        colors = [
            "green" if gap < 0.05 else "orange" if gap < 0.10 else "red"
            for gap in df["Overfitting Gap"]
        ]
        ax.barh(df["Dataset"], df["Overfitting Gap"], color=colors, alpha=0.7)
        ax.set_xlabel("Train-Test Gap", fontweight="bold")
        ax.set_title("Generalization Quality", fontweight="bold", fontsize=12)
        ax.axvline(x=0.05, color="green", linestyle="--", label="Excellent (<0.05)")
        ax.axvline(x=0.10, color="orange", linestyle="--", label="Good (<0.10)")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="x")

        # Panel 3: Train vs. Test Scatter
        ax = axes[1, 0]
        ax.scatter(df["Train Score"], df["Test Score"], s=100, alpha=0.6)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect fit")
        ax.set_xlabel("Train Score", fontweight="bold")
        ax.set_ylabel("Test Score", fontweight="bold")
        ax.set_title(
            "Train vs Test (Perfect line = no overfitting)",
            fontweight="bold",
            fontsize=12,
        )
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Panel 4: Stability (Std Deviation)
        ax = axes[1, 1]
        colors = [
            "green" if std < 0.01 else "orange" if std < 0.03 else "red"
            for std in df["Test Std"]
        ]
        ax.barh(df["Dataset"], df["Test Std"], color=colors, alpha=0.7)
        ax.set_xlabel("Standard Deviation", fontweight="bold")
        ax.set_title("Cross-Fold Stability", fontweight="bold", fontsize=12)
        ax.grid(True, alpha=0.3, axis="x")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"✓ Saved to {save_path}")

        plt.show()


def main():
    """Run comprehensive non-linear relationship testing."""

    print("\n" + "=" * 80)
    print("Non-Linear Benchmark Suite for MaxCorr")
    print("=" * 80)

    tester = NonLinearIndicatorTester(verbose=True)

    # Run Head-to-Head Comparison for SK and DK
    df_results = tester.test_all_datasets(
        algorithms=["sk", "dk"],
        n_splits=5,
        random_state=42,
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
    )


if __name__ == "__main__":
    main()
