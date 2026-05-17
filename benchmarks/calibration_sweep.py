"""
Empirical Baseline Calibrator for MaxCorr.

This script runs asymptotic sweeps across increasing polynomial degrees
to empirically determine the maximum possible correlation score (the ceiling)
for a given dataset before structural limits or severe overfitting occur.
"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import pandas as pd
from benchmark_nonlinear import NonLinearDatasetGenerator

from maxcorr.cross_validation import find_best_params

warnings.filterwarnings("ignore")


class BaselineCalibrator:
    """Runs complexity sweeps to find expected range ceilings."""

    def __init__(self, n_samples: int = 500, n_splits: int = 5):
        self.n_samples = n_samples
        self.n_splits = n_splits
        self.generator = NonLinearDatasetGenerator()

    def run_asymptotic_sweep(
        self,
        dataset_name: str,
        max_degree: int = 15,
    ) -> pd.DataFrame:
        """
        Sweeps polynomial degrees from 2 to max_degree to find the empirical ceiling.
        """
        print(f"\n{'=' * 80}")
        print(f"Running Asymptotic Sweep for: {dataset_name}")
        print(f"Sweeping Polynomial Degrees 2 through {max_degree}...")
        print(f"{'=' * 80}")

        # Fetch the dataset
        datasets = {
            name: (a, b)
            for a, b, name in self.generator.get_all_datasets(self.n_samples)
        }
        if dataset_name not in datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found.")

        a, b = datasets[dataset_name]
        results = []

        # Force the algorithm to test one specific degree at a time
        for degree in range(2, max_degree + 1):
            print(f"Testing Degree {degree:2d}...", end="\r")

            # Lock the grid to exactly this degree
            param_grid = {"kernel_a": [degree], "kernel_b": [degree]}

            # Run cross-validation
            best_params, cv_result = find_best_params(
                a,
                b,
                algorithm="dk",
                param_grid=param_grid,
                n_splits=self.n_splits,
                verbose=0,
            )

            results.append(
                {
                    "Degree": degree,
                    "Train Score": cv_result.mean_train_score,
                    "Test Score": cv_result.mean_test_score,
                    "Overfitting Gap": abs(
                        cv_result.mean_train_score - cv_result.mean_test_score
                    ),
                }
            )

        print(f"Testing Degree {max_degree:2d}... Complete!      ")
        return pd.DataFrame(results)

    @staticmethod
    def plot_sweep(df: pd.DataFrame, dataset_name: str, save_path: str | None = None):
        """Plots the learning curve to visually prove the empirical ceiling."""
        plt.figure(figsize=(10, 6))

        # Plot Train and Test scores
        plt.plot(
            df["Degree"], df["Train Score"], "b.-", label="Train Score", linewidth=2
        )
        plt.plot(
            df["Degree"],
            df["Test Score"],
            "g.-",
            label="Test Score (Cross-Validated)",
            linewidth=2,
        )

        # Highlight the Empirical Ceiling (Max Test Score)
        max_test_idx = df["Test Score"].idxmax()
        max_test_score = df.loc[max_test_idx, "Test Score"]
        optimal_degree = df.loc[max_test_idx, "Degree"]

        plt.axhline(
            y=max_test_score,
            color="r",
            linestyle="--",
            alpha=0.5,
            label=f"Empirical Ceiling: {max_test_score:.2f}",
        )
        plt.plot(optimal_degree, max_test_score, "ro", markersize=10)

        # Formatting
        plt.title(
            f"Complexity Sweep: {dataset_name}\n(Finding the Empirical Ceiling)",
            fontweight="bold",
        )
        plt.xlabel("Polynomial Kernel Degree")
        plt.ylabel("Maximal Correlation Score")
        plt.xticks(df["Degree"])
        plt.ylim([0, 1.05])
        plt.grid(True, alpha=0.3)
        plt.legend(loc="lower right")

        # Fill the overfitting gap area
        plt.fill_between(
            df["Degree"],
            df["Test Score"],
            df["Train Score"],
            color="red",
            alpha=0.1,
            label="Overfitting Penalty",
        )

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved plot to {save_path}")

        plt.show()


def main():
    calibrator = BaselineCalibrator(n_samples=500)

    # Example 1: Finding the structural ceiling for the Swiss Roll
    df_swiss = calibrator.run_asymptotic_sweep("SwissRoll", max_degree=12)
    calibrator.plot_sweep(df_swiss, "SwissRoll", "sweep_swissroll.png")

    # Example 2: Proving the noise floor on the Independent dataset
    df_indep = calibrator.run_asymptotic_sweep("Independent", max_degree=10)
    calibrator.plot_sweep(df_indep, "Independent Noise", "sweep_independent.png")


if __name__ == "__main__":
    main()
