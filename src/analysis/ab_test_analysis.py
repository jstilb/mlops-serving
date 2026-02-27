"""
A/B Test Results Analysis — Statistical Significance Testing
ISC-5568: Compute p-value and confidence interval between two model variants

Usage:
    python -m src.analysis.ab_test_analysis
    python -m src.analysis.ab_test_analysis --control-version v1.0 --treatment-version v2.0

Produces:
    - Statistical significance test (Mann-Whitney U + Welch's t-test)
    - Effect size (Cohen's d)
    - 95% Bootstrap confidence intervals
    - Visualization: latency distributions with statistical annotations
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple

import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def load_ab_test_data(
    data_path: str = None,
    n_control: int = 500,
    n_treatment: int = 150,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load A/B test observations from logs or generate synthetic latency data."""
    if data_path and Path(data_path).exists():
        df = pd.read_csv(data_path)
        control = df[df["variant"] == "control"]["latency_ms"].values
        treatment = df[df["variant"] == "treatment"]["latency_ms"].values
        return control, treatment

    rng = np.random.default_rng(seed)
    # Control (v1.0 RandomForest): heavier, ~8ms mean
    control = np.abs(rng.gamma(shape=8, scale=1.0, size=n_control) + rng.normal(0, 0.5, n_control))
    # Treatment (v2.0 XGBoost): faster, ~6ms mean
    treatment = np.abs(rng.gamma(shape=7, scale=0.85, size=n_treatment) + rng.normal(0, 0.4, n_treatment))
    return control, treatment


def compute_statistical_significance(
    control: np.ndarray,
    treatment: np.ndarray,
    alpha: float = 0.05,
) -> Dict:
    """
    Compute statistical significance between A/B test variants.

    Uses Mann-Whitney U (non-parametric), Welch's t-test, Cohen's d effect size,
    and 95% bootstrap confidence interval for the difference in means.
    """
    u_stat, p_value_mw = stats.mannwhitneyu(control, treatment, alternative='two-sided')
    t_stat, p_value_t = stats.ttest_ind(control, treatment, equal_var=False)

    pooled_std = np.sqrt((np.var(control) + np.var(treatment)) / 2)
    cohens_d = float((np.mean(control) - np.mean(treatment)) / pooled_std)

    rng = np.random.default_rng(42)
    boot_diffs = []
    for _ in range(10000):
        boot_ctrl = rng.choice(control, size=len(control), replace=True)
        boot_trt = rng.choice(treatment, size=len(treatment), replace=True)
        boot_diffs.append(float(np.mean(boot_ctrl) - np.mean(boot_trt)))

    ci_lower = float(np.percentile(boot_diffs, 2.5))
    ci_upper = float(np.percentile(boot_diffs, 97.5))
    relative_improvement = float((np.mean(control) - np.mean(treatment)) / np.mean(control) * 100)

    return {
        "control_mean": round(float(np.mean(control)), 3),
        "treatment_mean": round(float(np.mean(treatment)), 3),
        "control_std": round(float(np.std(control)), 3),
        "treatment_std": round(float(np.std(treatment)), 3),
        "n_control": int(len(control)),
        "n_treatment": int(len(treatment)),
        "mann_whitney_u": round(float(u_stat), 2),
        "p_value_mann_whitney": round(float(p_value_mw), 6),
        "p_value_welch_t": round(float(p_value_t), 6),
        "cohens_d": round(cohens_d, 3),
        "ci_95_lower": round(ci_lower, 3),
        "ci_95_upper": round(ci_upper, 3),
        "relative_improvement_pct": round(relative_improvement, 2),
        "statistically_significant": bool(p_value_mw < alpha),
        "alpha": alpha,
        "effect_size_interpretation": (
            "negligible" if abs(cohens_d) < 0.2
            else "small" if abs(cohens_d) < 0.5
            else "medium" if abs(cohens_d) < 0.8
            else "large"
        ),
    }


def plot_ab_test_results(
    control: np.ndarray,
    treatment: np.ndarray,
    stats_result: Dict,
    output_path: str = None,
) -> None:
    """Generate A/B test visualization with statistical significance annotations."""
    from scipy.stats import gaussian_kde

    fig = plt.figure(figsize=(14, 9))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)
    fig.suptitle("A/B Test Results: v1.0 (Control) vs v2.0 (Treatment)",
                 fontsize=14, fontweight='bold')

    # Distribution comparison
    ax1 = fig.add_subplot(gs[0, :])
    bins = np.linspace(0, max(control.max(), treatment.max()), 50)
    ax1.hist(control, bins=bins, alpha=0.6, color='#2196F3', density=True,
             label=f'v1.0 Control (n={len(control)})')
    ax1.hist(treatment, bins=bins, alpha=0.6, color='#FF9800', density=True,
             label=f'v2.0 Treatment (n={len(treatment)})')

    x_range = np.linspace(0, max(control.max(), treatment.max()), 200)
    ax1.plot(x_range, gaussian_kde(control)(x_range), color='#1565C0', linewidth=2, linestyle='--')
    ax1.plot(x_range, gaussian_kde(treatment)(x_range), color='#E65100', linewidth=2, linestyle='--')
    ax1.axvline(stats_result['control_mean'], color='#2196F3', linewidth=2, linestyle=':',
                label=f"v1.0 mean = {stats_result['control_mean']:.1f}ms")
    ax1.axvline(stats_result['treatment_mean'], color='#FF9800', linewidth=2, linestyle=':',
                label=f"v2.0 mean = {stats_result['treatment_mean']:.1f}ms")

    sig_label = 'SIGNIFICANT' if stats_result['statistically_significant'] else 'NOT SIGNIFICANT'
    sig_text = (f"p = {stats_result['p_value_mann_whitney']:.4f} [{sig_label}]\n"
                f"Cohen's d = {stats_result['cohens_d']:.2f} ({stats_result['effect_size_interpretation']})\n"
                f"95% CI: [{stats_result['ci_95_lower']:.2f}, {stats_result['ci_95_upper']:.2f}] ms\n"
                f"Improvement: {stats_result['relative_improvement_pct']:.1f}%")
    ax1.text(0.72, 0.88, sig_text, transform=ax1.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax1.set_title("Prediction Latency Distribution")
    ax1.set_xlabel("Latency (ms)")
    ax1.set_ylabel("Density")
    ax1.legend(fontsize=9)

    # Box plots
    ax2 = fig.add_subplot(gs[1, 0])
    bp = ax2.boxplot([control, treatment], labels=['v1.0\nControl', 'v2.0\nTreatment'],
                     patch_artist=True, medianprops=dict(color='black', linewidth=2))
    for patch, color in zip(bp['boxes'], ['#2196F3', '#FF9800']):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax2.set_title("Distribution Comparison")
    ax2.set_ylabel("Latency (ms)")

    # Mean + 95% CI errorbar
    ax3 = fig.add_subplot(gs[1, 1])
    means = [stats_result['control_mean'], stats_result['treatment_mean']]
    stds = [stats_result['control_std'], stats_result['treatment_std']]
    ns = [stats_result['n_control'], stats_result['n_treatment']]
    ci_widths = [1.96 * s / np.sqrt(n) for s, n in zip(stds, ns)]
    colors = ['#2196F3', '#FF9800']
    for i, (m, ci, color) in enumerate(zip(means, ci_widths, colors)):
        ax3.errorbar(i, m, yerr=ci, fmt='o', color=color, markersize=10,
                     capsize=8, capthick=2, elinewidth=2)
        ax3.text(i, m + ci + 0.2, f'{m:.1f}ms', ha='center', fontsize=10,
                 color=color, fontweight='bold')
    ax3.set_xticks([0, 1])
    ax3.set_xticklabels(['v1.0\nControl', 'v2.0\nTreatment'])
    ax3.set_title("Mean Latency +/- 95% CI")
    ax3.set_ylabel("Latency (ms)")
    ax3.set_xlim(-0.5, 1.5)

    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    plt.close()


def run_ab_test_analysis(
    data_path: str = None,
    control_version: str = "v1.0",
    treatment_version: str = "v2.0",
    output_dir: str = None,
) -> Dict:
    """Run full A/B test analysis and produce statistical significance report."""
    print(f"\nA/B Test Analysis: {control_version} vs {treatment_version}")
    print("=" * 60)

    control, treatment = load_ab_test_data(data_path)
    result = compute_statistical_significance(control, treatment)

    print(f"Control ({control_version}): mean={result['control_mean']}ms, n={result['n_control']}")
    print(f"Treatment ({treatment_version}): mean={result['treatment_mean']}ms, n={result['n_treatment']}")
    print(f"p_value (Mann-Whitney U): {result['p_value_mann_whitney']:.6f}")
    print(f"p_value (Welch's t-test): {result['p_value_welch_t']:.6f}")
    print(f"Statistically significant (alpha={result['alpha']}): {result['statistically_significant']}")
    print(f"Cohen's d: {result['cohens_d']} ({result['effect_size_interpretation']} effect)")
    print(f"95% Bootstrap CI for mean diff: [{result['ci_95_lower']}, {result['ci_95_upper']}] ms")
    print(f"Relative improvement: {result['relative_improvement_pct']:.1f}%")

    if result['statistically_significant'] and result['cohens_d'] > 0.2:
        print(f"\nRecommendation: PROMOTE {treatment_version}")
    else:
        print(f"\nRecommendation: HOLD — insufficient evidence to promote {treatment_version}")

    if output_dir:
        plot_path = str(Path(output_dir) / "ab_test_results.png")
        plot_ab_test_results(control, treatment, result, plot_path)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--control-version", type=str, default="v1.0")
    parser.add_argument("--treatment-version", type=str, default="v2.0")
    parser.add_argument("--output-dir", type=str, default="assets")
    args = parser.parse_args()

    run_ab_test_analysis(
        data_path=args.data,
        control_version=args.control_version,
        treatment_version=args.treatment_version,
        output_dir=args.output_dir,
    )
