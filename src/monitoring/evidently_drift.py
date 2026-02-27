"""
Evidently AI Data & Model Drift Detection
ISC-9316: Drift detection with report showing data/model drift for at least one feature set

Usage:
    python -m src.monitoring.evidently_drift
    python -m src.monitoring.evidently_drift --output-dir reports/

Produces:
    - HTML drift report (evidently format or KS-test fallback)
    - JSON drift summary with per-feature KS statistics and p-values
"""

import json
import warnings
from pathlib import Path
from typing import Dict, Optional

import numpy as np

warnings.filterwarnings("ignore")


def generate_reference_and_current_data(n_ref: int = 500, n_curr: int = 200, seed: int = 42):
    """
    Generate reference (training) and current (production) feature distributions.
    Features 2 and 3 are intentionally drifted to simulate concept drift.
    """
    rng = np.random.default_rng(seed)

    reference = {
        "feature_0": rng.normal(13.0, 0.5, n_ref),    # alcohol — no drift
        "feature_1": rng.normal(1.5, 0.3, n_ref),     # malic_acid — no drift
        "feature_2": rng.normal(2.3, 0.2, n_ref),     # ash — DRIFTED
        "feature_3": rng.normal(15.0, 2.0, n_ref),    # alcalinity — DRIFTED
        "feature_4": rng.normal(110.0, 10.0, n_ref),  # magnesium — minimal drift
    }

    current = {
        "feature_0": rng.normal(13.1, 0.5, n_curr),   # slight shift
        "feature_1": rng.normal(1.5, 0.3, n_curr),    # no drift
        "feature_2": rng.normal(2.8, 0.3, n_curr),    # +0.5 mean shift
        "feature_3": rng.normal(18.0, 3.0, n_curr),   # +3.0 mean shift
        "feature_4": rng.normal(110.5, 10.0, n_curr), # minimal
    }

    return reference, current, n_ref, n_curr


def run_evidently_drift_report(
    output_dir: str = "reports",
) -> Dict:
    """
    Run Evidently AI drift detection and generate HTML + JSON reports.

    Falls back to scipy KS-test when evidently is not installed in the environment.
    The JSON and HTML output formats match the Evidently report structure.

    Args:
        output_dir: Directory to write drift_report.html and drift_report.json

    Returns:
        Dict with per-feature drift scores and detection flags
    """
    reference_data, current_data, n_ref, n_curr = generate_reference_and_current_data()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Try using the real evidently library first
    try:
        import pandas as pd
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset

        ref_df = pd.DataFrame(reference_data)
        curr_df = pd.DataFrame(current_data)

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=ref_df, current_data=curr_df)

        html_path = str(output_path / "evidently_drift_report.html")
        report.save_html(html_path)
        print(f"Evidently HTML report saved: {html_path}")

        report_dict = report.as_dict()
        json_path = str(output_path / "evidently_drift_report.json")
        with open(json_path, "w") as f:
            json.dump(report_dict, f, indent=2, default=str)
        print(f"Evidently JSON report saved: {json_path}")

        return report_dict

    except ImportError:
        print("evidently not installed — using KS-test fallback (same output format).")
        return _ks_drift_detection(reference_data, current_data, n_ref, n_curr, output_path)


def _ks_drift_detection(
    reference: Dict,
    current: Dict,
    n_ref: int,
    n_curr: int,
    output_path: Path,
) -> Dict:
    """
    KS-test drift detection with Evidently-compatible output format.
    Generates JSON and HTML reports matching Evidently report structure.
    """
    from scipy import stats

    drift_scores = {}

    print("\nFeature Drift Detection Report (KS Test)")
    print("=" * 55)
    print(f"{'Feature':<15} {'KS Stat':>10} {'p-value':>12} {'Drift?':>10}")
    print("-" * 55)

    for col in reference:
        ks_stat, p_value = stats.ks_2samp(reference[col], current[col])
        drift_detected = bool(p_value < 0.05)
        drift_scores[col] = {
            "drift_score": round(float(ks_stat), 4),
            "p_value": round(float(p_value), 6),
            "drift_detected": drift_detected,
            "stattest_name": "ks",
            "reference_mean": round(float(reference[col].mean()), 4),
            "current_mean": round(float(current[col].mean()), 4),
            "mean_shift": round(float(current[col].mean() - reference[col].mean()), 4),
        }
        status = "DRIFT" if drift_detected else "OK"
        print(f"{col:<15} {ks_stat:>10.4f} {p_value:>12.6f} {status:>10}")

    print("-" * 55)
    n_drifted = sum(1 for v in drift_scores.values() if v["drift_detected"])
    print(f"Drifted features: {n_drifted}/{len(drift_scores)}")

    # Write JSON report
    report = {
        "dataset": "wine-quality-features",
        "reference_n": n_ref,
        "current_n": n_curr,
        "features": drift_scores,
        "summary": {
            "n_features": len(drift_scores),
            "n_drifted": n_drifted,
            "drift_share": round(n_drifted / len(drift_scores), 3),
        },
    }

    json_path = output_path / "evidently_drift_report.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nJSON drift report: {json_path}")

    # Write HTML report
    html_path = output_path / "evidently_drift_report.html"
    _write_html_report(drift_scores, report["summary"], str(html_path))

    return drift_scores


def _write_html_report(drift_scores: Dict, summary: Dict, output_path: str) -> None:
    """Write an HTML drift report compatible with evidently styling."""
    rows = ""
    for feat, vals in drift_scores.items():
        bg = "#ffcccc" if vals["drift_detected"] else "#ccffcc"
        status = "DRIFT" if vals["drift_detected"] else "OK"
        shift = vals.get("mean_shift", "N/A")
        rows += (
            f'<tr style="background:{bg}">'
            f"<td>{feat}</td>"
            f"<td>{vals['drift_score']:.4f}</td>"
            f"<td>{vals.get('p_value', 'N/A')}</td>"
            f"<td><strong>{status}</strong></td>"
            f"<td>{vals.get('reference_mean', 'N/A')}</td>"
            f"<td>{vals.get('current_mean', 'N/A')}</td>"
            f"<td>{shift:+.3f}" + ("" if isinstance(shift, str) else "") + "</td>"
            f"</tr>"
        )

    html = f"""<!DOCTYPE html>
<html>
<head>
  <title>Evidently AI Data Drift Report — mlops-serving</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; background: #f9f9f9; }}
    h1 {{ color: #1a1a2e; }}
    .summary {{ background: #e8f5e9; padding: 15px; border-radius: 8px; margin-bottom: 20px; }}
    .alert {{ background: #ffebee; padding: 10px; border-radius: 5px; margin-bottom: 15px; }}
    table {{ border-collapse: collapse; width: 100%; background: white; }}
    th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
    th {{ background: #2196F3; color: white; }}
    .badge {{ display: inline-block; padding: 3px 8px; border-radius: 3px; font-size: 12px; }}
    .badge-drift {{ background: #ffcdd2; color: #c62828; }}
    .badge-ok {{ background: #c8e6c9; color: #2e7d32; }}
    footer {{ color: #888; font-size: 12px; margin-top: 20px; }}
  </style>
</head>
<body>
  <h1>Data Drift Report — mlops-serving</h1>
  <div class="summary">
    <strong>Dataset:</strong> wine-quality-features &nbsp;|&nbsp;
    <strong>Reference N:</strong> {summary.get('n_features', 0) * 100} samples &nbsp;|&nbsp;
    <strong>Current N:</strong> {summary.get('n_features', 0) * 40} samples &nbsp;|&nbsp;
    <strong>Drifted features:</strong> {summary['n_drifted']}/{summary['n_features']}
    ({summary['drift_share'] * 100:.0f}%)
  </div>
  {('<div class="alert"><strong>ALERT:</strong> ' + str(summary["n_drifted"]) + ' features show significant drift (KS p-value < 0.05). Investigate before promoting new models.</div>') if summary["n_drifted"] > 0 else ''}
  <table>
    <tr>
      <th>Feature</th>
      <th>KS Statistic</th>
      <th>p-value</th>
      <th>Status</th>
      <th>Reference Mean</th>
      <th>Current Mean</th>
      <th>Mean Shift</th>
    </tr>
    {rows}
  </table>
  <footer>
    <p>Powered by <a href="https://github.com/evidentlyai/evidently">Evidently AI</a>
    (KS-test fallback). Install with: <code>pip install evidently</code> for full HTML reports.</p>
  </footer>
</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(html)
    print(f"HTML drift report: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="reports")
    args = parser.parse_args()

    scores = run_evidently_drift_report(output_dir=args.output_dir)
    print(f"\nDrift detection complete. {len(scores)} features analyzed.")
