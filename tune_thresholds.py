"""
tune_thresholds.py

PS153 - Infiltration Prediction Engine
Sweeps rule_weight (rule-based vs baseline-deviation blend) and
alert_threshold against the six labelled scenarios in
infiltration_engine.SCENARIOS, and scores each combination on:

  1. stage_accuracy   - fraction of scenarios whose final predicted
                         stage matches the expected ATT&CK stage
  2. alert_correctness - benign scenario should NOT fire the alert,
                         every attack scenario SHOULD
  3. separation        - gap between the benign max_probability and
                         the *lowest* attack max_probability. Bigger
                         gap = cleaner, more convincing demo (no
                         borderline calls near the threshold line)

Run: python3 tune_thresholds.py
"""

from __future__ import annotations
import itertools
import numpy as np

from infiltration_engine import (
    run_all_scenarios, SCENARIOS, DEFAULT_RULE_THRESHOLDS,
)

# Map each scenario name to the stage label it SHOULD produce.
EXPECTED_STAGE = {
    "benign": "Benign",
    "reconnaissance": "Reconnaissance",
    "initial_access": "Initial Access",
    "lateral_movement": "Lateral Movement",
    "command_and_control": "Command & Control",
    "exfiltration": "Exfiltration",
}


def score_config(rule_weight: float, alert_threshold: float,
                  rule_thresholds: dict | None = None, k: int = 6, seed: int = 42) -> dict:
    results = run_all_scenarios(k=k, seed=seed, rule_weight=rule_weight,
                                 alert_threshold=alert_threshold, rule_thresholds=rule_thresholds)

    correct = sum(1 for name, res in results.items() if res["peak_stage"] == EXPECTED_STAGE[name])
    stage_accuracy = correct / len(results)

    benign_prob = results["benign"]["max_probability"]
    benign_alert_ok = results["benign"]["overall_infiltration_flag"] is False
    attack_names = [n for n in results if n != "benign"]
    attack_probs = [results[n]["max_probability"] for n in attack_names]
    attacks_alert_ok = all(results[n]["overall_infiltration_flag"] for n in attack_names)

    separation = (min(attack_probs) - benign_prob) if attack_probs else 0.0
    alert_correctness = int(benign_alert_ok) + sum(
        int(results[n]["overall_infiltration_flag"]) for n in attack_names
    )
    alert_correctness /= len(results)

    return {
        "rule_weight": rule_weight,
        "alert_threshold": alert_threshold,
        "stage_accuracy": stage_accuracy,
        "alert_correctness": alert_correctness,
        "separation": separation,
        "benign_prob": benign_prob,
        "min_attack_prob": min(attack_probs) if attack_probs else None,
        "results": results,
    }


def grid_search():
    rule_weights = [0.0, 0.25, 0.5, 0.75, 1.0]
    alert_thresholds = [0.5, 0.6, 0.7]

    rows = []
    for rw, at in itertools.product(rule_weights, alert_thresholds):
        rows.append(score_config(rw, at))

    # Rank: correctness first (must classify + alert correctly), then
    # by separation (bigger = safer margin around the threshold line).
    rows.sort(key=lambda r: (r["stage_accuracy"], r["alert_correctness"], r["separation"]), reverse=True)
    return rows


def print_table(rows):
    header = f"{'rule_w':>7} {'alert_th':>9} {'stage_acc':>10} {'alert_ok':>9} {'separation':>11} {'benign_p':>9} {'min_atk_p':>10}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['rule_weight']:>7.2f} {r['alert_threshold']:>9.2f} {r['stage_accuracy']:>10.2f} "
              f"{r['alert_correctness']:>9.2f} {r['separation']:>11.3f} {r['benign_prob']:>9.3f} "
              f"{r['min_attack_prob']:>10.3f}")


def print_scenario_breakdown(best):
    print(f"\nScenario breakdown for the best config "
          f"(rule_weight={best['rule_weight']}, alert_threshold={best['alert_threshold']}):")
    print(f"{'scenario':>22} {'expected':>18} {'peak_stage':>18} {'max_prob':>9} {'alert':>6}")
    for name, res in best["results"].items():
        expected = EXPECTED_STAGE[name]
        ok = "OK" if res["peak_stage"] == expected else "MISS"
        print(f"{name:>22} {expected:>18} {res['peak_stage']:>18} "
              f"{res['max_probability']:>9.3f} {str(res['overall_infiltration_flag']):>6}  [{ok}]")


if __name__ == "__main__":
    rows = grid_search()
    print("=" * 70)
    print("Grid search: rule_weight x alert_threshold")
    print("(sorted best-first: stage_accuracy, then alert_correctness, then separation)")
    print("=" * 70)
    print_table(rows)

    best = rows[0]
    print_scenario_breakdown(best)

    print(f"\nRECOMMENDED CONFIG: rule_weight={best['rule_weight']}, "
          f"alert_threshold={best['alert_threshold']}")
    print(f"  -> stage_accuracy={best['stage_accuracy']:.0%}, "
          f"alert_correctness={best['alert_correctness']:.0%}, "
          f"separation={best['separation']:.3f} "
          f"(benign={best['benign_prob']:.3f} vs weakest attack={best['min_attack_prob']:.3f})")
