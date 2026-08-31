"""
horizon_experiment.py

PS153 - Infiltration Prediction Engine
Runs evaluate_horizon_stability() across MULTIPLE attack scenarios
(not just one), so the "optimal K" choice generalises instead of
being tuned to a single trajectory. Produces:

  1. a per-scenario table of mean final-state variance and mean
     max-probability at each K
  2. an aggregate table averaged across scenarios
  3. a drafted paragraph for the 2-page architecture doc, filled in
     with the real numbers from this run

Run: python3 horizon_experiment.py
"""

from __future__ import annotations
import numpy as np

from infiltration_engine import (
    evaluate_horizon_stability, build_scenario_state, SCENARIOS,
)

K_VALUES = (3, 5, 7, 10, 15)
N_TRIALS = 20
SCENARIOS_TO_TEST = ["reconnaissance", "initial_access", "lateral_movement",
                      "command_and_control", "exfiltration"]  # skip benign - nothing to escalate


def run_experiment():
    per_scenario = {}
    for name in SCENARIOS_TO_TEST:
        state = build_scenario_state(name)
        per_scenario[name] = evaluate_horizon_stability(
            state, k_values=K_VALUES, n_trials=N_TRIALS, base_seed=hash(name) % 1000
        )
    return per_scenario


def aggregate(per_scenario: dict) -> dict:
    agg = {}
    for k in K_VALUES:
        variances = [per_scenario[name][k]["mean_final_variance"] for name in SCENARIOS_TO_TEST]
        probs = [per_scenario[name][k]["mean_max_probability"] for name in SCENARIOS_TO_TEST]
        agg[k] = {
            "mean_variance_across_scenarios": float(np.mean(variances)),
            "mean_max_probability_across_scenarios": float(np.mean(probs)),
        }
    return agg


def pick_recommended_k(agg: dict, variance_budget: float = 0.003) -> int:
    """Largest K whose mean variance stays under the budget - i.e. the
    furthest we can forecast before compounding error gets noisy enough
    to undermine trust in the probability timeline."""
    candidates = [k for k in K_VALUES if agg[k]["mean_variance_across_scenarios"] <= variance_budget]
    return max(candidates) if candidates else min(K_VALUES)


def print_tables(per_scenario, agg):
    print("=" * 78)
    print(f"Per-scenario horizon stability (n_trials={N_TRIALS} per K)")
    print("=" * 78)
    header = f"{'scenario':>20} " + " ".join(f"K={k:<4}" for k in K_VALUES)
    print(header)
    for name in SCENARIOS_TO_TEST:
        row = f"{name:>20} "
        row += " ".join(f"{per_scenario[name][k]['mean_final_variance']:<6.4f}" for k in K_VALUES)
        print(row)

    print("\n" + "=" * 78)
    print("Aggregate across all attack scenarios")
    print("=" * 78)
    print(f"{'K':>4} {'mean_variance':>15} {'mean_max_probability':>22}")
    for k in K_VALUES:
        print(f"{k:>4} {agg[k]['mean_variance_across_scenarios']:>15.5f} "
              f"{agg[k]['mean_max_probability_across_scenarios']:>22.3f}")


def draft_paragraph(agg: dict, recommended_k: int) -> str:
    var_at_rec = agg[recommended_k]["mean_variance_across_scenarios"]
    var_at_next = agg.get(K_VALUES[K_VALUES.index(recommended_k) + 1]) if recommended_k != K_VALUES[-1] else None
    lowest_k, highest_k = K_VALUES[0], K_VALUES[-1]
    var_low = agg[lowest_k]["mean_variance_across_scenarios"]
    var_high = agg[highest_k]["mean_variance_across_scenarios"]
    growth_factor = var_high / var_low if var_low > 0 else float("inf")

    para = (
        f"To choose the forecast horizon K, we ran the autoregressive rollout "
        f"{N_TRIALS} times per horizon across {len(SCENARIOS_TO_TEST)} representative attack "
        f"trajectories (Reconnaissance, Initial Access, Lateral Movement, Command & Control, "
        f"Exfiltration), sweeping K over {K_VALUES}, and measured how much the predicted final "
        f"state varied across runs at each horizon (compounding error). Mean variance rose from "
        f"{var_low:.4f} at K={lowest_k} to {var_high:.4f} at K={highest_k} - roughly a "
        f"{growth_factor:.1f}x increase - confirming that prediction uncertainty compounds as "
        f"expected with autoregressive feedback. We selected K={recommended_k} as the forecast "
        f"horizon, the largest tested value where mean variance stays low "
        f"({var_at_rec:.4f}) while still giving defenders a meaningful lead window before "
        f"predicted compromise."
    )
    return para


if __name__ == "__main__":
    per_scenario = run_experiment()
    agg = aggregate(per_scenario)
    print_tables(per_scenario, agg)

    recommended_k = pick_recommended_k(agg)
    print(f"\nRecommended K: {recommended_k}")

    print("\n" + "=" * 78)
    print("Draft paragraph for the 2-page architecture doc")
    print("=" * 78)
    print(draft_paragraph(agg, recommended_k))
