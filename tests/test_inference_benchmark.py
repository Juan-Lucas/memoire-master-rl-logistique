from __future__ import annotations

import pandas as pd

from memoire_master_rl_logistique.experiments.statistical_tests import summarise_energy_efficiency
from memoire_master_rl_logistique.rl.evaluate_agent import benchmark_policy_inference


def test_benchmark_policy_inference_returns_metrics() -> None:
    def dummy_policy(obs, info=None):
        return 0

    metrics = benchmark_policy_inference(dummy_policy, n_calls=5, warmup_calls=1)

    assert metrics["n_calls"] == 5
    assert metrics["mean_ms"] >= 0.0
    assert metrics["std_ms"] >= 0.0
    assert metrics["min_ms"] >= 0.0
    assert metrics["max_ms"] >= metrics["min_ms"]


def test_summarise_energy_efficiency_returns_metrics(tmp_path) -> None:
    csv_path = tmp_path / "benchmark_results.csv"
    pd.DataFrame(
        [
            {"scenario": "nominal", "policy": "DQN", "specific_fuel_l_per_ton": 0.0405},
            {"scenario": "nominal", "policy": "Fixed", "specific_fuel_l_per_ton": 0.0431},
            {"scenario": "nominal", "policy": "DQN", "specific_fuel_l_per_ton": 0.0408},
            {"scenario": "nominal", "policy": "Fixed", "specific_fuel_l_per_ton": 0.0429},
        ]
    ).to_csv(csv_path, index=False)

    metrics = summarise_energy_efficiency(csv_path)

    assert metrics["scenario"] == "nominal"
    assert metrics["policy_a"] == "DQN"
    assert metrics["policy_b"] == "Fixed"
    assert metrics["delta_pct"] < 0.0
    assert 0.0 <= metrics["p_value"] <= 1.0
