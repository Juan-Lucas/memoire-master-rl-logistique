from __future__ import annotations

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

