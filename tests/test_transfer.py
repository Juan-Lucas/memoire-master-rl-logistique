"""Tests de T27 (M1) : évaluation des agents spécialisés et des agents nominaux transférés."""

import dataclasses
from pathlib import Path

import pandas as pd
import pytest

from memoire_master_rl_logistique.analysis.statistical_tests import load_results, select_mode
from memoire_master_rl_logistique.analysis.transfer import transfer_table
from memoire_master_rl_logistique.experiments import policies
from memoire_master_rl_logistique.experiments import run_benchmark as rb
from memoire_master_rl_logistique.experiments.scenarios import (
    SCENARIOS,
    SPECIALISED,
    TRANSFERRED,
    evaluation_mode,
    training_regimes,
)
from memoire_master_rl_logistique.rl.train_q_learning import train_q_learning
from memoire_master_rl_logistique.rl.train_sarsa import train_sarsa

V1_MODELS = Path("data/results_v1_article")
HEURISTICS = set(policies.HEURISTICS)


def test_regimes_d_entrainement_par_scenario():
    """Régime perturbé : spécialisés et nominaux ; non vu : nominaux ; sinon : les siens."""
    assert training_regimes("high_load") == ["high_load", "nominal"]
    assert training_regimes("high_breakdown") == ["high_breakdown", "nominal"]
    assert training_regimes("unseen_medium_load") == ["nominal"]
    assert training_regimes("nominal") == ["nominal"]
    assert training_regimes("ablation_w4_00") == ["ablation_w4_00"]


def test_mode_d_evaluation():
    assert evaluation_mode("high_load", "high_load") == SPECIALISED
    assert evaluation_mode("high_load", "nominal") == TRANSFERRED
    assert evaluation_mode("unseen_medium_load", "nominal") == TRANSFERRED
    assert evaluation_mode("nominal", "nominal") == SPECIALISED


def _tiny_train(calls):
    def tiny_train(algorithm, scenario, train_seed, output_dir):
        calls.append(output_dir.name)
        trainer = train_q_learning if algorithm == "Q-Learning" else train_sarsa
        trainer(n_episodes=1, seed=train_seed, output_dir=str(output_dir),
                truck_count=scenario.truck_count, shovel_count=scenario.shovel_count,
                dump_count=scenario.dump_count, episode_minutes=scenario.episode_minutes,
                breakdown_probability=scenario.breakdown_probability,
                reward_weights=scenario.reward_weights)
    return tiny_train


@pytest.mark.parametrize("regime", ["high_load", "high_breakdown"])
def test_regime_perturbe_specialises_et_transferes(regime, monkeypatch, tmp_path):
    """Chaque agent est évalué deux fois (spécialisé, transféré) sur les mêmes graines ;
    les heuristiques une seule fois, sans mode. En forte charge, cela suppose une
    observation indépendante du nombre de camions (T18)."""
    calls = []
    monkeypatch.setattr(rb, "_train", _tiny_train(calls))
    scenario = dataclasses.replace(SCENARIOS[regime], seeds=[42, 43])

    rows = rb.run_benchmark(scenario, tmp_path, train_ppo_flag=False, train_seeds=(1,))

    assert sorted(calls) == sorted([f"q_learning_{regime}_seed1", f"sarsa_{regime}_seed1",
                                    "q_learning_nominal_seed1", "sarsa_nominal_seed1"])
    df = pd.DataFrame(rows)
    heuristics = df[df["policy"].isin(HEURISTICS)]
    assert heuristics["mode"].isna().all() and len(heuristics) == len(HEURISTICS) * 2
    agents = df[~df["policy"].isin(HEURISTICS)]
    assert agents.groupby(["policy", "mode"]).size().to_dict() == {
        ("Q-Learning", SPECIALISED): 2, ("Q-Learning", TRANSFERRED): 2,
        ("SARSA", SPECIALISED): 2, ("SARSA", TRANSFERRED): 2}
    # Les mêmes graines d'évaluation dans les deux modes : comparaison appariée possible.
    seeds = agents.groupby("mode")["seed"].apply(sorted).to_dict()
    assert seeds[SPECIALISED] == seeds[TRANSFERRED] == [42, 42, 43, 43]
    # Le CSV accepte le même agent dans deux modes (pas un doublon).
    rb.write_results(rows, tmp_path, run_id="t27")


def test_scenario_non_vu_seulement_transfere(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(rb, "_train", _tiny_train(calls))
    scenario = dataclasses.replace(SCENARIOS["unseen_medium_load"], seeds=[42])

    rows = rb.run_benchmark(scenario, tmp_path, train_ppo_flag=False, train_seeds=(1,))

    assert sorted(calls) == ["q_learning_nominal_seed1", "sarsa_nominal_seed1"]
    modes = {r["mode"] for r in rows if r["policy"] not in HEURISTICS}
    assert modes == {TRANSFERRED}


def test_seuls_les_algorithmes_du_scenario_sont_entraines_et_evalues(monkeypatch, tmp_path):
    """Ablation sur w4 : seuls les algorithmes du scénario sont entraînés et évalués
    (ici Q-Learning, pour rester rapide) ; les heuristiques le sont toujours."""
    calls = []
    monkeypatch.setattr(rb, "_train", _tiny_train(calls))
    scenario = dataclasses.replace(SCENARIOS["ablation_w4_05"], seeds=[42],
                                   algorithms=("Q-Learning",))

    rows = rb.run_benchmark(scenario, tmp_path, train_ppo_flag=True, train_seeds=(1,))

    assert calls == ["q_learning_ablation_w4_05_seed1"]
    assert {r["policy"] for r in rows} == HEURISTICS | {"Q-Learning"}


def test_modele_d_avant_t26_jamais_entraine(monkeypatch, tmp_path):
    """train_seed=None désigne un modèle v1 : absent, il est signalé, pas entraîné."""
    monkeypatch.setattr(rb, "_train", lambda *a: pytest.fail("ne doit pas entraîner"))
    scenario = dataclasses.replace(SCENARIOS["nominal"], seeds=[42])
    with pytest.raises(FileNotFoundError, match="avant T26"):
        rb.run_benchmark(scenario, tmp_path, train_ppo_flag=False, train_seeds=(None,))


@pytest.mark.skipif(not V1_MODELS.exists(), reason="modèles de l'article v1 absents")
def test_q_tables_v1_transferees_en_fortes_pannes():
    """Faisable avec les modèles v1 (plan) : les Q-tables nominales v1 s'évaluent en
    fortes pannes sans réentraînement. (PPO et DQN v1 ne peuvent plus être rejoués
    depuis T18 ; leur vérification historique est dans le journal de T27.)"""
    scenario = dataclasses.replace(SCENARIOS["high_breakdown"], seeds=[42, 43])
    # Les Q-tables v1 n'ont pas de metadata.json : le chargeur doit avertir (T8).
    with pytest.warns(UserWarning, match="allow_legacy"):
        rows = rb.run_benchmark(scenario, V1_MODELS, train_ppo_flag=False,
                                train_seeds=(None,), allow_legacy_models=True)
    agents = pd.DataFrame([r for r in rows if r["policy"] not in HEURISTICS])
    assert set(agents["mode"]) == {SPECIALISED, TRANSFERRED}
    assert (agents["productivity_tph"] > 0).all()


# --- Analyse ---------------------------------------------------------------------

def _results(diff=-10.0, train_seeds=(1,), seeds=(42, 43, 44)):
    rows = [{"scenario": "high_load", "policy": "Fixed", "mode": None, "train_seed": None,
             "seed": s, "productivity_tph": 50.0} for s in seeds]
    for ts in train_seeds:
        for s in seeds:
            base = 100.0 + s + ts
            rows.append({"scenario": "high_load", "policy": "PPO", "mode": SPECIALISED,
                         "train_seed": ts, "seed": s, "productivity_tph": base})
            rows.append({"scenario": "high_load", "policy": "PPO", "mode": TRANSFERRED,
                         "train_seed": ts, "seed": s, "productivity_tph": base + diff})
    return pd.DataFrame(rows)


def test_select_mode():
    """Par défaut, les spécialisés et les heuristiques ; None garde tout ; CSV v1 intact."""
    data = _results()
    assert set(select_mode(data)["mode"].dropna()) == {SPECIALISED}
    assert len(select_mode(data)) == 6
    assert len(select_mode(data, None)) == 9
    v1 = data.drop(columns="mode")
    assert len(select_mode(v1)) == len(v1)


def test_load_results_ne_melange_pas_les_modes(tmp_path):
    """Les tableaux principaux (mode par défaut) ne voient pas les agents transférés ;
    avec mode=None, spécialisé et transféré ne sont pas des doublons."""
    path = tmp_path / "r.csv"
    _results().to_csv(path, index=False)

    assert set(load_results(path, n_expected=3)["mode"].dropna()) == {SPECIALISED}
    assert len(load_results(path, n_expected=3, mode=None)) == 9


def test_cout_du_transfert():
    """Écart transféré − spécialisé connu (−10 t/h sur chaque graine)."""
    table = transfer_table(_results(diff=-10.0))

    row = table.iloc[0]
    assert (row["scenario"], row["policy"], row["n_seeds"]) == ("high_load", "PPO", 3)
    assert row["diff"] == pytest.approx(-10.0)
    assert (row["ci_low"], row["ci_high"]) == pytest.approx((-10.0, -10.0))
    assert row["specialised"] == pytest.approx(144.0)  # 100 + moyenne(42, 43, 44) + 1
    assert row["diff_pct"] == pytest.approx(-1000 / 144)
    assert "Fixed" not in set(table["policy"])  # rien à transférer


def test_cout_du_transfert_plusieurs_graines_d_entrainement():
    """Avec deux graines d'entraînement, une ligne par graine d'évaluation (moyenne des
    deux modèles) : n_seeds reste le nombre de graines d'évaluation."""
    table = transfer_table(_results(diff=5.0, train_seeds=(1, 2)))
    assert table.iloc[0]["n_seeds"] == 3
    assert table.iloc[0]["diff"] == pytest.approx(5.0)
    assert table.iloc[0]["specialised"] == pytest.approx(144.5)


def test_cout_du_transfert_couples_incomplets_ignores():
    """Une graine évaluée dans un seul mode n'entre pas dans la comparaison."""
    data = _results()
    data = data.drop(data.index[(data["mode"] == TRANSFERRED) & (data["seed"] == 44)])
    assert transfer_table(data).iloc[0]["n_seeds"] == 2


def test_cout_du_transfert_csv_sans_mode():
    with pytest.raises(ValueError, match="T27"):
        transfer_table(_results().drop(columns="mode"))
