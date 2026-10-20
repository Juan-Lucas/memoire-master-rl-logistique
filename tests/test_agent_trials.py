"""Essais de la mise au point des agents et mesures de l'analyse de leur échec."""

from dataclasses import replace
import json

import pytest

from memoire_master_rl_logistique.analysis import reward_diagnostics
from memoire_master_rl_logistique.analysis.reward_diagnostics import (
    DIAGNOSTIC_SEEDS,
    constant_action_policy,
    q_value_gaps,
    reference_policies,
    reward_by_policy,
)
from memoire_master_rl_logistique.config import hyperparameters
from memoire_master_rl_logistique.experiments.agent_trials import (
    ARCHIVED_TRIALS,
    RESULT_FILE,
    TRIALS,
    list_trials,
    run_trial,
)
from memoire_master_rl_logistique.experiments.scenarios import SCENARIOS
from memoire_master_rl_logistique.rl.model_loading import model_file
from memoire_master_rl_logistique.rl.train_dqn import train_dqn
from memoire_master_rl_logistique.seeds import EVAL_SEEDS, TUNING_SEEDS

SHORT = replace(SCENARIOS["nominal"], episode_minutes=60.0)


# --- Registre des essais ------------------------------------------------------------------

def test_tous_les_essais_du_journal_sont_declares():
    """Essais 0 à 9 du tableau du journal : refaisables (5 à 9) ou archivés (0 à 4)."""
    assert sorted([*ARCHIVED_TRIALS, *TRIALS]) == list(range(10))
    assert all(f"{n} [" in list_trials() for n in range(10))


def test_essais_6_a_9_utilisent_la_configuration_de_la_campagne():
    """Le journal dit que les essais 6 à 9 sont la configuration de la campagne : ils
    n'écrasent aucun hyperparamètre de configs/."""
    assert all(not TRIALS[n].overrides for n in (6, 7, 8, 9))


def test_essai_5_ne_differe_de_la_campagne_que_par_la_piste_2():
    """Piste 1 seule = configuration de la campagne sans γ = 0,995 ni normalisation."""
    campaign = hyperparameters("PPO")
    overrides = TRIALS[5].overrides
    assert {k for k, v in overrides.items() if campaign[k] != v} == {"gamma", "normalize_reward"}
    assert overrides["gamma"] == 0.99 and overrides["normalize_reward"] is False


@pytest.mark.parametrize("number, budget", [(5, 1_024), (8, 3)])
def test_un_essai_ecrit_son_resultat_et_ne_se_reentraine_pas(tmp_path, number, budget):
    """Sur un budget minuscule : résultat complet, évalué sur les graines de sélection ;
    relancé, l'essai réutilise le modèle au lieu de le réentraîner."""
    seeds = TUNING_SEEDS[:2]
    result = run_trial(number, tmp_path, budget=budget, scenario=SHORT, seeds=seeds)

    saved = json.loads((tmp_path / f"essai_{number}" / RESULT_FILE).read_text(encoding="utf-8"))
    assert saved == result
    assert result["budget"] == budget and result["seeds"] == list(seeds)
    assert not set(result["seeds"]) & set(EVAL_SEEDS)
    assert sum(result["cycle_share_pct"].values()) == pytest.approx(100.0)
    assert 1 <= len(result["training_curve_tph"]) <= 10
    assert result["resumed"] is False

    model = model_file(tmp_path / f"essai_{number}", TRIALS[number].algorithm)
    mtime = model.stat().st_mtime_ns
    again = run_trial(number, tmp_path, budget=budget, scenario=SHORT, seeds=seeds)
    assert model.stat().st_mtime_ns == mtime
    assert again["productivity_mean"] == result["productivity_mean"]


# --- Mesures de l'analyse -------------------------------------------------------------------

def test_mesure_du_journal_la_recompense_voit_peu_la_duree_des_cycles():
    """Refait la mesure notée dans le journal (graines de sélection 20 000 à 20 004) :
    la récompense par décision classe MTWT devant la politique de DQN (toujours
    pelle 3 → verse 1), mais aller toujours à la pelle 3 plutôt qu'à la pelle 1 coûte
    9 % de productivité pour seulement 0,007 de récompense par décision."""
    scenario = SCENARIOS["nominal"]
    table = reward_by_policy(reference_policies(scenario), scenario).set_index("policy")

    expected = {  # productivité (t/h), récompense par décision (journal, 19 octobre 2026)
        "MTWT": (4448, 0.871), "LP": (4410, 0.863), "Fixed": (4221, 0.862),
        "Nearest": (3913, 0.820), "Toujours action 0": (3913, 0.820),
        "Toujours action 2": (3790, 0.822), "Toujours action 4": (3566, 0.813),
    }
    for policy, (tph, per_decision) in expected.items():
        assert table.loc[policy, "productivity_tph"] == pytest.approx(tph, abs=0.5)
        assert table.loc[policy, "reward_per_decision"] == pytest.approx(per_decision, abs=5e-4)
    terms = table.filter(like="per_decision_").sum(axis=1)
    assert terms.to_numpy() == pytest.approx(table["reward_per_decision"].to_numpy())
    assert set(TUNING_SEEDS[:5]) == set(DIAGNOSTIC_SEEDS)


def test_ecarts_des_valeurs_q(tmp_path):
    model = train_dqn(total_timesteps=1_000, episode_minutes=60.0, output_dir=str(tmp_path),
                      progress_bar=False, checkpoint_freq=None)
    gaps = q_value_gaps(model, SHORT)
    assert sum(gaps["chosen_actions"]) == gaps["decisions"] > 0
    assert 0 <= gaps["gap_best_second"] <= gaps["gap_best_worst"]
    assert len(gaps["mean_q_by_action"]) == 7


def test_politique_constante_et_refus_d_ecraser(tmp_path):
    assert constant_action_policy(4)(None, None, None) == 4
    (tmp_path / "reward_by_policy.csv").write_text("")
    with pytest.raises(FileExistsError):
        reward_diagnostics.main(tmp_path)
