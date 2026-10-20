"""Essais de la mise au point des agents et mesures de l'analyse de leur échec."""

from dataclasses import replace
import json

import pandas as pd
import pytest

from memoire_master_rl_logistique.analysis import reward_diagnostics
from memoire_master_rl_logistique.analysis.reward_diagnostics import (
    DIAGNOSTIC_SEEDS,
    constant_action_policy,
    option_b_rankings,
    q_value_gaps,
    rank_agreement,
    reference_policies,
    reward_by_policy,
)
from memoire_master_rl_logistique.config import hyperparameters
from memoire_master_rl_logistique.experiments.agent_trials import (
    ARCHIVED_TRIALS,
    BEFORE_OPTION_B,
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
    """Essais 0 à 13 du tableau du journal : archivés (0 à 4), faits avec la récompense
    d'avant l'option B (5 à 9, commit 2461c7c), contrôles de l'option B (10 à 13)."""
    assert sorted([*ARCHIVED_TRIALS, *TRIALS]) == list(range(14))
    assert all(f"{n} [" in list_trials() for n in range(14))
    assert all(TRIALS[n].commit == BEFORE_OPTION_B for n in range(5, 10))
    assert all(TRIALS[n].commit is None for n in range(10, 14))


def test_essais_de_controle_avec_la_configuration_de_la_campagne():
    """Essais 6 à 9 (avant l'option B) et 10 à 13 (contrôles de l'option B) : configuration
    de la campagne, aucun hyperparamètre écrasé ; les contrôles reprennent les mêmes agents
    (règle, point 13 : seule la récompense change)."""
    assert all(not TRIALS[n].overrides for n in (6, 7, 8, 9, 10, 11, 12, 13))
    assert [TRIALS[n].algorithm for n in (6, 7, 8, 9)] ==         [TRIALS[n].algorithm for n in (10, 11, 12, 13)]


def test_un_essai_d_avant_l_option_b_ne_tourne_pas_sur_le_code_actuel(tmp_path):
    with pytest.raises(ValueError, match="2461c7c"):
        run_trial(6, tmp_path)


def test_essai_5_ne_differe_de_la_campagne_que_par_la_piste_2():
    """Piste 1 seule = configuration de la campagne sans γ = 0,995 ni normalisation."""
    campaign = hyperparameters("PPO")
    overrides = TRIALS[5].overrides
    assert {k for k, v in overrides.items() if campaign[k] != v} == {"gamma", "normalize_reward"}
    assert overrides["gamma"] == 0.99 and overrides["normalize_reward"] is False


@pytest.mark.parametrize("number, budget", [(10, 1_024), (12, 3)])
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

def test_mesures_du_journal_avant_et_apres_l_option_b():
    """Refait les mesures notées dans le journal (graines de sélection 20 000 à 20 004).

    Avant l'option B (19 octobre) : la récompense par décision classe MTWT devant la
    politique de DQN (toujours pelle 3 → verse 1), mais aller toujours à la pelle 3 plutôt
    qu'à la pelle 1 coûte 9 % de productivité pour seulement 0,007 par décision.
    Avec l'option B (20 octobre, tableau 1 de docs/option_b_recompense.md) : l'écart entre
    ces deux politiques passe à 0,103, et le temps des camions est le même pour toutes."""
    scenario = SCENARIOS["nominal"]
    table = reward_by_policy(reference_policies(scenario), scenario).set_index("policy")

    expected = {  # t/h, récompense par décision avant l'option B, avec l'option B
        "MTWT": (4448, 0.871, -0.063), "LP": (4410, 0.863, -0.079),
        "Fixed": (4221, 0.862, -0.118), "Nearest": (3913, 0.820, -0.235),
        "Toujours action 0": (3913, 0.820, -0.235), "Toujours action 2": (3790, 0.822, -0.266),
        "Toujours action 4": (3566, 0.813, -0.338),
    }
    for policy, (tph, before_b, option_b) in expected.items():
        row = table.loc[policy]
        assert row["productivity_tph"] == pytest.approx(tph, abs=0.5)
        assert row["reward_per_decision_before_b"] == pytest.approx(before_b, abs=5e-4)
        assert row["reward_per_decision"] == pytest.approx(option_b, abs=5e-4)
    assert table["truck_min_per_shift"].to_numpy() == pytest.approx(12 * 480)  # temps fixe
    terms = table.filter(regex="^per_decision_").sum(axis=1)
    assert terms.to_numpy() == pytest.approx(table["reward_per_decision"].to_numpy())
    assert set(TUNING_SEEDS[:5]) == set(DIAGNOSTIC_SEEDS)


def test_option_b_classe_les_politiques_comme_la_productivite():
    """Tableau 2 de docs/option_b_recompense.md : corrélation de Spearman entre la
    productivité et la récompense par décision, 12 politiques, avant et avec l'option B."""
    rankings = option_b_rankings().set_index("scenario")
    expected = {"nominal": (0.965, 1.0), "high_load": (0.654, 0.767),
                "high_breakdown": (0.972, 1.0)}
    for scenario, (before_b, option_b) in expected.items():
        assert rankings.loc[scenario, "n_policies"] == 12
        assert rankings.loc[scenario, "spearman_before_b"] == pytest.approx(before_b, abs=5e-4)
        assert rankings.loc[scenario, "spearman_option_b"] == pytest.approx(option_b, abs=5e-4)
    table = reward_by_policy({"a": constant_action_policy(0)}, seeds=TUNING_SEEDS[:1])
    assert rank_agreement(pd.concat([table, table.assign(productivity_tph=0,
                                                         reward_per_decision=-1)]),
                          "reward_per_decision") == pytest.approx(1.0)


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
