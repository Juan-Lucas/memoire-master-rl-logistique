"""Lanceur de la campagne finale (T31) et configuration de sa répétition."""

import pytest
import yaml

from memoire_master_rl_logistique.config import experiment_config, hyperparameters
from memoire_master_rl_logistique.experiments import campaign
from memoire_master_rl_logistique.experiments.campaign import (
    EVALUATED_SCENARIOS,
    TrainJob,
    campaign_jobs,
    missing_models,
    rehearsal_config,
    train,
)
from memoire_master_rl_logistique.seeds import EVAL_SEEDS, TRAIN_SEEDS, TUNING_SEEDS


def test_entrainements_de_la_campagne():
    """60 entraînements dans les régimes principaux (4 agents × 3 régimes × 5 graines) et
    30 pour l'ablation sur w4 (PPO et DQN × 3 valeurs × 5 graines) ; aucun en double."""
    jobs = campaign_jobs()
    assert len(jobs) == len(set(jobs)) == 90
    main = [j for j in jobs if not j.regime.startswith("ablation")]
    assert len(main) == 60 and {j.algorithm for j in main} == {"Q-Learning", "SARSA", "DQN", "PPO"}
    ablation = [j for j in jobs if j.regime.startswith("ablation")]
    assert len(ablation) == 30 and {j.algorithm for j in ablation} == {"DQN", "PPO"}
    assert {j.train_seed for j in jobs} == set(TRAIN_SEEDS)
    assert not any(j.regime.startswith("unseen") for j in jobs)  # modèles nominaux réutilisés
    # Les plus longs d'abord : tabulaires, puis PPO, puis DQN.
    order = [j.algorithm for j in jobs]
    assert order.index("PPO") > order.index("SARSA") and order.index("DQN") > order.index("PPO")


def test_regimes_evalues():
    assert EVALUATED_SCENARIOS == ["nominal", "high_load", "high_breakdown",
                                   "unseen_medium_load", "unseen_medium_breakdown",
                                   "ablation_w4_00", "ablation_w4_01", "ablation_w4_05"]


def _fake_runner(job, root):
    """Écrit un fichier de modèle vide au bon endroit, sans entraîner."""
    from memoire_master_rl_logistique.rl.model_loading import model_file

    directory = job.directory(root)
    directory.mkdir(parents=True, exist_ok=True)
    model_file(directory, job.algorithm).write_bytes(b"")
    return job, 0.0


def test_entrainement_saute_les_modeles_deja_faits(tmp_path):
    """Relancer ne refait que ce qui manque ; la configuration est archivée."""
    jobs = [TrainJob("PPO", "nominal", 1), TrainJob("SARSA", "high_load", 2)]
    assert train(tmp_path, 1, jobs, _fake_runner) == jobs
    assert missing_models(tmp_path, jobs) == []
    assert train(tmp_path, 1, jobs, _fake_runner) == []
    assert (tmp_path / "configs_campagne" / "experiment_config.yaml").exists()


def test_un_echec_n_arrete_pas_les_autres(tmp_path):
    def runner(job, root):
        if job.algorithm == "DQN":
            raise ValueError("panne simulée")
        return _fake_runner(job, root)

    jobs = [TrainJob("DQN", "nominal", 1), TrainJob("PPO", "nominal", 1)]
    with pytest.raises(RuntimeError, match="DQN_nominal_seed1"):
        train(tmp_path, 1, jobs, runner)
    assert missing_models(tmp_path, jobs) == [TrainJob("DQN", "nominal", 1)]


def test_evaluation_refusee_s_il_manque_un_modele(tmp_path):
    with pytest.raises(FileNotFoundError, match="modèle"):
        campaign.evaluate(tmp_path, 1)


def test_un_seul_fichier_de_resultats(tmp_path):
    with pytest.raises(FileNotFoundError):
        campaign.results_csv(tmp_path)
    (tmp_path / "benchmark_1.csv").write_text("x")
    assert campaign.results_csv(tmp_path).name == "benchmark_1.csv"
    (tmp_path / "benchmark_2.csv").write_text("x")
    with pytest.raises(FileNotFoundError):
        campaign.results_csv(tmp_path)


def test_configuration_de_repetition(tmp_path):
    """Budgets réduits, mêmes hyperparamètres, et graines d'évaluation disjointes de
    celles de la campagne, de la sélection et de l'entraînement."""
    directory = rehearsal_config(tmp_path / "conf")
    config = experiment_config(directory)
    assert config["scenario_defaults"]["total_timesteps"] == campaign.REHEARSAL_TIMESTEPS
    evaluation = config["seeds"]["evaluation"]
    rehearsal_seeds = set(range(evaluation["start"], evaluation["start"] + evaluation["count"]))
    assert len(rehearsal_seeds) == len(EVAL_SEEDS)
    assert not rehearsal_seeds & set(EVAL_SEEDS) and not rehearsal_seeds & set(TUNING_SEEDS)
    assert max(rehearsal_seeds) < min(TRAIN_SEEDS) * config["seeds"]["training_episode_stride"]
    assert hyperparameters("SARSA", directory)["n_episodes"] == campaign.REHEARSAL_EPISODES
    assert hyperparameters("SARSA", directory)["alpha"] == hyperparameters("SARSA")["alpha"]
    assert hyperparameters("PPO", directory) == hyperparameters("PPO")
    # Seuls ces trois nombres changent dans experiment_config.yaml.
    original = yaml.safe_load((campaign.config_dir() / "experiment_config.yaml").read_text(encoding="utf-8"))
    assert {k: v for k, v in config.items() if k not in ("scenario_defaults", "seeds")} == \
        {k: v for k, v in original.items() if k not in ("scenario_defaults", "seeds")}
    with pytest.raises(FileExistsError):
        rehearsal_config(tmp_path / "conf")


def test_repetition_demande_un_dossier_vide(tmp_path):
    (tmp_path / "x").write_text("x")
    with pytest.raises(FileExistsError, match="vide"):
        campaign.rehearse(tmp_path)


def test_journal_ecrit_pendant_l_entrainement(tmp_path, monkeypatch):
    """Le journal d'un entraînement se lit pendant le calcul, pas seulement à la fin : une
    ligne affichée sans flush (comme les tableaux de Stable-Baselines3) est déjà sur le
    disque avant la fin de l'entraînement."""
    import sys

    from memoire_master_rl_logistique.experiments import run_benchmark

    job = TrainJob("PPO", "nominal", 1)
    log = tmp_path / campaign.LOG_DIR / f"{job.label()}.log"
    seen = {}

    def fake_train(algorithm, scenario, train_seed, output_dir):
        print("| rollout/ep_rew_mean | 123 |")          # stdout, sans flush
        print("progression 50 %", file=sys.stderr)       # tqdm écrit sur stderr
        seen["pendant"] = log.read_text(encoding="utf-8")

    monkeypatch.setattr(run_benchmark, "_train", fake_train)
    log.parent.mkdir(parents=True)
    campaign._run_job(job, str(tmp_path))
    assert "ep_rew_mean" in seen["pendant"] and "progression 50 %" in seen["pendant"]
    assert "=== fin" in log.read_text(encoding="utf-8")
