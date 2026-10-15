import pytest

from memoire_master_rl_logistique.env.mine_env import MineEnv
from memoire_master_rl_logistique.experiments.scenarios import SCENARIOS
from memoire_master_rl_logistique.rl.model_loading import ALGORITHMS, load_model, model_file
from memoire_master_rl_logistique.rl.model_metadata import (
    METADATA_FILENAME,
    EnvConfig,
    MissingMetadataError,
    ModelCompatibilityError,
    build_metadata,
    write_metadata,
)
from memoire_master_rl_logistique.rl.train_q_learning import QLearningPolicy, train_q_learning

NOMINAL = EnvConfig.from_scenario(SCENARIOS["nominal"])


@pytest.fixture(scope="module")
def trained_q_learning(tmp_path_factory):
    """Un vrai modèle Q-Learning (1 épisode), avec son metadata.json, partagé par les tests."""
    model_dir = tmp_path_factory.mktemp("q_learning_nominal")
    train_q_learning(n_episodes=1, output_dir=str(model_dir))
    return model_dir


def test_charge_un_modele_compatible(trained_q_learning):
    """Modèle et environnement nominaux : le modèle est chargé."""
    agent = load_model(trained_q_learning, "Q-Learning", MineEnv())

    assert isinstance(agent, QLearningPolicy)


def test_refus_avant_tout_chargement(tmp_path):
    """Métadonnées périmées : refus immédiat, sans même ouvrir le fichier du modèle (absent ici)."""
    perimee = EnvConfig(12, 3, 2, 480.0, 0.02, (1.0, 0.1, 0.05, 1.0))
    write_metadata(tmp_path, build_metadata("DQN", perimee, 147, seed=42))

    with pytest.raises(ModelCompatibilityError, match="reward_weights"):
        load_model(tmp_path, "DQN", MineEnv())


def test_modele_v1_sans_metadonnees(trained_q_learning, tmp_path):
    """Un modèle antérieur à T8 (fichier présent, metadata.json absent) est refusé par défaut."""
    legacy_dir = tmp_path / "q_learning_v1"
    legacy_dir.mkdir()
    model_file(legacy_dir, "Q-Learning").write_bytes(
        model_file(trained_q_learning, "Q-Learning").read_bytes()
    )

    with pytest.raises(MissingMetadataError):
        load_model(legacy_dir, "Q-Learning", MineEnv())
    with pytest.warns(UserWarning):
        assert isinstance(
            load_model(legacy_dir, "Q-Learning", MineEnv(), allow_legacy=True), QLearningPolicy
        )
    assert not (legacy_dir / METADATA_FILENAME).exists()  # aucune métadonnée inventée


def test_regime_d_entrainement_explicite(trained_q_learning):
    """Modèle nominal évalué en fortes pannes : refusé, sauf si l'on déclare le régime d'entraînement."""
    pannes = SCENARIOS["high_breakdown"]
    env = MineEnv(breakdown_probability=pannes.breakdown_probability)

    with pytest.raises(ModelCompatibilityError, match="breakdown_probability"):
        load_model(trained_q_learning, "Q-Learning", env)
    assert isinstance(load_model(trained_q_learning, "Q-Learning", env, trained_on=NOMINAL),
                      QLearningPolicy)


def test_observation_de_taille_differente_toujours_refusee(trained_q_learning):
    """15 camions donnent une observation de 180 au lieu de 147 : même en déclarant le régime
    d'entraînement, le modèle ne peut pas lire cette observation (défaut corrigé par T18)."""
    env = MineEnv(truck_count=15)

    with pytest.raises(ModelCompatibilityError, match="observation_size"):
        load_model(trained_q_learning, "Q-Learning", env, trained_on=NOMINAL)


def test_algorithme_inconnu():
    """Un nom d'algorithme inconnu est une erreur explicite."""
    with pytest.raises(ValueError, match="inconnu"):
        model_file("models", "A2C")
    assert set(ALGORITHMS) == {"Q-Learning", "SARSA", "DQN", "PPO"}
