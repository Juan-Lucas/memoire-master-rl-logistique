# Au-delà des heuristiques statiques : Conception et mise en œuvre d'un agent d'apprentissage par renforcement pour l'optimisation holistique et dynamique de la logistique minière

Présenté par **Jean-Luc MUPASA KALUNGA**

Supervisé par **Dr. Olfa FERCHICHI**

Université Don Bosco de Lubumbashi (UDBL) — Master 2 Data Science, spécialité Logistique — Année académique 2025-2026

---

## Introduction

Dans un contexte minier où l'optimisation des coûts opérationnels et la réduction de l'empreinte environnementale sont devenues des impératifs stratégiques, les systèmes de gestion de flotte (Fleet Management Systems - FMS) traditionnels montrent leurs limites. Bien que des solutions industrielles comme Caterpillar MineStar Dispatch et Komatsu Jigsaw aient révolutionné la coordination des flottes, ces systèmes reposent principalement sur des heuristiques statiques, des règles prédéfinies et des zones tampons pour gérer les flux de véhicules.

Ces approches, bien qu'efficaces dans des conditions stables, peinent à s'adapter dynamiquement aux micro-conditions évolutives : état dégradé des routes, variations stochastiques du trafic, conditions météorologiques imprévisibles ou encore style de conduite et état mécanique des véhicules.

Ce projet propose de dépasser ces limites en développant un système d'aide à la décision basé sur l'Apprentissage par Renforcement (Reinforcement Learning - RL), capable d'apprendre une politique de dispatching adaptative qui affecte dynamiquement chaque camion à une pelle et à un point de déchargement en fonction de l'état courant de la flotte.

---

## Problématique

Les FMS actuels optimisent localement (affectation camion-pelle, gestion des files d'attente), mais ne parviennent pas à optimiser globalement l'ensemble de la flotte en temps réel, ni à anticiper et s'adapter aux perturbations dynamiques telles que la dégradation des routes ou les congestions imprévues.

Un camion suivant une règle statique peut être affecté à une pelle déjà saturée et attendre inutilement en file alors qu'une autre pelle est disponible, gaspillant carburant et temps. Une approche adaptative pourrait réaffecter dynamiquement les camions vers les ressources les moins congestionnées.

---

## Hypothèse

L'utilisation d'un agent autonome basé sur l'Apprentissage par Renforcement permettra de dépasser les performances des systèmes heuristiques actuels en apprenant une politique d'affectation adaptative des camions aux pelles et aux points de déchargement, qui minimise la consommation de carburant, réduit le temps d'attente improductif et s'adapte en temps réel aux conditions changeantes de l'environnement minier.

Cette hypothèse sera validée par une comparaison rigoureuse entre l'agent d'apprentissage par renforcement et des baselines représentatives des approches actuelles (heuristiques statiques et logique "Dispatch" simplifiée).

> **Résultat (campagne finale, octobre 2026) : hypothèse réfutée.** Aucun agent ne dépasse la meilleure heuristique (MTWT). PPO s'en approche à moins de 1 %, consomme moins de carburant par tonne et transfère presque sans perte vers les régimes perturbés. Voir [Résultats](#résultats-de-la-campagne-finale).

---

## Questions de recherche

1. Comment modéliser l'environnement minier complexe (réseau routier, zones tampons, événements stochastiques) dans un cadre d'apprentissage par renforcement compatible avec les standards industriels ?

2. Quelle architecture d'agent d'apprentissage par renforcement est la plus adaptée pour apprendre une politique de dispatching adaptative (affectation camion → pelle → point de déchargement) dans un environnement minier stochastique ?

3. Comment concevoir une fonction de récompense qui capture fidèlement les coûts réels (carburant, usure, temps) tout en pénalisant les comportements non souhaitables ?

4. Quelles métriques et protocoles d'évaluation permettront de démontrer quantitativement la supériorité de l'approche d'apprentissage par renforcement par rapport aux approches classiques, notamment dans des scénarios de perturbation ?

5. Comment intégrer un tel système d'apprentissage par renforcement comme couche d'intelligence additionnelle dans l'écosystème des FMS existants, sans nécessiter un remplacement complet de l'infrastructure ?

---

## Résultats de la campagne finale

Campagne du 22 au 28 octobre 2026 : 10 politiques (6 heuristiques, Q-Learning, SARSA, DQN, PPO), 3 régimes (nominal, forte charge, fortes pannes), 5 graines d'entraînement par agent, évaluation unique sur 30 réplications appariées (graines 42 à 71).

| Politique | Nominal (t/h) | Forte charge (t/h) | Fortes pannes (t/h) | Carburant nominal (L/t) |
|---|---|---|---|---|
| MTWT (attente prévue minimale) | **4 437** | **6 365** | **4 184** | 0,244 |
| PPO | 4 415 | 6 310 | 4 164 | **0,237** |
| LP (programme linéaire) | 4 404 | 6 351 | 4 150 | 0,235 |
| Affectation fixe | 4 246 | 6 205 | 3 999 | 0,270 |
| Q-Learning | 4 088 | 5 788 | 3 840 | 0,271 |
| SARSA | 3 573 | 5 410 | 3 353 | 0,304 |
| DQN | 1 287 | 985 | 1 579 | 0,565 |

- MTWT est la plus productive dans les trois régimes ; PPO est 0,5 à 0,9 % en dessous (écart significatif mais petit), et consomme 1,4 à 2,8 % de carburant de moins par tonne.
- PPO entraîné en conditions nominales garde ses performances sans réentraînement en forte charge, en fortes pannes et dans deux régimes jamais vus.
- DQN s'effondre sur l'action ATTENDRE dans 4 modèles nominaux sur 5 ; les agents tabulaires se dégradent quand l'exploration diminue.

Données : CSV de l'évaluation, analyses, configurations et métadonnées des modèles dans `data/results_v2/campagne/` (les modèles eux-mêmes restent hors de git).

Détails : article `reports/article/RL_article.tex` (anglais, version soumise) et `reports/article/RL_article_FR_v3.tex` (français) ; analyses dans `data/results_v2/campagne/analyses/`.

---

## Reproduire les résultats

**Environnement** (Python 3.11.11, versions exactes dans `requirements.txt`) :

```bash
conda create -n rl-test python=3.11.11
conda activate rl-test
pip install -r requirements.txt
pip install -e .
pytest                      # suite de tests
```

Tous les paramètres (régimes, graines, hyperparamètres, poids de la récompense) sont lus dans `configs/*.yaml` et archivés avec chaque jeu de résultats.

**Campagne** (`experiments/campaign.py`, environ 5 jours de calcul sur un portable à 8 cœurs) :

```bash
python -m memoire_master_rl_logistique.experiments.campaign train     DOSSIER --workers 4  # 90 entraînements
python -m memoire_master_rl_logistique.experiments.campaign evaluate  DOSSIER --workers 4  # évaluation unique
python -m memoire_master_rl_logistique.experiments.campaign analyze   DOSSIER              # tests, agrégation, transfert, ablation
python -m memoire_master_rl_logistique.experiments.campaign behaviour DOSSIER --workers 4  # répartition entre pelles (rejeu vérifié)
python -m memoire_master_rl_logistique.experiments.campaign rehearse  DOSSIER_VIDE         # répétition à petite échelle (~30 min)
```

`train` reprend là où il s'est arrêté ; `evaluate` refuse de démarrer s'il manque un modèle. Les essais de la mise au point des agents se refont avec `experiments.agent_trials`.

**Figures de l'article** (option `--lang en` pour la version anglaise) :

```bash
python -m memoire_master_rl_logistique.plots.benchmark_comparisons CSV reports/article/v3/figures
python -m memoire_master_rl_logistique.plots.robustness            CSV reports/article/v3/figures
python -m memoire_master_rl_logistique.plots.campaign_curves       DOSSIER/analyses/courbes_productivity_tph.csv reports/article/v3/figures
python -m memoire_master_rl_logistique.plots.behaviour             DOSSIER/analyses/comportement/behaviour_summary.csv reports/article/v3/figures
python -m memoire_master_rl_logistique.plots.scenarios_schematic   reports/article/v3/figures
```

Les sources draw.io des schémas sont dans `reports/article/v3/figures_src/`.

---

## Organisation du dépôt

```
memoire_master_rl_logistique/
  env/          environnement Gymnasium (MineEnv, état de 23 composantes)
  simulation/   moteur à événements, réseau routier, carburant, récompense, indicateurs
  rl/           entraînement PPO, DQN, Q-Learning, SARSA (actualisation par minute)
  baselines/    FIFO, Affectation fixe, Pelle la plus proche, Plus court chemin, MTWT, LP
  experiments/  régimes, benchmark, campagne, mise au point des agents
  analysis/     tests appariés, agrégation sur les graines, transfert, comportement
  plots/        figures de l'article
configs/        paramètres (YAML)
tests/          tests automatiques
reports/        article (français et anglais), mémoire, figures
docs/           plan et journal des corrections (versionnés), notes de travail (locales)
```
