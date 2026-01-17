# Au-delà des Heuristiques Statiques : Conception et Mise en Œuvre d'un Agent d'Apprentissage par Renforcement pour l'Optimisation Holistique et Dynamique de la Logistique Minière

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11.11](https://img.shields.io/badge/python-3.11.11-blue.svg)](https://www.python.org/downloads/)
[![Anaconda](https://img.shields.io/badge/Anaconda-%2344A833.svg?style=flat&logo=anaconda&logoColor=white)](https://www.anaconda.com/)
<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

**Mémoire de Master 2 — Apprentissage par Renforcement appliqué à la Logistique Minière**

Présenté par : **Jean-Luc MUPASA KALAUNGA**  
Supervisé par : **Dr. Olfa FERCHICHI**  

Université : **Université Don Bosco de Lubumbashi (UDBL)** — Année académique 2025-2026

---

## Table des Matières

- [Introduction](#introduction)
- [Problématique](#problématique)
- [Hypothèse](#hypothèse)
- [Questions de Recherche](#questions-de-recherche)
- [Résumé du Projet](#résumé-du-projet)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Méthodologie](#méthodologie)
- [Résultats Attendus](#résultats-attendus)
- [Références Clés](#références-clés)
- [Structure du Projet](#structure-du-projet)
- [Licence](#licence)
- [Contact](#contact)

---

## Introduction

Dans un contexte minier où l'optimisation des coûts opérationnels et la réduction de l'empreinte environnementale sont devenues des impératifs stratégiques, les systèmes de gestion de flotte (Fleet Management Systems - FMS) traditionnels montrent leurs limites. Bien que des solutions industrielles comme **Caterpillar MineStar Dispatch** et **Komatsu Jigsaw** aient révolutionné la coordination des flottes, ces systèmes reposent principalement sur des heuristiques statiques, des règles prédéfinies et des zones tampons pour gérer les flux de véhicules.

Ces approches, bien qu'efficaces dans des conditions stables, peinent à s'adapter dynamiquement aux **micro-conditions** évolutives :
- État dégradé des routes (nids-de-poule, boue, pentes variables)
- Variations stochastiques du trafic aux points névralgiques
- Conditions météorologiques imprévisibles
- Style de conduite et état mécanique des véhicules

Ce projet propose de **dépasser ces limites** en développant un système d'aide à la décision basé sur l'**Apprentissage par Renforcement (Reinforcement Learning - RL)**, capable d'apprendre une politique de conduite holistique optimisant simultanément l'itinéraire ET le rythme de déplacement des véhicules.

---

## Problématique

Les FMS actuels optimisent localement (affectation camion → pelle, gestion des files d'attente), mais ne parviennent pas à :
1. **Optimiser globalement** l'ensemble de la flotte en temps réel.
2. **Anticiper et s'adapter** aux perturbations dynamiques (dégradation route, congestion imprévue).
3. **Minimiser le temps d'attente improductif** (moteur tournant au ralenti dans les zones tampons).
4. **Apprendre continuellement** des données opérationnelles pour améliorer les décisions futures.

**Constat critique** : Un camion suivant une règle statique peut sprinter vers sa destination pour finalement attendre dans une zone tampon, gaspillant carburant et temps. Une approche adaptative pourrait ajuster le rythme de déplacement pour arriver "juste à temps".

---

## Hypothèse

**L'utilisation d'un agent autonome basé sur l'Apprentissage par Renforcement (RL) permettra de dépasser les performances des systèmes heuristiques actuels en apprenant une politique de navigation adaptative qui :**
- Minimise la consommation de carburant par tonne-kilomètre
- Réduit drastiquement le temps d'attente improductif
- S'adapte en temps réel aux conditions changeantes de l'environnement minier

Cette hypothèse sera validée par une comparaison rigoureuse entre l'agent RL et des baselines représentatives des approches actuelles (heuristiques statiques + logique "Dispatch" simplifiée).

---

## Questions de Recherche

1. **Comment modéliser l'environnement minier complexe** (réseau routier, zones tampons, événements stochastiques) dans un cadre RL compatible avec les standards industriels (Gymnasium/OpenAI Gym) ?

2. **Quelle architecture d'agent RL** (DQN, PPO, SAC, etc.) est la plus adaptée pour apprendre une politique de navigation holistique intégrant à la fois le choix d'itinéraire et le contrôle de vitesse ?

3. **Comment concevoir une fonction de récompense** qui capture fidèlement les coûts réels (carburant, usure, temps) tout en pénalisant les comportements non souhaitables (attentes prolongées, accélérations brutales) ?

4. **Quelles métriques et protocoles d'évaluation** permettront de démontrer quantitativement la supériorité de l'approche RL par rapport aux heuristiques statiques et aux systèmes "Dispatch" actuels, notamment dans des scénarios de perturbation (dégradation route, pic de trafic) ?

5. **Comment intégrer un tel système RL** comme couche d'intelligence additionnelle dans l'écosystème des FMS existants, sans nécessiter un remplacement complet de l'infrastructure ?

---

## Résumé du Projet

Ce projet vise à développer un **agent d'Apprentissage par Renforcement (RL)** capable d'optimiser de manière holistique et dynamique la logistique d'une flotte de camions miniers. Contrairement aux approches traditionnelles qui appliquent des règles fixes (ex : "toujours prendre le chemin le plus court"), notre agent apprendra une politique adaptative qui :

- **Choisit l'itinéraire optimal** en fonction de l'état actuel du réseau routier, du trafic et de l'occupation des destinations.
- **Ajuste le rythme de déplacement** (vitesse cible) pour minimiser la consommation tout en évitant les attentes inutiles.
- **S'adapte en temps réel** aux perturbations (dégradation route, congestion, météo).

### Architecture Globale

```
┌─────────────────────────────────────────────────────────────┐
│                   Environnement de Simulation               │
│  ┌────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │  Réseau    │  │  Véhicules   │  │  Événements      │   │
│  │  Routier   │  │  (Flotte)    │  │  Dynamiques      │   │
│  │  (Graphe)  │  │              │  │  (Stochastiques) │   │
│  └────────────┘  └──────────────┘  └──────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            ↓ Observations
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                       Agent RL (PPO/DQN)                    │
│  ┌────────────────────────────────────────────────────┐    │
│  │  Policy Network : State → (Itinéraire, Vitesse)   │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                            ↓ Actions
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   Fonction de Récompense                    │
│  R = -α·Fuel - β·Time - γ·WaitTime - δ·Wear               │
└─────────────────────────────────────────────────────────────┘
```

### Comparaison avec les Approches Existantes

| Caractéristique              | Heuristiques Statiques (ex: Dijkstra) | FMS Actuels (Dispatch) | Agent RL (Notre Approche) |
|------------------------------|----------------------------------------|------------------------|----------------------------|
| Choix d'itinéraire           | Plus court chemin fixe                 | Plus court + gestion files | Adaptatif (état réseau)   |
| Contrôle de vitesse          | ❌ Non                                  | ⚠️ Limité (règles)      | ✅ Oui (optimal dynamique) |
| Adaptation aux perturbations | ❌ Non                                  | ⚠️ Réactive (re-routing) | ✅ Anticipative (apprentissage) |
| Minimisation attentes        | ❌ Non                                  | ⚠️ Zones tampons        | ✅ Synchronisation prédictive |
| Apprentissage continu        | ❌ Non                                  | ❌ Non                  | ✅ Oui (amélioration continue) |

---

## Installation

### Prérequis

- Python 3.11.11 (via Anaconda)
- Anaconda ou Miniconda
- Git
- (Optionnel) CUDA pour GPU (entraînement accéléré)

### Étapes

1. **Cloner le dépôt**
   ```bash
   git clone https://github.com/Juan-Lucas/memoire-master-rl-logistique.git
   cd memoire-master-rl-logistique
   ```

2. **Créer un environnement conda**
   ```bash
   conda create -n datascience python=3.11.11
   conda activate datascience
   ```

3. **Installer les dépendances**
   ```bash
   pip install -r requirements.txt
   # ou avec conda
   conda install --file requirements.txt
   ```

4. **Installer le package en mode développement**
   ```bash
   pip install -e .
   ```

### Dépendances Principales

- `gymnasium` : Environnements RL standardisés
- `stable-baselines3` : Implémentations RL (PPO, DQN, SAC)
- `torch` : Deep Learning backend
- `networkx` : Manipulation de graphes (réseau routier)
- `pandas`, `numpy` : Traitement de données
- `matplotlib`, `seaborn` : Visualisation
- `pytest` : Tests unitaires

---

## Utilisation

### 1. Lancer une Simulation de Baseline (Dijkstra)

```bash
python -m memoire_master_rl_logistique.baselines.dijkstra --scenario=default
```

### 2. Entraîner un Agent RL (PPO)

```bash
python -m memoire_master_rl_logistique.agents.ppo_agent --train \
    --timesteps=1000000 \
    --save-path=models/ppo_mine_agent
```

### 3. Évaluer et Comparer les Modèles

```bash
python -m memoire_master_rl_logistique.evaluate \
    --models ppo,dqn,dijkstra,dispatch \
    --scenarios perturbation_route,pic_trafic \
    --output reports/comparison.csv
```

### 4. Visualiser les Résultats

```bash
python -m memoire_master_rl_logistique.visualize \
    --input reports/comparison.csv \
    --output reports/figures/
```

### 5. Lancer l'Interface de Démonstration (Streamlit)

```bash
streamlit run app.py
```

---

## Méthodologie

Le projet suit la méthodologie **CRISP-DM (Cross-Industry Standard Process for Data Mining)** adaptée au Machine Learning et au Reinforcement Learning :

### Phase 1 : Compréhension Métier et Données (Business & Data Understanding)
- **Revue de littérature approfondie** :
  - Étude des FMS existants (Caterpillar MineStar, Komatsu Jigsaw)
  - Application du RL au Vehicle Routing Problem (VRP)
  - Modèles de consommation de carburant pour engins lourds
- **Définition des KPIs** :
  - Consommation (L/t·km)
  - Temps de cycle moyen (min)
  - Temps d'attente improductif (min)
  - Taux d'utilisation des assets (%)
- **Analyse des contraintes réelles** : zones tampons, capacités de chargement, règles de sécurité

### Phase 2 : Préparation des Données et Simulation
- **Modélisation du réseau routier** : Graphe orienté pondéré (NetworkX) avec attributs dynamiques
- **Création de l'environnement de simulation** (Gymnasium) :
  - Espace d'état : `(position, destination, charge, trafic_local, état_route, occupation_cibles)`
  - Espace d'action : `(prochain_noeud, vitesse_cible)`
  - Fonction de récompense : `R = -α·Fuel - β·Time - γ·WaitTime - δ·Wear`
- **Modélisation des événements stochastiques** :
  - Dégradation progressive des routes
  - Variations aléatoires de trafic
  - Conditions météorologiques

### Phase 3 : Modélisation (Modeling)
- **Implémentation de l'agent RL** :
  - Algorithmes testés : PPO, DQN, SAC (Stable-Baselines3)
  - Architecture réseau : MLP (Multi-Layer Perceptron) ou CNN si grille spatiale
- **Développement des baselines** :
  - Baseline 1 : Dijkstra (plus court chemin statique)
  - Baseline 2 : Heuristique "Dispatch" (simulation FMS actuel avec zones tampons)
- **Entraînement et tuning** : Hyperparameter search (Optuna), curriculum learning

### Phase 4 : Évaluation (Evaluation)
- **Scénarios de test** :
  - Scénario nominal (conditions stables)
  - Scénario de perturbation (dégradation route soudaine)
  - Scénario de pic de trafic
  - Scénario météo adverse
- **Métriques comparatives** :
  - Fuel consumption (L), Time (min), Wait time (min), Wear cost (€)
- **Analyses statistiques** : Tests de significativité (t-test, ANOVA), intervalles de confiance

### Phase 5 : Déploiement (Deployment - Conceptuel)
- **Prototype de démonstration** (Streamlit/Gradio)
- **Intégration conceptuelle** : Proposition d'architecture pour intégration dans FMS existant
- **Documentation** : Rapport de mémoire, présentation de soutenance

---

## Résultats Attendus

À l'issue de ce projet, nous anticipons :

1. **Validation quantitative** : Démonstration d'une réduction de 10-25% de la consommation de carburant et 30-50% du temps d'attente par rapport aux baselines statiques.

2. **Preuve de concept robuste** : Un environnement de simulation réaliste et un agent RL entraîné capable de généraliser à divers scénarios de perturbation.

3. **Contribution scientifique** : Publication des résultats dans une conférence/journal (ex: IEEE Intelligent Transportation Systems, Applied Energy).

4. **Recommandations industrielles** : Proposition d'une architecture d'intégration de l'agent RL comme module additionnel aux FMS actuels.

5. **Code open-source** : Mise à disposition du code (sous licence MIT) pour reproductibilité et extension par la communauté.

---

## Références Clés

### Systèmes de Gestion de Flotte Minière
- Caterpillar. (2023). *MineStar System Architecture*. Technical Documentation.
- Komatsu. (2022). *Jigsaw Fleet Management System*. Product Whitepaper.

### Apprentissage par Renforcement pour la Logistique
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
- Nazari, M., et al. (2018). "Reinforcement Learning for Solving the Vehicle Routing Problem". *NeurIPS*.

### Optimisation de la Consommation de Carburant
- Alarie, S., & Gamache, M. (2002). "Overview of Solution Strategies Used in Truck Dispatching Systems for Open Pit Mines". *International Journal of Surface Mining*.

### Modélisation et Simulation
- OpenAI. (2023). *Gymnasium: A Standard API for Reinforcement Learning*. Documentation.

*(Voir `references/` pour la liste complète des articles PDF)*

---

## Structure du Projet

Ce projet suit la structure **Cookiecutter Data Science** pour garantir reproductibilité et maintenabilité :

```
memoire-master-rl-logistique/
│
├── LICENSE                          <- Licence MIT
├── Makefile                         <- Commandes automatisées (make train, make test)
├── README.md                        <- Ce fichier
├── pyproject.toml                   <- Configuration projet (Poetry/setuptools)
├── requirements.txt                 <- Dépendances Python
├── setup.cfg                        <- Configuration flake8, pytest
│
├── data/
│   ├── external/                    <- Données tierces (cartes, benchmarks)
│   ├── interim/                     <- Données intermédiaires (graphes traités)
│   ├── processed/                   <- Données finales prêtes pour entraînement
│   └── raw/                         <- Données brutes originales
│
├── docs/                            <- Documentation (MkDocs)
│   ├── mkdocs.yml
│   ├── getting-started.md
│   └── index.md
│
├── memoire_master_rl_logistique/    <- Code source principal
│   ├── __init__.py
│   ├── config.py                    <- Variables et configuration globales
│   ├── dataset.py                   <- Scripts de génération de données
│   ├── features.py                  <- Feature engineering
│   │
│   ├── envs/                        <- Environnements de simulation (Gymnasium)
│   │   ├── __init__.py
│   │   ├── mine_env.py              <- Environnement principal
│   │   └── road_network.py          <- Modélisation du graphe routier
│   │
│   ├── agents/                      <- Implémentation des agents RL
│   │   ├── __init__.py
│   │   ├── ppo_agent.py             <- Agent PPO (Stable-Baselines3)
│   │   └── dqn_agent.py             <- Agent DQN
│   │
│   ├── baselines/                   <- Modèles de référence
│   │   ├── __init__.py
│   │   ├── dijkstra.py              <- Plus court chemin statique
│   │   └── dispatch_heuristic.py    <- Simulation FMS actuel
│   │
│   ├── models/                      <- Modèles physiques (consommation, usure)
│   │   ├── __init__.py
│   │   └── fuel_model.py            <- Modèle de consommation de carburant
│   │
│   ├── modeling/                    <- Pipeline d'entraînement et prédiction
│   │   ├── __init__.py
│   │   ├── train.py                 <- Code d'entraînement
│   │   └── predict.py               <- Inférence avec modèles entraînés
│   │
│   └── utils/                       <- Utilitaires (visualisation, métriques)
│       ├── __init__.py
│       ├── plots.py                 <- Code de visualisation
│       └── metrics.py               <- Calcul des KPIs
│
├── models/                          <- Modèles entraînés sauvegardés
│   ├── ppo_mine_agent.zip
│   └── dqn_mine_agent.zip
│
├── notebooks/                       <- Notebooks Jupyter (CRISP-DM)
│   ├── 1.0-business-and-data-understanding.ipynb
│   ├── 2.0-exploratory-data-analysis.ipynb
│   ├── 3.0-modeling.ipynb
│   └── 4.0-evaluation.ipynb
│
├── references/                      <- Articles scientifiques, documentation FMS
│   ├── Alarie_Gamache_2002.pdf
│   └── ...
│
├── reports/                         <- Rapports générés, figures pour le mémoire
│   ├── figures/
│   │   ├── comparison_fuel.png
│   │   └── comparison_waittime.png
│   └── final_report.pdf
│
└── tests/                           <- Tests unitaires
    ├── __init__.py
    ├── test_env.py
    └── test_agents.py
```

---

## Contributions

Les contributions sont bienvenues ! Si vous souhaitez :
- Signaler un bug → Ouvrir une [issue](https://github.com/Juan-Lucas/memoire-master-rl-logistique/issues)
- Proposer une amélioration → Créer une [pull request](https://github.com/Juan-Lucas/memoire-master-rl-logistique/pulls)
- Discuter du projet → [Discussions](https://github.com/Juan-Lucas/memoire-master-rl-logistique/discussions)

---

## Licence

Ce projet est sous licence **MIT**. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

Vous êtes libre de :
- ✅ Utiliser ce code pour vos propres recherches
- ✅ Modifier et distribuer
- ✅ Utiliser à des fins commerciales

À condition de :
- 📄 Inclure une copie de la licence et du copyright
- 🙏 Citer ce travail dans vos publications

### Citation

Si vous utilisez ce code dans vos recherches, merci de citer :

```bibtex
@mastersthesis{mupasa2026rl-mining,
  author  = {Jean-Luc MUPASA KALUNGA},
  title   = {Au-delà des Heuristiques Statiques : Conception et Mise en Œuvre 
             d'un Agent d'Apprentissage par Renforcement pour l'Optimisation 
             Holistique et Dynamique de la Logistique Minière},
  school  = {Université Don Bosco de Lubumbashi (UDBL)},
  year    = {2026},
  type    = {Mémoire de Master},
  url     = {https://github.com/Juan-Lucas/memoire-master-rl-logistique}
}
```

---

## Contact

**Jean-Luc Mupasa Kalunga**  
📧 Email : [16mk293@esisalama.org](mailto:16mk293@esisalama.org)  
🔗 GitHub : [@Juan-Lucas](https://github.com/Juan-Lucas)  
🎓 Université : Université Don Bosco de Lubumbashi (UDBL)

Pour toute question concernant ce projet, n'hésitez pas à me contacter ou à ouvrir une issue sur GitHub.

---

## Remerciements

- **Dr. Olfa FERCHICHI** pour l'encadrement et les conseils avisés
- **Communauté Stable-Baselines3** pour l'excellent framework RL
- **Auteurs des articles de référence** listés dans `references/`
- **Collègues du Master** pour les discussions enrichissantes

---

<p align="center">
  <i>Ce projet est développé dans le cadre d'un mémoire de Master 2 en Data science spécialité Logistique.<br>
  Dernière mise à jour : Octobre 2025</i>
</p>

<p align="center">
  <a href="#table-des-matières">⬆ Retour en haut</a>
</p>

--------

