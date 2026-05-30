# Paramètres de Simulation et Hypothèses

Ce document synthétise les paramètres réellement utilisés par la simulation et les choix présentés dans les chapitres 4, 5 et 6 du mémoire.

## 1. Tableau des Paramètres Utiles

| Paramètre                        | Valeur / Distribution         | Source / Justification                                                                 |
|----------------------------------|------------------------------|---------------------------------------------------------------------------------------|
| Capacité camion                  | 140 tonnes                   | Configuration nominale du simulateur / Chapitre 5                                   |
| Capacité godet pelle             | 15 tonnes                    | Spécification Hitachi / Chapitre 5                                                    |
| Nombre de camions (nominal)      | 12                           | Scénario nominal / Chapitre 6                                                         |
| Nombre de camions (high_load)    | 18                           | Scénario surcharge / Chapitre 6                                                       |
| Nombre de pelles                 | 3                            | Scénario nominal / Chapitre 6                                                         |
| Nombre de dumps                  | 2                            | Scénario nominal / Chapitre 6                                                         |
| Durée épisode                    | 8 heures (480 min)           | Configuration du poste / Chapitre 5                                                   |
| Temps de chargement pelle        | N(2, 0.3) minutes            | Chapitre 5                                                                           |
| Temps de déchargement            | N(1, 0.2) minutes            | Chapitre 5                                                                           |
| Temps de trajet                  | Log-normal (σ = 0.12)        | Chapitre 5                                                                           |
| Probabilité de panne (nominal)   | 2% par cycle                 | Chapitre 6                                                                           |
| Probabilité de panne (breakdown) | 10% par cycle                | Chapitre 6                                                                           |
| Durée panne                      | Uniforme U[10, 30] minutes   | Chapitre 5                                                                           |
| Pente route                      | 3-8%                         | Graphe routier / Chapitre 5                                                           |
| Distance route                   | 1.5-3.2 km                   | Graphe routier / Chapitre 5                                                           |
| Observation                     | 138 dimensions               | Section 4.3 / code `observation_size()`                                               |
| Action space                    | 7 actions                    | 3 pelles × 2 dumps + ATTENDRE                                                        |

## 2. Hyperparamètres RL

| Paramètre | Q-Learning | SARSA | DQN | PPO |
|-----------|------------|-------|-----|-----|
| Taux d'apprentissage (α) | 0.2 | 0.2 | 3×10⁻⁴ | 3×10⁻⁴ |
| Facteur d'actualisation (γ) | 0.99 | 0.99 | 0.99 | 0.99 |
| Taux d'exploration initial (ε) | 1.0 | 1.0 | - | - |
| Taux d'exploration final (ε) | 0.01 | 0.01 | - | - |
| Taille du batch | - | - | 64 | 64 |
| Taille du replay buffer | - | - | 200 000 | - |
| Période d'entraînement | 30 000 épisodes | 30 000 épisodes | 2 000 000 steps | 2 000 000 steps |
| Bins de discrétisation | 8 | 8 | - | - |
| Features d'état tabulaire | 5 | 5 | - | - |
| Architecture réseau | - | - | MLP 128×128 ReLU | MLP 128×128 ReLU |
| GAE λ | - | - | - | 0.95 |
| PPO clipping (ε) | - | - | - | 0.2 |

## 3. Hypothèses et justifications

- Les temps de trajet sont modélisés par une distribution log-normal, stabilisée par σ = 0.12.
- La probabilité de panne nominale est 2% par cycle ; 10% dans le scénario `high_breakdown`.
- Le vecteur d'observation compte 138 dimensions pour la configuration nominale.
- L'espace d'action discret encode les paires `(pelle, dump)` plus une action `ATTENDRE`.
- La fonction de récompense utilise les poids `w1 = 1.0`, `w2 = 0.1`, `w3 = 0.05` et `w4 = 1.0`.
- Les méthodes tabulaires réduisent l'état à 5 features discrétisées (8 bins chacune).
- Les résultats sont mesurés sur 10 réplications indépendantes (seeds 42..51) pour assurer la reproductibilité.

## 4. Limites et simplifications

- La simulation ne modélise pas explicitement la météo ou les incidents terrain.
- Les pannes sont uniquement des pannes camion, pas des pannes de pelle ou de dump.
- La consommation de carburant est estimée par modèle simple et normalisée.
- Le simulateur actuel ne couvre pas encore l’adaptation à topologies très différentes.
- Les résultats de DQN et PPO sont limités aux scénarios définis; le transfert de domaine n’a pas été testé exhaustivement.

## 5. Points d’attention

- La capacité camion est **140 tonnes**, valeur industrielle.
- Les durées de chargement et déchargement sont stochastiques.
- Le `Match Factor` nominal est 12 / 3 = 4, ce qui explique la performance élevée de `Fixed Assignment` en nominal.
- La normalisation des observations est essentielle pour la stabilité des agents Deep RL.
- Le terme `w4` dans la récompense permet de rendre le signal discriminant entre actions.

## 6. Sources et références

- Chapitre 5 : implémentation du simulateur et protocole expérimental.
- Chapitre 6 : résultats et analyse comparatives.
- Stable-Baselines3 : implémentation de DQN et PPO.
- Discussion théorique du choix des KPI dans les chapitres 4 et 6.

---

*Ce document doit suivre les choix réels de simulation et non les valeurs ponctuelles extraites d’un article tiers.*
