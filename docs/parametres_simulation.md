# Paramètres de Simulation et Hypothèses

Ce document synthétise tous les paramètres utiles à la construction d'une simulation logistique minière réaliste pour l'apprentissage par renforcement, extraits des fiches de lecture, articles, thèses et rapports du projet.

## 1. Tableau des Paramètres Utiles

| Paramètre                        | Valeur / Distribution         | Source / Justification                                                                 |
|----------------------------------|------------------------------|---------------------------------------------------------------------------------------|
| Capacité camion (Type 1)         | 140 tonnes                   | Caterpillar 785C, Afrapoli & Askari-Nasab (2017)                                      |
| Vitesse moyenne à vide           | LOGN(32, 26) km/h            | Table 4.4, Afrapoli (2019)                                                            |
| Temps de cycle pelle             | NORM(17, 0.5) s              | Table 4.3, Afrapoli (2019), Hitachi 2500                                              |
| Consommation au ralenti          | 10 L/heure                   | Standard diesel, article X                                                            |
| Probabilité de panne camion      | 2% par shift                 | Hypothèse réaliste, discussion article Y                                              |
| Pente route                      | 8%                           | Mohtasham et al. (2023), layout Sungun mine                                           |
| Temps d’attente zone tampon      | NORM(5, 2) min               | Synthèse articles simulation                                                          |
| Temps de trajet                  | LOGN(12, 4) min              | Distribution ajustée, Afrapoli & Askari-Nasab (2017)                                  |
| Capacité godet pelle             | 15 tonnes                    | Spécification Hitachi, Table 2, Mohtasham et al. (2023)                               |
| Production cible                 | 10 000 t/jour                | Cas d’étude, Mohtasham et al. (2023)                                                  |
| Probabilité dégradation route    | 5% par heure                 | Hypothèse de travail, justifiée pour robustesse                                       |
| Nombre de camions                | 12                           | Cas Kansanshi, Kangwa (2021)                                                          |
| Nombre de pelles                 | 3                            | Cas Kansanshi, Kangwa (2021)                                                          |
| Temps de chargement pelle        | NORM(2, 0.3) min             | Table 2, Mohtasham et al. (2023)                                                      |
| Temps de déchargement            | NORM(1, 0.2) min             | Table 2, Mohtasham et al. (2023)                                                      |
| Match Factor (MF)                | 0.85                         | KPI central, simulation minière, Ozdemir & Kumral (2019)                              |
| Utilisation flotte camions       | 75%                          | Simulation, Ozdemir & Kumral (2019)                                                   |
| Utilisation pelles               | 80%                          | Simulation, Ozdemir & Kumral (2019)                                                   |
| Temps de cycle camion            | NORM(15, 3) min              | Synthèse simulation, Zeng et al. (2022)                                               |
| Productivité                     | 8 000 t/shift                | Simulation, Abolghasemian et al. (2020)                                               |
| ...                              | ...                          | ...         

## 6. Hyperparamètres RL (Tableau 4.6 du mémoire)

| Paramètre | Q-Learning | SARSA | DQN | PPO |
|-----------|------------|-------|-----|-----|
| Taux d'apprentissage (α) | 0.1 | 0.1 | 0.0001 | 0.0003 |
| Facteur d'actualisation (γ) | 0.99 | 0.99 | 0.99 | 0.99 |
| Taux d'exploration initial (ε) | 1.0 | 1.0 | - | - |
| Taux d'exploration final | 0.01 | 0.01 | - | - |
| Taille du batch | - | - | 64 | 64 |
| Taille du replay buffer | - | - | 10000 | - |
| Période d'entraînement (épisodes) | 1000 | 1000 | 100 | 100 |
| Architecture réseau | - | - | MLP 128×128 ReLU | MLP 128×128 ReLU |
| GAE λ | - | - | - | 0.95 |
| PPO clipping (ε) | - | - | - | 0.2 |                                                                          |

*Remplir et compléter ce tableau au fur et à mesure de la lecture des articles et rapports.*

## 2. Hypothèses et Justifications

- Les distributions de temps de cycle sont supposées lognormales ou normales selon la littérature.
- La probabilité de panne est fixée à 2% par shift, faute de données précises, mais justifiée par les discussions d’articles sur la robustesse.
- La dégradation des routes est modélisée par une probabilité de 5% par heure, pour tester la robustesse de l’agent RL.
- Les valeurs non trouvées dans la littérature sont fixées par hypothèse de travail et clairement justifiées.

## 3. Limites et Simplifications

- Modélisation simplifiée des pannes et files d’attente.
- Pas de prise en compte explicite de la météo ou des retards opérationnels réels.
- Simulation sur des quarts de 8 heures, 100 réplications par scénario.
- Modèle de consommation de carburant simplifié (charge, pente, vitesse).

## 4. Exemples de Modélisation Stochastique

- Temps de trajet : `temps_trajet = np.random.lognormal(moyenne, ecart_type)`
- Temps de chargement : `temps_chargement = np.random.normal(moyenne, ecart_type)`
- Panne camion : `if np.random.rand() < 0.02: panne = True`
- Dégradation route : `if np.random.rand() < 0.05: route_degradee = True`

## 5. Sources et Références

- Afrapoli & Askari-Nasab (2017), Tableaux 2, 4.3, 4.4
- Mohtasham et al. (2023), Tableaux 1, 2, layout Sungun mine
- Kangwa (2021), Tableaux 1.1, 2.1, cas Kansanshi
- Ozdemir & Kumral (2019), Simulation truck-shovel
- Abolghasemian et al. (2020), Simulation hauling system
- Zeng, Baafi & Fan (2022), Simulation truck allocation
- Spécifications Caterpillar, Hitachi, Komatsu (PDF techniques)
- Hypothèses de travail clairement justifiées

---

*Ce document est évolutif et doit être enrichi à chaque nouvelle lecture ou extraction de données pertinente.*

*La crédibilité de la simulation repose sur la justification de chaque paramètre par une source ou une hypothèse explicitement reconnue.*
