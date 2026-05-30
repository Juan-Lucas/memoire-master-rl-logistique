# Cahier des Charges de la Simulation

## 1. Objectif de la simulation
- Reproduire le fonctionnement d’un système logistique minier réel sous forme de simulateur Gymnasium.
- Évaluer et comparer des stratégies de dispatching classiques (FIFO, Fixed Assignment, Nearest Shovel, Shortest Path) avec des approches RL (Q-Learning, SARSA, DQN, PPO).
- Mesurer la performance sur des scénarios opérationnels et perturbés (nominal, high_load, high_breakdown).
- Fournir des résultats exploitables pour le mémoire (chapitres 5 et 6).

## 2. Périmètre fonctionnel
- Modélisation des entités : camions, pelles, dumps, routes, états de disponibilité.
- Paramétrage des ressources : nombre de camions, nombre de pelles, nombre de dumps, capacité, temps de cycle.
- Gestion des événements : dispatching, trajet à vide, chargement, trajet chargé, déchargement, retour, pannes.
- Simulation de scénarios : conditions nominales, surcharge, perturbations de pannes.
- Export des résultats : CSV, logs, graphiques et tableaux pour analyse.

## 3. KPIs à mesurer
- Productivité horaire (t/h).
- Temps d’attente moyen par camion (minutes).
- Consommation spécifique (litres par tonne transportée).
- Taux d’utilisation des camions (%).
- Coût moyen par cycle (litres par cycle).
- Reward cumulée de l’agent RL.
- Robustesse et stabilité statistiques sur 10 réplications.

## 4. Contraintes techniques
- Interface Gymnasium compatible Stable-Baselines3.
- Observations normalisées dans [0, 1].
- Actions discrètes : paires (pelle, dump) + ACTION ATTENDRE.
- Reproductibilité : seeds fixes 42..51.
- Modularité : séparation simulation, baselines, RL, evaluation.

## 5. Scénarios expérimentaux
- **Nominal** : 12 camions, 3 pelles, 2 dumps, p_b = 2%.
- **High Load** : 18 camions, 3 pelles, 2 dumps, p_b = 2%.
- **High Breakdown** : 12 camions, 3 pelles, 2 dumps, p_b = 10%.

## 6. Livrables attendus
- Code source complet du simulateur et des agents.
- Fiches techniques des modules principaux.
- Résultats expérimentaux chiffrés (tableaux, figures).
- Rapport d’analyse des résultats pour les chapitres 5 et 6.
- Documentation de validation et guide de reproduction.

## 7. Critères de validation
- Les KPI sont calculés correctement et reproduisibles.
- Les scénarios de test sont identiques pour toutes les méthodes.
- Les baselines produisent des résultats comparables à ceux du mémoire.
- L’agent PPO est évalué sur résultats moyens ± écart-type.
- La comparaison met en évidence la robustesse en high_breakdown.

---

*Ce cahier des charges formalise les exigences opérationnelles et expérimentales du mémoire, en s’appuyant sur les chapitres 4, 5 et 6.*
