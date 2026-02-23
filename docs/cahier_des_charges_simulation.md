# Cahier des Charges de la Simulation

## 1. Objectif de la simulation
- Reproduire le fonctionnement d’un système logistique minier (flotte camions-pelles, zones tampon, routes, points de chargement/déchargement).
- Évaluer différentes stratégies de dispatching et d’optimisation (heuristiques, RL, simulation dynamique).
- Mesurer l’impact sur les KPIs : productivité, temps d’attente, utilisation, coûts, robustesse.

## 2. Périmètre fonctionnel
- Modélisation des entités : camions, pelles, zones tampon, routes, points de chargement/déchargement.
- Paramétrage des ressources : nombre, capacités, vitesses, contraintes opérationnelles.
- Gestion des événements : dispatching, chargement, transport, déchargement, attente, congestion, incidents.
- Simulation de scénarios : baseline, optimisation heuristique, RL, perturbations (pannes, variations de demande).

## 3. KPIs à mesurer
- Productivité (tonnage transporté, cycles).
- Temps d’attente (zone tampon, pelles, déchargement).
- Utilisation des équipements.
- Consommation de carburant.
- Coût opérationnel.
- Robustesse et adaptation.
- Sécurité (incidents).

## 4. Contraintes techniques
- Simulation paramétrable (nombre d’entités, topologie, règles).
- Export des résultats (CSV, graphiques, logs).
- Intégration avec modules d’optimisation (RL, heuristiques).
- Reproductibilité des scénarios.

## 5. Livrables attendus
- Code source de la simulation.
- Documentation technique et utilisateur.
- Jeux de données de test.
- Rapport d’analyse des résultats.

## 6. Critères de validation
- Respect du périmètre fonctionnel.
- Mesure correcte des KPIs.
- Robustesse face aux perturbations.
- Clarté de la documentation.
- Reproductibilité des résultats.

---

*Ce cahier des charges définit les exigences pour la simulation logistique minière, servant de base à l’évaluation des stratégies d’optimisation et à la rédaction du mémoire.*
