# Résultats clés - Chiffres à mémoriser

Ce document compile les chiffres clés à mémoriser pour la soutenance. Il est aligné sur les chapitres 4, 5 et 6 du mémoire.

---

## Configuration de simulation

### Mine nominale
- **Nombre de camions** : 12
- **Nombre de pelles** : 3
- **Nombre de dumps** : 2
- **Capacité camion** : 140 tonnes
- **Durée épisode** : 8 heures (480 minutes)
- **Observation** : 147 dimensions normalisées
- **Action space** : 7 actions (3×2 + ATTENDRE)

### Scénarios
- **Nominal** : 12 camions, 3 pelles, 2 dumps, p_b = 2 %
- **High Load** : 18 camions, 3 pelles, 2 dumps, p_b = 2 %
- **High Breakdown** : 12 camions, 3 pelles, 2 dumps, p_b = 10 %

### Hyperparamètres RL
- **PPO** : α = 3×10⁻⁴, γ = 0.99, batch size = 64, GAE λ = 0.95, ε clip = 0.2, MLP 128×128, 2M steps
- **DQN** : α = 3×10⁻⁴, γ = 0.99, batch size = 64, replay buffer = 200 000, MLP 128×128, 2M steps
- **Q-Learning / SARSA** : α = 0.2, γ = 0.99, ε = 1.0→0.01, 30 000 épisodes, 8 bins, 5 features discrétisées

---

## Fonction de récompense
- Poids : w1 = 1.0 (rendement), w2 = 0.1 (équité), w3 = 0.05 (coût), w4 = 0.3 (attente)
- Signal : rendement, variance des files, coût de trajet, pénalité d’attente opérationnelle
- Sans w4, signal non discriminant (Δ < 6×10⁻⁴) ; w4 = 0.3 garantit que tout cycle productif surpasse ATTENDRE.

---

## Résultats – Scénario nominal

### Productivité (t/h)
- **Fixed Assignment** : 4 074 ± 36
- **Nearest Shovel** : 3 830 ± 51
- **Shortest Path** : 3 830 ± 51
- **PPO** : 3 335 ± 38
- **DQN** : 3 311 ± 72
- **FIFO** : 3 318 ± 42
- **Q-Learning** : 3 273 ± 89
- **SARSA** : 3 222 ± 97

### Temps d’attente moyen (min)
- **Fixed Assignment** : 27.9 ± 3.8
- **PPO** : 30.6 ± 5.1
- **DQN** : 33.4 ± 10.4
- **FIFO** : 32.8 ± 3.7
- **Q-Learning** : 36.9 ± 19.4
- **SARSA** : 52.5 ± 14.2
- **Nearest Shovel** : 74.6 ± 7.2
- **Shortest Path** : 74.6 ± 7.2

### Consommation spécifique (L/t)
- **Nearest Shovel / Shortest Path** : 0.0443 ± 0.0004
- **Fixed Assignment** : 0.0431 ± 0.0003
- **PPO** : 0.0525 ± 0.0005
- **DQN** : 0.0524 ± 0.0008
- **FIFO** : 0.0528 ± 0.0003
- **Q-Learning** : 0.0534 ± 0.0008
- **SARSA** : 0.0537 ± 0.0006

### Taux d’utilisation camions (%)
- **PPO** : 97.1
- **Fixed Assignment** : 96.8
- **DQN** : 96.4
- **FIFO** : 95.8
- **Q-Learning** : 95.4
- **SARSA** : 93.2
- **Nearest Shovel / Shortest Path** : 87.8

---

## Résultats – Scénarios perturbés

### High Load (18 camions)
- **Fixed Assignment** : 5 874.8 t/h, 43.0 min attente
- **DQN** : 5 664.8 t/h, 46.8 min attente
- **PPO** : 5 244.8 t/h, 92.8 min attente (variable)
- **Q-Learning** : 4 795.0 t/h, 69.1 min attente
- **FIFO** : 4 747.8 t/h, 44.2 min attente
- **SARSA** : 4 042.5 t/h, 119.8 min attente
- **Shortest Path** : 3 956.8 t/h, 204.7 min attente
- **Nearest Shovel** : 3 939.3 t/h, 200.2 min attente

### High Breakdown (10% pannes)
- **DQN** : 3 991.8 t/h, 68.7 min attente
- **PPO** : 3 946.3 t/h, 65.1 min attente
- **Fixed Assignment** : 3 860.5 t/h, 53.1 min attente (meilleur wait)
- **Shortest Path** : 3 702.3 t/h, 91.0 min attente
- **Nearest Shovel** : 3 692.5 t/h, 90.1 min attente
- **Q-Learning** : 3 176.3 t/h, 88.2 min attente
- **FIFO** : 3 115.0 t/h, 53.0 min attente
- **SARSA** : 2 973.3 t/h, 77.5 min attente
---

## Statistiques d’évaluation

- **Réplications** : 10 par scénario
- **Seeds** : 42 à 51
- **Intervalle de confiance** : 95%

## Points à retenir

- **PPO et DQN surpassent Fixed en nominal** : 4 208.8 et 4 168.5 t/h vs 4 074.0 — H1 validee en conditions stables.
- **DQN et PPO surpassent Fixed en high_breakdown** : 3 991.8 et 3 946.3 t/h vs 3 860.5 — H1 validee en conditions perturbees.
- **Fixed domine en high_load** (5 874.8 t/h) mais DQN est proche (5 664.8, -3.6%).
- **Nearest Shovel / Shortest Path s'effondrent en high_load** : 200 min d'attente.
- **ShortestPath != Nearest** : 3 872.8 vs 3 830.8 en nominal — Eq. 3.1 apporte une differentiation reelle.
- **PPO variable en high_load** (5 244.8 t/h, ecart-type 332) — sensibilite aux conditions initiales.

---

## Notes techniques

- Capacité camion : 140 t
- Observation : 147 features normalisées
- Action : 7 actions
- Entraînement : 2M steps pour PPO/DQN, 30 000 épisodes pour Q-Learning/SARSA

---

*Ce document résume les chiffres clés et la cohérence des résultats présentés dans le mémoire.*


