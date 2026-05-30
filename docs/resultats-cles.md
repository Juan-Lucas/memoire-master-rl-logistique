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
- **Observation** : 138 dimensions normalisées
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
- Poids : w1 = 1.0 (rendement), w2 = 0.1 (équité), w3 = 0.05 (coût), w4 = 1.0 (attente)
- Signal : rendement, variance des files, coût de trajet, pénalité d’attente à la pelle
- Sans w4, signal non discriminant ; avec w4 = 1.0, signal efficace.

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
- **Random** : 3 148 ± 45

### Temps d’attente moyen (min)
- **Fixed Assignment** : 27.9 ± 3.8
- **PPO** : 30.6 ± 5.1
- **DQN** : 33.4 ± 10.4
- **FIFO** : 32.8 ± 3.7
- **Q-Learning** : 36.9 ± 19.4
- **SARSA** : 52.5 ± 14.2
- **Random** : 50.6 ± 3.9
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
- **Random** : 0.0551 ± 0.0005

### Taux d’utilisation camions (%)
- **PPO** : 97.1
- **Fixed Assignment** : 96.8
- **DQN** : 96.4
- **FIFO** : 95.8
- **Q-Learning** : 95.4
- **SARSA** : 93.2
- **Random** : 92.8
- **Nearest Shovel / Shortest Path** : 87.8

---

## Résultats – Scénarios perturbés

### High Load (18 camions)
- **Fixed Assignment** : 5 875 ± 54 t/h
- **Q-Learning** : 4 772 ± 45 t/h
- **DQN** : 4 772 ± 73 t/h
- **FIFO** : 4 772 ± 72 t/h
- **PPO** : 4 748 ± 59 t/h
- **SARSA** : 4 636 ± 62 t/h
- **Random** : 4 547 ± 46 t/h
- **Nearest Shovel / Shortest Path** : 3 919 ± 51 t/h (201.1 ± 11.4 min d’attente)

### High Breakdown (10% pannes)
- **Fixed Assignment** : 3 865 ± 28 t/h
- **Nearest Shovel / Shortest Path** : 3 687 ± 37 t/h
- **PPO** : 3 166 ± 52 t/h (meilleur temps d’attente : 47.7 min)
- **DQN** : 3 162 ± 35 t/h
- **Q-Learning** : 3 146 ± 44 t/h
- **FIFO** : 3 118 ± 60 t/h
- **SARSA** : 3 083 ± 56 t/h
- **Random** : 2 998 ± 38 t/h

---

## Statistiques d’évaluation

- **Réplications** : 10 par scénario
- **Seeds** : 42 à 51
- **Intervalle de confiance** : 95%

## Points à retenir

- **Fixed Assignment domine en nominal** grâce au Match Factor 12/3 = 4.
- **PPO est le meilleur agent en robustesse d’attente** sur le scénario `high_breakdown`.
- **Nearest Shovel / Shortest Path s’effondrent en high_load** (≈ 3 919 t/h, 201 min d’attente).
- **Q-Learning et SARSA** sont utiles pour l’analyse méthodologique ; PPO et DQN sont plus stables sur l’espace continu.

---

## Notes techniques

- Capacité camion : 140 t
- Observation : 138 features normalisées
- Action : 7 actions
- Entraînement : 2M steps pour PPO/DQN, 30 000 épisodes pour Q-Learning/SARSA

---

*Ce document résume les chiffres clés et la cohérence des résultats présentés dans le mémoire.*

