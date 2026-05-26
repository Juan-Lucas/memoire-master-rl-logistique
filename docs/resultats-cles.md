# Résultats clés - Chiffres à mémoriser

Ce document compile les chiffres clés à mémoriser pour la soutenance. Organisez-vous pour les connaître par cœur.

---

## Configuration de simulation

### Mine typique
- **Nombre de camions** : 12
- **Nombre de pelles** : 3
- **Nombre de dumps** : 2
- **Capacité camion** : 140 tonnes (Caterpillar 785C)
- **Capacité godet pelle** : 15 tonnes (Hitachi 2500)

### Paramètres temporels
- **Durée épisode** : 1 shift (8 heures = 480 minutes)
- **Pas de décision Δt** : 5 minutes
- **Nombre de pas par épisode** : 96 (480 / 5)
- **Nombre d'épisodes entraînement** : 100 (PPO), 1000 (Q-Learning/SARSA)

### Distributions stochastiques
- **Temps de trajet** : LOGN(12, 4) minutes (moyenne 12 min)
- **Temps de chargement** : N(2, 0.3) minutes
- **Temps de déchargement** : N(1, 0.2) minutes
- **Probabilité de panne** : 2% par shift

---

## Hyperparamètres RL

### PPO (méthode principale)
- **Learning rate (α)** : 0.0003
- **Facteur d'actualisation (γ)** : 0.99
- **Batch size** : 64
- **GAE λ** : 0.95
- **PPO clipping (ε)** : 0.2
- **Architecture** : MLP 128×128 avec ReLU
- **Nombre d'épisodes** : 100

### Q-Learning / SARSA
- **Learning rate (α)** : 0.1
- **Facteur d'actualisation (γ)** : 0.99
- **Taux d'exploration initial (ε)** : 1.0
- **Taux d'exploration final (ε)** : 0.01
- **Nombre d'épisodes** : 1000

### DQN
- **Learning rate (α)** : 0.0001
- **Facteur d'actualisation (γ)** : 0.99
- **Batch size** : 64
- **Taille replay buffer** : 10000
- **Architecture** : MLP 128×128 avec ReLU
- **Nombre d'épisodes** : 100

---

## Fonction de récompense

### Poids des composantes
- **w1 (rendement)** : 1.0
- **w2 (équité)** : 0.1
- **w3 (coût)** : 0.05

### Coût de trajet (Eq. 3.1)
- **α (temps)** : 0.5
- **β (distance)** : 0.3
- **γ (énergie)** : 0.2

### Exemple calcul récompense
- **Rendement** : 280 tonnes (2 camions × 140 t)
- **Équité** : -5.2 (pénalité files inégales)
- **Coût** : -120 (pénalité distance)
- **R_total** : 1.0 × 280 + 0.1 × (-5.2) + 0.05 × (-120) = 273.48

---

## Résultats - Scénario nominal

### Productivité (t/h)
- **PPO** : 8000 t/h
- **FIFO** : 6950 t/h
- **Shortest Path** : 7300 t/h
- **Fixed Assignment** : 6500 t/h
- **Nearest Shovel** : 7100 t/h
- **Gain PPO vs FIFO** : +15%
- **Gain PPO vs Fixed Assignment** : +23%

### Temps d'attente moyen (min)
- **PPO** : 3.2 min
- **FIFO** : 4.1 min
- **Shortest Path** : 4.3 min
- **Fixed Assignment** : 4.8 min
- **Nearest Shovel** : 3.9 min
- **Gain PPO vs Shortest Path** : -25%
- **Gain PPO vs Fixed Assignment** : -33%

### Consommation spécifique (L/t)
- **PPO** : 0.85 L/t
- **FIFO** : 0.91 L/t
- **Shortest Path** : 0.88 L/t
- **Fixed Assignment** : 0.94 L/t
- **Nearest Shovel** : 0.89 L/t
- **Gain PPO vs Fixed Assignment** : -10%
- **Gain PPO vs FIFO** : -7%

### Taux d'utilisation camions (%)
- **PPO** : 82%
- **FIFO** : 75%
- **Shortest Path** : 78%
- **Fixed Assignment** : 70%
- **Nearest Shovel** : 76%
- **Gain PPO vs Fixed Assignment** : +12 points

### Taux d'utilisation pelles (%)
- **PPO** : 85%
- **FIFO** : 78%
- **Shortest Path** : 80%
- **Fixed Assignment** : 72%
- **Nearest Shovel** : 77%
- **Gain PPO vs Fixed Assignment** : +13 points

---

## Résultats - Robustesse scénarios perturbés

### Scénario High-load (15 camions au lieu de 12)
- **PPO** : 92% de performance nominale
- **FIFO** : 85% de performance nominale
- **Shortest Path** : 88% de performance nominale
- **Gain PPO** : +7 points vs FIFO

### Scénario Low-load (8 camions au lieu de 12)
- **PPO** : 95% de performance nominale
- **FIFO** : 90% de performance nominale
- **Shortest Path** : 92% de performance nominale
- **Gain PPO** : +5 points vs FIFO

### Scénario High-breakdown (5% pannes au lieu de 2%)
- **PPO** : 88% de performance nominale
- **FIFO** : 78% de performance nominale
- **Shortest Path** : 82% de performance nominale
- **Gain PPO** : +10 points vs FIFO

### Scénario Single-shovel (1 pelle au lieu de 3)
- **PPO** : 85% de performance nominale
- **FIFO** : 70% de performance nominale
- **Shortest Path** : 75% de performance nominale
- **Gain PPO** : +15 points vs FIFO

### Scénario Short-shift (4 heures au lieu de 8)
- **PPO** : 90% de performance nominale
- **FIFO** : 82% de performance nominale
- **Shortest Path** : 85% de performance nominale
- **Gain PPO** : +8 points vs FIFO

---

## Statistiques d'évaluation

### Significativité statistique
- **Nombre de réplications** : 10
- **Seeds** : Fixées pour reproductibilité
- **Test** : t-test de Student apparié
- **Niveau de signification** : p < 0.05
- **Intervalle de confiance** : 95%

### Effet de taille (Cohen's d)
- **Productivité PPO vs FIFO** : d = 1.2 (effet large)
- **Temps d'attente PPO vs Shortest Path** : d = 0.9 (effet large)
- **Consommation PPO vs Fixed Assignment** : d = 0.8 (effet large)

### Variabilité (écart-type)
- **Productivité PPO** : 8000 ± 250 t/h
- **Productivité FIFO** : 6950 ± 300 t/h
- **Temps d'attente PPO** : 3.2 ± 0.4 min
- **Temps d'attente FIFO** : 4.1 ± 0.5 min

---

## Coûts computationnels

### Entraînement
- **PPO** : 1-2 heures sur GPU (RTX 3080)
- **DQN** : 1.5-2.5 heures sur GPU
- **Q-Learning** : 30-45 minutes sur CPU
- **SARSA** : 30-45 minutes sur CPU

### Inférence (temps réel)
- **PPO** : < 1 ms par décision
- **DQN** : < 1 ms par décision
- **Q-Learning** : < 0.1 ms par décision (lookup table)
- **SARSA** : < 0.1 ms par décision (lookup table)

### Mémoire
- **PPO** : ~500 MB (réseau + buffer)
- **DQN** : ~1 GB (replay buffer)
- **Q-Learning** : ~10 MB (table Q)
- **SARSA** : ~10 MB (table Q)

---

## Comparaison RL tabulaire vs profond

### Espace d'état
- **Dimension** : ~50 features (files, positions, disponibilités)
- **Taille discrétisée** : 10^50 états possibles
- **RL tabulaire** : impossible (explosion combinatoire)
- **RL profond** : gérable via approximation

### Performance
- **Q-Learning** : 7200 t/h (limité par taille table)
- **SARSA** : 7100 t/h (limité par taille table)
- **DQN** : 7800 t/h (meilleur que tabulaire)
- **PPO** : 8000 t/h (meilleur performance globale)

### Temps d'entraînement
- **Q-Learning** : 30-45 min (convergence lente)
- **SARSA** : 30-45 min (convergence lente)
- **DQN** : 1.5-2.5 h (plus stable)
- **PPO** : 1-2 h (plus rapide convergence)

---

## KPIs industriels clés

### Match Factor (MF)
- **Définition** : Nombre de camions / Nombre de pelles
- **Configuration** : 12 / 3 = 4
- **Optimal théorique** : 3-5
- **Résultat PPO** : MF effectif = 3.8 (proche optimal)

### Cycle Time
- **Moyenne théorique** : 30 min
- **Résultat PPO** : 28 min
- **Résultat FIFO** : 33 min
- **Gain** : -15% vs FIFO

### Tonnage par shift
- **Objectif** : 10 000 t/shift
- **Résultat PPO** : 9 200 t/shift
- **Résultat FIFO** : 8 000 t/shift
- **Gain** : +15% vs FIFO

---

## Impact économique estimé

### Gains par jour
- **Productivité** : +1 200 t/jour (15% de 8 000 t)
- **Valeur** : 1 200 t × $50/t = $60 000/jour
- **Carburant** : -10% → économie ~$5 000/jour
- **Total gains** : ~$65 000/jour

### Coûts annuels
- **Développement** : ~$50 000 (temps ingénieur)
- **Entraînement** : négligeable
- **Maintenance** : ~$10 000/an
- **Total coûts** : ~$60 000 (année 1)

### ROI
- **Gains annuels** : $65 000 × 250 jours = $16.25M
- **Coûts année 1** : $60 000
- **ROI année 1** : 27 000% (théorique)
- **ROI réaliste** : > 100% (avec hypothèses conservatrices)

---

## Exemples numériques à maîtriser

### Exemple 1 : Coût de trajet (Eq. 3.1)
- α = 0.5, β = 0.3, γ = 0.2
- T_ij = 15 min, D_ij = 2.5 km, E_ij = 0.8
- C_ij = 0.5 × 15 + 0.3 × 2.5 + 0.2 × 0.8 = 8.41

### Exemple 2 : Temps stochastique (Eq. 3.2)
- T̄_ij = 12 min, σ = 2.5
- ε_ij(t) = 2.3 (échantillon N(0, 2.5))
- T_ij(t) = 12 + 2.3 = 14.3 min

### Exemple 3 : Q-Learning (Eq. 4.3)
- Q = 50, r = 10, γ = 0.99, max Q' = 55, α = 0.1
- Q ← 50 + 0.1 × (10 + 0.99 × 55 - 50) = 51.45

### Exemple 4 : Espace d'action (Eq. 4.1)
- 3 pelles, 2 dumps
- |A| = 3 × 2 + 1 = 7 actions (0-6)
- Action 4 → shovel_idx = 2, dump_idx = 0

---

## Méthode de mémorisation

### Flash cards
- Créez des cartes avec question au recto, chiffre au verso
- Exemple : "Productivité PPO nominal ?" → "8000 t/h"

### Regroupement thématique
- **Configuration** : camions, pelles, temps
- **Hyperparamètres** : PPO, Q-Learning, DQN
- **Résultats nominaux** : productivité, attente, consommation
- **Robustesse** : % performance sur scénarios perturbés
- **Économie** : gains, coûts, ROI

### Répétition espacée
- Jour 1 : Toutes les catégories
- Jour 2 : Focus résultats nominaux
- Jour 3 : Focus robustesse
- Jour 4 : Focus économie
- Jour 5 : Révision complète

### Test à blanc
- Fermez ce document
- Essayez de réciter 10 chiffres clés
- Vérifiez et répétez les erreurs

---

## Checklist de mémorisation

### Configuration
- [ ] 12 camions, 3 pelles, 2 dumps
- [ ] Capacité camion 140 t
- [ ] Durée épisode 8h (480 min)
- [ ] Pas de décision 5 min
- [ ] 96 pas par épisode

### Hyperparamètres PPO
- [ ] α = 0.0003
- [ ] γ = 0.99
- [ ] Batch = 64
- [ ] λ = 0.95
- [ ] ε_clip = 0.2
- [ ] MLP 128×128

### Résultats nominaux
- [ ] PPO 8000 t/h
- [ ] Gain +15% vs FIFO
- [ ] Temps attente 3.2 min
- [ ] Consommation 0.85 L/t
- [ ] Utilisation camions 82%

### Robustesse
- [ ] High-load : 92% performance
- [ ] Low-load : 95% performance
- [ ] High-breakdown : 88% performance
- [ ] Single-shovel : 85% performance

### Économie
- [ ] Gains $65 000/jour
- [ ] ROI > 100%
- [ ] Coût entraînement 1-2h GPU
- [ ] Inférence < 1ms

---

**Score de mémorisation :** ____ / 20 cases cochées

**Objectif minimal avant soutenance :** 18 / 20
**Objectif idéal :** 20 / 20
