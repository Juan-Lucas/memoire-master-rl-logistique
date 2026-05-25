# Guide de Test et d'Évaluation — Commandes CLI

Toutes les commandes sont à taper depuis la racine du projet.

---

## Prérequis

```bash
pip install -e .
```

Vérifier l'installation :

```bash
python -m memoire_master_rl_logistique.main --check-env
```

Si cette commande affiche `check_env OK`, tout est installé correctement.

---

## 1. Simulation DES (validation du moteur)

```bash
python -m memoire_master_rl_logistique.main --sim-only
```

**Résultats attendus** (seed=42, 12 camions, 3 pelles, 2 dumps, 8h) :

| KPI | Valeur attendue |
|-----|-----------------|
| Tonnage total | 26 320 t |
| Productivité | 3 290 t/h |
| Attente moyenne/camion | 54.4 min |
| Utilisation | 89.5 % |
| Consommation spécifique | 0.05 L/t |

Chaque camion doit afficher entre 14 et 16 cycles complétés.

---

## 2. Validation de l'environnement Gymnasium

```bash
python -m memoire_master_rl_logistique.main --check-env
```

**Résultat attendu** :
```
check_env OK
  Observation : Box(0.0, 1.0, (54,), float32)
  Action      : Discrete(7)
  Camions=12, Pelles=3, Dumps=2
```

---

## 3. Évaluation des baselines heuristiques

```bash
python -m memoire_master_rl_logistique.rl.evaluate_agent
```

**Résultats attendus** (10 épisodes, 12 camions, 3 pelles, 2 dumps) :

| Baseline | Productivité (t/h) | Attente moy. (min) | Utilisation (%) |
|----------|--------------------:|--------------------:|----------------:|
| Fixed | ~4 074 ± 41 | ~27.9 ± 3.7 | ~96.5 ± 0.8 |
| Nearest | ~3 831 ± 55 | ~72.6 ± 7.4 | ~87.8 ± 1.5 |
| QueueAware | ~3 831 ± 55 | ~72.6 ± 7.4 | ~87.8 ± 1.5 |

Le message `Modèle PPO non trouvé` est normal à cette étape — il faut d'abord entraîner les agents (étapes suivantes).

---

## 4. Entraînement Q-Learning (Section 4.4.1)

### Entraînement rapide (test, ~1-2 minutes)

```bash
python -m memoire_master_rl_logistique.main --train-q-learning --episodes 1000
```

### Entraînement complet (pour le Chapitre 6, ~10-15 minutes)

```bash
python -m memoire_master_rl_logistique.main --train-q-learning
```

**Résultat attendu** :
- Q-table sauvegardée dans `models/q_learning/q_table.pkl`
- Récompenses d'entraînement dans `models/q_learning/training_rewards.csv`
- La récompense moyenne des 100 derniers épisodes devrait augmenter au fil de l'entraînement

---

## 5. Entraînement SARSA (Section 4.4.3)

### Entraînement rapide (test, ~1-2 minutes)

```bash
python -m memoire_master_rl_logistique.main --train-sarsa --episodes 1000
```

### Entraînement complet (pour le Chapitre 6, ~10-15 minutes)

```bash
python -m memoire_master_rl_logistique.main --train-sarsa
```

**Résultat attendu** :
- Table SARSA sauvegardée dans `models/sarsa/sarsa_table.pkl`
- Récompenses d'entraînement dans `models/sarsa/training_rewards.csv`

---

## 6. Entraînement DQN (Section 4.5.2)

### Entraînement rapide (test, ~1-2 minutes)

```bash
python -m memoire_master_rl_logistique.main --train-dqn --timesteps 5000
```

### Entraînement complet (pour le Chapitre 6, ~5-10 minutes)

```bash
python -m memoire_master_rl_logistique.main --train-dqn
```

**Résultat attendu** :
- Modèle DQN sauvegardé dans `models/dqn_mine/dqn_mine_agent.zip`
- Logs d'entraînement dans `models/dqn_mine/training_kpis.csv`
- Logs TensorBoard dans `models/dqn_mine/tb_logs/`

---

## 7. Entraînement PPO (Section 4.5.4)

### Entraînement rapide (test, ~30 secondes)

```bash
python -m memoire_master_rl_logistique.main --train-only --timesteps 5000
```

### Entraînement complet (pour le Chapitre 6, ~2-5 minutes)

```bash
python -m memoire_master_rl_logistique.main --train-only
```

**Résultat attendu** : Le modèle est sauvegardé dans `models/ppo_mine/ppo_mine_agent.zip` et les logs d'entraînement dans `models/ppo_mine/training_kpis.csv`.

---

## 8. Benchmark complet — 7 méthodes × 6 scénarios

```bash
python -m memoire_master_rl_logistique.experiments.run_benchmark
```

**Résultat attendu** : Fichier `data/results/benchmark_results.csv` contenant les KPIs pour 7 méthodes × 6 scénarios × 10 seeds.

Les 7 méthodes comparées :

| Catégorie | Méthode | Section du mémoire |
|-----------|---------|-------------------|
| Heuristique | Fixed | 4.7.1 |
| Heuristique | Nearest | 4.7.1 |
| Heuristique | QueueAware | 4.7.1 |
| RL classique | Q-Learning | 4.4.1 |
| RL classique | SARSA | 4.4.3 |
| Deep RL | DQN | 4.5.2 |
| Deep RL | PPO | 4.5.4 |

Les 6 scénarios testés :

| Scénario | Camions | Pelles | Dumps | Shift | Pannes |
|----------|---------|--------|-------|-------|--------|
| nominal | 12 | 3 | 2 | 8h | 2% |
| high_load | 18 | 3 | 2 | 8h | 2% |
| low_load | 6 | 3 | 2 | 8h | 2% |
| high_breakdown | 12 | 3 | 2 | 8h | 10% |
| single_shovel | 12 | 1 | 1 | 8h | 2% |
| short_shift | 12 | 3 | 2 | 4h | 2% |

Les KPIs calculés (Tableau 4.8) :

| KPI | Description |
|-----|-------------|
| Productivité (t/h) | Tonnage transporté par heure |
| Temps d'attente moyen (min) | Attente moyenne par camion |
| Consommation spécifique (L/t) | Litres de carburant par tonne |
| Coût moyen par cycle (L) | Consommation moyenne par cycle |
| Utilisation (%) | Taux d'utilisation des camions |
| Récompense cumulée | Performance RL globale |

---

## 9. Génération des figures (Chapitre 6)

```bash
python -m memoire_master_rl_logistique.experiments.stats_report
```

**Résultats attendus** dans `data/results/` :
- `summary_table.csv` — Tableau de synthèse (moyenne ± écart-type par méthode)
- `kpi_comparison_nominal.png` — Barplots comparatifs des 6 KPIs (7 méthodes)
- `learning_curve.png` — Courbe d'apprentissage PPO

---

## 10. Pipeline complet en une commande

```bash
python -m memoire_master_rl_logistique.main
```

Exécute dans l'ordre : simulation → entraînement PPO → benchmark.

### Options disponibles

```bash
python -m memoire_master_rl_logistique.main --check-env              # Valider l'environnement
python -m memoire_master_rl_logistique.main --sim-only               # Simulation seule
python -m memoire_master_rl_logistique.main --train-q-learning       # Entraîner Q-Learning
python -m memoire_master_rl_logistique.main --train-sarsa            # Entraîner SARSA
python -m memoire_master_rl_logistique.main --train-dqn              # Entraîner DQN
python -m memoire_master_rl_logistique.main --train-only             # Entraîner PPO
python -m memoire_master_rl_logistique.main --benchmark-only         # Benchmark seul
python -m memoire_master_rl_logistique.main --truck-count 18         # 18 camions
python -m memoire_master_rl_logistique.main --shovel-count 5         # 5 pelles
python -m memoire_master_rl_logistique.main --dump-count 3           # 3 dumps
python -m memoire_master_rl_logistique.main --timesteps 100000       # Plus de steps (DQN/PPO)
python -m memoire_master_rl_logistique.main --episodes 20000         # Plus d'épisodes (Q-Learning/SARSA)
```

---

## 11. Visualisation TensorBoard (optionnel)

```bash
tensorboard --logdir models/ppo_mine/tb_logs/
```

Pour DQN :

```bash
tensorboard --logdir models/dqn_mine/tb_logs/
```

Ouvrir `http://localhost:6006` dans le navigateur pour voir la courbe d'apprentissage en temps réel.

---

## 12. Ordre recommandé pour tout tester

| Étape | Commande | Durée |
|-------|----------|-------|
| 1 | `python -m memoire_master_rl_logistique.main --sim-only` | ~1 sec |
| 2 | `python -m memoire_master_rl_logistique.main --check-env` | ~1 sec |
| 3 | `python -m memoire_master_rl_logistique.rl.evaluate_agent` | ~10 sec |
| 4 | `python -m memoire_master_rl_logistique.main --train-q-learning --episodes 1000` | ~1-2 min |
| 5 | `python -m memoire_master_rl_logistique.main --train-sarsa --episodes 1000` | ~1-2 min |
| 6 | `python -m memoire_master_rl_logistique.main --train-dqn --timesteps 5000` | ~1-2 min |
| 7 | `python -m memoire_master_rl_logistique.main --train-only --timesteps 5000` | ~30 sec |
| 8 | `python -m memoire_master_rl_logistique.rl.evaluate_agent` | ~10 sec |
| 9 | `python -m memoire_master_rl_logistique.experiments.run_benchmark` | ~30-60 min |
| 10 | `python -m memoire_master_rl_logistique.experiments.stats_report` | ~5 sec |

---

## Dépannage

| Problème | Solution |
|----------|----------|
| `ModuleNotFoundError` | `pip install -e .` |
| `Modèle PPO non trouvé` | Lancer d'abord `python -m memoire_master_rl_logistique.main --train-only` |
| `Modèle DQN non trouvé` | Lancer d'abord `python -m memoire_master_rl_logistique.main --train-dqn` |
| PPO/DQN ne converge pas | Augmenter les timesteps : `--timesteps 200000` |
| Q-Learning/SARSA ne converge pas | Augmenter les épisodes : `--episodes 20000` |
| Figures non générées | Vérifier que le benchmark a été exécuté avant |
| `pkg_resources` manquant | `pip install setuptools==75.8.2` |
