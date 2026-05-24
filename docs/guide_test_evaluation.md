# Guide de Test et d'Évaluation — Commandes CLI

Toutes les commandes sont à taper depuis la racine du projet.

---

## Prérequis

```bash
pip install -e .
```

Vérifier l'installation :

```bash
python -c "import gymnasium, stable_baselines3, numpy, pandas, matplotlib; print('OK')"
```

**Résultat attendu** : `OK`

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
python -c "from gymnasium.utils.env_checker import check_env; from memoire_master_rl_logistique.env.mine_env import MineEnv; check_env(MineEnv(truck_count=12, shovel_count=3, dump_count=2), skip_render_check=True); print('check_env OK')"
```

**Résultat attendu** : `check_env OK` (aucune erreur)

---

## 3. Évaluation des baselines

```bash
python -m memoire_master_rl_logistique.rl.evaluate_agent
```

**Résultats attendus** (10 épisodes, 12 camions, 3 pelles, 2 dumps) :

| Baseline | Productivité (t/h) | Attente moy. (min) | Utilisation (%) |
|----------|--------------------:|--------------------:|----------------:|
| Fixed | ~4 074 ± 41 | ~27.9 ± 3.7 | ~96.5 ± 0.8 |
| Nearest | ~3 831 ± 55 | ~72.6 ± 7.4 | ~87.8 ± 1.5 |
| QueueAware | ~3 831 ± 55 | ~72.6 ± 7.4 | ~87.8 ± 1.5 |

Le message `Modèle PPO non trouvé` est normal à cette étape — il faut d'abord entraîner l'agent (étape suivante).

---

## 4. Entraînement de l'agent PPO

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

## 5. Évaluation PPO vs Baselines (après entraînement)

```bash
python -m memoire_master_rl_logistique.rl.evaluate_agent
```

**Résultat attendu** : Les 3 baselines + PPO sont évalués. PPO doit montrer une productivité compétitive ou supérieure aux baselines.

---

## 6. Benchmark complet sur tous les scénarios

```bash
python -m memoire_master_rl_logistique.experiments.run_benchmark
```

**Résultat attendu** : Fichier `data/results/benchmark_results.csv` contenant les KPIs pour 4 méthodes × 6 scénarios × 10 seeds.

Les 6 scénarios testés :

| Scénario | Camions | Pelles | Dumps | Shift | Pannes |
|----------|---------|--------|-------|-------|--------|
| nominal | 12 | 3 | 2 | 8h | 2% |
| high_load | 18 | 3 | 2 | 8h | 2% |
| low_load | 6 | 3 | 2 | 8h | 2% |
| high_breakdown | 12 | 3 | 2 | 8h | 10% |
| single_shovel | 12 | 1 | 1 | 8h | 2% |
| short_shift | 12 | 3 | 2 | 4h | 2% |

---

## 7. Génération des figures (Chapitre 6)

```bash
python -m memoire_master_rl_logistique.experiments.stats_report
```

**Résultats attendus** dans `data/results/` :
- `summary_table.csv` — Tableau de synthèse (moyenne ± écart-type par méthode)
- `kpi_comparison_nominal.png` — Barplots comparatifs des KPIs
- `learning_curve.png` — Courbe d'apprentissage PPO

---

## 8. Pipeline complet en une commande

```bash
python -m memoire_master_rl_logistique.main
```

Exécute dans l'ordre : simulation → entraînement PPO → benchmark.

### Options disponibles

```bash
python -m memoire_master_rl_logistique.main --sim-only              # Simulation seule
python -m memoire_master_rl_logistique.main --train-only             # Entraînement seul
python -m memoire_master_rl_logistique.main --benchmark-only         # Benchmark seul
python -m memoire_master_rl_logistique.main --truck-count 18         # 18 camions
python -m memoire_master_rl_logistique.main --shovel-count 5         # 5 pelles
python -m memoire_master_rl_logistique.main --dump-count 3           # 3 dumps
python -m memoire_master_rl_logistique.main --timesteps 100000       # Plus d'entraînement
```

---

## 9. Visualisation TensorBoard (optionnel)

```bash
tensorboard --logdir models/ppo_mine/tb_logs/
```

Ouvrir `http://localhost:6006` dans le navigateur pour voir la courbe d'apprentissage en temps réel.

---

## 10. Ordre recommandé pour tout tester

| Étape | Commande | Durée |
|-------|----------|-------|
| 1 | `python -m memoire_master_rl_logistique.main --sim-only` | ~1 sec |
| 2 | `python -c "from gymnasium.utils.env_checker import check_env; from memoire_master_rl_logistique.env.mine_env import MineEnv; check_env(MineEnv(), skip_render_check=True); print('OK')"` | ~1 sec |
| 3 | `python -m memoire_master_rl_logistique.rl.evaluate_agent` | ~10 sec |
| 4 | `python -m memoire_master_rl_logistique.rl.train_ppo` | ~2-5 min |
| 5 | `python -m memoire_master_rl_logistique.rl.evaluate_agent` | ~10 sec |
| 6 | `python -m memoire_master_rl_logistique.experiments.run_benchmark` | ~15-30 min |
| 7 | `python -m memoire_master_rl_logistique.experiments.stats_report` | ~5 sec |

---

## Dépannage

| Problème | Solution |
|----------|----------|
| `ModuleNotFoundError` | `pip install -e .` |
| `Modèle PPO non trouvé` | Lancer d'abord `python -m memoire_master_rl_logistique.rl.train_ppo` |
| PPO ne converge pas | Augmenter les timesteps : `--timesteps 200000` |
| Figures non générées | Vérifier que le benchmark a été exécuté avant |
