# Guide de Test et d'Évaluation Complet

Guide étape par étape pour tester le système de dispatching minier et valider les résultats des chapitres 4, 5 et 6.

Toutes les commandes sont à lancer depuis la racine du projet : `c:\Devs\python\Master 2\memoire-master-rl-logistique`

---

## Prérequis

### 1. Installation des dépendances

```bash
pip install -e .
```

### 2. Vérification de l'installation

```bash
python -m memoire_master_rl_logistique.main --check-env
```

**Résultat attendu** :
```
============================================================
Validation de l'environnement Gymnasium
============================================================
check_env OK
  Observation : Box(0.0, 1.0, (138,), float32)
  Action      : Discrete(7)
  Camions=12, Pelles=3, Dumps=2
```

Si cette commande affiche `check_env OK`, l’environnement est correctement installé.

---

## ÉTAPE 1 : Validation de l'environnement Gymnasium

**Objectif** : Vérifier la conformité de l’environnement avec le modèle du mémoire.

```bash
python -m memoire_master_rl_logistique.main --check-env
```

**À vérifier** :
- Observation : `Box(0.0, 1.0, (138,), float32)`
- Action : `Discrete(7)` (3 pelles × 2 dumps + ATTENDRE)
- Interface Gymnasium standard : `reset()`, `step()`, `render()`

**Si succès** : poursuivre.
**Si erreur** : vérifier l’installation et les dépendances.

---

## ÉTAPE 2 : Évaluation des baselines heuristiques

**Objectif** : Évaluer les baselines et établir les points de référence du chapitre 6.

```bash
python -m memoire_master_rl_logistique.rl.evaluate_agent
```

**Résultats attendus** :
- **Fixed Assignment** : productivité ≈ 4 074 t/h, attente ≈ 28 min
- **Nearest Shovel** / **Shortest Path** : productivité ≈ 3 830 t/h, attente ≈ 75 min
- **FIFO** : productivité ≈ 3 318 t/h, attente ≈ 33 min

**Vérifie** :
- baselines exécutables
- calculs KPI corrects
- valeurs cohérentes avec les chapitres du mémoire

---

## ÉTAPE 3 : Entraînement Q-Learning (test rapide)

**Objectif** : contrôler le fonctionnement du Q-Learning sans attendre 30 000 épisodes.

```bash
python -m memoire_master_rl_logistique.main --train-q-learning --episodes 1000
```

**Durée** : 1 à 2 minutes

**Résultats attendus** :
- progression de la récompense
- fichier Q-table sauvegardé
- logs de récompense générés

---

## ÉTAPE 4 : Entraînement Q-Learning (complet)

**Objectif** : reproduire l’entraînement complet du chapitre 6.

```bash
python -m memoire_master_rl_logistique.main --train-q-learning
```

**Paramètres attendus** :
- n_episodes = 30 000
- alpha = 0.2
- gamma = 0.99
- n_bins = 8

**Résultats attendus** :
- Q-table dans `models/q_learning/q_table.pkl`
- récompenser finale ≈ 170
- ≈ 6 401 états effectifs visités

---

## ÉTAPE 5 : Entraînement SARSA (test rapide)

**Objectif** : valider la mise en œuvre de SARSA.

```bash
python -m memoire_master_rl_logistique.main --train-sarsa --episodes 1000
```

**Durée** : 1 à 2 minutes

**Résultats attendus** :
- progression de la récompense
- fichier SARSA sauvegardé
- logs de récompense générés

---

## ÉTAPE 6 : Entraînement SARSA (complet)

**Objectif** : reproduire l’entraînement complet SARSA.

```bash
python -m memoire_master_rl_logistique.main --train-sarsa
```

**Paramètres attendus** :
- n_episodes = 30 000
- alpha = 0.2
- gamma = 0.99
- n_bins = 8

**Résultats attendus** :
- table SARSA dans `models/sarsa/sarsa_table.pkl`
- récompense finale ≈ 133

---

## ÉTAPE 7 : Entraînement DQN (test rapide)

**Objectif** : vérifier DQN avec un budget réduit.

```bash
python -m memoire_master_rl_logistique.main --train-dqn --timesteps 1000000
```

**Durée** : 1 à 2 minutes

**Résultat attendu** :
- modèle sauvegardé
- logs de formation présents

---

## ÉTAPE 8 : Entraînement DQN (complet)

**Objectif** : reproduire le DQN du chapitre 6.

```bash
python -m memoire_master_rl_logistique.main --train-dqn
```

**Paramètres attendus** :
- total_timesteps = 2 000 000
- batch_size = 64
- learning_rate = 3e-4

**Résultats attendus** :
- modèle DQN sauvegardé
- logs d’entraînement disponibles

---

## ÉTAPE 9 : Entraînement PPO (test rapide)

**Objectif** : vérifier que PPO fonctionne avec un budget réduit.

```bash
python -m memoire_master_rl_logistique.main --train-ppo --timesteps 1000000
```

**Résultat attendu** :
- modèle PPO sauvegardé
- logs TensorBoard générés

---

## ÉTAPE 10 : Entraînement PPO (complet)

**Objectif** : reproduire l’agent PPO du chapitre 6.

```bash
python -m memoire_master_rl_logistique.main --train-ppo
```

**Paramètres attendus** :
- total_timesteps = 2 000 000
- learning_rate = 3e-4
- n_steps = 1024

**Résultats attendus** :
- modèle PPO sauvegardé dans `models/ppo_mine`
- reward variance élevée en début d’entraînement puis stabilisation

---

## Notes de validation

- L’environnement doit être stable et reproductible.
- Les scénarios doivent être évalués avec 10 seeds fixes.
- Les KPI doivent être présentés en moyenne ± écart-type.
- Le scénario `high_breakdown` est crucial pour valider la robustesse de PPO.

*Ce guide doit permettre de vérifier que les résultats du mémoire sont reproductibles dans le code.*
**Si erreur** : Vérifiez que Stable-Baselines3 est installé.

**Résultats attendus** (10 épisodes, scénario nominal) :

| Méthode | Productivité (t/h) | Attente moy. (min) | Utilisation (%) | Conso spécifique (L/t) |
|---------|--------------------:|--------------------:|----------------:|-----------------------:|
| Fixed | 4074 ± 36 | 28 ± 4 | 96.8 | 0.043 ± 0.000 |
| PPO | 3335 ± 38 | 31 ± 5 | 97.1 | 0.053 ± 0.001 |
| DQN | 3311 ± 72 | 33 ± 10 | 96.4 | 0.052 ± 0.001 |
| FIFO | 3318 ± 42 | 33 ± 4 | 95.8 | 0.053 ± 0.000 |
| Q-Learning | 3273 ± 89 | 37 ± 19 | 95.4 | 0.053 ± 0.001 |
| SARSA | 3222 ± 97 | 53 ± 14 | 93.2 | 0.054 ± 0.001 |
| Random | 3148 ± 45 | 51 ± 4 | 92.8 | 0.055 ± 0.001 |
| Nearest | 3830 ± 57 | 75 ± 7 | 87.8 | 0.043 ± 0.000 |
| ShortestPath | 3830 ± 57 | 75 ± 7 | 87.8 | 0.043 ± 0.000 |

**Ce qui est vérifié** :
- Fixed Assignment domine (Match Factor optimal)
- PPO a les meilleurs temps d'attente parmi les agents RL
- Q-Learning surpasse SARSA (off-policy vs on-policy)
- Nearest = ShortestPath (bug connu, voir Chapitre 6)

**Si succès** : Passez à l'étape 12.
**Si erreur** : Vérifiez que tous les modèles sont sauvegardés.

---

## ÉTAPE 12 : Benchmark complet — 8 méthodes × 3 scénarios

**Objectif** : Exécuter le benchmark complet pour le Chapitre 6 (Section 5.4).

```bash
python -m memoire_master_rl_logistique.main --benchmark-only
```

**Durée** : ~10-20 heures (DQN/PPO = 2-4h par scénario)

**Résultat attendu** :
- Fichier `data/results/benchmark_results.csv` contenant les KPIs pour 8 méthodes × 3 scénarios × 10 seeds = 240 lignes

Les 8 méthodes comparées :

| Catégorie | Méthode | Section du mémoire |
|-----------|---------|-------------------|
| Heuristique | FIFO | 4.7.1 |
| Heuristique | Fixed | 4.7.1 |
| Heuristique | Nearest | 4.7.1 |
| Heuristique | ShortestPath | 4.7.1 |
| RL classique | Q-Learning | 4.4.1 |
| RL classique | SARSA | 4.4.3 |
| Deep RL | DQN | 4.5.2 |
| Deep RL | PPO | 4.5.4 |

Les 3 scénarios testés :

| Scénario | Camions | Pelles | Dumps | Shift | Pannes | Description |
|----------|---------|--------|-------|-------|--------|-------------|
| nominal | 12 | 3 | 2 | 8h | 2% | Conditions nominales |
| high_load | 18 | 3 | 2 | 8h | 2% | Charge élevée (surcharge) |
| high_breakdown | 12 | 3 | 2 | 8h | 10% | Taux de pannes élevé (robustesse) |

Les KPIs calculés (Tableau 4.8) :

| KPI | Description | Formule |
|-----|-------------|--------|
| Productivité (t/h) | Tonnage transporté par heure | total_tonnage / episode_hours |
| Temps d'attente moyen (min) | Attente moyenne par camion | total_wait / truck_count |
| Consommation spécifique (L/t) | Litres de carburant par tonne | total_fuel / total_tonnage |
| Coût moyen par cycle (L) | Consommation moyenne par cycle | total_fuel / total_cycles |
| Utilisation (%) | Taux d'utilisation des camions | (total_active / max_available) × 100 |
| Récompense cumulée | Performance RL globale | Σ r_t |

**Si succès** : Passez à l'étape 13.
**Si erreur** : Vérifiez que tous les modèles sont entraînés et sauvegardés.

---

## ÉTAPE 13 : Analyse des résultats

**Objectif** : Analyser les résultats du benchmark pour le Chapitre 6.

### Ouvrir le fichier CSV

```bash
# Sur Windows
notepad data/results/benchmark_results.csv

# Ou utiliser Excel / autre tableur
```

### Points à vérifier

1. **Scénario nominal** : Fixed Assignment domine (4074 t/h) grâce au Match Factor optimal
2. **Robustesse** : PPO obtient les meilleurs temps d'attente en high_breakdown (47.7 min vs 52.9 min pour Fixed)
3. **Scalabilité** : En high_load, Q-Learning = DQN ≈ FIFO (4772 t/h), Fixed domine (5875 t/h)
4. **Consistance** : Écarts-types acceptables, PPO stable (explained_variance = 0.999)

### Calculs statistiques (optionnel)

Vous pouvez utiliser Python pour calculer les moyennes et écarts-types :

```python
import pandas as pd

df = pd.read_csv('data/results/benchmark_results.csv')

# Moyennes par méthode sur scénario nominal
nominal = df[df['scenario'] == 'nominal']
summary = nominal.groupby('policy').agg({
    'productivity_tph': ['mean', 'std'],
    'avg_wait_min_per_truck': ['mean', 'std'],
    'utilization_pct': ['mean', 'std'],
    'specific_fuel_l_per_ton': ['mean', 'std']
})
print(summary)
```

**Si succès** : Passez à l'étape 14.
**Si erreur** : Vérifiez que le fichier CSV existe et est bien formé.

---

## ÉTAPE 14 : Visualisation TensorBoard (optionnel)

**Objectif** : Visualiser les courbes d'apprentissage PPO et DQN.

### Pour PPO

```bash
tensorboard --logdir data/results/ppo_nominal/tb_logs/
```

### Pour DQN

```bash
tensorboard --logdir data/results/dqn_nominal/tb_logs/
```

### Pour voir tous les scénarios PPO

```bash
tensorboard --logdir data/results/ppo_nominal/tb_logs/,data/results/ppo_high_load/tb_logs/,data/results/ppo_high_breakdown/tb_logs/
```

Ouvrir `http://localhost:6006` dans le navigateur pour voir :
- Courbe d'apprentissage (reward)
- KPIs au fil de l'entraînement
- Évolution des politiques

**Si succès** : Passez à l'étape 15.
**Si erreur** : Vérifiez que TensorBoard est installé (`pip install tensorboard`).

---

## ÉTAPE 15 : Pipeline complet en une commande

**Objectif** : Exécuter tout le pipeline d'un coup (validation + entraînement + benchmark).

```bash
python -m memoire_master_rl_logistique.main
```

**Durée** : ~30-60 minutes

**Ce qui est exécuté** :
1. Validation de l'environnement Gymnasium
2. Entraînement PPO (2_000_000 timesteps)
3. Benchmark complet (8 méthodes × 3 scénarios × 10 seeds)

**Si succès** : Votre système est entièrement testé et fonctionnel.
**Si erreur** : Exécutez les étapes individuellement pour identifier le problème.

---

## Options CLI disponibles

```bash
python -m memoire_master_rl_logistique.main --check-env              # Valider l'environnement
python -m memoire_master_rl_logistique.main --train-q-learning       # Entraîner Q-Learning
python -m memoire_master_rl_logistique.main --train-sarsa            # Entraîner SARSA
python -m memoire_master_rl_logistique.main --train-dqn              # Entraîner DQN
python -m memoire_master_rl_logistique.main --train-only             # Entraîner PPO
python -m memoire_master_rl_logistique.main --benchmark-only         # Benchmark seul
python -m memoire_master_rl_logistique.main --truck-count 18         # 18 camions
python -m memoire_master_rl_logistique.main --shovel-count 5         # 5 pelles
python -m memoire_master_rl_logistique.main --dump-count 3           # 3 dumps
python -m memoire_master_rl_logistique.main --timesteps 2000000      # Plus de steps (DQN/PPO)
python -m memoire_master_rl_logistique.main --episodes 30000        # Plus d'épisodes (Q-Learning/SARSA)
python -m memoire_master_rl_logistique.main --visualize-graph        # Générer figure du graphe
```

---

## Dépannage

| Problème | Solution |
|----------|----------|
| `ModuleNotFoundError: No module named 'memoire_master_rl_logistique.baselines.queue_aware_policy'` | Déjà corrigé dans baselines/__init__.py |
| `ModuleNotFoundError` | `pip install -e .` |
| `Modèle PPO non trouvé` | Lancer d'abord `python -m memoire_master_rl_logistique.main --train-only` |
| `Modèle DQN non trouvé` | Lancer d'abord `python -m memoire_master_rl_logistique.main --train-dqn` |
| PPO/DQN ne converge pas | Augmenter les timesteps : `--timesteps 200000` |
| Q-Learning/SARSA ne converge pas | Augmenter les épisodes : `--episodes 20000` |
| Benchmark très lent | Réduire le nombre de scénarios ou de seeds |
| `pkg_resources` manquant | `pip install setuptools==75.8.2` |
| Problème GPU PyTorch | Installer CUDA ou utiliser CPU (plus lent) |

---

## Résumé du test complet

| Étape | Commande | Durée | Objectif |
|-------|----------|-------|----------|
| 1 | `--check-env` | ~1 sec | Validation Gymnasium |
| 2 | `evaluate_agent` | ~10 sec | Baselines heuristiques |
| 3 | `--train-q-learning --episodes 1000` | ~1-2 min | Test Q-Learning |
| 4 | `--train-q-learning` | ~15-20 min | Entraînement Q-Learning complet (30k épisodes) |
| 5 | `--train-sarsa --episodes 1000` | ~1-2 min | Test SARSA |
| 6 | `--train-sarsa` | ~15-20 min | Entraînement SARSA complet (30k épisodes) |
| 7 | `--train-dqn --timesteps 1_000_000` | ~1-2 min | Test DQN |
| 8 | `--train-dqn` | ~60-90 min | Entraînement DQN complet (2M steps) |
| 9 | `--train-only --timesteps 1_000_000` | ~30 sec | Test PPO |
| 10 | `--train-only` | ~60-90 min | Entraînement PPO complet (2M steps) |
| 11 | `evaluate_agent` | ~10 sec | Évaluation complète |
| 12 | `--benchmark-only` | ~10-20 h | Benchmark complet (3 scénarios) |
| 13 | Analyse CSV | ~5 min | Analyse résultats |
| 14 | TensorBoard | ~5 min | Visualisation |
| 15 | Pipeline complet | ~10-20 h | Test final |

**Durée totale estimée** : ~10-20 heures pour un test complet (à cause de DQN/PPO)

---

## Prochaines étapes après test réussi

1. **Compléter le Chapitre 5** (Implémentation et expérimentation) du mémoire — figures manquantes à insérer
2. **Compléter le Chapitre 6** (Résultats et analyse) du mémoire — déjà rédigé avec résultats du benchmark
3. **Compléter la Conclusion générale** — actuellement vide
4. **Insérer les 5 figures** dans les chapitres 5 et 6
5. **Pratiquer la soutenance** avec `guide-soutenance.md` et `faq-jury.md`
