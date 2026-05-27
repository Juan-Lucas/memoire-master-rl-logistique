# Guide de Test et d'Évaluation Complet

Guide étape par étape pour tester totalement votre système de dispatching minier par RL.

Toutes les commandes sont à taper depuis la racine du projet : `c:\Devs\python\Master 2\memoire-master-rl-logistique`

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
  Observation : Box(0.0, 1.0, (54,), float32)
  Action      : Discrete(7)
  Camions=12, Pelles=3, Dumps=2
```

Si cette commande affiche `check_env OK`, tout est installé correctement.

---

## ÉTAPE 1 : Validation de l'environnement Gymnasium

**Objectif** : Vérifier que l'environnement Gymnasium est conforme aux spécifications du mémoire (Section 4.3).

```bash
python -m memoire_master_rl_logistique.main --check-env
```

**Ce qui est vérifié** :
- Observation space : Box(0.0, 1.0, (54,), float32)
- Action space : Discrete(7) pour 3 pelles × 2 dumps + 1 action ATTENDRE
- Interface Gymnasium standard (reset, step, render)

**Si succès** : Passez à l'étape 2.
**Si erreur** : Vérifiez l'installation des dépendances avec `pip install -e .`

---

## ÉTAPE 2 : Évaluation des baselines heuristiques

**Objectif** : Évaluer les 4 baselines heuristiques (Section 4.7.1) pour établir un point de référence.

```bash
python -m memoire_master_rl_logistique.rl.evaluate_agent
```

**Résultats attendus** (10 épisodes, 12 camions, 3 pelles, 2 dumps) :

| Baseline | Productivité (t/h) | Attente moy. (min) | Utilisation (%) |
|----------|--------------------:|--------------------:|----------------:|
| FIFO | ~3 800-4 100 ± 50 | ~25-35 ± 5 | ~90-95 ± 2 |
| Fixed | ~4 000-4 200 ± 50 | ~20-30 ± 5 | ~92-97 ± 2 |
| Nearest | ~3 700-3 900 ± 60 | ~60-80 ± 8 | ~85-90 ± 2 |
| ShortestPath | ~3 800-4 000 ± 60 | ~50-70 ± 8 | ~87-92 ± 2 |

**Ce qui est vérifié** :
- Les 4 baselines s'exécutent sans erreur
- Les KPIs sont calculés correctement
- Les résultats sont cohérents avec la littérature

**Si succès** : Passez à l'étape 3.
**Si erreur** : Vérifiez que tous les fichiers baselines existent et sont correctement importés.

---

## ÉTAPE 3 : Entraînement Q-Learning (test rapide)

**Objectif** : Tester l'entraînement Q-Learning tabulaire (Section 4.4.1) avec peu d'épisodes.

```bash
python -m memoire_master_rl_logistique.main --train-q-learning --episodes 1000
```

**Durée** : ~1-2 minutes

**Résultat attendu** :
```
Début de l'entraînement Q-Learning (1000 épisodes)...
  Épisode 1/1000 — récompense moy. (100 derniers) = X.XX, ε = 0.XXXX, |Q-table| = XXX
  Épisode 1000/1000 — récompense moy. (100 derniers) = X.XX, ε = 0.XXXX, |Q-table| = XXX
Q-table sauvegardée : models/q_learning/q_table.pkl
Récompenses d'entraînement : models/q_learning/training_rewards.csv
```

**Ce qui est vérifié** :
- La Q-table se remplit progressivement
- La récompense moyenne augmente (convergence)
- Les fichiers sont sauvegardés correctement

**Si succès** : Passez à l'étape 4.
**Si erreur** : Vérifiez que numpy est installé et que la discrétisation fonctionne.

---

## ÉTAPE 4 : Entraînement Q-Learning (complet)

**Objectif** : Entraînement complet de Q-Learning pour le Chapitre 6.

```bash
python -m memoire_master_rl_logistique.main --train-q-learning
```

**Durée** : ~10-15 minutes

**Résultat attendu** :
- Q-table sauvegardée dans `models/q_learning/q_table.pkl`
- Récompenses d'entraînement dans `models/q_learning/training_rewards.csv`
- La récompense moyenne des 100 derniers épisodes devrait augmenter significativement

**Si succès** : Passez à l'étape 5.
**Si erreur** : Vérifiez que vous avez suffisamment de mémoire RAM.

---

## ÉTAPE 5 : Entraînement SARSA (test rapide)

**Objectif** : Tester l'entraînement SARSA tabulaire (Section 4.4.3) avec peu d'épisodes.

```bash
python -m memoire_master_rl_logistique.main --train-sarsa --episodes 1000
```

**Durée** : ~1-2 minutes

**Résultat attendu** :
```
Début de l'entraînement SARSA (1000 épisodes)...
  Épisode 1/1000 — récompense moy. (100 derniers) = X.XX, ε = 0.XXXX, |Q-table| = XXX
  Épisode 1000/1000 — récompense moy. (100 derniers) = X.XX, ε = 0.XXXX, |Q-table| = XXX
SARSA table sauvegardée : models/sarsa/sarsa_table.pkl
Récompenses d'entraînement : models/sarsa/training_rewards.csv
```

**Ce qui est vérifié** :
- La table SARSA se remplit progressivement
- La récompense moyenne augmente (convergence)
- Les fichiers sont sauvegardés correctement

**Si succès** : Passez à l'étape 6.
**Si erreur** : Vérifiez que les imports SARSA sont corrects.

---

## ÉTAPE 6 : Entraînement SARSA (complet)

**Objectif** : Entraînement complet de SARSA pour le Chapitre 6.

```bash
python -m memoire_master_rl_logistique.main --train-sarsa
```

**Durée** : ~10-15 minutes

**Résultat attendu** :
- Table SARSA sauvegardée dans `models/sarsa/sarsa_table.pkl`
- Récompenses d'entraînement dans `models/sarsa/training_rewards.csv`

**Si succès** : Passez à l'étape 7.
**Si erreur** : Vérifiez que vous avez suffisamment de mémoire RAM.

---

## ÉTAPE 7 : Entraînement DQN (test rapide)

**Objectif** : Tester l'entraînement DQN (Section 4.5.2) avec peu de timesteps.

```bash
python -m memoire_master_rl_logistique.main --train-dqn --timesteps 1_000_000
```

**Durée** : ~1-2 minutes

**Résultat attendu** :
```
Début de l'entraînement DQN (5000 steps)...
...
Modèle DQN sauvegardé : models/dqn_mine/dqn_mine_agent.zip
```

**Ce qui est vérifié** :
- Le réseau de neurones DQN s'entraîne
- Les logs TensorBoard sont générés
- Le modèle est sauvegardé correctement

**Si succès** : Passez à l'étape 8.
**Si erreur** : Vérifiez que Stable-Baselines3 et PyTorch sont installés.

---

## ÉTAPE 8 : Entraînement DQN (complet)

**Objectif** : Entraînement complet de DQN pour le Chapitre 6.

```bash
python -m memoire_master_rl_logistique.main --train-dqn
```

**Durée** : ~5-10 minutes

**Résultat attendu** :
- Modèle DQN sauvegardé dans `models/dqn_mine/dqn_mine_agent.zip`
- Logs d'entraînement dans `models/dqn_mine/training_kpis.csv`
- Logs TensorBoard dans `models/dqn_mine/tb_logs/`

**Si succès** : Passez à l'étape 9.
**Si erreur** : Vérifiez que PyTorch a accès au GPU (optionnel mais recommandé).

---

## ÉTAPE 9 : Entraînement PPO (test rapide)

**Objectif** : Tester l'entraînement PPO (Section 4.5.4) avec peu de timesteps.

```bash
python -m memoire_master_rl_logistique.main --train-only --timesteps 1_000_000
```

**Durée** : ~30 secondes

**Résultat attendu** :
```
Début de l'entraînement PPO (5000 steps)...
...
Modèle sauvegardé : models/ppo_mine/ppo_mine_agent.zip
```

**Ce qui est vérifié** :
- Le réseau de neurones PPO s'entraîne
- Les callbacks fonctionnent
- Le modèle est sauvegardé correctement

**Si succès** : Passez à l'étape 10.
**Si erreur** : Vérifiez que Stable-Baselines3 est installé.

---

## ÉTAPE 10 : Entraînement PPO (complet)

**Objectif** : Entraînement complet de PPO pour le Chapitre 6.

```bash
python -m memoire_master_rl_logistique.main --train-only
```

**Durée** : ~2-5 minutes

**Résultat attendu** :
- Modèle PPO sauvegardé dans `models/ppo_mine/ppo_mine_agent.zip`
- Logs d'entraînement dans `models/ppo_mine/training_kpis.csv`
- Logs TensorBoard dans `models/ppo_mine/tb_logs/`

**Si succès** : Passez à l'étape 11.
**Si erreur** : Vérifiez que PyTorch a accès au GPU (optionnel mais recommandé).

---

## ÉTAPE 11 : Réévaluation complète avec agents RL entraînés

**Objectif** : Évaluer tous les agents (baselines + RL) pour comparaison.

```bash
python -m memoire_master_rl_logistique.rl.evaluate_agent
```

**Résultats attendus** (10 épisodes, 12 camions, 3 pelles, 2 dumps) :

| Méthode | Productivité (t/h) | Attente moy. (min) | Utilisation (%) | Conso spécifique (L/t) |
|---------|--------------------:|--------------------:|----------------:|-----------------------:|
| FIFO | ~3 800-4 100 ± 50 | ~25-35 ± 5 | ~90-95 ± 2 | ~0.04-0.05 ± 0.005 |
| Fixed | ~4 000-4 200 ± 50 | ~20-30 ± 5 | ~92-97 ± 2 | ~0.04-0.05 ± 0.005 |
| Nearest | ~3 700-3 900 ± 60 | ~60-80 ± 8 | ~85-90 ± 2 | ~0.05-0.06 ± 0.005 |
| ShortestPath | ~3 800-4 000 ± 60 | ~50-70 ± 8 | ~87-92 ± 2 | ~0.05-0.06 ± 0.005 |
| Q-Learning | ~4 100-4 300 ± 60 | ~15-25 ± 5 | ~94-98 ± 2 | ~0.04-0.05 ± 0.005 |
| SARSA | ~4 050-4 250 ± 60 | ~18-28 ± 5 | ~93-97 ± 2 | ~0.04-0.05 ± 0.005 |
| DQN | ~4 150-4 350 ± 60 | ~12-22 ± 5 | ~95-99 ± 2 | ~0.04-0.05 ± 0.005 |
| PPO | ~4 200-4 400 ± 60 | ~10-20 ± 5 | ~96-99 ± 2 | ~0.04-0.05 ± 0.005 |

**Ce qui est vérifié** :
- Les agents RL surpassent les baselines
- Les KPIs sont cohérents avec l'objectif de maximisation de productivité
- Les écarts-types sont acceptables (< 10%)

**Si succès** : Passez à l'étape 12.
**Si erreur** : Vérifiez que tous les modèles sont sauvegardés.

---

## ÉTAPE 12 : Benchmark complet — 8 méthodes × 6 scénarios

**Objectif** : Exécuter le benchmark complet pour le Chapitre 6 (Section 5.4).

```bash
python -m memoire_master_rl_logistique.main --benchmark-only
```

**Durée** : ~30-60 minutes

**Résultat attendu** :
- Fichier `data/results/benchmark_results.csv` contenant les KPIs pour 8 méthodes × 6 scénarios × 10 seeds = 480 lignes

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

Les 6 scénarios testés :

| Scénario | Camions | Pelles | Dumps | Shift | Pannes | Description |
|----------|---------|--------|-------|-------|--------|-------------|
| nominal | 12 | 3 | 2 | 8h | 2% | Conditions nominales |
| high_load | 18 | 3 | 2 | 8h | 2% | Charge élevée (surcharge) |
| low_load | 6 | 3 | 2 | 8h | 2% | Charge faible |
| high_breakdown | 12 | 3 | 2 | 8h | 10% | Taux de pannes élevé (robustesse) |
| single_shovel | 12 | 1 | 1 | 8h | 2% | Une seule pelle (goulot) |
| short_shift | 12 | 3 | 2 | 4h | 2% | Shift court (4h) |

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

1. **Scénario nominal** : PPO et DQN devraient surpasser les baselines
2. **Robustesse** : PPO et DQN devraient performer bien sur high_breakdown
3. **Scalabilité** : Performance sur high_load et low_load
4. **Consistance** : Écarts-types faibles (< 10% de la moyenne)

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
tensorboard --logdir models/ppo_mine/tb_logs/
```

### Pour DQN

```bash
tensorboard --logdir models/dqn_mine/tb_logs/
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
2. Entraînement PPO (50 000 timesteps)
3. Benchmark complet (8 méthodes × 6 scénarios × 10 seeds)

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
python -m memoire_master_rl_logistique.main --timesteps 100000       # Plus de steps (DQN/PPO)
python -m memoire_master_rl_logistique.main --episodes 20000         # Plus d'épisodes (Q-Learning/SARSA)
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
| 4 | `--train-q-learning` | ~10-15 min | Entraînement Q-Learning complet |
| 5 | `--train-sarsa --episodes 1000` | ~1-2 min | Test SARSA |
| 6 | `--train-sarsa` | ~10-15 min | Entraînement SARSA complet |
| 7 | `--train-dqn --timesteps 5000` | ~1-2 min | Test DQN |
| 8 | `--train-dqn` | ~5-10 min | Entraînement DQN complet |
| 9 | `--train-only --timesteps 5000` | ~30 sec | Test PPO |
| 10 | `--train-only` | ~2-5 min | Entraînement PPO complet |
| 11 | `evaluate_agent` | ~10 sec | Évaluation complète |
| 12 | `--benchmark-only` | ~30-60 min | Benchmark complet |
| 13 | Analyse CSV | ~5 min | Analyse résultats |
| 14 | TensorBoard | ~5 min | Visualisation |
| 15 | Pipeline complet | ~30-60 min | Test final |

**Durée totale estimée** : ~1-2 heures pour un test complet

---

## Prochaines étapes après test réussi

1. **Compléter le Chapitre 5** (Implémentation et expérimentation) du mémoire
2. **Compléter le Chapitre 6** (Résultats et analyse) du mémoire avec les résultats du benchmark
3. **Créer les figures** décrites dans `diagrammes-systeme.md`
4. **Pratiquer la soutenance** avec `guide-soutenance.md` et `faq-jury.md`
