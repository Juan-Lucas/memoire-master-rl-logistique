# Guide de Test et d'Évaluation du Système de Dispatching Minier

Ce guide permet de tester chaque composante du système implémenté et d'évaluer les résultats pour le Chapitre 6 du mémoire.

## Prérequis

```bash
# Depuis la racine du projet
pip install -e .
```

Vérifier que tout est installé :

```bash
python -c "import gymnasium, stable_baselines3, numpy, pandas, matplotlib; print('OK')"
```

---

## 1. Test du Moteur de Simulation (DES)

### 1.1 Simulation nominale (12 camions, 3 pelles, 2 dumps)

```python
from memoire_master_rl_logistique.simulation.events import build_simulation

sim = build_simulation(seed=42, truck_count=12, shovel_count=3, dump_count=2)
result = sim.run_episode(episode_minutes=480.0)

print("=== KPIs globaux ===")
for k, v in result.kpis.items():
    print(f"  {k}: {v:.2f}")

print("\n=== Détails par camion ===")
for t in result.per_truck:
    print(f"  {t['truck_id']}: {int(t['cycles_completed'])} cycles, "
          f"{t['total_tonnage_t']:.0f}t, wait={t['total_wait_min']:.1f}min")
```

**Résultats attendus** (seed=42) :
- Productivité : ~3 290 t/h
- Tonnage total : ~26 320 t sur 8h
- Attente moyenne/camion : ~54 min
- Utilisation : ~89.5 %
- Consommation spécifique : ~0.05 L/t

### 1.2 Test de reproductibilité

Vérifier que la même seed donne les mêmes résultats :

```python
from memoire_master_rl_logistique.simulation.events import build_simulation

results = []
for _ in range(3):
    sim = build_simulation(seed=42, truck_count=12, shovel_count=3, dump_count=2)
    r = sim.run_episode(episode_minutes=480.0)
    results.append(r.kpis["total_tonnage_t"])

assert all(r == results[0] for r in results), "Reproductibilité échouée !"
print(f"Reproductibilité OK : {results[0]:.0f}t (identique sur 3 runs)")
```

### 1.3 Test avec différentes configurations

```python
from memoire_master_rl_logistique.simulation.events import build_simulation

configs = [
    {"truck_count": 6,  "shovel_count": 2, "dump_count": 1},
    {"truck_count": 12, "shovel_count": 3, "dump_count": 2},
    {"truck_count": 18, "shovel_count": 5, "dump_count": 3},
]

for cfg in configs:
    sim = build_simulation(seed=42, **cfg)
    r = sim.run_episode()
    print(f"{cfg['truck_count']}T/{cfg['shovel_count']}P/{cfg['dump_count']}D → "
          f"prod={r.kpis['productivity_tph']:.0f} t/h, "
          f"wait={r.kpis['avg_wait_min_per_truck']:.1f} min/camion")
```

**Ce qu'il faut vérifier** :
- Plus de camions → plus de tonnage, mais aussi plus d'attente
- Plus de pelles → moins d'attente par camion
- Les résultats doivent être cohérents avec la littérature (Kangwa 2021, Mohtasham 2023)

---

## 2. Test de l'Environnement Gymnasium (MDP)

### 2.1 Validation avec check_env

```python
from gymnasium.utils.env_checker import check_env
from memoire_master_rl_logistique.env.mine_env import MineEnv

env = MineEnv(truck_count=12, shovel_count=3, dump_count=2)
check_env(env, skip_render_check=True)
print("check_env OK — l'environnement est conforme à l'API Gymnasium")
```

### 2.2 Inspection de l'observation et de l'espace d'action

```python
from memoire_master_rl_logistique.env.mine_env import MineEnv

env = MineEnv(truck_count=12, shovel_count=3, dump_count=2)
obs, info = env.reset(seed=42)

print(f"Dimension de l'observation : {obs.shape[0]}")
print(f"  = {env.shovel_count} pelles + {env.dump_count} dumps "
      f"+ {env.truck_count}×4 features camions + 1 temps")
print(f"  = {env.shovel_count} + {env.dump_count} + {env.truck_count*4} + 1 "
      f"= {obs.shape[0]}")
print(f"Espace d'action : {env.action_space}")
print(f"  Actions 0..{env.shovel_count-1} = assigner à pelle i")
print(f"  Action {env.shovel_count} = ATTENDRE")
print(f"Plage des observations : [{obs.min():.3f}, {obs.max():.3f}]")
print(f"Info initiale : {info}")
```

**Correspondance avec le mémoire** :
- Observation = s_t = ({q_p}, {x_c}, {z_r}, t_courant) — Eq. 3.3
- Action = Assignment(c → p) ou ATTENDRE — Eq. 3.5

### 2.3 Exécution d'un épisode complet avec politique aléatoire

```python
from memoire_master_rl_logistique.env.mine_env import MineEnv

env = MineEnv(truck_count=12, shovel_count=3, dump_count=2)
obs, info = env.reset(seed=42)

total_reward = 0.0
steps = 0

while True:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    steps += 1
    if terminated or truncated:
        break

print(f"Épisode terminé en {steps} steps")
print(f"Récompense totale : {total_reward:.2f}")
print(f"Tonnage total : {info['total_tonnage_t']:.0f} t")
print(f"Temps final : {info['current_time_min']:.1f} min")
print(f"Cycles totaux : {info['total_cycles']}")
```

### 2.4 Test de la récompense pondérée (Eq. 3.8)

```python
from memoire_master_rl_logistique.env.mine_env import MineEnv

# Tester différents poids (w1, w2, w3)
weight_configs = [
    ((1.0, 0.0, 0.0), "Rendement pur"),
    ((0.0, 1.0, 0.0), "Équité pure"),
    ((0.0, 0.0, 1.0), "Coût pur"),
    ((1.0, 0.1, 0.05), "Pondération du mémoire"),
]

for weights, desc in weight_configs:
    env = MineEnv(truck_count=12, shovel_count=3, dump_count=2,
                  reward_weights=weights)
    obs, _ = env.reset(seed=42)
    obs, reward, _, _, _ = env.step(0)
    print(f"w={weights} ({desc}) → reward={reward:.4f}")
```

---

## 3. Test des Baselines (Section 4.5 du mémoire)

### 3.1 Test individuel de chaque baseline

```python
from memoire_master_rl_logistique.env.mine_env import MineEnv
from memoire_master_rl_logistique.baselines.fixed_policy import FixedAssignmentPolicy
from memoire_master_rl_logistique.baselines.nearest_policy import NearestShovelPolicy
from memoire_master_rl_logistique.baselines.queue_aware_policy import QueueAwarePolicy

def run_episode(env, policy_fn, seed=42):
    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    while True:
        action = policy_fn(obs, info, env)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    tonnage = sum(t.total_tonnage_t for t in env.trucks)
    wait = sum(t.total_wait_min for t in env.trucks) / env.truck_count
    return total_reward, tonnage, wait

# Fixed Assignment
env = MineEnv(truck_count=12, shovel_count=3, dump_count=2)
fixed = FixedAssignmentPolicy(num_shovels=3)
r, ton, wait = run_episode(env, lambda o, i, e: fixed.predict(o, i))
print(f"Fixed:      reward={r:.2f}, tonnage={ton:.0f}t, attente_moy={wait:.1f}min")

# Nearest Shovel
env = MineEnv(truck_count=12, shovel_count=3, dump_count=2)
obs, info = env.reset(seed=42)
nearest = NearestShovelPolicy(graph=env.graph, shovel_node_ids=[s.node_id for s in env.shovels])
r, ton, wait = run_episode(env,
    lambda o, i, e: nearest.predict(o, i, e.truck_locations[e.current_truck_idx]))
print(f"Nearest:    reward={r:.2f}, tonnage={ton:.0f}t, attente_moy={wait:.1f}min")

# Queue-Aware
env = MineEnv(truck_count=12, shovel_count=3, dump_count=2)
obs, info = env.reset(seed=42)
qa = QueueAwarePolicy(graph=env.graph, shovels=env.shovels)
r, ton, wait = run_episode(env,
    lambda o, i, e: qa.predict(o, i, e.truck_locations[e.current_truck_idx], e.current_time_min))
print(f"QueueAware: reward={r:.2f}, tonnage={ton:.0f}t, attente_moy={wait:.1f}min")
```

**Ce qu'il faut vérifier** :
- QueueAware doit être meilleur que Nearest, qui doit être meilleur que Fixed
- Si ce n'est pas le cas, la politique de l'agent PPO a d'autant plus de marge d'amélioration

---

## 4. Entraînement de l'Agent PPO (Section 4.4)

### 4.1 Entraînement rapide (test)

```bash
# Entraînement rapide (5 000 steps, ~30 secondes)
python -c "
from memoire_master_rl_logistique.rl.train_ppo import train_ppo
model = train_ppo(total_timesteps=5_000, seed=42, truck_count=12, shovel_count=3, dump_count=2,
                  output_dir='models/ppo_test')
print('Entraînement test terminé')
"
```

### 4.2 Entraînement complet (pour le Chapitre 6)

```bash
# Entraînement complet (50 000 steps, ~2-5 minutes CPU)
python -m memoire_master_rl_logistique.rl.train_ppo
```

Les fichiers générés :
- `models/ppo_mine/ppo_mine_agent.zip` — modèle entraîné
- `models/ppo_mine/training_kpis.csv` — KPIs par épisode (pour la learning curve)
- `models/ppo_mine/tb_logs/` — logs TensorBoard

### 4.3 Visualisation de la courbe d'apprentissage

```bash
# Via TensorBoard
tensorboard --logdir models/ppo_mine/tb_logs/
# Ouvrir http://localhost:6006 dans le navigateur
```

Ou via le script de rapport :

```python
from pathlib import Path
from memoire_master_rl_logistique.experiments.stats_report import plot_learning_curve

plot_learning_curve("models/ppo_mine/training_kpis.csv", Path("data/results"))
print("Figure sauvegardée dans data/results/learning_curve.png")
```

---

## 5. Évaluation Comparative (Chapitre 6)

### 5.1 Évaluation PPO vs Baselines

```python
from memoire_master_rl_logistique.rl.evaluate_agent import evaluate_all_baselines, evaluate_ppo

# Évaluer les baselines (10 épisodes)
print("=== Baselines ===")
results = evaluate_all_baselines(n_episodes=10, truck_count=12, shovel_count=3, dump_count=2)
for r in results:
    print(f"\n{r['policy_name']}:")
    print(f"  Productivité : {r['productivity_tph_mean']:.1f} ± {r['productivity_tph_std']:.1f} t/h")
    print(f"  Attente moy.  : {r['avg_wait_min_per_truck_mean']:.1f} ± {r['avg_wait_min_per_truck_std']:.1f} min")
    print(f"  Utilisation   : {r['utilization_pct_mean']:.1f} ± {r['utilization_pct_std']:.1f} %")
    print(f"  Conso spéc.   : {r['specific_fuel_l_per_ton_mean']:.4f} ± {r['specific_fuel_l_per_ton_std']:.4f} L/t")

# Évaluer PPO (si le modèle est entraîné)
import os
if os.path.exists("models/ppo_mine/ppo_mine_agent.zip"):
    print("\n=== PPO ===")
    ppo = evaluate_ppo("models/ppo_mine/ppo_mine_agent.zip", n_episodes=10,
                       truck_count=12, shovel_count=3, dump_count=2)
    print(f"  Productivité : {ppo['productivity_tph_mean']:.1f} ± {ppo['productivity_tph_std']:.1f} t/h")
    print(f"  Attente moy.  : {ppo['avg_wait_min_per_truck_mean']:.1f} ± {ppo['avg_wait_min_per_truck_std']:.1f} min")
    print(f"  Utilisation   : {ppo['utilization_pct_mean']:.1f} ± {ppo['utilization_pct_std']:.1f} %")
    print(f"  Conso spéc.   : {ppo['specific_fuel_l_per_ton_mean']:.4f} ± {ppo['specific_fuel_l_per_ton_std']:.4f} L/t")
```

### 5.2 Benchmark complet sur tous les scénarios

```bash
# Lance les 4 méthodes × 6 scénarios × 10 seeds
python -m memoire_master_rl_logistique.experiments.run_benchmark
```

Résultats sauvegardés dans `data/results/benchmark_results.csv`.

### 5.3 Benchmark sur un seul scénario (plus rapide)

```python
from memoire_master_rl_logistique.experiments.run_benchmark import run_all_benchmarks

# Scénario nominal uniquement (le plus rapide)
run_all_benchmarks(
    output_dir="data/results",
    train_ppo_flag=True,
    scenario_names=["nominal"],
)
```

---

## 6. Génération des Figures et Tableaux (Chapitre 6)

### 6.1 Générer le rapport statistique complet

```bash
python -m memoire_master_rl_logistique.experiments.stats_report
```

Fichiers générés dans `data/results/` :
- `summary_table.csv` — Tableau de synthèse (moyenne ± écart-type par méthode)
- `kpi_comparison_nominal.png` — Barplots comparatifs des KPIs
- `learning_curve.png` — Courbe d'apprentissage PPO

### 6.2 Générer les figures manuellement

```python
import pandas as pd
from pathlib import Path
from memoire_master_rl_logistique.experiments.stats_report import (
    generate_summary_table,
    plot_kpi_comparison,
    plot_learning_curve,
)

# Charger les résultats du benchmark
df = pd.read_csv("data/results/benchmark_results.csv")
out = Path("data/results")

# Tableau de synthèse
summary = generate_summary_table(df, out)
print(summary[["scenario", "policy", "productivity_tph_display",
               "avg_wait_min_per_truck_display"]].to_string(index=False))

# Barplots
plot_kpi_comparison(df, out, scenario_name="nominal")

# Learning curve
plot_learning_curve("models/ppo_mine/training_kpis.csv", out)
```

---

## 7. Pipeline Complet en une Commande

```bash
# Simulation + Entraînement PPO + Benchmark (tout d'un coup)
python -m memoire_master_rl_logistique.main

# Ou étape par étape :
python -m memoire_master_rl_logistique.main --sim-only          # Simulation DES seule
python -m memoire_master_rl_logistique.main --train-only        # Entraînement PPO seul
python -m memoire_master_rl_logistique.main --benchmark-only    # Benchmark seul

# Options de configuration :
python -m memoire_master_rl_logistique.main --truck-count 18 --shovel-count 5 --dump-count 3
python -m memoire_master_rl_logistique.main --timesteps 100000  # Plus d'entraînement
```

---

## 8. Scénarios Expérimentaux Disponibles

| Scénario | Camions | Pelles | Dumps | Shift | Pannes | Objectif |
|----------|---------|--------|-------|-------|--------|----------|
| `nominal` | 12 | 3 | 2 | 8h | 2% | Conditions normales |
| `high_load` | 18 | 3 | 2 | 8h | 2% | Surcharge (congestion) |
| `low_load` | 6 | 3 | 2 | 8h | 2% | Sous-utilisation |
| `high_breakdown` | 12 | 3 | 2 | 8h | 10% | Robustesse aux pannes |
| `single_shovel` | 12 | 1 | 1 | 8h | 2% | Goulot d'étranglement |
| `short_shift` | 12 | 3 | 2 | 4h | 2% | Shift court |

---

## 9. KPIs à Analyser (Section 4.5 du mémoire)

| KPI | Formule | Unité | Interprétation |
|-----|---------|-------|----------------|
| **Productivité P_h** | Tonnage / Temps | t/h | Plus c'est élevé, mieux c'est |
| **Attente moyenne E_t** | Σ attente / N_camions | min | Plus c'est bas, mieux c'est |
| **Utilisation** | Temps actif / Temps dispo | % | Objectif > 85% |
| **Conso spécifique E_e** | Carburant / Tonnage | L/t | Plus c'est bas, mieux c'est |

---

## 10. Checklist de Validation pour le Chapitre 6

- [ ] Simulation DES fonctionne (Section 1)
- [ ] Reproductibilité vérifiée (même seed = même résultat)
- [ ] check_env() passe pour l'environnement Gymnasium
- [ ] Les 3 baselines fonctionnent et produisent des résultats cohérents
- [ ] PPO est entraîné (50 000 steps minimum)
- [ ] Learning curve montre une amélioration au fil des épisodes
- [ ] Benchmark comparatif exécuté (au moins scénario nominal)
- [ ] Tableau de synthèse généré (summary_table.csv)
- [ ] Figures générées (barplots + learning curve)
- [ ] PPO surpasse au moins une baseline sur la productivité
- [ ] Les résultats sont commentés et interprétés

---

## Dépannage

**Erreur "ModuleNotFoundError"** : Vérifier que le package est installé avec `pip install -e .`

**PPO ne converge pas** : Essayer avec plus de timesteps (`--timesteps 200000`) ou ajuster les poids de récompense.

**Résultats identiques pour toutes les baselines** : Normal si le scénario a une seule pelle (single_shovel) — il n'y a qu'une seule action possible.

**Figures non générées** : Vérifier que le dossier `data/results/` existe et que le benchmark a été exécuté au préalable.
