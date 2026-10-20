# Essais 0 à 4 de la mise au point des agents (16 octobre 2026)

Ces essais ont été faits **avant** la piste 1 :

- l'observation avait 17 composantes ;
- pour les essais 1, 2 et 4, la récompense était une variante d'essai, jamais commitée.

Ils ne tournent donc pas sur le code actuel. Ce dossier garde de quoi les refaire, pour que **tous** les essais déclarés dans l'article restent vérifiables.

Les essais 5 à 9 tournent sur le code actuel : `python -m memoire_master_rl_logistique.experiments.agent_trials --list`.

Conditions communes à tous les essais :

- régime nominal ;
- graine d'entraînement 1 ;
- critère : productivité sur les graines de sélection 20 000 à 20 009 (jamais celles de l'évaluation, 42 à 71).

| Essai | Ce qui est testé | Code | Commande | Résultat noté (journal) |
|---|---|---|---|---|
| 0 | Réglage T25 bis (4 configurations PPO, 4 DQN, 200 000 pas) | `157f97b` | `python -m memoire_master_rl_logistique.experiments.tune_hyperparams DOSSIER`, puis `tuning_diag.py` pour les parts de cycles | PPO 3 953, DQN 3 992 t/h |
| 1 | Récompense estimée à la décision (1b), 200 000 pas | `157f97b` + patch | `python reward_timing_test.py {PPO,DQN} DOSSIER` | PPO 3 915, DQN 4 191 |
| 2 | 1b + coût des minutes en file, 200 000 pas | `157f97b` + patch | `python reward_queue_test.py {PPO,DQN} DOSSIER` | PPO 3 981, DQN 4 106 |
| 3 | PPO, 2 000 000 pas, récompense en fin de cycle (1a) | `157f97b` + patch | `python diag_full.py cycle_end DOSSIER` | 4 260 |
| 4 | PPO, 2 000 000 pas, récompense à la décision (1b) | `157f97b` + patch | `python diag_full.py decision DOSSIER` | 3 943 |

Le patch `essai_reward_timing.patch` ajoute à `MineEnv` deux options :

- `reward_timing="decision"` : récompense du cycle estimée et versée à la décision ;
- `queue_opportunity=True` : coût des minutes en file.

Avec `reward_timing="cycle_end"` (valeur par défaut), le comportement est celui du commit `157f97b`. L'essai 3 a été lancé avec le patch, par le même script que l'essai 4, mais il utilise bien la récompense commitée.

`tuning_diag.py` lit les modèles du réglage dans `data/results_v2/tuning` (chemin écrit dans le script).

## Refaire un essai

Depuis la racine du dépôt :

```bash
git worktree add --detach ../essais_157f97b 157f97b
cd ../essais_157f97b
git apply ../memoire-master-rl-logistique/archives/mise_au_point_agents/essai_reward_timing.patch
PYTHONPATH=. python ../memoire-master-rl-logistique/archives/mise_au_point_agents/diag_full.py cycle_end DOSSIER
```

Après les calculs, supprimer le worktree avec `git worktree remove ../essais_157f97b`.

Le patch s'applique sans erreur sur `157f97b` (vérifié le 20 octobre 2026 avec `git apply --check`). Les résultats n'ont pas été recalculés.

Ces scripts sont gardés **tels qu'ils ont tourné**, à deux corrections près dans leur docstring : le titre de `reward_timing_test.py` et de `reward_queue_test.py`, et la ligne d'usage de `reward_queue_test.py`. Le code n'a pas été modifié.

Les mesures en lecture seule de l'analyse de l'échec des agents à valeurs (19 octobre) sont dans le code : `memoire_master_rl_logistique/analysis/reward_diagnostics.py`. Elles sont vérifiées par `tests/test_agent_trials.py`.
