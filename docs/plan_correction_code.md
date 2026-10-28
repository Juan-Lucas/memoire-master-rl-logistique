# Plan de correction du code (dispatching RL minier)

Version fusionnée du 7 octobre 2026. Elle combine :

- le plan en 31 tâches T1 à T31, qui sert de référence ;
- les numéros de ligne et le niveau d'impact de l'ancien plan ;
- les corrections issues d'une vérification dans le code.

Ce plan te fait corriger toi-même ton code en **34 tâches réparties en 8 phases** (T1 à T31, plus T12 bis, T16 bis et T25 bis). Chaque correction est testée avant la suivante, et aucun agent n'est réentraîné avant la phase 7.

Chaque tâche renvoie au point qu'elle corrige :

- M ou m : un point du rapport de relecture ;
- N : un point du rapport d'audit ;
- Dir. corr. ou Dir. plan : une remarque de ta directrice (document de correction scientifique ou plan « Article Q1 »).

Elle renvoie aussi au module de ton plan de maîtrise.

> Les numéros de ligne correspondent au code du commit `424ff39` (7 octobre 2026). Ils bougeront dès tes premières modifications : fie-toi au nom de la fonction.

> **Réorganisation du 15 octobre 2026 (après T7).** Le dossier `experiments/` a été découpé par responsabilité. Quand une tâche cite un ancien chemin, utilise la table ci-dessous (tous les chemins sont relatifs à `memoire_master_rl_logistique/`).
>
> | Ancien chemin | Nouveau chemin |
> |---|---|
> | `experiments/scenarios.py`, `run_benchmark.py`, `ablation_study.py` | inchangés |
> | `experiments/statistical_tests.py` | `analysis/statistical_tests.py` |
> | `experiments/robustness_analysis.py` | calculs : `analysis/robustness.py` ; heatmap et ligne de commande : `plots/robustness.py` |
> | `experiments/compute_sample_efficiency.py` | `analysis/sample_efficiency.py` |
> | `experiments/action_distribution.py` | `analysis/action_distribution.py` |
> | `experiments/plot_<nom>.py` | `plots/<nom>.py` (ex. `plot_benchmark_comparisons.py` → `plots/benchmark_comparisons.py`) |
> | `experiments/visualize_mine_graph.py` | `plots/mine_graph.py` |
> | `experiments/plot_utils.py`, `experiments/reporting.py` | `plots/common.py` |
> | `experiments/train_diagnostics.py`, `train_diagnostics_nominal.py`, `train_losscurve.py`, `export_convergence_qlearning_sarsa.py` | `rl/diagnostics/` (mêmes noms) |
> | `model_version.py` (T8) | remplacé par `rl/model_metadata.py` |

---

## Mode d'emploi

### Règles de travail

- **Une tâche = un commit** (ou une branche courte), avec un message qui cite son code, par exemple « T10 M2 : séparer les attentes ».
- **Écris le test avant la correction** : il doit échouer, puis réussir. Le dossier `tests/` et `pytest` suffisent.
- **Tu écris chaque ligne toi-même.** Tu peux demander à l'IA d'expliquer une fonction, de relire ton code ou de te poser des questions, jamais de l'écrire à ta place.
- **Avant de modifier, écris ta prédiction** : quel chiffre va bouger, et dans quel sens. Après le test, compare. Une prédiction fausse est ce qui t'apprend le plus.
- **Après chaque tâche, une entrée dans ton journal de maîtrise** : ce que tu as changé, pourquoi, ce que le test vérifie.
- **Chaque tâche a des questions « Savoir expliquer »** : tant que tu ne peux pas y répondre à voix haute sans notes, la tâche n'est pas finie.

### Pourquoi cet ordre

1. **Phases 0 à 2** : elles ne changent pas le comportement des agents. Tu peux les faire tout de suite, et tes résultats actuels restent valables.
2. **Phases 3 à 6** : elles modifient le simulateur, l'observation, les heuristiques et le protocole. Elles invalident les résultats actuels.
3. **Phase 7** : tous les entraînements sont relancés **une seule fois**, après les phases 3 à 6. Relancer avant, ce serait payer deux fois les 2 heures par entraînement.

### Légendes

**Priorité**

- **Indispensable** : exigé pour la soumission.
- **Recommandé** : attendu par un relecteur.
- **Optionnel** : renforce l'article, à décider avec ta directrice.

**Impact** (ce que la tâche oblige à relancer)

| Étiquette | Signification |
|---|---|
| **[MESURE]** | Change seulement ce qui est calculé ou rapporté. Les modèles entraînés restent valables. |
| **[RÉÉVAL]** | Change la simulation ou une heuristique sans changer ce que l'agent voit. Il faut relancer l'évaluation, pas l'entraînement. |
| **[RÉENTR]** | Change l'observation, la récompense ou la dynamique apprise par l'agent. Il faut réentraîner. |

**Règle d'or : toutes les tâches [RÉENTR] doivent être finies avant la phase 7.**

---

## Phase 0. Préparation (½ jour)

**Objectif :** pouvoir modifier le code sans jamais perdre la version qui a produit l'article actuel.

### T1. Figer la version actuelle

**Indispensable · Aucun impact**

- **À faire :** créer un tag Git `v1-article` sur l'état actuel, puis une branche `corrections-v2` pour tout le travail.
- **Savoir expliquer :** la différence entre un tag et une branche ; comment revenir à `v1-article` ; pourquoi on ne travaille jamais sur la version qui a produit les résultats publiés.
- [x] Fait

### T2. Geler les résultats v1

**Indispensable · Aucun impact · ✅ Déjà fait le 7 octobre**

- **État :** `data/results/` a été déplacé vers **`data/results_v1_article/`** (135 fichiers, 124 Mo). `data/results/` est vide. Garde ce nom partout dans le plan : ne crée pas un second dossier `results_v1`.
- **Reste à faire :** une copie de `data/results_v1_article/` hors du projet (disque externe ou cloud), car `data/` est ignoré par Git. Les nouveaux résultats iront dans `data/results_v2/`.
- **Attention :** une vingtaine de scripts écrivent `data/results/` en dur dans leurs chemins. Tant que tu ne passes pas explicitement l'archive, `run_benchmark` ne trouvera plus aucun modèle et **réentraînera tout**.
- **Savoir expliquer :** quels fichiers de `results_v1_article` ont servi à chaque tableau de l'article ; pourquoi les modèles DQN datent de mai et les modèles PPO d'août (N11).
- [x] Copie externe faite

### T3. Environnement reproductible

**Indispensable · Aucun impact**

- **À faire :** un fichier `requirements.txt` avec les versions exactes enregistrées dans tes modèles (fichier `system_info.txt` des `.zip`, vérifié) : Python 3.11.11, stable-baselines3 2.8.0, torch 2.5.1, gymnasium 1.2.3, numpy 2.4.6. Ajoute pandas, scipy, statsmodels, networkx et matplotlib. Le `Makefile` (cible `requirements`) attend déjà ce fichier.
- **À corriger aussi :** `configs/experiment_config.yaml` annonce Python 3.10 et Gymnasium 0.26, ce qui est faux.
- **Test :** créer un environnement neuf, installer le fichier, lancer `python -m memoire_master_rl_logistique.main --check-env`. Cette option existe déjà dans `main.py`.
- **Savoir expliquer :** pourquoi une version différente de PyTorch peut changer un modèle entraîné avec la même graine.
- [x] Fait

### T4. Tests de non-régression

**Indispensable · Aucun impact**

- **Fichiers :** nouveau `tests/test_regression.py`.
- **À faire :** lancer l'Affectation fixe et FIFO sur le régime nominal avec les graines 42 à 46. Pour chaque graine, comparer `productivity_tph` et `total_reward` aux valeurs de `data/results_v1_article/benchmark_results.csv`, à 10⁻⁶ près. Attention : ce CSV contient chaque ligne en double (T5), donc prends la première occurrence.
- **Usage :** ce test doit rester vert pendant les phases 1 et 2. Il deviendra rouge en phase 3 : c'est normal et voulu. Tu remplaceras alors les valeurs de référence et tu noteras pourquoi dans ton journal.
- **Savoir expliquer :** ce qu'est un test de non-régression ; pourquoi la même graine doit redonner exactement le même résultat (quelle ligne de `mine_env.py` le garantit ?).
- [x] Fait

---

## Phase 1. Résultats et statistiques (2 jours)

**Objectif :** qu'un résultat ne puisse plus jamais être compté deux fois, et que les tests statistiques correspondent à ton plan expérimental. C'est le module 7 de ton plan de maîtrise.

### T5. Fichier de résultats sans doublons

**Indispensable · [MESURE] · M3**

- **Fichiers :** `experiments/run_benchmark.py`, fin de `run_all_benchmarks` (lignes 330 à 339).
- **Problème :** le CSV est ouvert en mode ajout (`open("a")`, ligne 334). Tes deux exécutions ont produit 1 440 lignes au lieu de 720.
- **À faire :**
  - écrire chaque exécution dans un nouveau fichier daté (`benchmark_YYYYMMDD_HHMM.csv`), en mode écriture, avec une colonne `run_id` ;
  - juste avant l'écriture, vérifier qu'aucun triplet (scénario, politique, graine) n'apparaît deux fois.
- **Test :** lancer deux fois un mini-benchmark (2 graines) et vérifier qu'on obtient deux fichiers et aucun doublon.
- **Savoir expliquer :** pourquoi un doublon ne change pas la moyenne mais divise la largeur d'un intervalle de confiance par environ 1,4 (√2).
- [x] Fait

### T6. Tests statistiques appariés

**Indispensable · [MESURE] · M3, M4, M5, M6, M22**

- **Fichiers :** `experiments/statistical_tests.py`, à réécrire en grande partie.

| Ligne | Problème actuel |
|---|---|
| 42 à 44 (`load_results`) | Aucune vérification de *n* ni des doublons |
| 114, 372 | Welch indépendant **unilatéral** (`"greater"`) : p = 1 quand l'écart va dans l'autre sens |
| 394, 426 | Tests carburant unilatéraux (`"less"`) |
| 357 à 360 | Welch ou Kruskal–Wallis selon la normalité : un cadre différent par scénario |
| 149 (`dunn_test`) | Post-hoc pour groupes indépendants |
| 310 (`bootstrap_ci`) | Intervalle sur chaque moyenne séparément, pas sur la différence appariée |

- **À faire, dans cet ordre :**
  1. charger le CSV et **refuser de continuer** si un groupe n'a pas exactement 30 lignes ;
  2. construire un tableau croisé graine × politique pour chaque régime (`pivot`) ;
  3. test de Friedman par régime (`scipy.stats.friedmanchisquare`) ;
  4. tests de Wilcoxon **appariés et bilatéraux** (`scipy.stats.wilcoxon`) sur les paires d'intérêt, avec correction de Holm (`statsmodels.stats.multitest.multipletests`) ;
  5. intervalle bootstrap apparié : rééchantillonner les 30 différences par graine, 10 000 fois ;
  6. interaction politique × régime : modèle `ols("prod_par_camion ~ C(policy) * C(scenario) + C(bloc)")` puis `anova_lm` ;
  7. appliquer **le même cadre** au carburant, et afficher `p < 0,001` au lieu de 10⁻⁴⁶.
- **Test :** sur ton CSV dédoublonné, tu dois retrouver la section 7.6 de l'article v2. Par exemple, PPO − Affectation fixe en nominal : +105,0 t/h, intervalle [88,1 ; 120,8].
- **Savoir expliquer :** pourquoi les données sont appariées ; la différence entre unilatéral et bilatéral ; pourquoi la plus petite valeur p d'un Wilcoxon avec 30 paires vaut environ 2 × 10⁻⁹ (2 / 2³⁰).
- [x] Fait

### T7. Robustesse et figures

**Indispensable · [MESURE] · M15, M2, M19**

- **Fichiers :** `experiments/robustness_analysis.py` et `experiments/plot_benchmark_comparisons.py` (dossier de sortie à la ligne 188).
- **À faire :**
  - dédoublonner à la lecture ;
  - calculer le ratio de robustesse en forte charge sur la **productivité par camion** ;
  - tracer le Pareto **par régime** ;
  - retirer le seuil de 30 min de la figure d'attente.

  Tu peux t'inspirer des quatre figures de `figures_v2.zip`, mais refais-les avec ton propre code.
- **Test :** comparer tes ratios au tableau 13 de l'article v2. Par exemple, Affectation fixe en forte charge : 0,962.
- **Savoir expliquer :** pourquoi un ratio supérieur à 1 en forte charge ne prouvait rien.
- [x] Fait (commit `c9f1332` ; la colonne « Δ attente en file » attend T10)

### T8. Tracer la version de chaque modèle

**Recommandé · [MESURE] · N6, N11**

- **Fichiers :** `model_version.py` (constante à la ligne 7, fonctions aux lignes 18 à 30), `rl/train_ppo.py`, `rl/train_dqn.py`, `experiments/run_benchmark.py` (chargements aux lignes 223 à 225 et 251 à 253).
- **Problème :** `model_version.py` existe, mais aucune fonction n'est jamais appelée. Un ancien modèle peut être rechargé sans que tu le saches.
- **À faire :**
  - à chaque entraînement, écrire à côté du modèle un `metadata.json` : date, commit Git, régime, graine, poids de récompense, taille d'observation ;
  - au chargement, refuser un modèle dont les métadonnées ne correspondent pas à la configuration demandée.
- **Test :** un faux modèle avec des métadonnées périmées doit être refusé.
- **Vérification N11 à faire tout de suite :** `git log -p -- memoire_master_rl_logistique/env/mine_env.py` entre fin mai et début août 2026. La fonction de récompense a-t-elle changé entre l'entraînement de DQN et celui de PPO ?
- **Savoir expliquer :** comment tu prouverais à un relecteur que DQN et PPO ont été entraînés avec la même récompense.
- [x] Fait (commit `b902df2` ; chargement unique par `rl/model_loading.py`)

### T9. Petites corrections

**Recommandé · [MESURE] · N8, N9, m20**

- **`main.py` :** le pipeline complet écrit PPO dans `models/ppo_mine` (valeur par défaut de `train_ppo`, ligne 31), un dossier que le benchmark ne relit pas. Aligne les deux chemins.
- **`rl/evaluate_agent.py` :** écart-type avec `ddof=1`. Supprime `benchmark_policy_inference` (ligne 75) : elle appelle `policy_fn(None, None)` et mesure donc une politique factice. Adapte ensuite `tests/test_inference_benchmark.py`.
- **`run_benchmark.py`, lignes 55 à 57 :** mesurer séparément le temps de construction de l'état (`_get_obs`) et le temps de décision.
- **`experiments/scenarios.py` :** `unseen_medium_load` (lignes 96 à 100) plante avec les modèles nominaux, qui attendent 147 entrées contre 180 à 15 camions. Retire-le, ou garde-le pour après T18.
- **Savoir expliquer :** ce que mesurent exactement les 10 ms de latence rapportées dans l'article.
- [x] Fait (commit `823559f` ; latence réelle ≈ 0,9 ms, à corriger dans l'article en T31)

---

## Phase 2. Indicateurs (1 à 2 jours)

**Objectif :** que chaque indicateur mesure exactement ce que dit l'article. Ces tâches ajoutent des mesures sans changer les décisions : le test T4 doit rester vert. C'est le module 1 de ton plan de maîtrise.

### T10. Décomposer le temps d'attente

**Indispensable · [MESURE] · M2, M17**

- **Fichiers :** `simulation/entities.py` (classe `Truck`, lignes 14 à 30), `env/mine_env.py`, `simulation/kpi.py`.
- **Problème :** `total_wait_min` mélange tout, puis `kpi.py` (ligne 19) divise par le nombre de camions.

| `mine_env.py`, lignes | Ce qui est ajouté à `total_wait_min` |
|---|---|
| 215 à 219 | File à la pelle |
| 241 à 245 | File à la verse |
| 266 à 270 | Durée de **réparation** d'une panne |
| 322 (`_apply_wait_action`) | Durée de l'action ATTENDRE |

- **À faire :**
  - trois compteurs dans `Truck` : `queue_wait_min`, `repair_min` et `voluntary_wait_min`, chacun incrémenté à sa place ;
  - `total_wait_min` devient leur somme ;
  - `compute_kpis` renvoie les trois, par camion et par poste, plus la file par cycle.
- **Ne touche pas** à `operational_wait_min` dans l'appel à la récompense (ligne 288). Sinon la tâche devient [RÉENTR].
- **Test :**
  - sur un épisode, la somme des trois compteurs égale l'ancien `total_wait_min` ;
  - avec l'Affectation fixe en nominal, sur 30 graines : environ 22,2 min de file et 7,2 min de panne par camion, soit environ 1,1 min de file par cycle.
- **Savoir expliquer :** pourquoi « 29 min par cycle » était faux ; combien de cycles fait un camion par poste.
- [x] Fait (commit `74767d8` ; 22,17 / 7,18 / 1,14 min retrouvés ; Δ file Fixed FC = +60,44 %, l'article écrit +60,5)

### T11. Utilisation bornée au poste et utilisation des pelles

**Indispensable · [MESURE] · M2 bis, m18**

- **Fichiers :**
  - `env/mine_env.py` : lignes 207, 225, 233, 251 et 259 (`total_active_min += ...`) et lignes 274 à 277 (le tonnage n'est compté que si le cycle finit avant 480 min) ;
  - `simulation/entities.py` (classe `Shovel`) ;
  - `simulation/kpi.py`, ligne 20.
- **Problème :** le temps actif du dernier cycle est compté même après 480 min. Cela gonfle l'utilisation : 96,4 % au lieu d'environ 93,9 % pour l'Affectation fixe.
- **À faire :**
  - n'ajouter à `total_active_min` que la part de chaque activité située avant la fin du poste ;
  - ajouter un compteur `busy_min` aux pelles, borné de la même façon ;
  - `compute_kpis` renvoie l'utilisation des camions et celle des pelles.
- **Test :** pour chaque camion, temps actif + attentes ≤ 480 min ; l'utilisation des pelles reste inférieure à 100 %.
- **Savoir expliquer :** pourquoi les pelles sont occupées moins de 50 % du temps même avec 18 camions, et ce que cela dit du facteur d'appariement.
- [x] Fait (commit `e8e12d8` ; utilisation des camions 94,1 %, des pelles 33,9 % et 49,0 % ; attentes aussi bornées, donc les valeurs de T10 deviennent 21,80 / 6,72 / 1,13 min)

### T12. Journal des cycles et test de cohérence

**Recommandé · [MESURE] · M2, M11**

- **Fichiers :** `env/mine_env.py` (`step`, lignes 164 à 300) et nouveau `tests/test_cycle.py`.
- **À faire :** un journal optionnel, activé par un paramètre `log_cycles=True`. Pour chaque cycle, il enregistre : camion, début, fin, trajets, attente à la pelle, chargement, attente à la verse, déversement, panne.
- **Test :** pour chaque cycle, fin − début = somme des composantes, à 10⁻⁶ près.
- **Savoir expliquer :** ce qui se passe, minute par minute, entre le moment où un camion reçoit une affectation et celui où il redevient disponible (première question du module 1).
- [x] Fait (commit `112e2a9` ; `MineEnv(log_cycles=True)` → `env.cycle_log`, chronologie continue vérifiée)

### T12 bis. Analyse comportementale sur 30 réplications

**Recommandé · [MESURE] · M16, C3**

Cette tâche produit le tableau `tab:behaviour` et la figure `fig:behavioral` de l'article v2 (section 7.3).

- **Fichiers :**
  - `experiments/action_distribution.py` : ne porte aujourd'hui que sur 10 graines (42 à 51) ;
  - `experiments/plot_behavioral_analysis.py` : un seul épisode (graine 42), avec une analyse par l'entropie ;
  - nouveau script, par exemple `experiments/behavior_stats.py`.
- **Problème :** le tableau comportemental de la v2 vient de l'audit, pas de ton code. La figure repose sur un seul épisode, et sa légende (panneau C : « adaptive load balancing », « greedy → hot spot ») interprète plus que ce qu'elle montre (balise verte de la v2).
- **À faire :**
  - pour chaque politique et chaque régime, sur les 30 graines : la part des cycles par pelle et la part des décisions ATTENDRE. Si T10 et T19 sont faites, ajoute la décomposition de l'attente (file, panne, ATTENDRE) ;
  - écrire le résultat dans un CSV, puis produire `tab:behaviour` à partir de ce fichier ;
  - refaire la figure : retirer les mentions interprétatives du panneau C, ou remplacer l'entropie d'un épisode par les parts de cycles par pelle sur 30 réplications ;
  - si tu gardes l'entropie, écrire sa formule exacte avant de coder.
- **Test :** avec les modèles v1, tu retrouves le tableau de la v2 : Pelle la plus proche et Plus court chemin à 100 % sur la pelle 1 dans les trois régimes ; en forte charge, ATTENDRE dans 87 % des décisions pour PPO et 64 % pour DQN.
- **Savoir expliquer :** pourquoi ATTENDRE domine chez PPO en forte charge (lien avec T19) ; ce que la figure démontre et ce qu'elle ne démontre pas.
- [x] Fait (commit `d1cec48` ; `tab:behaviour` reproduit à l'identique par `analysis/behavior.py`)

---

## Phase 3. Simulateur (3 à 5 jours)

**Objectif :** un simulateur dont les aléas sont comparables d'une politique à l'autre, et dont le modèle correspond à la description de l'article. À partir d'ici, le test T4 devient rouge : mets à jour ses valeurs de référence après chaque tâche, en notant pourquoi. Ce sont les modules 1 et 6 de ton plan de maîtrise.

### T13. Un flux aléatoire par processus

**Indispensable · [RÉÉVAL] · M4**

La loi de l'environnement ne change pas, donc tes modèles restent valables en principe. Mais tous les chiffres bougent.

- **Fichiers :** tous les tirages passent aujourd'hui par le même `self.rng`.

| Fichier, lignes | Tirage |
|---|---|
| `mine_env.py`, 119 | Création : `Random(effective_seed)` |
| `mine_env.py`, 266 et 267 | Panne et durée de réparation |
| `mine_env.py`, 435 à 437, puis `graph_model.py`, 176 à 187 | Temps de trajet |
| `mine_env.py`, 445 et 446 | Dégradation de la piste |
| `mine_env.py`, 449 à 454 | Chargement et déversement |

- **Problème :** dès que deux politiques prennent une décision différente, leurs aléas divergent. La première panne n'est identique que pour 5 graines sur 30.
- **À faire :**
  - dans `reset`, dériver de la graine des générateurs indépendants avec `numpy.random.SeedSequence(seed).spawn(...)` : un pour les trajets, un pour les chargements et déversements, un pour la dégradation des pistes, et un par camion pour les pannes ;
  - pour les pannes, tirer à l'avance, pour chaque camion, la liste « le k-ième cycle tombe-t-il en panne, et pour combien de temps ».
- **Test :** avec la même graine, le camion 3 tombe en panne au même numéro de cycle sous FIFO et sous l'Affectation fixe, avec la même durée de réparation.
- **Savoir expliquer :** ce que sont des nombres aléatoires communs ; pourquoi ils réduisent la variance d'une différence ; pourquoi une panne ne peut quand même pas survenir à la même minute pour deux politiques.
- [x] Fait (commit `a2a8700` ; références de non-régression régénérées dans `tests/fixtures/`)

### T14. Carburant paramétrable et calibré

**Recommandé · [RÉENTR] · M20**

Le carburant fait partie de l'observation (`observation_builder.py`, ligne 87) : changer les constantes change ce que voit l'agent.

- **Fichiers :** `simulation/fuel_model.py` (0,65 et 0,45 L/km à la ligne 12, 10 L/h au ralenti à la ligne 18) et `simulation/constants.py`.
- **À faire :** regrouper ces constantes dans `constants.py`, avec leur source en commentaire. Chercher la consommation horaire publiée pour un tombereau de 140 t, puis ajuster les constantes pour s'en approcher, ou garder les valeurs et l'écrire comme limite.
- **Savoir expliquer :** combien de litres consomme un cycle dans ton simulateur (environ 6 L) et comment ce chiffre se compare à la réalité.
- [x] Fait (commit `5177693` ; calibré à 95 L/h sur le Cat 785D ; observation de l'agent inchangée, donc pas de réentraînement : T14 devient [MESURE])

### T15. Servir les files dans l'ordre d'arrivée

**Optionnel · [RÉENTR] · N4**

- **Fichiers :** `env/mine_env.py`. Les réservations se font par `available_at_min` (lignes 215, 228, 241 et 254).
- **Problème :** le cycle est simulé d'un seul tenant au moment de la décision ; pelles et verses servent dans l'ordre des décisions, pas des arrivées.
- **À faire :** soit découper le cycle en événements traités dans l'ordre chronologique (file de priorité `heapq`, ou SimPy), soit garder le modèle actuel et le décrire comme une approximation. C'est la tâche la plus longue du plan : **décide avec ta directrice** si elle en vaut la peine.
- **Savoir expliquer :** la différence entre ton moteur par réservation et une simulation à événements discrets avec file d'événements.
- [x] Fait (commit `157f97b`, avec T16 bis : moteur à événements, récompense versée à la fin du cycle)

### T16. Pannes de pelles

**Optionnel · [RÉENTR] · N5**

- **Fichiers :** `simulation/entities.py`, `env/mine_env.py`, `experiments/scenarios.py`.
- **À faire :**
  - ajouter aux pelles une probabilité de panne et une durée de réparation, avec leur propre flux aléatoire (T13) ;
  - exposer leur disponibilité dans l'état ;
  - si ta directrice le valide, ajouter un régime « panne de pelle ». C'est aussi ce qui donnerait un sens à une réaffectation de secours pour l'Affectation fixe (T22).
- **Savoir expliquer :** pourquoi la perte d'une pelle est une perturbation plus sévère que la panne d'un camion avec un facteur d'appariement de 0,35.
- [x] Décidé : non fait, écrit comme limite de l'article (16 octobre 2026)

### T16 bis. Prendre la décision à la verse (option b de T21)

**À décider avec T21 · [RÉENTR] · M16, M25**

- **Fichiers :** `env/mine_env.py`, lignes 256 à 263.
- **Problème :** à la fin de chaque cycle, le camion **refait le trajet verse → même pelle** avant même la décision suivante. Il est donc toujours à la pelle qu'il vient de servir quand il reçoit sa nouvelle affectation. C'est la cause de la dégénérescence de Pelle la plus proche et de Plus court chemin (T21).
- **À faire, si ta directrice choisit l'option (b) de T21 :**
  - terminer le cycle à la verse, après le déversement ;
  - le retour à vide devient le premier trajet du cycle suivant, vers la pelle choisie ;
  - la position du camion au moment de la décision devient la verse.

  Cela change la dynamique pour toutes les politiques, agents compris : c'est pourquoi la tâche est ici, en phase 3, et non en phase 5.
- **Test :** chaque cycle commence à une verse (sauf le premier, au parc) ; la part des cycles par pelle de Pelle la plus proche ne dépasse plus 60 % pour aucune pelle.
- **Savoir expliquer :** où se trouve un camion, dans une vraie mine, quand le répartiteur lui donne sa prochaine affectation.
- [x] Fait (commit `157f97b`, avec T15 ; Pelle la plus proche reste à 100 % sur la pelle 1 : géométrie, voir le journal)

---

## Phase 4. Observation et actions (2 à 3 jours)

**Objectif :** un état sans composantes inutiles et de taille indépendante du nombre de camions, pour pouvoir évaluer PPO et DQN sans réentraînement en forte charge. C'est le module 2 de ton plan de maîtrise.

> **Ordre important :** fais **T18 avant T17**. Si tu choisis les variables agrégées en T18, les blocs par camion disparaissent, et avec eux la plupart des composantes constantes de T17.

### T17. Retirer les composantes constantes

**Indispensable · [RÉENTR] · N1, N2, N3**

- **Fichiers :** `env/observation_builder.py`.
- **Problème :** 54 composantes sur 147 ne changent jamais.

| Composante | Lignes | Raison |
|---|---|---|
| Statut des camions | 76 | Toujours 0 : `mine_env.py`, ligne 281, remet le statut à libre avant que l'observation soit construite (ligne 298) |
| Colonnes « intersection » et « verses » du codage de position | 90 à 92 | Le camion n'est jamais qu'au parc ou à une pelle (`mine_env.py`, lignes 156 et 262) |
| Les 6 distances pelle–verse | 100 à 104 | Constantes par construction |
| Tonnage | 20 et 84 | Divisé par 2 000 t, il sature avant la fin du poste |

- **À faire :**
  - avec l'option « remplissage » de T18 : supprimer ces composantes et normaliser le tonnage par le maximum réaliste d'un poste (environ 20 cycles × 140 t = 2 800 t, à vérifier dans tes résultats) ;
  - avec l'option « agrégats » : vérifier que la nouvelle observation n'en contient aucune.

  Dans les deux cas, mettre à jour `observation_size` (lignes 112 à 129) et `rl/discretize.py`, qui lit l'observation **depuis la fin** (lignes 33 à 39).
- **Test :**
  - sur 10 épisodes, aucune composante de l'état n'a un écart-type nul ;
  - pour les **variables continues seulement**, aucune ne vaut exactement 1 plus de 5 % du temps. Ne l'applique pas aux colonnes d'un codage one-hot : elles valent légitimement 1 souvent.
- **Savoir expliquer :** pourquoi le statut valait toujours 0 ; pourquoi une composante constante ne coûte rien à la performance, mais affaiblit la description de l'article.
- [x] Fait (commit `d778e30`, avec T18 : aucune composante constante ni saturée sur 10 épisodes × 3 politiques × 3 régimes)

### T18. Observation de taille fixe

**Indispensable · [RÉENTR] · M1**

- **Fichiers :** `env/observation_builder.py` (bloc par camion aux lignes 73 à 92 ; `observation_size` = 15 + 11N) et `env/mine_env.py` (`observation_space`, lignes 91 à 94).
- **Problème :** un réseau entraîné avec 12 camions (147 entrées) ne peut pas recevoir l'état de 18 camions (213 entrées).
- **À faire :** choisir une des deux solutions, et justifier ton choix par écrit.
  1. **Variables agrégées (recommandée) :**
     - par pelle : le temps avant disponibilité, le nombre de camions qui s'y dirigent ou y attendent (divisé par un N maximal, par exemple 24), et la distance du camion qui décide ;
     - par verse : le temps avant disponibilité ;
     - pour le camion qui décide : sa position, son tonnage et son carburant ;
     - enfin, le temps du poste.

     La taille ne dépend plus de N.
  2. **Remplissage :** réserver 24 emplacements de camions, remplir les emplacements vides par des zéros et ajouter un indicateur « camion présent ». Limite à écrire dans l'article : les emplacements 13 à 18 n'ont jamais été actifs pendant l'entraînement.
- **Argument de coût pour ta directrice :** avec une observation indépendante de N, un entraînement en nominal suffit pour évaluer les trois régimes. Si l'article se concentre sur le transfert, la campagne passe de 30 à 10 entraînements.
- **Test :** `observation_space.shape` est identique pour 12 et 18 camions ; un modèle PPO entraîné brièvement avec 12 camions s'exécute sans erreur dans un environnement à 18 camions.
- **Après T18 :** réintroduire le scénario `unseen_medium_load` (15 camions) dans `experiments/scenarios.py`, retiré en T9 (le commentaire en place donne sa définition), et retirer le test `test_observation_de_taille_differente_toujours_refusee` de `tests/test_model_loading.py`, qui documente le défaut.
- **Savoir expliquer :** pourquoi ce choix conditionne toute l'affirmation de transfert sans réentraînement de l'article ; ce que l'agent ne voit plus avec des variables agrégées.
- [x] Fait (commit `d778e30` ; variables agrégées, 18 composantes ; `unseen_medium_load` réintroduit ; à confirmer avec la directrice)

### T19. Masque d'actions et action ATTENDRE

**Recommandé · [MESURE] pour la journalisation, [RÉENTR] si le masque est utilisé · M16**

- **Fichiers :** `env/action_mask.py` et `env/mine_env.py` (`_apply_wait_action`, lignes 302 à 334).
- **Problème :** `compute_action_mask` autorise tout et n'est utilisé nulle part. En forte charge, PPO choisit ATTENDRE dans 87 % de ses décisions.
- **À faire :**
  - supprimer le fichier inutilisé, ou l'utiliser vraiment avec `MaskablePPO` (bibliothèque sb3-contrib), par exemple pour interdire ATTENDRE quand une pelle est libre ;
  - enregistrer dans `info` le nombre d'actions ATTENDRE par épisode.
- **Savoir expliquer :** pourquoi la récompense actuelle peut rendre ATTENDRE intéressante pour l'agent. Indice : regarde le coût de distance passé à `_compute_reward` pour une attente (ligne 331).
- [x] Fait (mesure seulement : `action_mask.py` supprimé, ATTENDRE compté dans `info` et les KPI ; cause dans la récompense, à corriger en T25)

### T20. Représentation tabulaire documentée

**Recommandé · [MESURE] · M8**

- **Fichiers :**
  - `rl/discretize.py` : multiplication par 50 à la ligne 23, découpage en intervalles à la ligne 31 ;
  - `rl/tabular_training.py` : l'action 0 par défaut sur un état inconnu, ligne 52.
- **À faire :**
  - remplacer les constantes « magiques » (× 50, 8 intervalles) par des paramètres nommés et commentés, **sans changer leurs valeurs**, sinon la tâche devient [RÉENTR] ;
  - journaliser, pendant l'évaluation, la part d'états déjà vus et la part d'actions par défaut.
- **Savoir expliquer :** pourquoi la table compte 98 304 états possibles (8⁵ × 3) ; pourquoi l'agent tabulaire peut, lui, être transféré à 18 camions.
- [x] Fait (commit `56920b7` ; état inchangé ; action par défaut jamais utilisée en évaluation, même transférée à 18 camions)

---

## Phase 5. Heuristiques de référence (2 à 4 jours)

**Objectif :** des références qu'un ingénieur minier jugerait crédibles, décrites exactement comme elles sont codées. Les heuristiques n'apprennent rien : les modifier ne demande qu'une réévaluation. C'est le module 6 de ton plan de maîtrise.

### T21. Pelle la plus proche et Plus court chemin

**Indispensable · Impact selon l'option · M16, M25**

- **Fichiers :**
  - `baselines/nearest_policy.py` (lignes 52 à 56) ;
  - `baselines/shortest_path_policy.py` (lignes 82 à 92) ;
  - **et la cause réelle** : `env/mine_env.py`, lignes 256 à 263.
- **Problème :** un cycle se termine à la pelle servie, donc la pelle la plus proche est toujours celle où se trouve le camion. Depuis le parc, c'est la pelle 1. Résultat : 100 % des cycles à la pelle 1, dans les trois régimes.
- **À faire :** décider avec ta directrice entre deux options.
  - **(a)** Garder les règles et les présenter comme des bornes basses, ce que fait l'article v2. **Aucun impact.**
  - **(b)** Les rendre réalistes en prenant la décision **à la verse**. ⚠️ Modifier seulement l'heuristique pour mesurer la distance depuis la verse **ne suffit pas** : dans le simulateur actuel, le camion est déjà revenu à son ancienne pelle, et il ferait un trajet pelle → pelle irréaliste. Il faut changer le simulateur : c'est **T16 bis**, en phase 3, avec impact [RÉENTR].
- **Test :** compter la part des cycles par pelle sur 30 graines. Avec l'option (b), aucune pelle ne doit dépasser, disons, 60 %.
- **Savoir expliquer :** pourquoi la productivité de ces règles plafonne vers 3 900 t/h quel que soit le régime (capacité d'une seule pelle, environ 4 200 t/h).
- [x] Décision prise : option (b), réalisée par T16 bis (16 octobre 2026)
- [x] Fait (commit `157f97b`)

### T22. Docstrings de FIFO et de l'Affectation fixe

**Indispensable · Aucun impact (ou [RÉÉVAL] si tu ajoutes une réaffectation) · Dir. corr. 1**

- **Fichiers :**
  - `baselines/fifo_policy.py` : docstring aux lignes 1 à 13, règle aux lignes 48 à 52 ;
  - `baselines/fixed_policy.py` : docstring aux lignes 1 à 7, règle aux lignes 32 à 35.
- **À faire :** réécrire les docstrings pour décrire exactement la règle.
  - FIFO : « pelle libre le plus tôt, verse libre le plus tôt, sans temps de trajet ». Envisage aussi de la renommer, car ce n'est pas un « premier arrivé, premier servi ».
  - Affectation fixe : « pelle i mod 3, verse i mod 2, sans réaffectation ».

  Si tu fais T16 et que ta directrice tient à la réaffectation de secours, implémente-la ici, avec un test.
- **Savoir expliquer :** pourquoi l'Affectation fixe est si difficile à battre en forte charge.
- [x] Fait (commit `3d9b608` ; nom FIFO gardé et réaffectation non implémentée : à confirmer avec la directrice)

### T23. Règle du temps d'attente minimal des camions (MTWT)

**Recommandé · [RÉÉVAL] · M25**

- **Fichiers :** nouveau `baselines/mtwt_policy.py`, à ajouter dans `run_benchmark.py` (dictionnaire `policies`, lignes 154 à 159).
- **À faire :**
  - pour chaque pelle j, estimer l'heure d'arrivée du camion avec le temps de trajet moyen depuis sa position (`graph.mean_travel_time_minutes`, `graph_model.py`, lignes 160 à 174) ;
  - attente prévue = max(0, disponibilité de la pelle − arrivée) ;
  - choisir la pelle qui minimise l'attente, puis l'arrivée en cas d'égalité ;
  - même logique pour la verse.

  Cette règle combine files et trajets, comme les systèmes commerciaux (Zeng et al., 2022). Avec le simulateur actuel, le temps de trajet vers sa propre pelle vaut 0 : vérifie l'effet de T16 bis sur cette règle.
- **Test :** sur un état construit à la main (pelle 1 libre dans 10 min, pelle 2 libre maintenant mais à 3 min de trajet de plus), vérifier que la règle choisit la pelle 2.
- **Savoir expliquer :** en quoi cette règle diffère de ton FIFO actuel.
- [x] Fait (commit `199ad46` ; 5e heuristique ; file ÷ 3 mais productivité < Affectation fixe)

### T24. Politique fondée sur un programme linéaire

**Optionnel · [RÉÉVAL] · M25**

- **Fichiers :** nouveau `baselines/lp_policy.py`.
- **À faire :**
  - en début de poste, résoudre avec `scipy.optimize.linprog` un programme qui fixe des taux de flux cibles par couple pelle–verse : maximiser le tonnage sous les contraintes de capacité des pelles et d'heures-camions disponibles ;
  - à chaque décision, envoyer le camion vers le couple dont le flux réalisé est le plus en retard sur sa cible.

  C'est le principe de DISPATCH (Soumis et al., 1989).
- **Savoir expliquer :** la différence entre l'étage de planification (programme linéaire) et l'étage d'affectation en temps réel.
- [x] Fait (commit `6639c6b` ; plafond d'utilisation 0,7 choisi sur les graines de sélection)

---

## Phase 6. Récompense et protocole d'expérience (3 jours)

**Objectif :** un protocole qui permet des conclusions au niveau des algorithmes et une vraie évaluation sans réentraînement. Ce sont les modules 3, 5 et 6 de ton plan de maîtrise.

### T25. Module de récompense séparé

**Indispensable · [MESURE] · M10, M9**

Ce n'est qu'un déplacement de code sans changement de calcul : le test T4 doit rester vert sur `total_reward`. Tu peux donc faire cette tâche dès la phase 2.

- **Fichiers :** nouveau `simulation/reward.py` et `env/mine_env.py` (`_compute_reward`, lignes 346 à 373 ; distance utilisée pour le coût, lignes 196 à 200).
- **À faire :**
  - déplacer le calcul dans une fonction pure qui renvoie la récompense et ses quatre termes ;
  - écrire la formule de chaque terme dans la docstring (unité, normalisation) ;
  - renvoyer les termes dans `info` pour pouvoir les tracer.
- **Test :** la somme pondérée des quatre termes égale la récompense renvoyée par `step`. Sur un poste d'Affectation fixe en nominal, le terme de distance pèse environ −19.
- **Savoir expliquer :** les formules de la section 5.5 de l'article v2 sans les regarder ; pourquoi la récompense cumulée vaut à peu près le nombre de cycles ; combien de minutes représente γ = 0,99 par décision avec 12 puis 18 camions.
- [x] Fait (`simulation/reward.py`, valeurs inchangées, termes dans `info` et les KPI ; coût d'ATTENDRE inchangé : décision à prendre avec la directrice)

### T25 bis. Réglage des hyperparamètres documenté

**Indispensable (au moins la décision) · [RÉENTR] si tu fais le balayage · M12**

- **Fichiers :** nouveau `experiments/tune_hyperparams.py` ; `rl/train_ppo.py` (lignes 65 à 80) ; `rl/train_dqn.py` (lignes 65 à 81) ; dans l'article v2, le paragraphe « Sensibilité aux hyperparamètres » (balise violette, vers la ligne 1361).
- **Problème :** l'article dit que des essais préliminaires sur le coefficient d'entropie de PPO (0 ; 0,01 ; 0,02 ; 0,05) ont conduit à 0,02, mais aucun script du projet ne contient ces essais. DQN, lui, tourne avec des valeurs par défaut. Un relecteur demandera si PPO a été mieux réglé que les autres.
- **À faire :** décider avec ta directrice entre deux options.
  - **(a) Balayage documenté.** Un script qui teste, pour PPO **et** DQN, un même nombre de configurations, par exemple :
    - PPO : coefficient d'entropie et taux d'apprentissage ;
    - DQN : taux d'apprentissage et fraction d'exploration.

    Le critère de sélection est la productivité moyenne sur des graines **distinctes** de l'évaluation finale (par exemple 20 000 à 20 009). Le résultat devient un tableau chiffré : valeurs testées, régime, critère, meilleure configuration, moyenne ± écart-type. Coût : chaque configuration est un entraînement de 2 h ; réduis le nombre de pas pour le balayage et dis-le.
  - **(b) Retirer la mention** des essais préliminaires, garder les valeurs usuelles de PPO et de DQN, et écrire l'inégalité de réglage dans les limites.

  Fais ce choix **avant la phase 7**, car l'option (a) peut changer les hyperparamètres de la campagne finale.
- **Test (option a) :** relancer le script sur une configuration redonne la même valeur du critère (graines fixées).
- **Savoir expliquer :** pourquoi un réglage fait sur les graines d'évaluation biaise la comparaison ; pourquoi un budget de réglage inégal fausse « PPO contre DQN ».
- [x] Décision prise : option (a), balayage lancé le 16 octobre 2026 (résultats à reporter dans le journal)
- [x] Fait : script `experiments/tune_hyperparams.py` écrit et testé (commit `62a4be3`), pas encore lancé

### T26. Graines d'entraînement et d'évaluation séparées

**Indispensable · [RÉENTR] · M7**

- **Fichiers :** `experiments/scenarios.py` (ligne 31) et `experiments/run_benchmark.py`, où `seeds[0]` sert à l'entraînement (lignes 174, 203, 230 et 258). Concernés aussi :
  - `env/mine_env.py`, lignes 114 à 116 : sans graine, `reset` utilise un compteur 43, 44, 45… ;
  - `rl/train_q_learning.py`, ligne 72, et `rl/train_sarsa.py`, ligne 79 : `seed + ep`.
- **Problème :** il n'y a qu'une graine d'entraînement (42), et les graines d'évaluation 42 à 71 sont les premiers épisodes d'entraînement. De plus, l'entraînement est déterministe : relancer avec 42 redonne le même modèle.
- **À faire :**
  - `train_seeds = [1, 2, 3, 4, 5]` et `eval_seeds = range(10000, 10030)` ;
  - un dossier par (agent, régime, graine d'entraînement) ;
  - une colonne `train_seed` dans le CSV.
- **Test :** aucune graine d'évaluation n'est utilisée pendant un entraînement. Pour le vérifier, journalise les graines passées à `reset`, y compris celles du compteur interne et de `seed + ep`.
- **Savoir expliquer :** la différence entre variabilité d'entraînement et variabilité d'évaluation ; pourquoi on ne peut pas dire « PPO est meilleur que DQN » avec un seul entraînement chacun.
- [x] Fait (`afcc713`) : graines d'entraînement déplacées (1 à 5, épisodes s × 10⁶) au lieu des graines d'évaluation, qui restent 42 à 71 (voir `seeds.py`)

### T27. Évaluation sans réentraînement

**Indispensable · [MESURE] · M1**

- **Fichiers :** `experiments/run_benchmark.py` (`model_key`, ligne 106 ; scénarios non vus, lignes 326 et 327).
- **À faire :**
  - pour chaque régime perturbé, évaluer à la fois les agents entraînés dans ce régime et les agents nominaux (le paramètre `model_scenario` existe déjà) ;
  - ajouter au CSV une colonne `mode`, « spécialisé » ou « transféré ».

  Après T18, cela fonctionne aussi pour PPO et DQN en forte charge.
- **Faisable dès maintenant** avec les modèles v1 en fortes pannes : appelle `run_benchmark(..., output_dir=Path("data/results_v1_article"), model_scenario="nominal")`, en écrivant les résultats dans un **autre** fichier.
- **Test :** avec les modèles v1, sur les **30 graines d'évaluation** (42 à 71), l'agent PPO nominal en fortes pannes doit donner une productivité moyenne proche de 3 926 t/h (tableau 14 de l'article v2).
- **Savoir expliquer :** pourquoi l'article v1 mesurait l'adaptation et non la robustesse.
- [x] Fait (`da45df5`) : colonne `mode` ; test du plan fait sur le code d'avant T13 (PPO v1 nominal en fortes pannes = 3 926,4 t/h), impossible sur le code actuel depuis T18

### T28. Fichiers de configuration réellement utilisés

**Recommandé · [MESURE] · cohérence code/article**

- **Fichiers :** `configs/` et `experiments/scenarios.py`.
- **Problème :** le dossier `configs/` **existe déjà**, avec trois fichiers YAML (`ppo_hyperparameters.yaml`, `dqn_hyperparameters.yaml`, `experiment_config.yaml`), et l'article v1 l'annonce. Mais **aucun script ne les lit** : les valeurs sont recopiées en dur dans `scenarios.py` et `train_*.py`. Elles peuvent donc diverger sans que tu le voies, comme la version de Python déjà fausse (T3).
- **À faire :** décrire régimes, hyperparamètres et graines dans ces fichiers YAML. Chaque script lit sa configuration et la recopie dans le dossier de résultats.
- **Savoir expliquer :** comment un relecteur relancerait une expérience précise avec une seule commande.
- [x] Fait (`0ab2031`) : `config.py` lit et valide `configs/*.yaml` ; configuration archivée à chaque lancement ; `MINE_RL_CONFIG_DIR` pour rejouer

### T29. Parallélisation et mesure du temps de calcul

**Recommandé · [MESURE] pour l'évaluation, [RÉENTR] pour PPO à plusieurs environnements · M28**

- **Fichiers :** `experiments/run_benchmark.py`, `rl/train_ppo.py` et `rl/train_dqn.py` (`make_vec_env(..., n_envs=1)`, lignes 51 à 63).
- **À faire :**
  - exécuter les 30 réplications en parallèle (`multiprocessing.Pool`) ;
  - essayer `SubprocVecEnv` avec 4 environnements pour PPO ;
  - journaliser la durée de chaque entraînement et de chaque évaluation, et le temps moyen par décision pour 12, 18 et 24 camions.
- **Test :** les résultats parallèles sont identiques aux résultats séquentiels, graine par graine.
- **Savoir expliquer :** pourquoi changer le nombre d'environnements de PPO change le modèle obtenu, alors que paralléliser l'évaluation ne change rien.
- [x] Fait

### T30. Agrégation sur plusieurs graines et courbes d'apprentissage

**Recommandé · [MESURE] · M7, N7, Dir. plan**

- **Fichiers :** nouveau `experiments/aggregate_seeds.py`, `experiments/plot_convergence_deeprl.py`, `experiments/plot_policy_value_loss.py` et `experiments/plot_td_error.py`.
- **À faire :**
  - moyenne interquartile (IQM) et intervalles bootstrap stratifiés sur les 5 graines (bibliothèque `rliable`) ;
  - courbes de convergence en moyenne ± intervalle sur les graines ;
  - **N7 :** les figures de pertes et d'exploration viennent aujourd'hui de réentraînements séparés (`train_losscurve.py`, `train_diagnostics*.py`), pas des modèles évalués. Fais-les produire par les **mêmes** entraînements que ceux de la campagne T31, puis supprime ces scripts et les cibles correspondantes du `Makefile`.
- **Savoir expliquer :** ce qu'est la moyenne interquartile et pourquoi Agarwal et al. (2021) la recommandent.
- [x] Fait

---

## Phase 7. Relancer, vérifier, nettoyer (1 à 2 semaines, surtout du calcul)

### T31. Campagne finale et mise à jour de l'article

**Indispensable · nouveaux résultats**

- **Avant de lancer :**
  - [ ] tous les tests sont verts ;
  - [ ] chaque fichier modifié a été relu ligne par ligne ;
  - [ ] ta directrice a validé les choix de T15, T16, T16 bis/T21, T18, T24 et T25 bis ;
  - [ ] T8 est en place et les modèles v1 ne peuvent plus être rechargés par erreur ;
  - [ ] un entraînement court (20 000 pas, 1 graine) passe de bout en bout.
- **Coût à prévoir :** 2 agents profonds × 3 régimes × 5 graines = 30 entraînements, soit environ 34 h pour PPO et 28 h pour DQN en séquentiel. Ce sera moins si T29 permet d'en lancer plusieurs à la fois, ou si T18 permet de n'entraîner qu'en nominal. L'ablation à 3 graines par valeur de w₄ ajoute environ 50 h : décide si elle porte sur PPO seul.
- **À faire :**
  - lancer la campagne dans `data/results_v2/` ;
  - relancer T6, T7 et T12 bis ;
  - recopier chaque chiffre de l'article depuis les nouveaux fichiers, jamais à la main ;
  - **réécrire aussi le texte** : les phrases qui décrivent l'implémentation actuelle deviennent fausses après les phases 3 à 6. Le tableau ci-dessous dit, pour chaque partie de l'article v2, quelles tâches la modifient ;
  - supprimer les balises orange de l'article une à une ; avant la soumission, supprimer aussi toutes les autres balises (`\revmarksfalse`) et le tableau récapitulatif des modifications en fin d'article.

**Correspondance entre l'article v2 et les tâches.** Les numéros de ligne sont approximatifs ; fie-toi aux étiquettes `\label`.

| Partie de l'article v2 | Étiquettes (ligne ≈) | Tâches qui la modifient | Ce qui change |
|---|---|---|---|
| Simulateur et paramètres | `sec:simulator`, `tab:simparams`, `fig:simulation` (518 à 546) | T13, T14, T15, T16, T16 bis | Flux aléatoires, carburant, ordre des files, pannes de pelles, lieu de la décision |
| Processus de décision, état | `sec:smdp`, `tab:state`, `tab:mdp`, `fig:mdp` (568 à 646) | T16 bis, T17, T18 | Taille 15 + 11N et les 54 composantes constantes (lignes 577 et 579) ; intervalles de 1,96 et 1,36 min entre décisions (ligne 620) |
| Récompense | `sec:reward` (652) | T25 | Formules à vérifier contre le nouveau `simulation/reward.py` |
| Agents, hyperparamètres | `sec:agents`, `tab:hyper` (678 à 698), paragraphe « Sensibilité » (≈ 1357) | T25 bis, T26, T28, T29 | Graines, balayage ou retrait de la mention, nombre d'environnements |
| Heuristiques | `sec:baselines-def`, `tab:baselines` (817 à 830) | T16 bis, T21, T22, T23, T24 | Description des règles, nouvelles règles MTWT et programme linéaire |
| Calcul | `sec:compute`, `fig:architecture` (845 à 853) | T9, T29 | Latence, temps de calcul, parallélisation |
| Scénarios, protocole, métriques | `sec:scenarios` à `sec:proto` (881 à 957) | T10, T11, T13, T26, T27 | Définition des attentes et de l'utilisation, graines communes, graines d'entraînement et d'évaluation |
| Productivité | `sec:productivity`, `tab:productivity`, `tab:cycles_reward` (989 à 1035) | toutes | Tous les chiffres |
| Attente | `sec:waiting`, `tab:waiting`, `fig:waiting` (1061 à 1098) | T10, T13 | Décomposition file, panne, ATTENDRE |
| Comportement | `sec:behaviour`, `tab:behaviour`, `fig:behavioral` (1103 à 1143) | T12 bis, T16 bis, T19 | 100 % à la pelle 1, part d'ATTENDRE |
| Pareto | `sec:pareto`, `tab:pareto`, `fig:pareto` (1149 à 1175) | T7 | Un front par régime |
| Robustesse et transfert | `sec:robustness`, `tab:robustness`, `tab:transfer`, `fig:heatmap` (1180 à 1252) | T7, T18, T27 | Transfert de PPO et DQN en forte charge |
| Statistiques | `sec:stats`, `tab:stats` (1257 à 1268) | T6 | Toutes les valeurs, interaction F(14, 609) |
| Stabilité, convergence | `sec:stability`, `sec:convergence`, `fig:convergence` (1289 à 1321) | T26, T30 | Cinq graines, IQM, courbes moyennes |
| Ablation | `sec:ablation`, `tab:ablation` (1327 à 1338) | T31 (ablation multi-graines) | Plusieurs graines par valeur de w₄ |
| Discussion, hypothèses | `sec:discussion`, `tab:hypotheses` (1368 à 1388) | toutes | Revérifier H1 à H4 : les conclusions peuvent changer |
| Durabilité | `sec:sustainability`, `tab:efficiency`, `tab:fuel` (1421 à 1450) | T14 | Valeurs de carburant absolues |
| Limites | `sec:limitations` (1479) | T14, T15, T16, T16 bis, T25 bis | Retirer les limites corrigées, garder les autres |
| Résumé, conclusion | (début, 1500) | toutes | Chiffres et formulation des résultats principaux |
- **Nettoyage (N10) :**
  - retirer des commentaires les références au mémoire (« Section 4.3 », « Tableau 4.8 », « Eq. 3.7 ») ;
  - corriger le commentaire « MF = 4,33 » de `scenarios.py` (ligne 48), qui est un ratio camions/pelles et non un facteur d'appariement ;
  - écrire un `README` qui explique comment reproduire chaque tableau et chaque figure.
- **Savoir expliquer :** pour chaque tableau de l'article, quel script et quel fichier l'ont produit.
- [ ] Fait

---

## Calendrier proposé

1. **Semaine 1 :** phases 0, 1 et 2 (T1 à T12 bis), plus T25 et T27 pour les fortes pannes. Tes résultats actuels restent valables ; refais les statistiques de l'article avec ton propre code.
2. **Semaine 2 :** phase 3 (T13 et T14 ; T15, T16 et T16 bis seulement si validés). Prévois la réunion avec ta directrice **en début de semaine** pour trancher T15, T16, T18, T21 et T24.
3. **Semaine 3 :** phases 4 et 5 (T17 à T24).
4. **Semaine 4 :** phase 6 (T25 bis à T30), puis courts entraînements d'essai pour vérifier que tout fonctionne.
5. **Semaines 5 et 6 :** campagne finale (T31), pendant que tu rédiges les fiches des modules 4 et 5 de ton plan de maîtrise.
6. **Semaine 7 :** mise à jour de l'article v2 avec les nouveaux résultats, puis envoi à ta directrice.

## Décisions à prendre avec la directrice

| Tâche | Question | Options |
|---|---|---|
| T15 | Moteur à événements discrets ? | Réécrire / décrire comme approximation |
| T16 | Pannes de pelles et nouveau régime ? | Oui / non (limite écrite) |
| T18 | Quelle observation de taille fixe ? | Agrégats (recommandé) / remplissage / rester spécialisé |
| T21 + T16 bis | Pelle la plus proche et Plus court chemin | (a) bornes basses / (b) décision à la verse |
| T22 | Réaffectation de secours de l'Affectation fixe | Corriger le texte / l'implémenter (avec T16) |
| T24 | Politique par programme linéaire | Oui / non |
| T25 bis | Réglage des hyperparamètres (M12) | Balayage documenté pour PPO et DQN / retirer la mention |
| T31 | Ablation sur w₄ | PPO seul / PPO et DQN |

## Suivi

| Phase | Tâches | Effet sur les résultats | Statut |
|---|---|---|---|
| 0. Préparation | T1 à T4 | aucun | fait (T2 : reste la copie externe) |
| 1. Résultats et statistiques | T5 à T9 | statistiques seulement | fait |
| 2. Indicateurs | T10 à T12 bis | nouveaux indicateurs | fait |
| 3. Simulateur | T13 à T16 bis | invalide les résultats | fait (T15 et T16 bis faits ensemble ; T16 écrit comme limite) |
| 4. Observation et actions | T17 à T20 | invalide les résultats | fait |
| 5. Heuristiques | T21 à T24 | invalide les résultats | fait (T21 par T16 bis) |
| 6. Récompense et protocole | T25 à T30 (dont T25 bis) | invalide les résultats | fait (T25 bis : balayage lancé le 16 octobre ; T29 : n_envs = 1 gardé) |
| 7. Campagne finale | T31 | nouveaux résultats | à faire |
