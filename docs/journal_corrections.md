# Phase 0. Préparation (½ jour)

# Journal des corrections du code

Une entrée par tâche du plan (`docs/plan_correction_code.md`) : ce que j'ai changé, pourquoi, ce que le test vérifie, ce que j'ai appris.

---

## T1 — Figer la version actuelle (fait le 7 octobre 2026)

**Ce que j'ai fait :** tag Git `v1-article` posé sur le commit `cdd9883` (7 octobre 2026), puis création de la branche `corrections-v2` où se fait tout le travail de correction.

**Pourquoi :** garder un point fixe vers lequel revenir à tout moment pour retrouver le code exact qui a produit les résultats de l'article v1. On ne corrige jamais directement cette version.

**Vérification :** `git tag -n1` affiche `v1-article` ; `git branch --show-current` affiche `corrections-v2`.

**Note :** le commit `37fed45` (épuration des commentaires et docstrings) est postérieur au tag. Il ne touche pas au calcul, mais les numéros de ligne du plan (relevés sur `424ff39`) peuvent être décalés : se fier aux noms de fonctions.

**Questions « Savoir expliquer » à préparer :**
- [x] différence entre un tag et une branche : une branche est dynamique et change avec les commits, tandis qu'un tag est statique et reste fixe et figé sur un commit ;
- [x] comment revenir à `v1-article` : `git checkout v1-article` pour consulter ou exécuter le code de l'article. Cela place en mode *detached HEAD* : un commit fait dans cet état n'appartient à aucune branche et peut être perdu. Pour travailler à partir de cette version : `git switch -c <nouvelle-branche> v1-article` ;
- [x] pourquoi on ne travaille jamais sur la version qui a produit les résultats publiés : le tag n'empêche pas de modifier les fichiers, il marque seulement un commit. On ne travaille pas sur cette version pour qu'elle reste intacte : (1) si un relecteur demande le code exact qui a produit un tableau, je dois pouvoir le lui donner et il doit retrouver les mêmes chiffres ; (2) j'ai besoin de la v1 comme point de comparaison pour mesurer ce que chaque correction change (test T4).

---

## T2 — Geler les résultats v1 (fait le 7 octobre 2026)

**Ce que j'ai fait :** déplacement de `data/results/` vers `data/results_v1_article/` (135 fichiers, 124 Mo), puis copie de ce dossier hors du projet. `data/results/` est vide ; les nouveaux résultats iront dans `data/results_v2/`.

**Pourquoi :** `data/` est ignoré par Git, donc ces résultats n'existent nulle part ailleurs. Sans copie, une erreur de manipulation les ferait perdre définitivement.

**Incident (8 octobre 2026) :** le dossier `data/results_v1_article/` avait disparu du projet ; je l'ai restauré depuis la copie externe. Vérification après restauration : 135 fichiers, 124 Mo, `benchmark_results.csv` à 1 441 lignes (en-tête + 1 440 lignes, le doublon traité en T5).

**Emplacement de la copie externe :** _disque dur externe_

**Attention pour la suite :** une vingtaine de scripts écrivent encore `data/results/` en dur. Sans passer explicitement l'archive, `run_benchmark` ne trouve aucun modèle et réentraîne tout.

**Questions « Savoir expliquer » à préparer :**
- [x] quels fichiers de `results_v1_article` ont servi à chaque tableau de l'article :

  | Tableau ou figure (article v2) | Fichier source |
  |---|---|
  | `tab:productivity`, `fig:productivity`, `tab:cycles_reward` | `benchmark_results.csv` |
  | `tab:waiting`, `fig:waiting` | `benchmark_results.csv` |
  | `tab:pareto`, `fig:pareto` | `benchmark_results.csv` |
  | `tab:stats` | `benchmark_results.csv` |
  | `tab:efficiency`, `tab:fuel` | `benchmark_results.csv` |
  | `tab:robustness`, `fig:heatmap` | `robustness_ratios.csv` (v1) ; recalculés depuis `benchmark_results.csv` en v2 |
  | `tab:behaviour`, `fig:behavioral` | `action_distribution.csv` (10 graines seulement) ; le tableau v2 vient de l'audit, pas de mon code (T12 bis) |
  | `tab:transfer` | évaluation des modèles nominaux en fortes pannes, faite hors de mes scripts (à refaire en T27) |
  | `fig:convergence` | `convergence_deeprl_powerbi.csv` et `convergence_qlearning_sarsa_powerbi.csv` |
  | `tab:ablation` | `ablation/ablation_w4.csv` |
  | — | `sample_efficiency.csv` : utilisé dans la v1, plus cité dans la v2 |

  Attention : `benchmark_results.csv` contient chaque ligne en double (T5). Les valeurs de la v2 ont été recalculées après dédoublonnage par l'audit ; T6, T7 et T12 bis les referont avec mon propre code.
- [x] pourquoi les modèles DQN datent de mai (`dqn_nominal`, 30 mai 2026) et les modèles PPO d'août (`ppo_nominal`, 5 août 2026) — lien avec N11 et T8 :
  - DQN a été entraîné les 30 et 31 mai, juste après le commit `b13b2e1` (« Correctifs RL : observation, récompense… », 30 mai). PPO a été réentraîné les 5 et 6 août ; l'ancien PPO de mai existe encore (`ppo_nominal_asymétrique`, 31 mai). La raison du réentraînement de PPO en août n'est tracée nulle part (pas de commit, pas de métadonnées) ;
  - vérification N11 (`git log` sur `env/` et `simulation/` entre fin mai et début août) : un seul commit entre les deux entraînements, `0348a8b` (16 juin). Il change la valeur **par défaut** de w4 (1,0 → 0,3), mais les deux entraînements passaient `reward_weights=scenario.reward_weights` = (1,0 ; 0,1 ; 0,05 ; 0,3) explicitement, donc ce défaut n'a pas servi. Le calcul de `_compute_reward` n'a pas changé (docstring seulement) ; le reste du commit est du refactoring (observation, temps de trajet) annoncé à sortie identique ;
  - conclusion : d'après Git, DQN et PPO ont été entraînés avec la même récompense. Mais je ne peux pas le **prouver** : DQN a été entraîné à 20 h 02, après le commit de 11 h 41, et d'éventuelles modifications non commitées ne laissent aucune trace. C'est exactement ce que T8 corrige (un `metadata.json` avec le commit et les poids à côté de chaque modèle).

---

## T3 — Environnement reproductible (fait le 8 octobre 2026, commit `a22c42b`)

**Ce que j'ai changé :**
- `requirements.txt` : 15 paquets avec leur version exacte (stable-baselines3 2.8.0, torch 2.5.1, gymnasium 1.2.3, numpy 2.4.6, cloudpickle 3.1.2, pandas, scipy, statsmodels, networkx, matplotlib, seaborn, tqdm, rich, pytest, pyyaml) ;
- `configs/experiment_config.yaml` : Python 3.10 → 3.11.11, Gymnasium 0.26 → 1.2.3, SB3 2.x → 2.8.0.

**Pourquoi :** les versions viennent du `system_info.txt` des six modèles de l'article (PPO et DQN, trois régimes) : elles sont identiques partout. L'ancien YAML annonçait des versions fausses. seaborn (`robustness_analysis.py`), tqdm et rich (`progress_bar=True` dans `train_ppo.py` et `train_dqn.py`) sont nécessaires même si le plan ne les citait pas.

**Ce que le test vérifie :** dans un environnement neuf (`rl-test`), `pip install -r requirements.txt` puis `pip install -e .` suffisent pour que les 6 tests passent (`pytest`, 9 octobre 2026).

**Ce que j'ai appris :**
- deux fautes de frappe (`network` au lieu de `networkx`, `pyttest` au lieu de `pytest`) auraient fait échouer l'installation. Toujours vérifier le nom exact d'un paquet sur pypi.org (risque de *typosquatting*) ;
- sans `pip install -e .`, `pytest` ne trouve pas le paquet (`ModuleNotFoundError`). `python -m ...` marchait seulement parce que je lançais la commande depuis la racine du projet. Ordre d'installation : d'abord `requirements.txt`, ensuite `pip install -e .`.

**Décision sur `pyproject.toml` :** `requirements.txt` fait foi pour reproduire l'article. `pyproject.toml` ne déclare que networkx et matplotlib ; il sera complété au moment du README (T31).

**Questions « Savoir expliquer » à préparer :**
- [x] pourquoi une version différente de PyTorch peut changer un modèle entraîné avec la même graine : Fixer une graine aléatoire (seed) garantit la reproductibilité uniquement au sein d'une même version de PyTorch et sur un environnement matériel et logiciel identique, si la version de PyTorch change, le modèle final ou ses performances divergeront, car la documentation officielle de PyTorch précise qu'aucune garantie de reproductibilité n'est maintenue d'une version à une autre. Deux mécanismes l'expliquent :
  1. **L'initialisation** : à la création de `PPO(...)`, les poids du réseau sont tirés au hasard. Si une nouvelle version tire ces nombres dans un autre ordre ou avec une autre méthode, le réseau de départ est différent malgré la même graine ;
  2. **Les calculs** : l'addition de nombres à virgule flottante n'est pas associative ((a + b) + c peut différer de a + (b + c) à la dernière décimale), et l'ordre des opérations dans les produits matriciels change d'une version à l'autre. Une différence infime s'amplifie au fil des milliers de mises à jour du gradient, et le jour où elle fait choisir une autre action à l'agent, toute la trajectoire suivante change (autres états, autres aléas, autres données d'apprentissage).


---

## T4 — Tests de non-régression
*Indispensable · Aucun impact*

**Date :** 9 octobre 2026

**Prédiction (avant de modifier) :** les chiffres seront les mêmes, parce que jusqu'à maintenant le simulateur n'avait pas été modifié : depuis le tag `v1-article`, le seul commit de code est `37fed45`, qui ne touche que les commentaires et les docstrings. Ce que je ne savais pas encore : le CSV de référence date du 7 août 2026, et le commit `424ff39` (12 septembre) a modifié `run_benchmark.py` et `scenarios.py` après sa production. La vraie question était donc : le code d'aujourd'hui redonne-t-il les chiffres produits par le code du 7 août ?

**Ce que j'ai changé :** aucun fichier du simulateur. J'ai ajouté le fichier `tests/test_regression.py`.

**Pourquoi :** faire un état des lieux avant de modifier le code, et installer une alarme. Le test compare les résultats actuels aux résultats de référence de l'article v1, pour détecter toute modification qui changerait un chiffre alors qu'elle n'aurait pas dû.

**Test prévu par le plan :** Affectation fixe et FIFO, régime nominal, graines 42 à 46 : `productivity_tph` et `total_reward` égaux à ceux de `data/results_v1_article/benchmark_results.csv` à 10⁻⁶ près (première occurrence). Doit rester vert en phases 1 et 2, devenir rouge en phase 3 (mettre alors à jour les références et noter pourquoi).

**Ce que le test vérifie :** pour Fixed et FIFO, en régime nominal, graines 42 à 46 (10 cas), que `productivity_tph` et `total_reward` calculés par le code actuel sont égaux, à 10⁻⁶ près, à ceux de `data/results_v1_article/benchmark_results.csv` (première occurrence de chaque triplet). Si le CSV est absent, le test est ignoré au lieu de planter.

**Résultat comparé à ma prédiction :** 1 cas sur 10 vérifié à la main (Fixed, graine 42) : `productivity_tph = 4077.5` et `total_reward = 210.9715212200733`, identiques au CSV. Prédiction juste pour ce cas. Puis test automatique : **10 passed** sur les 10 cas (Fixed et FIFO, graines 42 à 46), en environ 11 s ; suite complète **16 passed** (6 anciens tests + 10 nouveaux). Prédiction juste : ni `424ff39` ni `37fed45` n'ont changé le régime nominal pour Fixed et FIFO.

**Preuve que l'alarme fonctionne (sabotage volontaire) :** référence décalée de +1 (`ref_prod + 1`) → **10 failed**, avec des messages du type `assert 4077.5 == 4078.5 ± 1.0e-06` ; sabotage annulé → de nouveau **10 passed**. Un test qui n'a jamais échoué ne prouve rien.

**Robustesse au dossier de lancement :** le chemin du CSV est construit à partir de `Path(__file__)`, donc le test passe aussi lancé depuis `tests/` (10 passed). Avec un chemin relatif, il aurait été ignoré en silence (`skipped`) au lieu de signaler un problème.

**Ce que j'ai appris :**
- lire un traceback de bas en haut : la dernière ligne dit quoi (`AttributeError … Did you mean: 'current_truck_idx'?`), les lignes au-dessus disent où ;
- un résultat de référence a une date : le code a pu changer entre sa production (7 août) et le tag (7 octobre). Il faut comparer à ce qui a produit les chiffres, pas seulement au dernier tag ;
- `from memoire_master_rl_logistique.env import MineEnv` fonctionne parce que `env/__init__.py` réexporte `MineEnv` ;
- `AM` dans `git status` : fichier ajouté à l'index (`git add`) puis modifié depuis. Il faut refaire `git add` avant de commiter, sinon on commite l'ancienne version ;
- `python script.py` ajoute le dossier du script à `sys.path`, `python -m` ajoute le dossier courant, `pytest` ajoute `tests/` : d'où l'importance de `pip install -e .` ;
- un adaptateur (`policy_fn`) traduit une interface en une autre : `run_single_episode` appelle `policy_fn(obs, info, env)`, alors que la politique attend `predict(obs, info, truck_location=...)`. `make_policy_fn` est une fonction qui fabrique une fonction : la fonction intérieure « se souvient » de la politique (fermeture, *closure*) ;
- `@pytest.mark.parametrize` empilés donnent toutes les combinaisons (2 politiques × 5 graines = 10 tests) ; `pytest.approx(..., abs=1e-6)` compare des flottants avec une tolérance et affiche obtenu, attendu et tolérance en cas d'échec ;
- `csv.DictReader` lit tout comme du texte : il faut convertir (`int(row["seed"])`, `float(...)`). Le `return` dans la boucle prend automatiquement la première occurrence, donc le doublon du CSV n'est jamais lu ;
- la productivité avance par paliers de 140 t / 8 h = **17,5 t/h** (4077,5 = 233 cycles ; 4042,5 = 231 cycles) : deux épisodes différents peuvent avoir exactement la même productivité. `total_reward` est beaucoup plus fin (il dépend de chaque minute d'attente et de chaque kilomètre) : c'est pour ça que le test compare les deux ;
- `ruff check --fix` lancé sur tout le paquet a modifié 11 fichiers de code (ordre des imports, imports inutilisés `mpatches`, `FancyArrowPatch` et `Scenario`, une f-string sans variable). Changements sans effet sur les calculs (16 tests verts), mais ils ne relèvent pas de T4 : commit séparé. Leçon : limiter un outil automatique au fichier qu'on travaille (`ruff check tests/test_regression.py --fix`).

**Commit :** _(à compléter : hash du commit « T4 : test de non-régression… »)_

**Questions « Savoir expliquer » :**
- [x] ce qu'est un test de non-régression : c'est un test automatique qui permet de vérifier que les résultats restent les mêmes après **chaque** modification du code. Il compare les résultats actuels à des **valeurs de référence enregistrées** (ici le CSV de l'article v1), pour détecter une modification qui change un résultat alors qu'elle n'aurait pas dû. Quand une modification **doit** changer les résultats (phase 3), le test devient rouge volontairement : on met alors à jour les références en notant pourquoi ;
- [x] pourquoi la même graine doit redonner exactement le même résultat (quelle ligne de `mine_env.py` le garantit ?) : la ligne 116 de `mine_env.py`, dans `reset` : `self.rng = Random(effective_seed)`. Tous les tirages aléatoires du simulateur (pannes, durées de réparation, temps de trajet, chargements, déversements, dégradation des pistes) passent par ce `self.rng`, et deux générateurs `Random` créés avec la même graine produisent exactement la même suite de nombres. Seconde condition : la politique doit être déterministe. C'est le cas de Fixed et FIFO, qui ne tirent rien au hasard ; donc mêmes décisions + mêmes tirages = même épisode, au bit près.

---

# Phase 1. Résultats et statistiques (2 jours)

## T5 — Fichier de résultats sans doublons
*Indispensable · [MESURE] · M3*

**Date :** 9 et 10 octobre 2026

**Prédiction (avant de modifier) :** les résultats seront toujours les mêmes, parce que c'est juste l'organisation du code qui change : seule la façon d'écrire le fichier change, pas le calcul. Le test T4 doit donc rester vert.

**Ce que j'ai changé :**
- `experiments/run_benchmark.py` : nouvelle fonction `write_results`, placée avant `run_all_benchmarks` ; l'ancien bloc d'écriture (nom fixe `benchmark_results.csv`, mode `"a"`, en-tête écrit seulement la première fois) est remplacé par un seul appel à `write_results` après la boucle des scénarios. Si aucun résultat n'est obtenu, `run_all_benchmarks` lève `ValueError("Aucun résultat n'a été obtenu.")` (choix : échouer bruyamment plutôt que continuer en silence, car un benchmark vide est presque toujours une erreur, par exemple un nom de régime mal orthographié). Docstring du module mise à jour ;
- `tests/test_results_writer.py` : nouveau fichier, 4 tests.

Spécification de la nouvelle fonction :

`write_results(rows, output_dir, run_id=None)` → chemin du fichier écrit
- refuse une liste vide (`ValueError`), avant toute écriture ;
- refuse un triplet (scénario, politique, graine) présent deux fois (`ValueError`, le message nomme le triplet), avant toute écriture ;
- si `run_id` est absent (`None`), il prend la date et l'heure au format `AAAAMMJJ_HHMMSS` (secondes comprises ; le plan proposait `AAAAMMJJ_HHMM`) ;
- écrit un nouveau fichier `benchmark_<run_id>.csv`, avec `run_id` en première colonne ;
- ouvre le fichier en mode `"x"` : si le fichier existe déjà (même `run_id`), `FileExistsError`, et rien n'est écrasé.

**Pourquoi :**
- **un fichier par lancement** : avec le mode `"a"`, deux lancements ont produit 1 440 lignes au lieu de 720 (le 2ᵉ ajoute ses 720 lignes aux 720 déjà présentes). Passer simplement en `"w"` sur le même nom supprimerait le doublon mais **effacerait** les résultats précédents, peut-être ceux qui ont produit un tableau de l'article. Avec un fichier par lancement, rien n'est jamais effacé et on sait quel lancement a produit quel tableau ;
- **un contrôle des doublons avant écriture** : dans le code actuel, un doublon à l'intérieur d'un lancement est difficile à produire. C'est une assurance contre le futur (graines lues dans un YAML en T28, où une faute de frappe suffit ; graines d'entraînement et mode spécialisé/transféré en T26 et T27). On vérifie juste avant d'écrire parce que c'est le dernier endroit où l'on peut arrêter l'erreur : un doublon ne fait pas planter les statistiques, il les rend faussement précises ;
- **séparer ce qui calcule de ce qui écrit** : tester l'écriture en lançant `run_all_benchmarks` aurait entraîné tous les modèles (`data/results/` est vide), soit des heures de calcul. Sortie dans sa propre fonction, l'écriture se teste avec 3 fausses lignes en une fraction de seconde.

**Test prévu par le plan :** lancer deux fois un mini-benchmark (2 graines) et vérifier qu'on obtient deux fichiers et aucun doublon.

**Ce que le test vérifie :** `tests/test_results_writer.py`, dans un dossier temporaire (`tmp_path`) :
1. deux appels avec deux `run_id` différents donnent deux fichiers différents ; le fichier relu contient `len(ROWS)` lignes, toutes avec `run_id == "a"` ;
2. une liste contenant deux fois le même triplet lève `ValueError` ;
3. deux appels avec le même `run_id` : le second lève `FileExistsError` ;
4. une liste vide lève `ValueError` **et** aucun fichier n'est créé (`list(tmp_path.iterdir()) == []`).

Le test prévu par le plan (mini-benchmark lancé deux fois) n'a pas été fait tel quel : il aurait déclenché l'entraînement des modèles. Les tests portent sur `write_results` ; le branchement dans `run_all_benchmarks` a été vérifié par relecture.

**Résultat comparé à ma prédiction :** prédiction juste. T4 est resté vert ; suite complète **20 passed** (16 + 4).

**Conséquence connue :** `plot_benchmark_comparisons.py`, `robustness_analysis.py` et `statistical_tests.py` lisent encore `data/results/benchmark_results.csv` en dur. Ils ne trouveront pas les nouveaux fichiers ; ils seront corrigés en T6 et T7.

**Ce que j'ai appris :**
- **valider tout avant d'écrire quoi que ce soit** : mon premier contrôle de la liste vide était à l'intérieur du `with open(..., "x")`. Le fichier était donc déjà créé quand l'erreur survenait, et il restait un fichier vide sur le disque, qui bloquait ensuite tout nouvel essai avec le même `run_id`. En plus, `rows[0]` levait `IndexError` avant même d'arriver au contrôle ; et je testais si la **première ligne** avait des colonnes au lieu de tester si **la liste** était vide ;
- **le mode `"x"` plutôt que `exists()` puis `"w"`** : il crée le fichier ou lève `FileExistsError` en une seule opération, sans intervalle où un autre programme pourrait créer le même fichier ;
- **`if run_id is None`** plutôt que `if not run_id` : dit exactement « si on ne l'a pas fourni » (`not` est aussi vrai pour une chaîne vide) ;
- **un test ne couvre que ce qu'il exécute** : mon appel à `write_results` était indenté **dans** la boucle des scénarios. Un lancement aurait produit 3 fichiers, le nominal présent dans les 3, c'est-à-dire le bug M3 sous une autre forme, avec 20 tests verts, parce qu'aucun test ne lance `run_all_benchmarks`. C'est la relecture qui l'a trouvé ;
- **un test peut se tromper lui-même** : `assert list(tmp_path.iterdir())` exige que le dossier **contienne** quelque chose (une liste vide vaut `False`). Le message `AssertionError: assert []` montrait que le dossier était vide, ce que je voulais ; c'est le test qui était inversé ;
- **deux tests ne doivent pas se contredire** : mon premier test « doublon » faisait le même scénario que le test 1 (deux appels, deux `run_id`) en attendant une erreur ; aucune fonction ne pouvait satisfaire les deux ;
- **un test vide (`pass`) est déclaré réussi** : il ne prouve rien ;
- **relire les imports ajoutés automatiquement** : l'éditeur avait ajouté `from openai.types.responses import response_output_refusal`, inutile, et qui aurait fait planter le fichier de test dans l'environnement `rl-test` (`openai` n'est pas dans `requirements.txt`) ;
- **concevoir l'interface avant d'écrire le code** : nom, paramètres, valeur renvoyée, ce qui est refusé et comment (lever une exception, pas renvoyer `False`). Passer `run_id` en paramètre facultatif permet au test de le fixer (`"a"`, `"b"`) au lieu de dépendre de l'heure.

**Commit :** `187125d`

**Questions « Savoir expliquer » :**
- [x] pourquoi un doublon ne change pas la moyenne mais divise la largeur d'un intervalle de confiance par environ 1,4 (√2) : parce que dupliquer **chaque ligne** du jeu de données ne modifie pas le centre (la moyenne), mais double artificiellement la taille de l'échantillon, ce qui réduit mathématiquement l'incertitude perçue par la formule de l'intervalle de confiance. En détail :
  1. **La moyenne ne change pas** : la somme double et le nombre de valeurs double, donc leur rapport reste le même ;
  2. **L'écart-type σ ne change presque pas** : les valeurs sont les mêmes, leur dispersion aussi (seul le dénominateur n − 1 bouge un peu) ;
  3. **D'où vient √2** : la largeur de l'intervalle est proportionnelle à σ / √n. Avec n → 2n et σ inchangé, √n devient √(2n) = √2 · √n, donc la largeur est divisée par √2 ≈ 1,41. Le calcul exact, avec n − 1, donne un rapport √((2n − 1)/(n − 1)) : 1,43 pour n = 30, qui tend vers √2 quand n grandit (d'où le « environ ») ;
  4. **Vérification** avec 1, 2, 3, 4, 5 : moyenne 3,0 dans les deux cas ; σ = 1,58 puis 1,49 une fois dupliqué ; σ / √n = 0,707 puis 0,471, soit un rapport de 1,5 (= √(9/4)). Avec 30 valeurs, le rapport est 1,43 ;
  5. **Pourquoi c'est grave** : l'intervalle paraît plus précis alors qu'**aucune information nouvelle** n'a été ajoutée (ce sont toujours les mêmes 30 épisodes). Les valeurs p deviennent trop petites : une différence entre deux politiques peut paraître « significative » uniquement parce que les données ont été comptées deux fois. C'est pour ça que M3 est indispensable.

## T6 — Tests statistiques appariés
*Indispensable · [MESURE] · M3, M4, M5, M6, M22*

**Date :** commencé le 11 octobre 2026

**Prédiction (avant de modifier) :** 1. l'IC de PPO − Fixed en nominal sera **plus large** qu'avec le CSV v1 brut ; 2. PPO restera **significativement** meilleur que Fixed en nominal.

**Ce que j'ai changé :**

*Conception retenue (étape 3) :*
1. chargement validé (chemin en paramètre, doublons, 30 lignes par groupe) ;
2. tableau croisé graine × politique par régime ;
3. Friedman par régime, toujours, sans pré-test de normalité ;
4. Wilcoxon signé **apparié et bilatéral** sur les paires d'intérêt, puis Holm ;
5. bootstrap apparié : rééchantillonner les 30 différences par graine, 10 000 fois, avec les percentiles 2,5 % et 97,5 % ;
6. interaction politique × régime ;
7. le même cadre pour le carburant, une seule fonction d'analyse appelée pour chaque régime et chaque métrique.

*Étape 4 — chargement validé (fait le 12 octobre 2026) :*
- **Découverte dans le CSV v1** : 720 triplets en double, mais **0** ligne entièrement identique. Seules `inference_time_ms_mean` et `inference_time_ms_max` diffèrent entre les deux copies. Ce sont des mesures de temps réel (horloge de l'ordinateur), que la graine ne contrôle pas, donc elles ne sont pas reproductibles. Cela prouve que le fichier contient deux lancements complets mis bout à bout. Un simple `drop_duplicates()` n'aurait **rien** retiré.
- `load_results(csv_path, n_expected=30)` :
  - constante de module `DYNAMIC_COLUMNS` (`run_id`, `inference_time_ms_mean`, `inference_time_ms_max`) : les colonnes autorisées à varier ;
  - retire les doublons identiques sur toutes les autres colonnes, en gardant la première occurrence ;
  - si un triplet (scenario, policy, seed) reste en double, le doublon est contradictoire : `ValueError` ;
  - si un groupe (scenario, policy) n'a pas `n_expected` lignes : `ValueError`, avec la liste des groupes fautifs, par exemple `{('nominal', 'Fixed'): 2}` ;
  - renvoie le DataFrame nettoyé, avec `reset_index(drop=True)`.
- `tests/test_statistical_tests.py`, 4 tests : fichier propre, doublons identiques retirés, doublon contradictoire, groupe incomplet. `pytest.raises(..., match="contradictoires" / "incomplets")`.
- `summarise_energy_efficiency` reçoit `n_expected` et le transmet à `load_results`. Son test (`tests/test_inference_benchmark.py`) a été mis à jour : ajout de la colonne `seed`, graines 42 et 43 pour chaque politique, `n_expected=2`.
- Sur le CSV v1 : **720 lignes**, 24 groupes de 30, aucune erreur. Suite complète : **24 passed**.

*Étape 5a — tableau apparié (fait le 13 octobre 2026) :*
- `paired_table(data, scenario, metric)` : garde le régime demandé, puis `pd.pivot(index="seed", columns="policy", values=metric)`. On obtient une ligne par graine et une colonne par politique.
- **Faille que `load_results` ne voit pas** : si PPO a les graines 42–44 et Fixed les graines 43–45, chaque groupe a bien 3 lignes, mais l'appariement est cassé. Dans le tableau croisé, cela donne des **cases vides** (NaN). `isna()` les détecte, `any(axis=1)` donne une valeur par graine, et le message cite les graines fautives : « Cases vides : les graines [44, 45] n'ont pas toutes les politiques. »
- 2 tests : la forme (3 × 2, et la case (42, PPO) vaut 85,5) et des graines différentes, qui lèvent `ValueError`.

*Étape 5b — test de Friedman (fait le 13 octobre 2026) :*
- `friedman_test(table)` : `stats.friedmanchisquare(*[table[col] for col in table.columns])`, qui renvoie `(statistic, pvalue)`.
- Test `test_friedman_classement_parfait` : 10 graines × 3 politiques, avec A < B < C sur chaque ligne. On attend χ² = 20 (valeur calculée à la main) et p < 0,001.
- **Sur le CSV v1**, je retrouve exactement l'article v2 (§7.6) : χ²(7) = **207,6** en nominal, **200,7** en forte charge, **202,6** en fortes pannes (p ≈ 10⁻⁴⁰ dans les trois cas, à écrire « p < 0,001 »).
- Suite complète : **27 passed**.

*Étape 5c — Wilcoxon apparié bilatéral et Holm (fait le 13 octobre 2026) :*
- `paired_comparisons(table, pairs)` : pour chaque paire (a, b), on calcule `diff = table[a] - table[b]`, puis `stats.wilcoxon(table[a], table[b])`, bilatéral par défaut. Elle renvoie un DataFrame avec les colonnes `pair`, `mean_diff` et `p_value`, puis `p_holm = multipletests(p_value, method="holm")[1]`. L'élément `[1]` est le tableau des p-values corrigées : `multipletests` renvoie 4 résultats.
- Comme dans l'article v2, il y a 3 paires par régime (PPO–Fixed, DQN–Fixed, PPO–DQN) et Holm s'applique sur ces 3 comparaisons.
- Tests avec `make_table()` : 30 graines, A = 100, B = 100 + 1…30, C = 100 + 2 × (1…30).
  1. B bat A partout : `mean_diff` = 15,5 et p = 2/2³⁰ ;
  2. inverser la paire ne change pas p, ce qui prouve que le test est bilatéral ;
  3. 3 p-values égales : chaque `p_holm` vaut 3p.
- **Sur le CSV v1**, je retrouve exactement les différences du tableau de l'article v2 : nominal +105,0 / +63,0 / +42,0 ; forte charge −724,5 / −222,25 / −502,25 ; fortes pannes PPO − Fixed +81,1. Toutes les p-values de Holm sont < 0,001.
- Suite complète : **30 passed**.
- Reste à faire : les docstrings des 3 tests Wilcoxon (vides).

*Étape 5d — IC bootstrap apparié (fait le 14 octobre 2026) :*
- `paired_bootstrap_ci(diff, n_boot=10_000, seed=0)` :
  - `rng = np.random.default_rng(seed)` ;
  - `rng.choice(diff, size=(n_boot, len(diff)), replace=True)` tire 10 000 rééchantillons de 30 différences ;
  - `mean(axis=1)` donne une moyenne par rééchantillon ;
  - `np.percentile(means, [2.5, 97.5])` donne les bornes, renvoyées en `float(low), float(high)`.
- 3 tests :
  1. différences toutes égales à 5, donc IC = (5, 5) : chaque rééchantillon vaut [5, …, 5], sa **moyenne** vaut 5 (et non sa somme, 150) ;
  2. même graine, même résultat, et graine 1, résultat différent. Le second `assert` a été ajouté parce qu'avec une fonction vide, `None == None` faisait passer le test ;
  3. borne basse < 15,5 < borne haute.
- **Sur le CSV v1**, PPO − Fixed en nominal donne **[88,08 ; 120,75]**. L'article v2 donne [88,1 ; 120,8] : c'est **identique**.

*Étape 5e — l'IC dans `paired_comparisons` :* colonnes `ci_low` et `ci_high`, et un test `test_comparisons_contient_ic`. Suite : 34 passed.

*Étape 6 — interaction politique × régime (fait le 14 octobre 2026) :*
- `interaction_test(df, metric="productivity_tph")` :
  - `y` = métrique / `truck_count`, parce que la forte charge a 18 camions au lieu de 12 ;
  - `bloc` = régime + "_" + graine, c'est l'appariement ;
  - modèle complet `y ~ C(bloc) + C(policy) + C(policy):C(scenario)` et modèle réduit sans l'interaction ;
  - `sm.stats.anova_lm(reduced, full).iloc[1]` donne le test F de l'amélioration ;
  - η²p = ss_diff / (ss_diff + ssr) ;
  - renvoie un dictionnaire `F`, `df_num`, `df_den`, `p_value`, `eta_p2`.
- Tests avec `make_interaction_rows(with_interaction)` : 3 régimes × 3 politiques × 5 graines, avec un bruit `rng.normal(0, 1)`. Avec interaction, le classement est inversé en s2 et les politiques sont égales en s3. On attend p < 0,001 avec df = (4, 24), et sans interaction p > 0,05.
- **Sur le CSV v1** : F(14, 609) = **430,7**, p < 0,001, η²p = **0,91**. L'article v2 écrit **430,4** : l'écart est minime et vient probablement d'un autre type de somme des carrés dans la v1. Les degrés de liberté et η²p sont identiques. **À corriger dans l'article (T31) : 430,7.**
- Suite : **36 passed**.

*Étape 7 — carburant :* rien à écrire, parce que les fonctions prennent la métrique en paramètre. Avec `"specific_fuel_l_per_ton"`, on retrouve l'article : DQN − Fixed −0,0024 L/t (−5,5 %) en nominal, −0,0033 L/t (−7,3 %) en fortes pannes, +0,2 % en forte charge. Le défaut **D6 est corrigé** : le carburant reçoit le même traitement que la productivité.
- Observation en forte charge, DQN − Fixed : p de Holm = 0,026, alors que l'IC bootstrap de la moyenne contient 0 ([−0,00003 ; +0,00019]). Wilcoxon teste si les différences sont plutôt d'un signe (à partir des rangs), le bootstrap porte sur la moyenne. L'écart est minuscule : « presque à égalité », comme l'écrit l'article.

*Étape 8 — nouvelle `main` et nettoyage (fait le 14–15 octobre 2026) :*
- 8a : suppression de deux imports ajoutés automatiquement par l'éditeur (`from rich import columns` et `from torch import return_types`). Le second chargeait PyTorch à chaque import du module.
- 8b : `format_p(p)` renvoie « p < 0,001 » si p < 0,001, sinon « p = 0,026 » (3 décimales, virgule). Elle a son test `test_format_p`.
- 8c : `main(csv_path)`. Constantes de module `SCENARIOS`, `PAIRS`, `METRICS`. Pour chaque métrique et chaque régime : `paired_table`, puis Friedman (χ²(k − 1)), puis `paired_comparisons` + d de Cohen. À la fin : l'interaction. Lancement : `python -m memoire_master_rl_logistique.experiments.statistical_tests <chemin du CSV>` (`sys.argv[1]`). Le test `test_main_retrouve_article` (avec `capsys` et `skipif` comme en T4) vérifie que la sortie contient « χ²(7) = 207,6 » et « F(14, 609) = 430,7 ».
- 8d : suppression des 13 anciennes fonctions (Shapiro, Levene, Welch, ANOVA, Kruskal, Dunn, ancien Holm, Hedges, Cliff, ancien bootstrap…), de `summarise_energy_efficiency` (Welch unilatéral indépendant, chemin inexistant) et de son test, puis des imports inutiles. On garde `cohens_d`. La docstring du module est réécrite. `statistical_tests.py` passe de **428 à 190 lignes**.
- Suite complète : **37 passed** (un test supprimé avec sa fonction).

**Pourquoi :** lecture de `main()` dans `experiments/statistical_tests.py` (lignes 314–395).

*Ce que fait `main` aujourd'hui, pour chacun des trois régimes (`nominal`, `high_load`, `high_breakdown`) :*
1. `print_group_stats` : statistiques descriptives et IC bootstrap **par politique** ;
2. `run_normality_tests` (Shapiro) et `run_levene_test` : prérequis de l'ANOVA ;
3. `run_anova` : ANOVA si **toutes** les politiques passent Shapiro, sinon Kruskal-Wallis ; Levene choisit entre ANOVA classique et ANOVA de Welch ;
4. 3 paires seulement (PPO–Fixed, DQN–Fixed, PPO–DQN) : `ttest_ind(..., alternative="greater")`, `run_t_test` et `run_effect_sizes` ;
5. `run_holm_correction` sur ces 3 valeurs p.

Le carburant (lignes 374–389) **ne suit pas** ce chemin. Il n'a qu'un Welch `"less"` sur 2 paires et un IC bootstrap par politique, sans ANOVA, sans Holm et sans taille d'effet.

*Les défauts :*

| # | Défaut | Conséquence |
|---|---|---|
| D1 | Il lit `data/results/benchmark_results.csv`, qui n'existe plus (v1 déplacé dans `data/results_v1_article/`, et depuis T5 les lancements écrivent `benchmark_<run_id>.csv`) | Le script affiche « ERREUR » et fait `return`, sans exception : l'échec passe inaperçu. `summarise_energy_efficiency` a le même chemin par défaut |
| D2 | Le CSV v1 contient chaque ligne deux fois, et rien ne les retire | n = 60 au lieu de 30 par groupe, erreur-type σ/√n divisée par environ √2, IC environ 29 % trop étroits, valeurs p trop petites |
| D3 | Tests pour groupes indépendants (`ttest_ind`, ANOVA/Kruskal) alors que les données sont **appariées par graine** | Le bruit dû aux graines reste dans la variance : le test est mal calibré pour ce plan d'expérience |
| D4 | Tests **unilatéraux** (`"greater"`, `"less"`) dont le sens a été choisi après avoir vu les résultats | Valeurs p divisées par 2 sans justification, risque de faux positif sous-estimé |
| D5 | Le choix ANOVA ou Kruskal est fait régime par régime | Les trois régimes ne sont pas forcément analysés avec le même outil |
| D6 | Le carburant a un traitement à part | Ses conclusions sont moins solides que celles de la productivité |

**Test prévu par le plan :** sur ton CSV dédoublonné, tu dois retrouver la section 7.6 de l'article v2. Par exemple, PPO − Affectation fixe en nominal : +105,0 t/h, intervalle [88,1 ; 120,8].

**Ce que le test vérifie :** `tests/test_statistical_tests.py`, **18 tests** :
- chargement : fichier propre, doublons identiques retirés, doublon contradictoire, groupe incomplet ;
- tableau apparié : forme, graines différentes ;
- Friedman : classement parfait, χ² = 20 ;
- Wilcoxon : domination totale, p = 2/2³⁰ ; bilatéral ; Holm sur 3 p égales, 3p ;
- bootstrap : différences constantes, IC = (5, 5) ; reproductible ; encadre la moyenne ; IC dans `paired_comparisons` ;
- interaction : forte, p < 0,001 avec df = (4, 24) ; absente, p > 0,05 ;
- `format_p` ;
- `main` sur le CSV v1 : elle retrouve l'article.

Vérification manuelle sur le CSV v1 avec `python -m memoire_master_rl_logistique.experiments.statistical_tests data/results_v1_article/benchmark_results.csv`. Je retrouve **toute la §7.6 de l'article v2** :
- Friedman 207,6 / 200,7 / 202,6 ;
- PPO − Fixed nominal +105,0 [88,1 ; 120,8], d = 1,93 ;
- DQN − Fixed +63,0 [45,5 ; 81,1], d = 1,10 ;
- PPO − DQN +42,0 [28,6 ; 56,6], d = 0,64 ;
- forte charge −724,5 / −222,2 / −502,2 ;
- fortes pannes +81,1 ;
- carburant −5,5 % / −7,3 % ;
- η²p = 0,91.

Seul écart : **F(14, 609) = 430,7** au lieu de 430,4 dans l'article. À corriger en T31.

Pour le carburant, plusieurs p valent exactement 1,862645 × 10⁻⁹ = **2/2³⁰**. La consommation est continue, donc il n'y a pas d'ex æquo, et scipy fait le calcul exact : on tombe sur le minimum théorique.

**Résultat comparé à ma prédiction :**
- **Prédiction 2** (« PPO restera significativement meilleur que Fixed en nominal ») : ✅ juste, p de Holm < 0,001.
- **Prédiction 1** (« l'IC sera plus large qu'avec le CSV v1 brut ») : ❌ **fausse**. Pour PPO − Fixed en nominal :

| Méthode | n | IC 95 % | Largeur |
|---|---|---|---|
| Indépendant, CSV brut (avec doublons) | 60 | [85,8 ; 124,3] | 38,5 |
| Indépendant, sans doublons | 30 | [77,6 ; 131,3] | 53,7 (×1,39 ≈ √2) |
| **Apparié, sans doublons (nouvelle méthode)** | 30 | **[88,1 ; 120,8]** | **32,7** |

  Deux effets jouent en sens opposé. **Retirer les doublons élargit** l'IC (38,5 → 53,7, le facteur √2 prévu), mais **apparier le resserre davantage** (53,7 → 32,7), parce que le bruit des graines disparaît des différences. L'IC correct est donc plus étroit que l'ancien, tout en étant honnête (30 vrais épisodes). Ce que j'en retiens : l'appariement apporte de la précision sans épisode supplémentaire. Dans la v1, les deux erreurs (doublons et tests indépendants) se compensaient en partie, par hasard.

**Ce que j'ai appris :**
- *Erreur-type et doublons* : dupliquer les données ne change presque pas σ, mais divise σ/√n par environ √2. On obtient une fausse précision sans aucune information nouvelle.
- *Bootstrap apparié* : on tire au hasard dans la liste des 30 **différences** par graine, et non dans chaque groupe séparément. Tirer une différence revient à tirer une graine entière, donc l'appariement est conservé.
- *Friedman* : il classe les politiques **à l'intérieur de chaque graine**. C'est un test apparié, et il n'a pas besoin de la normalité. On l'applique toujours, sans choisir le test d'après les données.
- *Masque pandas* : il faut d'abord **comparer** (`nb_groups != n_expected`, une Series de True/False), puis demander `.any()` **sur le masque**. Le filtrage `nb_groups[mask]` sert seulement à afficher les groupes fautifs. Avec `nb_groups.any() != n_expected`, `.any()` vaut True, et `True != 3` vaut True : l'erreur était levée à chaque fois.
- *`pytest.raises(ValueError)` peut passer pour une mauvaise raison* : l'erreur pandas « The truth value of a Series is ambiguous » est elle aussi une `ValueError`. `match=` avec un mot-clé de **mon** message rend le test précis. Pas de nombre ni de parenthèses dans `match`, puisque c'est une expression régulière.
- *Le bloc `pytest.raises` contient seulement la ligne censée lever l'erreur.* Un `assert` placé après dans le bloc n'est jamais exécuté.
- *On ne modifie pas un test pour qu'il colle au code*, sauf si le **contrat change volontairement**. Ici, un CSV sans `seed` ou avec 2 lignes par groupe n'est plus valide, donc le mini-CSV du test énergie a été mis à jour, avec les mêmes vérifications.
- *Le code du projet ne va pas dans le fichier de test.* Le test **importe** la fonction depuis `statistical_tests.py`. Une copie locale dans le test ne vérifie pas le code que `main()` utilise.
- *`axis=1` « écrase » les colonnes* : on obtient une valeur **par ligne**, donc par graine. `axis=0`, la valeur par défaut, donne une valeur par colonne, donc par politique. On veut les **graines**, parce que la graine est l'unité d'appariement.
- *L'opérateur `*` (unpacking)* : `f(*[a, b, c])` revient à `f(a, b, c)`. `friedmanchisquare` attend un argument par politique. Sans l'étoile, scipy reçoit 1 seul échantillon (« got 1 »). Et `[col for col in table.columns]` donne les **noms** des colonnes, alors que `[table[col] for ...]` donne leurs **valeurs**.
- *Un test compare à une valeur connue d'avance, calculée indépendamment.* Recoder la formule dans le test risque de reproduire la même erreur des deux côtés. Un test **sans `assert`** passe toujours, donc il ne prouve rien.
- *Lire une erreur de collecte* : « error during collection » veut dire que pytest n'a même pas pu lire le fichier. C'est une erreur de syntaxe, par exemple `x =` sans rien à droite, ou une `*` mal placée.
- *Compter sans lister* : `list(product(["+", "-"], repeat=30))` essaie de créer 2³⁰ ≈ 1,07 milliard de tuples, d'où le `MemoryError`. On vérifie le principe sur de petits n (2, 4, 8, 16, 32 : ça double à chaque fois), puis on raisonne : chaque différence a 2 signes possibles, indépendants, donc il y a 2³⁰ combinaisons.
- *La différence A − B* : pour chaque graine, de combien A bat B. C'est la base des trois outils appariés : Wilcoxon (signes et tailles des différences), `mean_diff` (leur moyenne) et le bootstrap apparié (tirage parmi elles).
- *`multipletests` renvoie un tuple de 4 éléments* : `[0]` rejet True/False, **`[1]` p-values corrigées**, `[2]` et `[3]` des seuils alpha ajustés. Holm : on trie les p-values, on multiplie la plus petite par m, la suivante par m − 1, etc., puis on impose qu'elles ne diminuent pas. Exemple : [0,01 ; 0,02 ; 0,04] devient [0,03 ; 0,04 ; 0,04].
- *Un test peut passer par coïncidence* : additionner les 3 p-values brutes donnait aussi 3 × 2/2³⁰, sans aucune correction de Holm. Il faut vérifier **la colonne qu'on veut tester** (`p_holm`), ligne par ligne.
- *Ex æquo et Wilcoxon* : sur les vraies données, les productivités varient par paliers de 17,5 t/h, donc plusieurs différences sont identiques. Avec des ex æquo, scipy abandonne le calcul exact pour une approximation normale. Les p-values tournent alors autour de 10⁻⁶, et non 2 × 10⁻⁹. Conclusion inchangée : on écrit p < 0,001.
- *Tableau numpy et `==`* : comparer deux tableaux numpy compare **case par case** (`array([True, True])`), et `assert` ne sait pas trancher (« truth value … is ambiguous »). C'est le même piège qu'avec les Series pandas. Une fonction annoncée `-> tuple[float, float]` doit renvoyer un vrai tuple : `low, high = np.percentile(...)`, puis `return float(low), float(high)`.
- *Un test peut passer avec une fonction vide* : `None == None` est vrai. On ajoute un `assert` qui échouerait avec `None` (ici : une autre graine doit donner un autre résultat).
- *Graine fixée = reproductibilité* : avec `default_rng(0)`, le bootstrap redonne exactement [88,08 ; 120,75] à chaque lancement, et le même IC que l'article.
- *Interaction = comparer deux modèles* : le modèle réduit dit que la politique a le même effet dans tous les régimes ; le modèle complet ajoute « l'effet dépend du régime ». Le test F mesure si l'ajout améliore nettement le modèle. df_num = (politiques − 1) × (régimes − 1) : 4 dans le test, 14 sur les vraies données.
- *Fonction d'aide de test* : elle va dans le fichier de test (pas dans le module) et doit **renvoyer** son résultat. Sans `return`, elle renvoie `None`. Avant d'écrire les tests, je vérifie les données dans la console (moyennes par régime et politique).
- *`KeyError` sur un nom de colonne* : afficher les vrais noms (`row.index`, `df.columns`) et les copier. `"Pr(>F)"` et `"PR(>F)"` sont deux noms différents.
- *Fonctions génériques* : passer la métrique en paramètre a suffi à traiter le carburant sans nouvelle ligne de code.
- *Un test ne prend pas de paramètres libres* : pour pytest, chaque paramètre d'un test est une **fixture** (`tmp_path`, `capsys`). `def test_format_p(p)` provoque une ERROR, car il n'existe pas de fixture `p`.
- *Tester une fonction qui affiche* : `capsys.readouterr().out` récupère le texte affiché, puis `assert "..." in out`.
- *Ordre des arguments* : `paired_table(df, metric, scenario)` au lieu de `(df, scenario, metric)` donnait un `KeyError`. Je relis la signature avant d'appeler.
- *Imports auto-ajoutés par l'éditeur* : après `rich` et `torch` (et `openai` en T5), je vérifie le haut du fichier avant chaque commit.
- *Supprimer du code mort* : une fonction que plus personne n'appelle, et qui contient les anciens défauts, est supprimée avec son test. Garder du code faux « au cas où », c'est risquer qu'il soit réutilisé.
- *Une prédiction fausse est utile* : je n'avais pas anticipé que l'appariement l'emporterait sur le retrait des doublons. Le tableau des trois IC le montre clairement.

**Commit :** `7617053` (final, 15 octobre 2026). Commits intermédiaires : `873913c` (chargement, tableau apparié, Friedman, Wilcoxon + Holm) et `0b0f4cc` (bootstrap apparié, interaction).

**Questions « Savoir expliquer » :**
- [x] pourquoi les données sont appariées : pour une graine donnée, PPO et Fixed partent du **même état du générateur aléatoire** (`self.rng = Random(seed)`), donc des mêmes conditions de départ : mêmes pannes et mêmes tirages, au moins tant que les décisions ne font pas diverger les tirages. Leurs résultats sont donc **corrélés** : la ligne `nominal, PPO, 42` va avec la ligne `nominal, Fixed, 42`. Pour une même graine, l'écart entre deux résultats mélange l'effet de la politique (ce qu'on veut mesurer) et l'effet de la graine (journée avec beaucoup ou peu de pannes, c'est du bruit). Un test apparié calcule la différence **graine par graine**, ce qui annule l'effet de la graine. Un test indépendant (`ttest_ind`) ne sait pas quelles lignes vont ensemble et garde ce bruit ;
- [x] la différence entre unilatéral et bilatéral : un test unilatéral (`"greater"`) ne cherche une différence que dans un sens (H1 : moyenne de x > moyenne de y). Un test bilatéral la cherche dans les deux sens. Quand l'écart va dans le sens attendu, la valeur p bilatérale est environ **le double** de l'unilatérale, puisqu'on additionne les deux queues symétriques. Choisir le sens **après** avoir vu les données revient à diviser la valeur p par 2 gratuitement : si le résultat était allé dans l'autre sens, on aurait écrit `"less"`, donc le vrai risque d'erreur est celui d'un test bilatéral. L'unilatéral n'est légitime que si le sens est fixé **avant** l'expérience **et** si l'autre sens n'intéresse pas. Ce n'est pas le cas ici : PPO pire que Fixed serait un résultat important, et pour PPO–DQN les deux sens comptent. D'où le passage au **bilatéral** ;
- [x] (en plus) le test de Friedman : pour **chaque graine**, on classe les k politiques (rang 1 pour la plus petite valeur, rang k pour la plus grande). On additionne ensuite les rangs de chaque politique sur les n graines : ce sont les sommes Rⱼ. Si les politiques sont équivalentes, les Rⱼ sont proches. La statistique χ² = 12 / (n·k·(k+1)) × ΣRⱼ² − 3·n·(k+1) mesure leur écart. Exemple : n = 10, k = 3, classement parfait, donc R = 10, 20, 30 et χ² = 0,1 × 1 400 − 120 = **20**. C'est un test **apparié**, puisqu'il compare les politiques à l'intérieur d'une même graine, et **sans hypothèse de normalité**, puisqu'il travaille sur des rangs. Les degrés de liberté valent k − 1 : 7 pour 8 politiques, d'où « χ²(7) » ;
- [x] pourquoi la plus petite valeur p d'un Wilcoxon avec 30 paires vaut environ 2 × 10⁻⁹ (2 / 2³⁰) : chacune des 30 différences peut être positive ou négative, donc il y a 2 × 2 × … × 2 = **2³⁰** répartitions de signes possibles (environ 1,07 milliard). Si les deux politiques sont équivalentes, ces répartitions sont toutes également probables. Le cas le plus extrême, « B gagne sur les 30 graines », ne correspond qu'à **1** répartition (tout +). Comme le test est bilatéral, on compte aussi l'extrême opposé (tout −) : **2** cas. Donc p = 2 / 2³⁰ ≈ 1,86 × 10⁻⁹. Avec 5 graines, ce serait 2/2⁵ ≈ 0,06 : même une victoire parfaite ne serait pas significative. C'est la **plus petite p-value possible** avec 30 paires, d'où l'absurdité des « 10⁻⁴⁶ » de la v1, qui venaient de tests t indépendants sur des données dupliquées.

## T7 — Robustesse et figures
*Indispensable · [MESURE] · M15, M2, M19*

**Date :** 15 octobre 2026

**Changement de méthode à partir de T7 (décidé le 15 octobre 2026).** Au rythme d'une tâche par jour environ, le plan n'aurait pas pu être fini à temps. Nouvelle répartition :
- l'IA écrit le code et les tests ;
- je relis, je lance les tests et je remplis le « Savoir expliquer », pour pouvoir défendre chaque correction ;
- j'ai écrit moi-même le calcul du ratio (`robustness_ratios`) et répondu à la question de fond avant le code.

**Prédiction (avant de modifier) :** pas de prédiction chiffrée écrite. Raisonnement fait avant le code (étape 1), avec une politique qui transporte 300 t/h par camion dans les deux régimes :
- 12 camions en nominal donnent 3 600 t/h ; 18 camions en forte charge donnent 5 400 t/h ;
- le ratio sur la productivité totale vaut 1,5, le ratio par camion vaut 1 ;
- c'est le ratio **par camion** qui montre que la politique n'a ni progressé ni régressé.

**Ce que j'ai changé :**
- `experiments/robustness_analysis.py` :
  - `robustness_ratios(df)` : ratio = productivité par camion dans le régime / en nominal (écrit par moi) ;
  - `_scenario_means` : lève `ValueError` si un régime manque ;
  - `wait_change_pct` : variation de l'attente moyenne par camion, en % du nominal ;
  - `robustness_table` ;
  - `plot_robustness_heatmap` : deux heatmaps côte à côte, productivité et attente. Le bleu signifie toujours « mieux » (`RdBu_r` pour l'attente) et les nombres affichés sont les vraies variations ;
  - `main(csv, dossier, overwrite)` : lit avec `load_results` (T6), qui dédoublonne.
- `experiments/plot_benchmark_comparisons.py` :
  - `summarise` : moyenne et IC bootstrap 95 % par (politique, régime), graine 0 ;
  - `pareto_front` : masque des politiques non dominées (attente minimale, productivité maximale) ;
  - `plot_grouped_bars` : une seule fonction pour les barres de productivité et d'attente ; le seuil de 30 min est retiré ;
  - `plot_pareto` : **un panneau par régime**, front de Pareto en pointillés, politiques non dominées en gras ;
  - `main`.
- `experiments/reporting.py` (nouveau) :
  - ordre, libellés français et couleurs des politiques et des régimes, pour des figures cohérentes ;
  - `order_policies` : une politique inconnue n'est jamais perdue en silence ;
  - `prepare_outputs` : **refuse d'écraser** un fichier existant sans `--overwrite`.
- Lancement des deux scripts : `python -m memoire_master_rl_logistique.experiments.<module> CSV DOSSIER [--overwrite]`. Le dossier de sortie est obligatoire.
- `tests/conftest.py` : backend matplotlib `Agg`, pour générer les figures sans fenêtre pendant les tests.
- **Incident pendant la tâche :** j'ai lancé l'analyse avec `reports/figures` comme dossier de sortie. La nouvelle heatmap a écrasé celle de l'article v1, restaurée avec `git restore`. C'est la raison de `prepare_outputs`.

**Pourquoi :**
- le ratio de la v1 était calculé sur la productivité **totale**, sur un CSV avec doublons ;
- le Pareto mélangeait les trois régimes, qui n'ont pas la même échelle ;
- la figure d'attente traçait un « seuil cible » de 30 min sans source ;
- le chemin du CSV (`data/results/benchmark_results.csv`) n'existait plus ;
- les scripts écrivaient directement dans `reports/figures`.

**Test prévu par le plan :** comparer tes ratios au tableau 13 de l'article v2. Par exemple, Affectation fixe en forte charge : 0,962.

**Ce que le test vérifie :**
- `tests/test_robustness.py`, 7 tests :
  - ratio par camion (A = 1, B = 2/3, fortes pannes 0,9) ;
  - variation d'attente (+50 %, −50 %, 0) ;
  - régime manquant → `ValueError` ;
  - ordre des politiques sans perte des inconnues ;
  - refus d'écraser ;
  - heatmap écrite ;
  - `main` sur le CSV v1 retrouve le tableau 13 (Fixed 0,962 / 0,938, PPO 0,822, attente DQN +9,3 %) et refuse un second lancement dans le même dossier.
- `tests/test_benchmark_plots.py`, 4 tests :
  - deux tests sur le front de Pareto (point dominé, égalité) ;
  - `summarise` (moyenne, IC nul sur des valeurs constantes) ;
  - les 3 figures sont écrites sur le CSV v1, et un second lancement refuse d'écraser.
- Suite complète : **48 passed**, avant et après la réorganisation.

**Résultat comparé à ma prédiction :** le raisonnement de l'étape 1 est confirmé. Les 16 ratios du tableau 13 sont **identiques** : FIFO 0,965 / 0,949, Fixed 0,962 / 0,938, Nearest 0,688 / 0,957, ShortestPath 0,683 / 0,955, Q-Learning 0,887 / 0,887, SARSA 0,874 / 0,966, DQN 0,911 / 0,955, PPO 0,822 / 0,934. La variation d'attente totale en forte charge est identique à la colonne « Immob. totale FC » (+35,0 … PPO +196,4).
- **Pas encore fait :** la colonne « Δ attente en file » et le Pareto « en file » dépendent de **T10**. Pour l'instant, c'est l'attente totale, et les docstrings le signalent.
- Lecture du Pareto : en forte charge, l'Affectation fixe **domine toutes les autres politiques** (plus de productivité et moins d'attente). En nominal, le front ne contient que PPO et l'Affectation fixe.

**Ce que j'ai appris :**
- *Normaliser avant de comparer* : quand un régime change la taille de la flotte, on compare par camion, sinon on mesure l'ajout de camions et non la politique.
- *Front de Pareto* : avec deux objectifs (moins d'attente, plus de productivité), une politique est dominée si une autre est au moins aussi bonne sur les deux et strictement meilleure sur l'un. Le front, ce sont les compromis non battus. On le trace par régime, parce que les échelles diffèrent.
- *Une figure ne doit rien affirmer sans source* : le seuil de 30 min n'était justifié nulle part.
- *Protéger les sorties* : ne jamais écrire par défaut dans un dossier qui contient des résultats publiés. Il faut un dossier explicite, et un refus d'écraser sauf demande.
- *Couleurs et sens* : le bleu doit toujours signifier « mieux ». Pour l'attente (plus bas = mieux), on inverse l'échelle de couleur, pas les nombres.
- *Tester le code de quelqu'un d'autre* : mon test sur les politiques A et B aurait révélé que l'ancienne heatmap perdait les politiques absentes de la liste d'ordre. D'où `order_policies`.

**Commit :** `c9f1332` (15 octobre 2026). Ensuite, `13efcdd` : réorganisation du code par responsabilité (voir plus bas).

**Questions « Savoir expliquer » :**
- [x] pourquoi un ratio supérieur à 1 en forte charge ne prouvait rien : en forte charge, on passe de 12 à 18 camions (+50 %). Le ratio calculé sur la productivité **totale** mesure surtout cet ajout. Une politique dont chaque camion transporte exactement autant qu'en nominal (300 t/h) passe de 3 600 à 5 400 t/h, donc ratio 1,5, alors qu'elle n'a rien amélioré. Un ratio supérieur à 1 ne prouvait donc pas la robustesse. En ramenant la productivité au nombre de camions, cette politique a un ratio de 1, et les vraies pertes apparaissent : Pelle la plus proche 0,688, Plus court chemin 0,683, PPO 0,822, contre 0,962 pour l'Affectation fixe. En fortes pannes, le nombre de camions ne change pas, donc les deux calculs donnent le même ratio.
- [x] (en plus) pourquoi le point (coût 25, gain 150) est dominé par (20, 200) dans `pareto_front` : on cherche à **minimiser** le coût (l'attente) et à **maximiser** le gain (la productivité). Le second point est au moins aussi bon sur les deux critères (20 ≤ 25 et 200 ≥ 150) **et** strictement meilleur sur au moins un (ici les deux). Personne n'a intérêt à choisir le premier : il est dominé. Le front de Pareto, ce sont les points qu'aucun autre ne domine, c'est-à-dire les vrais compromis. Deux points identiques ne se dominent pas, faute d'être strictement meilleurs l'un que l'autre.
- [x] (en plus) pourquoi un Pareto **par régime** : les régimes n'ont pas les mêmes échelles. En forte charge, la productivité tourne autour de 4 000 à 5 900 t/h et l'attente va jusqu'à 200 min ; en nominal, environ 3 100 à 4 200 t/h et 30 à 80 min. Sur un seul graphique, une politique « nominale » serait comparée à une politique « forte charge » dans des conditions différentes, ce qui n'a pas de sens. Lecture : en forte charge, l'Affectation fixe domine toutes les autres ; en nominal, le front ne contient que PPO et l'Affectation fixe.
- [x] (en plus) la case PPO / forte charge de la heatmap d'attente (+196 %) : elle est **rouge foncé**. Une hausse de l'attente est une dégradation, et on veut que le bleu signifie toujours « mieux ». On a donc inversé l'**échelle de couleur** (`RdBu_r`) sans inverser les nombres : la case affiche bien +196,4, la vraie variation.
- [x] (en plus) lancer deux fois la même commande avec le même dossier : le second lancement s'arrête **avant d'écrire quoi que ce soit**, avec `FileExistsError` (« Fichiers déjà présents (utiliser --overwrite) »). C'est `prepare_outputs` dans `plots/common.py`, ajouté après que la heatmap de l'article v1 a été écrasée. Pour écraser volontairement : `--overwrite`.

**Où se trouve le code de T7 après la réorganisation du 15 octobre 2026 (commit `13efcdd`) :**
- `analysis/robustness.py` : les calculs (`_scenario_means`, `robustness_ratios`, `wait_change_pct`, `robustness_table`), sans matplotlib ;
- `plots/robustness.py` : la heatmap et la ligne de commande, `python -m memoire_master_rl_logistique.plots.robustness CSV DOSSIER [--overwrite]` ;
- `plots/benchmark_comparisons.py` : barres et Pareto, `python -m memoire_master_rl_logistique.plots.benchmark_comparisons CSV DOSSIER [--overwrite]` ;
- `plots/common.py` : conventions des figures, `order_policies`, `prepare_outputs`, `save_figure`.

## T8 — Tracer la version de chaque modèle
*Recommandé · [MESURE] · N6, N11*

**Date :** 15 octobre 2026 (méthode 1 : code et tests écrits par l'IA, relus par moi)

**Prédiction (avant de modifier) :** pas de prédiction chiffrée écrite. Attendu :
- les modèles de l'article v1 seront refusés, puisqu'ils n'ont pas de métadonnées ;
- la vérification N11 confirmera la conclusion de T2 : même récompense, probable mais pas prouvable.

**Ce que j'ai changé :**
- `rl/model_metadata.py` (nouveau, remplace `model_version.py`, jamais appelé et supprimé) :
  - `EnvConfig` : nombres de camions, pelles et verses, durée, probabilité de panne, **poids de la récompense**. Les types sont normalisés : 480 = 480.0, liste = tuple ;
  - `ModelMetadata` : algorithme, `EnvConfig`, **taille d'observation**, graine, paramètres d'entraînement, **commit Git**, `git_dirty` (code non commité au moment de l'entraînement), date ;
  - `write_metadata` et `read_metadata` : écriture et lecture de `metadata.json` ;
  - `check_model_compatibility` : liste **tous** les écarts. La graine et le commit ne sont pas comparés : on évalue sur d'autres graines, et un nouveau commit ne rend pas un modèle faux.
- `rl/train_dqn.py`, `train_ppo.py`, `train_q_learning.py`, `train_sarsa.py` : écrivent `metadata.json` juste après la sauvegarde du modèle.
- `rl/model_loading.py` (nouveau) : `load_model(dossier, algorithme, env, trained_on=None, allow_legacy=False)` est le **seul** point de chargement d'un modèle. Il vérifie, puis il charge. Plus aucun `DQN.load`, `PPO.load` ou `…Policy.load` direct ailleurs dans le code.
- Utilisé par :
  - `experiments/run_benchmark.py` : pour les scénarios `unseen_*`, il compare au régime **d'entraînement** (nominal) ;
  - `analysis/action_distribution.py` ;
  - `plots/behavioral_analysis.py`, qui en plus exige un dossier de sortie et refuse d'écraser, comme en T7 ;
  - `rl/evaluate_agent.py`.
- Option `--allow-legacy-models` (dans `main.py --benchmark-only` et dans les scripts) : réévaluer **en connaissance de cause** les modèles v1, avec un avertissement. On n'invente jamais de métadonnées pour un ancien modèle.

**Pourquoi :** `model_version.py` existait, mais rien ne l'appelait. Un ancien modèle, entraîné avec une autre récompense ou une autre observation, pouvait être rechargé et évalué sans que personne le sache (N6). Et on ne pouvait pas prouver avec quelle récompense DQN et PPO avaient été entraînés (N11).

**Vérification N11** (`git log` sur `env/` entre fin mai et début août, dates des fichiers de modèles, lecture des `.zip`) :

| Fait | Source |
|---|---|
| 30 mai, **11 h 41** : le commit `b13b2e1` fixe la récompense actuelle (w4 = 0,3) et l'observation à **147** | `git log` |
| DQN nominal entraîné le 30 mai à **20 h 02**, donc **après** ce commit ; DQN forte charge et fortes pannes le 31 mai | dates des fichiers |
| PPO nominal entraîné le 5 août, forte charge le 5 août, fortes pannes le 6 août | dates des fichiers |
| Entre les deux, seul `0348a8b` (16 juin) touche l'environnement, et seulement la valeur **par défaut** de w4. Les entraînements passaient les poids explicitement | `git log -p` |
| DQN et PPO attendent tous deux une observation de taille **147** | lu dans les `.zip` |

Conclusion : même récompense et même observation, **très probablement**. Ce n'est **pas prouvable à 100 %**, car on ignore si le code avait des modifications non commitées le jour de l'entraînement. À partir de T8, `git_commit` + `git_dirty` apportent cette preuve.

**Test prévu par le plan :** un faux modèle avec des métadonnées périmées doit être refusé.

**Ce que le test vérifie :**
- `tests/test_model_metadata.py`, 10 tests :
  - aller-retour JSON ;
  - normalisation des types ;
  - modèle compatible accepté ;
  - **récompense périmée (w4 = 1,0) refusée** (le test du plan) ;
  - observation 138 refusée ;
  - autre algorithme refusé ;
  - tous les écarts listés ;
  - modèle sans métadonnées refusé, ou accepté avec avertissement ;
  - commit Git de 40 caractères ;
  - un vrai entraînement Q-Learning d'un épisode écrit des métadonnées acceptées.
- `tests/test_model_loading.py`, 6 tests :
  - chargement d'un modèle compatible ;
  - **refus avant toute lecture du fichier du modèle** ;
  - modèle v1 sans métadonnées : refusé par défaut, accepté avec `allow_legacy`, sans métadonnée inventée ;
  - régime d'entraînement explicite (`trained_on`) ;
  - observation de taille différente toujours refusée ;
  - algorithme inconnu.
- Vérification réelle : `plots/behavioral_analysis.py` **refuse** le PPO v1 (« Pas de metadata.json »), et produit la figure avec `--allow-legacy-models`.
- Suite complète : **64 passed**. Ruff ne signale plus rien sur tout le projet.

**Résultat comparé à ma prédiction :** les deux attentes sont confirmées : les modèles v1 sont refusés sans l'option, et N11 reste « probable, non prouvable ». C'est plus solide qu'en T2, grâce aux heures exactes et à la taille d'observation lue dans les `.zip`.
- **Découverte :** la taille de l'observation **dépend du nombre de camions** (12 → 147, 13 → 158, 15 → 180, 18 → 213). Le scénario de généralisation `unseen_medium_load` (15 camions) essayait d'évaluer des modèles nominaux (147) sur une observation de 180. Le contrôle le refuse désormais avec un message clair. C'est le défaut que corrige **T18** (observation de taille fixe).

**Ce que j'ai appris :**
- *Un résultat doit pouvoir dire d'où il vient* : sans « carte d'identité » à côté du modèle, impossible de prouver quelle récompense a produit quel agent.
- *Le commit ne suffit pas* : il faut aussi savoir si le code était **modifié sans commit** (`git_dirty`) au moment de l'entraînement. C'est exactement le trou de N11.
- *Refuser tôt, avec un message utile* : on vérifie avant de charger, et on liste tous les écarts. Une erreur de forme dans PyTorch n'aurait rien expliqué.
- *Un seul point d'entrée* : si chaque script charge les modèles à sa façon, un contrôle oublié suffit à tout invalider. `load_model` est le seul chemin.
- *Ne jamais inventer de métadonnées* pour un ancien modèle : l'option `--allow-legacy-models` dit honnêtement « sans contrôle ».
- *Les outils de vérification attrapent ce que les tests ne voient pas* : ruff a détecté un `PPO` non importé dans un script que les tests ne lancent pas, qui aurait planté à l'exécution.

**Commit :** `b902df2` (15 octobre 2026).

**Questions « Savoir expliquer » :**
- [x] comment tu prouverais à un relecteur que DQN et PPO ont été entraînés avec la même récompense :
  - **Pour les modèles de l'article v1 (avant T8),** je ne peux apporter qu'un faisceau de preuves, pas une preuve absolue :
    1. le commit `b13b2e1` (30 mai, 11 h 41) fixe la récompense actuelle (w4 = 0,3) ;
    2. DQN a été entraîné **après** ce commit (30 mai, 20 h 02), et PPO en août ;
    3. entre les deux, aucun commit ne change la formule de la récompense, seulement une valeur par défaut que l'entraînement n'utilisait pas ;
    4. les deux modèles attendent la même observation (147), lue dans leurs fichiers.

    Limite honnête : on ne sait pas si le code avait des modifications non commitées le jour de l'entraînement.
  - **Pour les nouveaux modèles (après T8),** la preuve est directe. Chaque modèle a un `metadata.json` qui contient les **poids de la récompense**, la taille d'observation, le **commit** et `git_dirty`. Si les deux fichiers ont les mêmes `reward_weights`, le même commit et `git_dirty = false`, les deux agents ont été entraînés avec exactement le même code. Et `load_model` refuse tout modèle dont la récompense ne correspond pas à celle demandée.

## T9 — Petites corrections
*Recommandé · [MESURE] · N8, N9, m20*

**Date :** 15 octobre 2026 (méthode 1 : code et tests écrits par l'IA, relus par moi)

**Prédiction (avant de modifier) :** pas de prédiction chiffrée écrite. Attendu :
- ces corrections ne changent aucun résultat de l'article, puisque ce sont des chemins, un écart-type dans un module non utilisé pour l'article, et une mesure de temps ;
- le test T4 (Fixed nominal, graine 42 : 4077,5 t/h) doit rester vert.

**Ce que j'ai changé :**
- **N8, chemins des modèles :**
  - `rl/model_paths.py` (nouveau) : `model_dir(results_dir, algo, scénario)` donne `data/results/<algo>_<scénario>`, à un seul endroit ;
  - les 4 entraînements l'utilisent par défaut (avant : `models/ppo_mine`, `models/dqn_mine`…) ;
  - le benchmark, `main.py` et `analysis/action_distribution.py` l'utilisent aussi.
- `main.py` :
  - le pipeline complet ne fait plus un entraînement PPO que personne ne relisait. Il lance le benchmark des 3 régimes, qui entraîne les modèles manquants au bon endroit ;
  - `training_dir` : une flotte nominale va dans le dossier relu par le benchmark, toute autre flotte dans `<algo>_custom_<N>t_<S>s_<D>d` ;
  - nouvelle option `--overwrite`.
- **Sécurité ajoutée** : `ensure_new_model` empêche les 4 entraînements d'écraser un modèle existant, sauf avec `overwrite=True`. Les scripts de diagnostic passent `overwrite=True`, parce qu'ils régénèrent volontairement leurs résultats.
- **N9, `rl/evaluate_agent.py` :**
  - écart-type d'échantillon `ddof=1`, et 0 pour un seul épisode ;
  - suppression de `benchmark_policy_inference`, qui chronométrait `policy_fn(None, None)`, donc une politique appelée **sans observation**. Son test (`tests/test_inference_benchmark.py`) est supprimé avec elle.
- **m20, mesure du temps :**
  - `MineEnv._get_obs` mémorise la durée de construction de l'observation dans `last_obs_time_ms`. C'est une mesure seulement, sans effet sur la simulation ;
  - `run_single_episode` enregistre séparément `decision_time_ms_mean/max` (l'appel de la politique) et `obs_time_ms_mean/max` (la construction de l'observation), à la place de `inference_time_ms_*` ;
  - ces colonnes sont ajoutées à `DYNAMIC_COLUMNS` de `load_results`.
- **`unseen_medium_load`** : retiré de `experiments/scenarios.py`, avec un commentaire qui donne sa définition. Une note dans le plan (T18) dit de le réintroduire.

**Pourquoi :**
- N8 : le modèle entraîné par le pipeline n'était pas celui évalué ;
- N9 : `np.std` sans `ddof=1` sous-estime l'écart-type d'un échantillon, et la mesure de latence portait sur une politique factice ;
- m20 : l'article annonçait une latence sans dire ce qu'elle mesurait ;
- `unseen_medium_load` plantait avec les modèles nominaux (observation de 180 au lieu de 147).

**Ce que le test vérifie :** `tests/test_small_fixes.py`, 12 tests :
- `model_dir` et algorithme inconnu ;
- pour chacun des 4 algorithmes, le dossier par défaut de l'entraînement est celui relu par le benchmark ;
- `training_dir` : nominal ou `custom_15t_3s_2d` ;
- `ensure_new_model` (avec ou sans `.zip`) ;
- un second entraînement dans le même dossier échoue avant de calculer, sauf avec `overwrite=True` ;
- écart-type sur 2 épisodes = |a − b| / √2 (ddof=1), et 0 sur un épisode ;
- `decision_time_ms` et `obs_time_ms` présents et positifs, l'ancienne colonne absente ;
- la mesure ne change pas la simulation (Fixed, graine 42 : 4077,5) ;
- tous les scénarios `unseen_*` restants ont la même taille d'observation que le nominal.

Suite complète : **75 passed**. Ruff ne signale rien.

**Résultat comparé à ma prédiction :** confirmée. Aucun résultat de l'article ne change, et T4 reste vert (4077,5).
- **Découverte sur la latence** (PPO v1, nominal, 3 épisodes) :

| | Moyenne | Maximum |
|---|---|---|
| Décision (appel de PPO) | **0,6 ms** | 1 ms (29 ms au premier épisode) |
| Construction de l'observation | **0,25 ms** | 0,6 ms |
| Latence complète | **≈ 0,9 ms** | |

  Le CSV v1 donne le même ordre de grandeur pour la décision : DQN 0,37 ms et PPO 0,60 ms en moyenne, maximum de 29 ms pour PPO. L'article annonce « environ 10 ms, maximum inférieur à 20 ms », ce qui **ne correspond à aucune mesure**. **À corriger en T31 :** décision ≈ 0,4 à 0,6 ms, observation ≈ 0,25 ms, et un pic au premier appel dû au démarrage de PyTorch.

**Ce que j'ai appris :**
- *Un seul endroit pour une convention* : quand l'entraînement et l'évaluation décident chacun du chemin d'un modèle, ils finissent par diverger (N8). `model_dir` supprime ce risque.
- *Protéger ce qui coûte cher* : un modèle, c'est jusqu'à 2 heures de calcul. Écraser doit être un choix explicite (`overwrite`).
- *`ddof=1`* : sur un échantillon, on divise par n − 1. `np.std` divise par n par défaut et sous-estime la dispersion.
- *Mesurer, c'est dire quoi* : une latence n'a de sens que si l'on précise ce qui est chronométré. Ici, la décision seule ou l'observation plus la décision.
- *Le premier appel est plus lent* : PyTorch s'initialise au premier `predict`. Un maximum isolé de 29 ms n'est pas représentatif.
- *Une mesure ne doit pas modifier ce qu'elle mesure* : le test T4 et `test_la_mesure_ne_change_pas_les_resultats` le vérifient.

**Commit :** `823559f` (15 octobre 2026).

**Questions « Savoir expliquer » :**
- [x] ce que mesurent exactement les 10 ms de latence rapportées dans l'article : **rien de ce qu'on peut retrouver dans les données.**
  - La v1 chronométrait seulement l'appel de la politique (`policy_fn`), sans la construction de l'observation. Le CSV v1 donne en moyenne 0,37 ms pour DQN et 0,60 ms pour PPO, avec un maximum de 29 ms pour PPO.
  - La fonction `benchmark_policy_inference` mesurait une politique appelée **sans observation**, donc rien de réel.
  - Depuis T9, on mesure séparément la décision (≈ 0,6 ms pour PPO) et la construction de l'observation (≈ 0,25 ms). La latence complète d'une décision est leur somme, environ 0,9 ms, et le pic de 29 ms vient du premier appel, au démarrage de PyTorch.
  - Phrase à mettre dans l'article : « Une décision de PPO prend en moyenne 0,9 ms (0,25 ms pour construire l'état, 0,6 ms pour l'appel du réseau), négligeable devant la durée d'un cycle de transport (environ 23 min). »

---

# Phase 2. Indicateurs (1 à 2 jours)

## T10 — Décomposer le temps d'attente
*Indispensable · [MESURE] · M2, M17*

**Date :** 15 octobre 2026 (méthode 1 : code et tests écrits par l'IA, relus par moi)

**Prédiction (avant de modifier) :** pas de prédiction chiffrée écrite. Les valeurs attendues étaient données par le plan (22,2 / 7,2 / 1,1 min). Attendu en plus :
- la simulation ne change pas : T4 reste vert et le total d'immobilisation reste identique à la v1 ;
- la récompense ne change pas, sinon il faudrait réentraîner.

**Ce que j'ai changé :**
- `simulation/entities.py` : `Truck` a 3 compteurs, `queue_wait_min` (file à la pelle ou à la verse), `repair_min` (réparation après une panne) et `voluntary_wait_min` (action ATTENDRE). `total_wait_min` devient une **propriété**, la somme des trois, et n'est plus un compteur.
- `env/mine_env.py` : chaque incrémentation va dans son compteur :
  - file à la pelle et file à la verse → `queue_wait_min` ;
  - réparation → `repair_min` ;
  - `_apply_wait_action` → `voluntary_wait_min`.

  L'appel à la récompense (`operational_wait_min = wait_load_min + wait_dump_min`) **n'est pas touché**. `_get_info` expose aussi les 3 totaux.
- `simulation/kpi.py` :
  - `compute_kpis` reçoit les 3 composantes et renvoie, en plus de l'ancien `avg_wait_min_per_truck` (gardé pour comparer avec la v1), `queue_wait_min_per_truck`, `repair_min_per_truck`, `voluntary_wait_min_per_truck` (par camion **et par poste**) et `queue_wait_min_per_cycle` ;
  - nouvelle fonction `fleet_kpis(camions, durée)`, utilisée par `run_benchmark.run_single_episode` et `evaluate_agent.evaluate_policy`, qui recopiaient les mêmes sommes.
- **Complément de T7 :**
  - `analysis/robustness.py` : `wait_column(df)` choisit l'**attente en file** si le CSV la contient (depuis T10), sinon l'immobilisation totale (CSV v1) ;
  - la heatmap, les barres et le Pareto utilisent cette colonne, et leur libellé dit laquelle est affichée (`WAIT_LABELS` dans `plots/common.py`).

**Pourquoi :** `total_wait_min` mélangeait la file, les réparations et l'action ATTENDRE. Puis `kpi.py` le divisait par le nombre de camions (pas par le nombre de cycles). Le « temps d'attente » de l'article était donc une immobilisation **cumulée sur tout le poste**, pannes comprises, qu'on pouvait lire à tort comme une attente par cycle (M2, M17).

**Test prévu par le plan :** sur un épisode, la somme des trois compteurs égale l'ancien `total_wait_min` ; avec l'Affectation fixe en nominal, sur 30 graines : environ 22,2 min de file et 7,2 min de panne par camion, soit environ 1,1 min de file par cycle.

**Ce que le test vérifie :** `tests/test_wait_decomposition.py`, 6 tests :
- `compute_kpis` : unités (par camion sur le poste, par cycle) ;
- `Truck.total_wait_min` = somme des trois compteurs ;
- **même total que la v1** : Fixed, nominal, graine 42, la somme des 3 compteurs vaut 303,611025 min, la valeur du CSV de l'article ;
- l'action ATTENDRE n'alimente que `voluntary_wait_min` ;
- **valeurs de référence du plan** sur 30 graines (voir ci-dessous) ;
- la robustesse utilise l'attente en file, et on retrouve le Δ du tableau 13.

Suite complète : **81 passed**. Ruff ne signale rien.

**Résultat comparé à ma prédiction :** confirmée. La simulation et la récompense sont inchangées (T4 vert, total identique à la v1).

| Affectation fixe, nominal, 30 graines | Plan | Mesuré |
|---|---|---|
| File par camion et par poste | ≈ 22,2 min | **22,17** |
| Réparation par camion et par poste | ≈ 7,2 min | **7,18** |
| ATTENDRE | 0 | **0** |
| File par cycle | ≈ 1,1 min | **1,14** |
| Immobilisation totale par camion et par poste | (le « 29 min » de la v1) | **29,36** |
| Cycles par camion et par poste | | **19,4** |

Autres régimes (Affectation fixe, par camion et par poste) :
- forte charge : file 35,58, réparation 7,58, 18,7 cycles ;
- fortes pannes : file 19,47, réparation 37,72, 18,2 cycles.

- **Δ attente en file du tableau 13**, Affectation fixe :
  - forte charge : **+60,44 %** (l'article écrit +60,5, l'arrondi correct est **+60,4**) ;
  - fortes pannes : −12,2 % (identique).

  **À corriger en T31.**
- **Limite :** le CSV v1 ne contient pas l'attente en file. Pour remplir toute la colonne « Δ attente en file » (8 politiques), il faut relancer le benchmark. Décision : attendre la campagne finale (T31), puisque la phase 3 invalidera ces résultats, sauf si ma directrice a besoin du tableau complet avant.

**Ce que j'ai appris :**
- *Un indicateur doit dire son unité* : « attente moyenne » ne veut rien dire tant qu'on ne précise pas par camion ou par cycle, par poste ou par épisode, et ce qu'elle contient (file, panne, attente volontaire).
- *Séparer les causes* : la file dépend de la politique de dispatching, la panne non (c'est le hasard du simulateur). Les mélanger attribuait aux politiques un temps qu'elles ne contrôlent pas.
- *Décomposer sans changer la simulation* : on remplace un compteur par trois, et un test vérifie que leur somme redonne exactement l'ancien total (303,611025).
- *Ne pas toucher la récompense* : la modifier aurait obligé à réentraîner tous les agents. Une correction de mesure doit rester une correction de mesure.
- *Vérifier un chiffre avec le code, pas de tête* : j'avais conclu trop vite que +60,5 % était retrouvé ; le test a montré +60,44 %.
- *Garder la compatibilité* : l'ancien CSV n'a pas la nouvelle colonne, donc l'analyse se rabat sur l'ancienne et le dit dans le libellé de la figure.

**Commit :** `74767d8` (15 octobre 2026).

> **Mise à jour par T11 (`e8e12d8`) :** les compteurs sont maintenant bornés à la fin du poste. Les valeurs ci-dessus deviennent, pour l'Affectation fixe en nominal : file 21,80, réparation 6,72, file par cycle 1,13 min ; Δ file +59,79 % en forte charge et −12,02 % en fortes pannes. Voir l'entrée T11.

**Questions « Savoir expliquer » :**
- [x] pourquoi « 29 min par cycle » était faux : 29,36 min, c'est l'immobilisation **totale d'un camion sur tout le poste** (8 h), et non l'attente d'un cycle. Il y a deux erreurs.
  1. **Mauvais dénominateur** : le total de la flotte était divisé par le nombre de camions, pas par le nombre de cycles.
  2. **Mauvais contenu** : le total mélangeait la file (22,17 min) et les réparations après panne (7,18 min), qui ne dépendent pas de la politique.

  La vraie attente en file par cycle est **1,14 min**, environ 25 fois moins.
- [x] combien de cycles fait un camion par poste : avec l'Affectation fixe, **19,4 cycles** en nominal (233 cycles pour 12 camions), 18,7 en forte charge, 18,2 en fortes pannes. Avec FIFO, environ 15,6 en nominal. Un cycle dure donc environ 480 / 19,4 ≈ 25 min, file et pannes comprises.

## T11 — Utilisation bornée au poste et utilisation des pelles
*Indispensable · [MESURE] · M2 bis, m18*

**Date :** 15 octobre 2026 (méthode 1 : code et tests écrits par l'IA, relus par moi)

**Prédiction (avant de modifier) :** pas de prédiction chiffrée écrite. Le plan annonçait une utilisation des camions d'environ 93,9 % au lieu de 96,4 % pour l'Affectation fixe. Attendu en plus : la production et la récompense ne changent pas, puisque seuls les compteurs sont bornés, pas la simulation.

**Ce que j'ai changé :**
- `env/mine_env.py` : nouvelle fonction `_in_shift(début, durée)` = max(0, min(durée, 480 − début)), c'est-à-dire la part d'une activité située avant la fin du poste. Elle est appliquée aux **9 compteurs de temps** :
  - temps actif : trajet vers la pelle, chargement, trajet vers la verse, déchargement, retour ;
  - file à la pelle et file à la verse ;
  - réparation ;
  - action ATTENDRE.

  La simulation avance toujours de la durée complète : seule la comptabilité est bornée.
- `simulation/entities.py` : `Shovel.busy_min`, le temps de chargement pendant le poste (borné de la même façon).
- `simulation/kpi.py` :
  - `compute_kpis` reçoit `shovel_busy_min` et `shovel_count`, et renvoie `shovel_utilization_pct` à côté de `utilization_pct` (camions) ;
  - `fleet_kpis(camions, pelles, durée)` ;
  - mise à jour des appels dans `run_benchmark` et `evaluate_agent`.
- **Choix de définition :** j'ai borné les **attentes aussi**, pas seulement le temps actif. Sans cela, une panne commençant à 475 min compterait 30 min de réparation dans un poste qui n'a plus que 5 min, et le test « actif + attentes ≤ 480 » échouerait.
- **Conséquence sur T10**, valeurs mises à jour dans `tests/test_wait_decomposition.py` :

| Affectation fixe | T10 (non borné) | T11 (borné au poste) |
|---|---|---|
| File par camion et par poste, nominal | 22,17 | **21,80** |
| Réparation par camion et par poste, nominal | 7,18 | **6,72** |
| File par cycle, nominal | 1,14 | **1,13** |
| Immobilisation totale par camion, nominal | 29,36 | **28,53** |
| Δ file, forte charge | +60,44 % | **+59,79 %** |
| Δ file, fortes pannes | −12,21 % | **−12,02 %** |
| Total d'immobilisation, graine 42 | 303,611 (= v1) | **299,886** |

- **Non borné volontairement :** le **carburant** du dernier cycle reste compté en entier. Le plan ne le demande pas ici, c'est le sujet de T14 (carburant paramétrable et calibré).

**Pourquoi :** le dernier cycle de chaque camion était compté jusqu'au bout, même après 480 min, ce qui gonflait l'utilisation (96,4 %) (M2 bis). Et on ne mesurait pas l'occupation des pelles, alors que c'est elle qui explique d'où viennent les files (m18).

**Test prévu par le plan :** pour chaque camion, temps actif + attentes ≤ 480 min ; l'utilisation des pelles reste inférieure à 100 %.

**Ce que le test vérifie :** `tests/test_shift_bounds.py`, 9 tests :
- `_in_shift` : activité entièrement dans le poste, à cheval sur la fin, entièrement après ;
- **test du plan**, pour les 3 régimes, avec l'Affectation fixe **et** une politique aléatoire qui utilise ATTENDRE, sur 3 graines : pour chaque camion, temps actif + attentes ≤ 480 min, et pour chaque pelle, 0 < temps de chargement ≤ 480 min (6 tests) ;
- la borne ne change pas la simulation : production 4077,5 t/h et récompense 210,9715… (références v1) ;
- utilisation, Affectation fixe sur 30 graines : camions 94,06 % en nominal ; pelles 33,92 % en nominal et 49,00 % en forte charge, toujours < 50 %.

Suite complète : **90 passed**. Ruff ne signale rien.

**Résultat comparé à ma prédiction :** confirmée. La simulation est inchangée (T4 vert).

| Affectation fixe, 30 graines | v1 | T11 |
|---|---|---|
| Utilisation des camions, nominal | 96,4 % | **94,1 %** (plan : ≈ 93,9 %) |
| Utilisation des camions, forte charge | 93,9 % | **91,3 %** |
| Utilisation des camions, fortes pannes | 90,8 % | **88,6 %** |
| Utilisation des pelles, nominal | — | **33,9 %** |
| Utilisation des pelles, forte charge | — | **49,0 %** |
| Utilisation des pelles, fortes pannes | — | **31,9 %** |

L'écart de 0,2 point avec les 93,9 % du plan est probablement dû à l'estimation du plan, faite autrement. L'ordre de grandeur (−2,3 points) est retrouvé.

**Ce que j'ai appris :**
- *Une période de mesure a des bornes* : un indicateur « par poste » ne doit compter que ce qui se passe pendant le poste. Le dernier cycle qui déborde n'appartient pas au poste.
- *Séparer comptabilité et simulation* : on peut borner ce qu'on **compte** sans changer ce qu'on **simule**. Le test de la production et de la récompense le prouve.
- *Une correction peut en modifier une autre* : borner les attentes a changé les chiffres de T10. Il faut le dire et mettre à jour les tests, pas le cacher.
- *Tester avec une politique « bizarre »* : la politique aléatoire utilise ATTENDRE, que l'Affectation fixe n'utilise jamais. Sans elle, le compteur `voluntary_wait_min` n'aurait pas été vérifié.
- *Regarder les deux côtés de la file* : l'utilisation des camions dit s'ils travaillent, celle des pelles dit si la capacité de chargement est le goulot d'étranglement. Ici, ce n'est pas le cas.

**Commit :** `e8e12d8` (15 octobre 2026).

**Questions « Savoir expliquer » :**
- [x] pourquoi les pelles sont occupées moins de 50 % du temps même avec 18 camions, et ce que cela dit du facteur d'appariement :
  - **Capacité des pelles :** une pelle charge 140 t en 2 min, soit 4 200 t/h, donc **12 600 t/h** pour les 3 pelles. L'Affectation fixe produit 4 078 t/h en nominal (≈ 32 % de la capacité) et 5 874 t/h en forte charge (≈ 47 %). C'est presque exactement l'utilisation mesurée (33,9 % et 49,0 %) : **les pelles passent la majorité du poste à attendre des camions.**
  - **Facteur d'appariement :** FA = (N × temps de chargement) / (S × temps de cycle) = (12 × 2) / (3 × 22,7) ≈ **0,35** en nominal et (18 × 2) / (3 × 22,7) ≈ **0,53** en forte charge. FA < 1 veut dire qu'il n'y a pas assez de camions pour occuper les pelles en continu. L'utilisation des pelles est d'ailleurs à peu près égale au FA.
  - **Ce que cela dit :** même la « forte charge » n'est pas une saturation des pelles. Les files observées ne viennent pas d'un manque de capacité de chargement, mais de la **mauvaise répartition** des camions entre les pelles. C'est pourquoi « Pelle la plus proche », qui envoie toute la flotte à la même pelle, crée de longues files alors que les deux autres pelles sont libres. La bonne politique n'est pas celle qui ajoute des camions, mais celle qui les **répartit**.

## T12 — Journal des cycles et test de cohérence
*Recommandé · [MESURE] · M2, M11*

**Date :** 15 octobre 2026 (méthode 1 : code et tests écrits par l'IA, relus par moi)

**Prédiction (avant de modifier) :** pas de prédiction chiffrée écrite. Attendu :
- la somme des composantes égale la durée de chaque cycle (c'est le test du plan) ;
- le journal ne change pas la simulation.

**Ce que j'ai changé :**
- `simulation/entities.py` : nouvelle classe `CycleRecord` (immuable), une entrée par décision d'un camion :
  - `kind` : `"cycle"` (affectation pelle → verse) ou `"wait"` (action ATTENDRE) ;
  - `truck_id`, `start_min`, `end_min`, `shovel_id`, `dump_id` ;
  - **9 composantes** : `travel_to_shovel_min`, `shovel_wait_min`, `load_min`, `travel_to_dump_min`, `dump_wait_min`, `unload_min`, `travel_back_min`, `repair_min`, `voluntary_wait_min` ;
  - `delivered_t` : 140 t, ou 0 si le cycle se termine après la fin du poste ;
  - `components_min` (la somme des 9) et `as_dict()` (pour construire un DataFrame).

  Suppression du champ `Truck.history`, jamais utilisé.
- `env/mine_env.py` : paramètre `log_cycles=False`. Activé, chaque `step` ajoute un `CycleRecord` à `env.cycle_log` (une entrée `"wait"` pour ATTENDRE) ; `reset` vide le journal. Les durées enregistrées sont les durées **simulées complètes** (sans la borne au poste de T11), pour que fin − début = somme exacte.

**Pourquoi :** les compteurs ne gardent que des totaux. Impossible de vérifier qu'un cycle est cohérent, ni de voir ce que fait un camion minute par minute (M2, M11). Le journal sert aussi de base à l'analyse comportementale (T12 bis).

**Test prévu par le plan :** pour chaque cycle, fin − début = somme des composantes, à 10⁻⁶ près.

**Ce que le test vérifie :** `tests/test_cycle.py`, 13 tests :
- journal désactivé par défaut ;
- **test du plan**, dans les 3 régimes, avec l'Affectation fixe et une politique aléatoire (6 tests) : fin − début = somme des composantes à 10⁻⁶ près, et toutes les composantes ≥ 0 ;
- **plus fort que le plan**, la chronologie continue de chaque camion (2 tests) : le premier cycle commence à 0, et chaque décision commence exactement à la fin de la précédente ;
- cohérence avec les compteurs : tonnage livré et nombre de cycles du journal = compteurs de l'épisode ;
- ATTENDRE journalisé (entrée `"wait"`, sans pelle, ≥ 0,2 min) ;
- le journal ne change pas la simulation (4077,5 t/h, même récompense) ;
- `reset` vide le journal.

Suite complète : **103 passed**. Ruff ne signale rien.

**Résultat comparé à ma prédiction :** confirmée. Sur un épisode de l'Affectation fixe (nominal, graine 42) :
- 245 entrées ;
- écart maximal entre fin − début et somme des composantes : **8,5 × 10⁻¹⁴** min ;
- aucun trou dans la chronologie d'aucun camion ;
- tonnage du journal 32 620 t = compteur.

**Ce que j'ai appris :**
- *Un total ne se vérifie pas, une décomposition oui* : avec les 9 composantes, chaque cycle se contrôle par une simple addition.
- *Tester une propriété plus forte que demandé* : la continuité de la chronologie prouve qu'aucune minute du poste n'est oubliée, ce que la somme par cycle seule ne prouve pas.
- *Un journal optionnel ne coûte rien* : désactivé par défaut, il n'alourdit pas les 2 millions de pas d'un entraînement.
- *Enregistrer la donnée brute, borner ensuite* : le journal garde les durées complètes, la borne au poste (T11) s'applique seulement aux indicateurs.
- *Un journal rend visibles des hypothèses cachées* : il montre que le camion retourne toujours à la même pelle avant sa décision suivante (voir ci-dessous).

**Commit :** `112e2a9` (15 octobre 2026).

**Questions « Savoir expliquer » :**
- [x] ce qui se passe, minute par minute, entre le moment où un camion reçoit une affectation et celui où il redevient disponible (première question du module 1). Premier cycle du camion T1, Affectation fixe, nominal, graine 42, d'après le journal :

| Minute | Ce qui se passe | Durée (min) |
|---|---|---|
| 0,00 | T1 reçoit l'affectation **pelle S1 → verse D1**, au dépôt | |
| 0,00 → 4,19 | trajet à vide du dépôt vers la pelle S1 | 4,19 |
| 4,19 | file à la pelle : S1 est libre, pas d'attente | 0 |
| 4,19 → 6,13 | chargement de 140 t (durée aléatoire, loi log-normale, moyenne 2 min) | 1,94 |
| 6,13 → 14,38 | trajet en charge vers la verse D1 (plus lent qu'à vide) | 8,25 |
| 14,38 | file à la verse : D1 est libre | 0 |
| 14,38 → 15,11 | déchargement (moyenne 1 min) | 0,73 |
| 15,11 → 22,22 | **retour à vide vers la même pelle S1** | 7,12 |
| 22,22 | tirage de panne (2 % en nominal, 10 % en fortes pannes) : pas de panne ici. Sinon, réparation de 10 à 30 min. | 0 |
| **22,22** | les 140 t sont comptées (le cycle finit avant 480 min). T1 redevient disponible **à la pelle S1** et attend sa décision suivante. | **total 22,22** |

  Ce qu'il faut savoir dire en plus :
  - s'il y a une file, l'attente se place **avant** le chargement ou le déchargement : le camion attend que la pelle ou la verse soit libre ;
  - une affectation engage le camion pour un **cycle complet** (aller, chargement, verse, retour). Il n'y a pas de réaffectation en route ;
  - le camion revient **toujours à la même pelle** : la décision suivante se prend donc à la pelle, pas à la verse comme dans une vraie mine. Dans le journal, `travel_to_shovel_min` ne vaut plus que 0 après le premier cycle. C'est la limite que corrige **T16 bis** (décision à la verse) ;
  - avec ATTENDRE, le camion reste immobile au maximum 5 min (au moins 0,2 min), jusqu'au prochain événement du système.

## T12 bis — Analyse comportementale sur 30 réplications
*Recommandé · [MESURE] · M16, C3*

**Date :** 15 octobre 2026 (méthode 1 : code et tests écrits par l'IA, relus par moi)

**Prédiction (avant de modifier) :** pas de prédiction chiffrée écrite. Le plan donnait les valeurs attendues : 100 % à la pelle 1 pour les deux règles de distance, et 87 % / 64 % d'ATTENDRE pour PPO / DQN en forte charge.

**Ce que j'ai changé :**
- `experiments/policies.py` (nouveau) : les 8 politiques construites à un seul endroit, avec la signature commune `policy_fn(obs, info, env)` :
  - `heuristic_policies(scénario)` ;
  - `agent_policies(scénario, dossier, …)`, qui charge sans jamais entraîner, avec le contrôle de T8 ;
  - `agent_policy_fn` et `make_env`.

  `run_benchmark` les utilise aussi : il n'y a plus de copie du code des heuristiques.
- `analysis/behavior.py` (nouveau) : à partir du journal des cycles (T12), pour chaque politique × régime × graine, il calcule :
  - la part des cycles par pelle et la part des décisions ATTENDRE ;
  - la durée moyenne d'un cycle et le nombre de décisions ;
  - la file, la réparation et l'ATTENDRE par camion (T10).

  Il écrit `behaviour_runs.csv` (une ligne par graine), `behaviour_summary.csv` et `tab_behaviour.tex`, c'est-à-dire **le tableau de l'article produit par mon code**.
- `plots/behaviour.py` (nouveau) : la figure sur 30 réplications, avec une barre empilée par politique (part des cycles par pelle) et la part d'ATTENDRE, **sans légende interprétative**.
- `plots/behavioral_analysis.py` (figure sur un seul épisode) : légendes « greedy → hot spot » et « adaptive load balancing » retirées, et formule de l'entropie écrite dans la docstring : H = −Σ p_s log₂ p_s / log₂ S.
- `analysis/action_distribution.py` : 30 graines au lieu de 10.

**Pourquoi :** le tableau `tab:behaviour` de la v2 venait de l'audit, pas de mon code. La figure reposait sur un seul épisode, avec des légendes qui interprétaient plus que ce qu'elle montrait (M16, C3).

**Test prévu par le plan :** avec les modèles v1, tu retrouves le tableau de la v2 : Pelle la plus proche et Plus court chemin à 100 % sur la pelle 1 dans les trois régimes ; en forte charge, ATTENDRE dans 87 % des décisions pour PPO et 64 % pour DQN.

**Ce que le test vérifie :** `tests/test_behavior.py`, 5 tests :
- les indicateurs d'un épisode, sur un journal fabriqué à la main ;
- les heuristiques dans les 3 régimes : 100 % à la pelle 1 pour Nearest et ShortestPath, répartition équilibrée pour FIFO et Fixed, aucun ATTENDRE ;
- le format exact d'une ligne du tableau LaTeX ;
- la figure ;
- **le test lent**, seulement avec `RUN_SLOW=1` (3 min 30) : modèles v1, forte charge, 30 graines, PPO 87 % et DQN 64 % d'ATTENDRE.

Suite complète : **107 passed + 1 lent passé**. Ruff ne signale rien.

**Résultat comparé à ma prédiction :** confirmée, et même mieux. Le calcul complet (8 politiques × 3 régimes × 30 graines, 6 min 30) reproduit **exactement, ligne pour ligne, le tableau `tab:behaviour` de l'article v2**, pour les 8 politiques. Il retrouve aussi la durée de cycle citée dans le texte : FIFO 29,8 min, Affectation fixe 24,1 min.
- Chiffres nouveaux, en forte charge : PPO prend **4 794 décisions par épisode**, contre environ 350 pour l'Affectation fixe, et cumule **54,3 min d'ATTENDRE par camion et par poste**. DQN : 996 décisions et 9,0 min.

**Ce que j'ai appris :**
- *Une figure sur un seul épisode illustre, elle ne prouve pas* : il faut des moyennes sur toutes les réplications, et une légende qui décrit au lieu d'interpréter.
- *Produire chaque chiffre de l'article avec son propre code* : on peut alors le vérifier, le refaire et l'expliquer.
- *Un pourcentage de décisions peut tromper* : 87 % d'ATTENDRE ne veut pas dire 87 % du temps à attendre. Ce sont surtout beaucoup de **petites** attentes (≥ 0,2 min).
- *Un seul endroit pour construire les politiques* : le benchmark et l'analyse font jouer exactement les mêmes politiques.
- *Séparer tests rapides et tests lents* : la suite quotidienne reste rapide, et le test lent se lance à la demande.

**Commit :** `d1cec48` (15 octobre 2026).

**Questions « Savoir expliquer » :**
- [x] pourquoi ATTENDRE domine chez PPO en forte charge (lien avec T19) :
  - **Les faits** : PPO prend 4 794 décisions par épisode (environ 350 pour l'Affectation fixe), dont 87 % d'ATTENDRE. Une ATTENDRE dure au minimum 0,2 min et s'arrête au prochain événement du système. Ce sont donc de très nombreuses attentes très courtes, soit 54 min par camion et par poste au total. Ses cycles restent répartis de façon équilibrée entre les pelles (34 / 34 / 32).
  - **L'explication** : dans la récompense, la pénalité d'attente d'une décision vaut au plus −1, et une ATTENDRE de 0,2 min ne coûte presque rien en production. En forte charge, partir tout de suite expose souvent à une file. L'agent a appris qu'attendre un peu est presque gratuit et évite parfois cette file. Il répète donc l'action.
  - **Le lien avec T19** : il faut encadrer l'action ATTENDRE. Un masque d'actions peut l'interdire quand rien ne la justifie, ou on peut lui donner un coût proportionnel à sa durée. Sinon, un agent peut « tricher » en multipliant les décisions vides.
- [x] ce que la figure démontre et ce qu'elle ne démontre pas :
  - **Elle démontre** la répartition moyenne des cycles entre les pelles et la fréquence d'ATTENDRE, pour chaque politique et chaque régime, mesurées sur 30 réplications. Par exemple : les deux règles de distance envoient 100 % des cycles à la pelle 1, et les agents profonds entraînés en fortes pannes n'utilisent jamais la pelle 3.
  - **Elle ne démontre pas** une stratégie, une intention ou un « équilibrage adaptatif ». Elle ne dit pas **pourquoi** une politique choisit une pelle, ni si ce choix est **bon** : c'est le rôle des comparaisons de productivité et d'attente (T6, T7). Elle ne montre pas non plus la **durée** des ATTENDRE, seulement leur nombre. L'ancienne figure, faite sur un seul épisode avec des légendes interprétatives, laissait croire le contraire.

---

# Phase 3. Simulateur (3 à 5 jours)

## T13 — Un flux aléatoire par processus
*Indispensable · [RÉÉVAL] · M4*

**Date :** 15 octobre 2026 (méthode 1 : code et tests écrits par l'IA, relus par moi). Première tâche de la phase 3.

**Prédiction (avant de modifier) :** pas de prédiction chiffrée écrite. Annoncé par le plan :
- tous les chiffres d'un épisode bougent, donc le test T4 (valeurs v1) passe au rouge ;
- la **loi** du simulateur ne change pas, donc les moyennes sur 30 graines restent proches, et les modèles entraînés restent valables en principe.

**Ce que j'ai changé :**
- `env/mine_env.py` : le générateur unique `self.rng = Random(graine)` est remplacé par des **flux indépendants**, créés à chaque `reset` avec `numpy.random.SeedSequence(graine).spawn(3 + N)` :
  - `rng_travel` : temps de trajet ;
  - `rng_service` : chargements et déversements ;
  - `rng_road` : dégradation des pistes ;
  - `rng_breakdowns[i]` : **un flux de pannes par camion**. À chaque cycle, il tire **toujours deux nombres** (« panne ou non », puis « durée »), même sans panne : le k-ième cycle d'un camion reçoit ainsi la même panne et la même durée quelle que soit la politique. C'est équivalent à tirer la liste à l'avance, comme le demande le plan, mais sans calculer une liste de longueur inconnue.

  Chaque flux est un `random.Random` initialisé par un enfant de la `SeedSequence` (fonction `_stream`), pour garder les lois existantes (`lognormvariate`, `uniform`).
- **Nouvelles références de non-régression**, suivies par Git, puisque `data/` ne l'est pas :
  - `tests/fixtures/regression_reference.csv` (Fixed et FIFO, nominal, graines 42 à 46 : productivité, récompense, immobilisation) ;
  - le script qui les produit, `tests/fixtures/generate_regression_reference.py`, avec sa docstring qui dit quand le relancer ;
  - `tests/reference_values.py` (accès partagé).
- `tests/test_regression.py` : il compare à ces références. Les valeurs v1 restent archivées dans `data/results_v1_article`.
- Les tests « ceci ne change pas la simulation » (journal des cycles T12, borne au poste T11, chronométrage T9) se comparent maintenant aux références, et non plus au 4077,5 de la v1.
- Les tests de T10 et T11 ont leurs valeurs recalculées, avec l'historique v1 → T11 → T13 dans chaque docstring.

**Pourquoi :** avec un seul générateur, les tirages sont consommés dans l'ordre des événements. Dès que deux politiques font un choix différent, l'ordre change et **tout le reste de l'épisode diverge** : même graine ne voulait plus dire mêmes aléas. La première panne n'était identique que pour 5 graines sur 30. L'appariement par graine de T6 perdait donc une partie de son intérêt (M4).

**Test prévu par le plan :** avec la même graine, le camion 3 tombe en panne au même numéro de cycle sous FIFO et sous l'Affectation fixe, avec la même durée de réparation.

**Ce que le test vérifie :** `tests/test_random_streams.py`, 33 tests :
- **le test du plan** : camion T3, fortes pannes, graine 42 ;
- **plus fort que le plan** (30 tests) : pour **chacune des 30 graines** et **chaque camion**, les pannes des cycles réalisés par les deux politiques sont identiques (numéro de cycle et durée) ;
- reproductibilité : même graine, même épisode, au bit près ; autre graine, autre épisode ;
- flux indépendants : 3 + N générateurs distincts, qui produisent des suites différentes.

Suite complète : **140 passed**. Ruff ne signale rien.

**Résultat comparé à ma prédiction :** confirmée.
- **Test du plan, camion T3** : 4ᵉ cycle sous FIFO (début 111,8 min) et sous l'Affectation fixe (début 99,3 min), réparation de **14,87 min** dans les deux cas. Même cycle, même durée, minute différente.
- **Chiffres d'un épisode** : ils changent tous. Fixed, nominal, graine 42 : 4 095,0 t/h au lieu de 4 077,5, et récompense 212,63 au lieu de 210,97. Le test T4 est passé au rouge, comme prévu, puis il a été rebasé sur les nouvelles références.
- **Moyennes sur 30 graines** (Affectation fixe) : elles changent peu, comme prévu.

| | T11 (ancien tirage) | T13 |
|---|---|---|
| File par camion, nominal | 21,80 | **22,26** |
| Réparation par camion, nominal | 6,72 | **7,37** |
| File par cycle, nominal | 1,13 | **1,15** |
| Utilisation des camions, nominal | 94,06 % | **93,83 %** |
| Utilisation des pelles, nominal / forte charge | 33,92 / 49,00 % | **33,79 / 48,81 %** |
| Δ file forte charge / fortes pannes | +59,79 / −12,02 % | **+56,46 / −13,78 %** |

- **Conséquence pour l'article** : tous les résultats chiffrés de la v2 seront à recalculer en T31. Les modèles restent utilisables en principe, puisque la loi est la même (tâche [RÉÉVAL], pas [RÉENTR]).

**Ce que j'ai appris :**
- *Un seul générateur couple tout* : un choix différent décale tous les tirages suivants, et deux politiques jouées « avec la même graine » ne voient pas les mêmes aléas.
- *Un flux par processus* : séparer les sources de hasard rend l'aléa de chaque processus indépendant des décisions prises ailleurs.
- *Aligner par événement, pas par minute* : tirer deux nombres à chaque cycle, même sans panne, garde la k-ième panne attachée au k-ième cycle.
- *Une référence de test peut devoir changer* : quand une modification voulue change la simulation, on régénère la référence avec un script versionné, et on écrit pourquoi. On ne relâche pas le test.
- *Les tests « ceci ne change rien » doivent se comparer à une référence partagée*, pas à un chiffre copié dans chaque fichier : il n'y a plus qu'un endroit à mettre à jour.

**Commit :** `a2a8700` (15 octobre 2026).

**Questions « Savoir expliquer » :**
- [x] ce que sont des nombres aléatoires communs : une technique de simulation qui consiste à faire affronter aux politiques comparées **les mêmes réalisations du hasard**, ici les mêmes pannes au même cycle de chaque camion, avec les mêmes durées. L'écart de performance mesuré vient alors de la politique, et non d'une différence de chance. En pratique : une graine par réplication, et un flux aléatoire séparé pour chaque processus.
- [x] pourquoi ils réduisent la variance d'une différence : Var(A − B) = Var(A) + Var(B) − 2 Cov(A, B). Avec les mêmes aléas, une « mauvaise journée » (beaucoup de pannes) pénalise les deux politiques, donc A et B sont **corrélés positivement**, et la covariance diminue la variance de la différence. Mesuré en fortes pannes, Fixed − FIFO sur 30 graines :
  - mêmes graines : corrélation **0,58**, écart-type de la différence **58,8 t/h** ;
  - graines différentes : corrélation **0,23**, écart-type **74,2 t/h**.

  On obtient une comparaison plus précise, avec le même nombre d'épisodes. C'est ce qu'exploitent les tests appariés de T6 (Wilcoxon apparié, bootstrap sur les différences).
- [x] pourquoi une panne ne peut quand même pas survenir à la même minute pour deux politiques : la panne est attachée au **numéro de cycle** du camion, pas à l'heure. Deux politiques envoient le camion vers des pelles et des verses différentes, avec des trajets et des files différents. Son 4ᵉ cycle ne commence donc pas à la même minute : 111,8 min sous FIFO contre 99,3 min sous l'Affectation fixe. Faire coïncider aussi la minute obligerait à imposer les mêmes trajets, donc la même politique, ce qui supprimerait la comparaison.

## T14 — Carburant paramétrable et calibré
*Recommandé · [RÉENTR] · M20*

**Date :** 15 octobre 2026 (méthode 1 : code et tests écrits par l'IA, relus par moi)

**Décision :** **calibrer** (mon choix, parmi « calibrer », « garder et documenter comme limite » ou « demander à la directrice »). Toutes les consommations sont multipliées par un **seul facteur**, pour viser le milieu de la plage publiée.

**Prédiction (avant de modifier) :** pas de prédiction chiffrée écrite. Attendu :
- le simulateur consomme beaucoup moins qu'un vrai tombereau ;
- un facteur unique ne change pas les écarts relatifs entre politiques.

**Ce que j'ai changé :**
- **Mesure de départ** (code de T13, Affectation fixe, nominal, 30 graines) : **14,635 L/h** par camion, **6,05 L par cycle**, 0,043 L/t.
- **Référence publiée** : Cat 785D (≈ 136 t de charge utile, le plus proche des 140 t), Caterpillar Performance Handbook (édition 2022, données télématiques VisionLink) : 54,2 à 81,4 L/h en usage léger, **81,4 à 108,5 L/h en usage moyen**, 108,5 à 135,6 L/h en usage intensif.
- `simulation/constants.py` : toutes les constantes du carburant regroupées, **avec leur source en commentaire** :
  - `FUEL_CALIBRATION_FACTOR` = 95 / 14,635 ≈ **6,49** ;
  - `FUEL_L_PER_KM_LOADED` = 0,65 × 6,49 ≈ 4,22 L/km ;
  - `FUEL_L_PER_KM_EMPTY` = 0,45 × 6,49 ≈ 2,92 L/km ;
  - `FUEL_IDLE_L_PER_H` = 10 × 6,49 ≈ 64,9 L/h ;
  - `FUEL_ROAD_STATE_PENALTY` = 0,5, non calibré faute de source, à écrire comme hypothèse ;
  - `FUEL_OBS_MAX_L` = 500 × 6,49 (normalisation de l'observation).
- `simulation/fuel_model.py` : `FuelParams` (dataclass), qui rend le modèle **paramétrable**, et `DEFAULT_FUEL`. Les deux fonctions reçoivent les paramètres en argument.
- `env/mine_env.py` : `MineEnv(fuel=DEFAULT_FUEL)`, transmis à tous les calculs de carburant.
- `env/observation_builder.py` : `_MAX_FUEL_L = FUEL_OBS_MAX_L`.
- `baselines/shortest_path_policy.py` : poids du carburant γ = 0,2 / 6,49 au lieu de 0,2. Sans cela, le carburant pesait 6,49 fois plus dans le coût de l'heuristique, et ses décisions changeaient dans **11 épisodes sur 30**.

**Pourquoi :** les constantes (0,65 et 0,45 L/km, 10 L/h) étaient marquées « non calibrées » et dispersées dans le code. Le simulateur consommait environ 6,5 fois moins qu'un vrai tombereau de 140 t, donc les valeurs absolues de carburant (L/t) n'avaient pas de sens physique (M20).

**Ce que le test vérifie :** `tests/test_fuel.py`, 9 tests :
- un seul facteur pour toutes les consommations (trajet chargé, à vide, ralenti) ;
- le modèle est paramétrable ;
- **calibration** : Affectation fixe, nominal, 30 graines, 95,0 L/h, dans la plage « usage moyen » du Cat 785D, et 39,3 L par cycle ;
- **l'agent voit les mêmes observations**, dans les 3 régimes (3 tests) : avec le carburant calibré, chaque observation d'un épisode complet est identique (à 10⁻⁵ près) à celle d'avant T14 ;
- **« Plus court chemin » décide comme avant**, dans les 3 régimes, sur 5 graines (3 tests) : même productivité, et carburant multiplié exactement par le facteur.

Suite complète : **149 passed**. Ruff ne signale rien.

**Résultat comparé à ma prédiction :** confirmée.

| | Avant T14 | Après T14 | Cat 785D (usage moyen) |
|---|---|---|---|
| L/h par camion (Affectation fixe, nominal) | 14,6 | **95,0** | 81,4 à 108,5 |
| L/h, FIFO et Fixed, 3 régimes | — | **93,7 à 95,2** | |
| L par cycle (Fixed / FIFO, nominal) | ≈ 6 | **39,3 / 48,5** | |
| L par tonne (Fixed / FIFO, nominal) | 0,043 | **0,281 / 0,347** | |

- **Découverte : T14 n'oblige plus à réentraîner.** Le plan la classait [RÉENTR], parce que le carburant fait partie de l'observation. En multipliant la normalisation par le même facteur que les consommations, l'agent voit exactement les mêmes nombres. La récompense n'utilise pas le carburant (elle utilise la distance). Les modèles restent donc valables : **T14 devient une tâche de mesure**.
- **Pour l'article :** les écarts relatifs entre politiques (DQN − Fixed : −5,5 % en nominal, −7,3 % en fortes pannes…) ne changent pas, puisque les décisions sont identiques et que toutes les consommations sont multipliées par le même facteur. Seules les **valeurs absolues** changent (L/t, L/cycle, « litres pour 1 000 t ») : **à mettre à jour en T31.**

**Ce que j'ai appris :**
- *Une constante sans source est une hypothèse cachée* : on écrit la source à côté de la valeur, ou on écrit qu'elle n'en a pas.
- *Calibrer avec un seul facteur* garde la structure du modèle (rapports entre trajet chargé, à vide et ralenti) : on corrige l'échelle sans inventer de nouvelles hypothèses.
- *Une mesure peut contaminer une décision* : le carburant servait aussi au coût de « Plus court chemin ». Changer son échelle changeait la politique. Il faut repérer tous les usages d'une constante avant de la modifier.
- *Préserver ce que voit l'agent* : en mettant à l'échelle la normalisation de l'observation, on évite un réentraînement qui aurait coûté des heures. Un test le prouve sur des épisodes complets.
- *Une calibration sur une moyenne ne valide pas le détail* : la répartition entre trajet et ralenti, et les effets de la pente et de la piste, restent des hypothèses.

**Commit :** `5177693` (15 octobre 2026).

**Questions « Savoir expliquer » :**
- [x] combien de litres consomme un cycle dans ton simulateur (environ 6 L) et comment ce chiffre se compare à la réalité :
  - **Avant T14 :** environ **6 L par cycle** (6,05 L pour l'Affectation fixe en nominal), soit 14,6 L/h par camion.
  - **Réalité :** un Cat 785D (≈ 136 t) consomme 81 à 109 L/h en usage moyen d'après le Caterpillar Performance Handbook. Un cycle dure environ 24 min, ce qui fait 95 L/h × 24/60 ≈ **38 L par cycle**. Le simulateur sous-estimait donc la consommation d'un facteur d'environ **6,5**.
  - **Après T14 :** **39,3 L par cycle** et 95 L/h, cohérent avec la référence.
  - **Ce qu'il faut en dire :** avant T14, les valeurs absolues (L/t) n'avaient pas de sens physique, mais les comparaisons **relatives** entre politiques restaient valables, et elles sont inchangées. La calibration porte sur une moyenne horaire : la répartition fine entre trajets et attentes moteur tournant n'est pas validée séparément, c'est une limite à écrire dans l'article.

## T15 — Servir les files dans l'ordre d'arrivée
*Optionnel · [RÉENTR] · N4*

**Date :** 16 octobre 2026 (méthode 1 : code et tests écrits par l'IA, relus par moi), fait **avec T16 bis**, dans le même commit.

**Prédiction (avant de modifier) :** je ne comptais pas faire T15 : la note à la directrice recommandait de le décrire comme une approximation. Je pensais que l'ordre des décisions et celui des arrivées différaient rarement.

**Décision :** **faire T15** (option « 1a »), prise par moi le 16 octobre 2026, à partir des recommandations de `docs/note_directrice_decisions.md`, **à présenter à la directrice**. La décision a changé en cours de route, à cause d'une mesure faite pendant T16 bis :
- avec la décision à la verse (T16 bis), environ 18 min séparent la décision de l'arrivée à la verse, au lieu d'environ 10 min ;
- le moteur par réservation servait alors **31 % des déversements dans le désordre des arrivées** pour l'Affectation fixe (18 % avant T16 bis), et sa file à la verse passait de 0,68 à 1,61 min par cycle ;
- garder T16 bis sans T15 aurait remplacé un biais (le détour) par un autre (des verses vides pendant qu'un camion attend).

Trois options ont été comparées : (1a) moteur à événements avec la récompense d'un cycle versée à sa fin ; (1b) le même moteur, avec une récompense estimée à la décision ; (2) revenir sans T16 bis. Choix : **1a**, car la récompense est alors exacte (vraies files, vrai tonnage) et le retard de la récompense reste faible (≈ 12 décisions, γ¹² ≈ 0,89).

**Ce que j'ai changé :**
- `env/mine_env.py`, **moteur à événements** :
  - la décision lance seulement le trajet à vide vers la pelle (`_start_cycle`) ;
  - à l'**arrivée à la pelle** : file (premier arrivé, premier servi), chargement, départ chargé (`_arrive_at_shovel`) ;
  - à l'**arrivée à la verse** : file, déversement, panne éventuelle, fin du cycle et calcul de sa récompense (`_arrive_at_dump`) ;
  - les arrivées de toute la flotte sont traitées dans l'ordre chronologique (`_run_until_decision`) ; à égalité, les arrivées passent avant les décisions, puis le plus petit indice ;
  - à la fin du poste, les cycles en cours sont menés à terme (`_finish_cycles`) : comme en v1, tout cycle décidé est simulé en entier, et son tonnage ne compte que s'il est livré avant 480 min ;
  - `truck_statuses` dit où en est chaque camion (en attente de décision, en route vers la pelle, engagé vers la verse), et `available_at_min` donne l'heure de son prochain événement.
- **Récompense (option 1a)** : celle d'un cycle est versée au `step` pendant lequel il se termine. La récompense d'un `step` est donc la somme des cycles terminés depuis la décision précédente (de n'importe quel camion), plus celle d'ATTENDRE, qui est connue tout de suite. `info["reward_terms"]` somme les termes du `step`.
- **Prévision de libération** (`forecast_free_at`) : la libération de chaque pelle et verse, camions **en route** compris (chacun, dans l'ordre de son arrivée, occupe la ressource pendant une durée moyenne de service). Elle alimente l'observation, `info` (lu par FIFO et MTWT) et le terme d'équilibre. Sans elle, FIFO envoyait plusieurs camions vers la même pelle « libre » (1,81 min de file par cycle à la pelle au lieu de 0,29).
- `simulation/reward.py` : durée de référence du coût d'opportunité mise à 23,2 min, celle d'un cycle de l'Affectation fixe sur le nouveau moteur (24,2 min avant).
- Tests : `tests/test_event_engine.py` (nouveau) et onze fichiers de tests mis à jour ; références de non-régression régénérées (voir T16 bis).

**Pourquoi :** le moteur par réservation simule tout le cycle au moment de la décision. Une pelle ou une verse est donc réservée dans l'ordre des **décisions** : un camion qui a décidé plus tôt mais arrive plus tard passe devant un camion déjà sur place, et la ressource reste vide pendant que ce dernier attend. C'est un artefact du simulateur, pas un comportement réel.

**Ce que le test vérifie :** `tests/test_event_engine.py`, 14 tests :
- **service dans l'ordre des arrivées**, aux pelles et aux verses, dans les trois régimes, avec une politique aléatoire qui utilise aussi ATTENDRE : premier arrivé, premier servi ; un seul camion à la fois ; servi dès que la ressource est libre (aucun temps mort) ;
- **toute la récompense est versée, une seule fois** : la somme des récompenses des `step` égale celle de tous les cycles ; chaque cycle décidé est terminé ; à la fin, plus aucun camion n'est en route ;
- la récompense d'un cycle n'est versée qu'à sa fin (0 pendant les premières décisions du poste) ;
- la prévision ajoute un chargement moyen par camion en route ;
- l'horloge des décisions ne recule jamais.

Suite complète : **328 passed, 1 skipped**. Ruff ne signale rien.

**Résultat comparé à ma prédiction :** infirmée. Le désordre n'était pas rare : 18 % des déversements de l'Affectation fixe dès la v1, 31 % avec la décision à la verse. Avec le moteur à événements, **0 %** aux pelles comme aux verses. Effets, Affectation fixe en nominal sur 30 graines :
- file par camion : 22,3 → **11,3 min** ; file à la verse : 0,68 → 0,13 min par cycle ;
- productivité : 4 084 → 4 246 t/h ;
- intervalle entre décisions : 1,97 → 1,89 min (12 camions), 1,36 → 1,29 min (18 camions) ;
- utilisation des pelles en forte charge : 48,8 → **50,7 %** (le « toujours moins de 50 % » de l'article tombe) ;
- carburant : 95,0 → 95,7 L/h. **Le facteur de calibration n'est pas recalé** : il fixe la consommation du camion lui-même (L/km, L/h au ralenti), pas un chiffre du simulateur à retrouver.

**Ce que j'ai appris :**
- *Un artefact peut en cacher un autre* : en corrigeant le détour (T16 bis), j'ai allongé l'écart entre décision et arrivée, et l'artefact de T15, jusque-là discret, est devenu important.
- *Mesurer avant de conclure* : l'hypothèse « l'Affectation fixe attend plus parce qu'elle est servie dans le désordre aux pelles » était fausse (6 % seulement) ; c'est la décomposition pelle / verse qui a montré la vraie cause (les verses).
- *Simuler à l'arrivée, pas à la décision* : une ressource partagée doit être attribuée au moment où le camion se présente, sinon l'ordre de service dépend de l'ordre de calcul.
- *Une récompense exacte mais différée vaut mieux qu'une récompense immédiate mais estimée* : l'agent doit optimiser ce qu'on mesure.
- *Garder ce qu'un vrai système sait* : un répartiteur connaît les camions en route ; sans la prévision, FIFO et MTWT devenaient artificiellement naïfs.

**Commit :** `157f97b` (16 octobre 2026), avec T16 bis.

**Questions « Savoir expliquer » :**
- [x] la différence entre ton moteur par réservation et une simulation à événements discrets avec file d'événements : le moteur par réservation calcule **tout le cycle d'un camion au moment où il décide** : il réserve la pelle puis la verse à l'heure où il prévoit d'y arriver. Les ressources sont donc attribuées dans l'ordre des **décisions**. Une simulation à événements discrets range les **événements** (arrivée à la pelle, arrivée à la verse, fin de cycle) par date, et les traite un par un dans l'ordre chronologique ; une ressource est attribuée au moment où le camion s'y présente, donc dans l'ordre des **arrivées**. Dans le nouveau moteur, la « file d'événements » est la liste des prochaines dates de chaque camion (`available_at_min`) : on traite toujours le plus tôt. Conséquence pour l'apprentissage : on ne connaît plus les files et le tonnage d'un cycle au moment de la décision, d'où la récompense versée à la fin du cycle (cadre SMDP).

## T16 — Pannes de pelles
*Optionnel · [RÉENTR] · N5*

**Date :** 16 octobre 2026

**Décision :** **ne pas faire T16** et l'écrire comme une limite de l'article, prise par moi le 16 octobre 2026, à partir des recommandations de `docs/note_directrice_decisions.md`, **à présenter à la directrice**. Raisons : l'article a déjà trois régimes ; un quatrième (« panne de pelle ») allongerait la campagne et le texte ; c'est une perspective naturelle.

**Ce que j'ai changé :** rien dans le code. Conséquence pour T22 : pas de réaffectation de secours pour l'Affectation fixe, puisqu'elle n'aurait de sens qu'avec des pannes de pelles.

**À écrire dans l'article (T31) :** seuls les camions tombent en panne ; les pelles sont toujours disponibles. Limite à mentionner avec sa conséquence (voir ci-dessous).

**Questions « Savoir expliquer » :**
- [x] pourquoi la perte d'une pelle est une perturbation plus sévère que la panne d'un camion avec un facteur d'appariement de 0,35 : le facteur d'appariement compare la capacité de chargement demandée par les camions à celle des pelles : MF = (camions × durée de chargement) / (pelles × durée de cycle) = (12 × 2) / (3 × 23) ≈ 0,35. Les **camions** sont donc la ressource limitante. Une panne de camion retire 1/12 de la capacité de transport pendant 10 à 30 min : la perte est petite, proportionnelle, et les autres camions ne sont pas touchés. Une panne de pelle retire **un tiers** de la capacité de chargement d'un coup : MF passe à (12 × 2) / (2 × 23) ≈ 0,52, toujours sous 1, donc la mine *pourrait* garder presque toute sa production, **à condition de réaffecter les camions**. C'est là que la perturbation est sévère : une politique qui ne réaffecte pas (l'Affectation fixe et ses 4 camions attachés à la pelle en panne) perd un tiers de sa production, et les autres doivent absorber des files plus longues sur deux pelles. La panne de pelle teste donc la capacité de **réaffectation** d'une politique, ce que la panne de camion ne teste presque pas.

## T16 bis — Prendre la décision à la verse (option b de T21)
*À décider avec T21 · [RÉENTR] · M16, M25*

**Date :** 16 octobre 2026 (méthode 1 : code et tests écrits par l'IA, relus par moi), avec T15, dans le même commit.

**Prédiction (avant de modifier) :** comme le plan, je pensais que Pelle la plus proche et Plus court chemin allaient à 100 % à la pelle 1 **parce que** le camion revenait à son ancienne pelle avant de décider. En décidant à la verse, les deux règles devaient répartir leurs cycles.

**Décision :** **option (b) de T21**, prise par moi le 16 octobre 2026, à partir des recommandations de `docs/note_directrice_decisions.md`, **à présenter à la directrice**. J'avais d'abord appliqué l'option (a) (garder les règles comme bornes basses) ; j'ai ensuite choisi (b).

**Ce que j'ai vérifié avant de coder, et qui a changé la justification :**
- **(b) ne corrige pas les règles de distance.** La pelle 1 est la plus proche **de partout** : du parc (2,30 km), de la verse 1 (3,44 km) et de la verse 2 (3,76 km). Plus court chemin choisit aussi (pelle 1, verse 1) depuis ces trois lieux. La cause est double : ces règles sont **statiques** (elles ignorent les files) et la géométrie favorise une pelle.
- **(b) corrige en revanche un biais important.** Un camion qui changeait de pelle faisait verse → **ancienne pelle** → nouvelle pelle. Sur 10 graines en nominal : FIFO change de pelle dans 69 % de ses cycles, avec un détour moyen de **3,65 km** ; MTWT, 69 % et 3,45 km ; l'Affectation fixe ne change jamais, donc ne paie jamais ce détour. Les agents RL, qui répartissent les camions, le payaient aussi.

J'ai gardé (b) pour cette seconde raison.

**Ce que j'ai changé :**
- `env/mine_env.py` : le cycle se termine à la verse, après le déversement ; il n'y a plus de retour à vide vers la pelle. Le trajet vers la pelle choisie part du lieu de la décision (parc ou verse). La panne éventuelle se produit à la verse. Le tonnage est livré au déversement.
- Distance du cycle (récompense) : lieu → pelle → verse ; le retour à vide est le premier trajet du cycle suivant.
- `env/observation_builder.py` : **17 composantes au lieu de 18**. Le bloc `position` décrit {parc, verse 1, verse 2} (au lieu de {parc, pelle 1, 2, 3}), puisqu'on ne décide plus jamais à une pelle ; `shovel_trucks` compte les camions **engagés** vers chaque pelle (en route, en file ou en chargement). Taille toujours indépendante du nombre de camions (3S + 2D + 4).
- `simulation/entities.py` : `CycleRecord` perd `travel_back_min` ; `travel_to_shovel_min` est le trajet depuis le lieu de la décision.
- `baselines/shortest_path_policy.py` : coût du cycle sur deux segments (lieu → pelle → verse).
- `tests/fixtures/regression_reference.csv` : **régénéré**, changement voulu de la simulation (Fixed et FIFO, nominal, graines 42 à 46 : 4 084,5 → 4 273,5 t/h et 3 293,5 → 4 263,0 t/h). L'historique est dans le générateur.

**Pourquoi :** dans une vraie mine, un camion reçoit son affectation à la verse ; le simulateur l'envoyait d'abord à son ancienne pelle, ce qui pénalisait toute politique qui change de pelle.

**Test prévu par le plan :** chaque cycle commence à une verse (sauf le premier, au parc) ; la part des cycles par pelle de Pelle la plus proche ne dépasse plus 60 % pour aucune pelle.

**Ce que le test vérifie :**
- `tests/test_event_engine.py` : les décisions se prennent au parc ou à une verse, jamais à une pelle ; chaque camion part une fois du parc ; la distance d'un cycle va de la verse précédente à la pelle puis à la verse (calculée à la main) ; **plus de détour** : les cycles de FIFO sont aussi courts que ceux de l'Affectation fixe (± 3 %), alors qu'ils duraient 29,8 min contre 24,1 avant ;
- `tests/test_observation.py` : 17 composantes, position à la verse après le premier cycle, distance depuis la verse, camions engagés vers une pelle ; les tests de T17 (aucune composante constante ni saturée) passent toujours ;
- `tests/test_behavior.py` : **la seconde moitié du test du plan ne peut pas réussir**, et le test dit pourquoi : Pelle la plus proche reste à 100 % sur la pelle 1 (géométrie) ; Plus court chemin descend entre 60 et 100 % (≈ 72 %) ;
- tests mis à jour avec leurs nouvelles valeurs, chacun avec l'historique des anciennes : Affectation fixe toujours sur le même couple pelle–verse (`test_baselines`), MTWT au-dessus de FIFO et de l'Affectation fixe (`test_mtwt`), récompense et intervalle entre décisions (`test_reward`), décomposition de l'attente et Δ attente en forte charge et fortes pannes (`test_wait_decomposition` : +68,81 % et −6,19 %), utilisation (`test_shift_bounds`), carburant (`test_fuel`).

**Résultat comparé à ma prédiction :** infirmée sur les règles de distance, confirmée et dépassée sur le reste. Heuristiques sur 30 graines, nouveau moteur (T15 + T16 bis), en t/h :

| Régime | FIFO | Affectation fixe | Pelle la plus proche | Plus court chemin | MTWT |
|---|---|---|---|---|---|
| Nominal | 4 244 | 4 246 | 3 893 | 4 035 | **4 437** |
| Forte charge | 6 224 | 6 205 | 4 023 | 4 675 | **6 365** |
| Fortes pannes | 4 001 | 3 999 | 3 754 | 3 867 | **4 184** |

- **MTWT devient la meilleure heuristique dans les trois régimes**, devant l'Affectation fixe (l'inverse en v1) ; FIFO fait jeu égal avec l'Affectation fixe (3 294 → 4 244 t/h en nominal).
- « L'Affectation fixe est difficile à battre », conclusion de la v1, venait en grande partie des deux biais du simulateur : le détour et les verses servies dans le désordre (T15).
- Les modèles entraînés avant ce changement ne sont plus compatibles (observation de 17 composantes) ; la campagne T31 réentraîne tout de toute façon.

**Ce que j'ai appris :**
- *Vérifier l'hypothèse d'un plan avant de coder* : un calcul de distances de deux minutes a montré que la cause annoncée était fausse.
- *Une correction peut être juste pour une autre raison que prévu* : (b) ne répare pas les règles de distance, mais supprime un biais qui touchait toutes les politiques sauf une.
- *Un biais qui épargne une seule politique fausse tout le classement* : le détour ne pénalisait jamais l'Affectation fixe.
- *Écrire dans le test pourquoi le test du plan ne peut pas réussir*, plutôt que de le retirer en silence.

**Commit :** `157f97b` (16 octobre 2026), avec T15.

**Questions « Savoir expliquer » :**
- [x] où se trouve un camion, dans une vraie mine, quand le répartiteur lui donne sa prochaine affectation : **à la verse (ou au concasseur), juste après avoir déversé**, au moment où il repart à vide. C'est là que les systèmes de gestion de flotte (DISPATCH de Modular Mining, par exemple) affectent le camion à une pelle : sa prochaine destination dépend de l'état des files à ce moment. Le camion ne retourne pas d'abord à sa pelle précédente, ce qui serait un trajet inutile dès qu'il change de pelle.

---

# Phase 4. Observation et actions (2 à 3 jours)

## T17 — Retirer les composantes constantes
*Indispensable · [RÉENTR] · N1, N2, N3*

**Date :** 15 octobre 2026, faite **avec T18** (option « variables agrégées »), comme le plan le prévoit.

**Prédiction (avant de modifier) :** avec les variables agrégées, les composantes constantes disparaissent avec le bloc par camion. Il reste seulement à le vérifier.

**Ce que j'ai changé :** rien de plus que T18. Les 54 composantes constantes sur 147 ont disparu avec la nouvelle observation :
- le **statut des camions** (toujours 0) : supprimé avec le bloc par camion ;
- les **colonnes « intersection » et « verses »** du codage de position : le nouveau one-hot ne porte que sur le parc et les pelles, les seuls lieux où une décision est prise ;
- les **6 distances pelle–verse** (constantes) : supprimées ;
- le **tonnage**, qui saturait (÷ 2 000 t) : divisé maintenant par 3 500 t (25 cycles × 140 t ; maximum observé : 23 cycles, 3 220 t).

`rl/discretize.py` lit l'observation par le nom de ses blocs (`observation_layout`), et non plus depuis la fin. `observation_size` ne dépend plus du nombre de camions.

**Pourquoi :** 54 composantes sur 147 ne changeaient jamais (N1, N2, N3). L'article décrivait donc un état plus riche que ce que l'agent recevait vraiment.

**Test prévu par le plan :** sur 10 épisodes, aucune composante de l'état n'a un écart-type nul ; pour les **variables continues seulement**, aucune ne vaut exactement 1 plus de 5 % du temps. Ne l'applique pas aux colonnes d'un codage one-hot : elles valent légitimement 1 souvent.

**Ce que le test vérifie :** `tests/test_observation.py`, partie T17, pour chacun des 3 régimes, sur 10 graines × 3 politiques (Fixed, FIFO et une politique aléatoire qui utilise ATTENDRE) :
- aucune des 18 composantes n'a un écart-type nul ;
- aucune variable continue (tout sauf le one-hot de position) ne vaut exactement 1 plus de 5 % du temps.

**Résultat comparé à ma prédiction :** confirmée dans les 3 régimes (6 400 à 9 700 observations par régime) : aucune composante constante, aucune saturation. Valeurs maximales : tonnage 0,92, carburant 0,71, camions par pelle 0,54 ; le temps du poste n'atteint 1 qu'à la toute dernière décision.

**Ce que j'ai appris :**
- *Vérifier ce que l'agent reçoit vraiment* : une composante peut être décrite dans l'article et pourtant valoir toujours la même chose.
- *L'ordre des opérations compte* : le statut était remis à « libre » juste avant la construction de l'observation, donc il valait toujours 0.
- *Faire T18 avant T17* évitait de nettoyer à la main des blocs que T18 allait supprimer.

**Commit :** `d778e30` (15 octobre 2026), commun avec T18.

**Questions « Savoir expliquer » :**
- [x] pourquoi le statut valait toujours 0 : dans `step`, le cycle entier du camion est simulé d'un seul coup. À la fin, le statut du camion est remis à « libre » (`STATUS_IDLE` = 0), **puis** l'observation est construite. Au moment où l'agent observe, aucun camion n'est donc jamais « en route » ou « en chargement » : le cycle est déjà terminé dans la simulation. La composante valait 0 pour tous les camions, à chaque décision.
- [x] pourquoi une composante constante ne coûte rien à la performance, mais affaiblit la description de l'article :
  - **Performance** : une entrée constante n'apporte aucune information. Le réseau apprend simplement à l'ignorer (son poids n'influence rien, ou il s'ajoute à un biais), donc la politique apprise n'est ni meilleure ni pire.
  - **Article** : il décrivait un état de 147 composantes « riche » (statut, position de chaque camion…), alors qu'un tiers ne portait aucune information. Un relecteur qui compare la description au code y voit une exagération de ce que l'agent « sait ». La description doit correspondre à ce que l'agent reçoit vraiment : maintenant, 18 composantes, toutes informatives.

## T18 — Observation de taille fixe
*Indispensable · [RÉENTR] · M1*

**Date :** 15 octobre 2026 (méthode 1 : code et tests écrits par l'IA, relus par moi). Fait **avant T17**, comme le plan le demande.

**Prédiction (avant de modifier) :** pas de prédiction chiffrée écrite. Attendu :
- la simulation ne change pas (seule l'observation change), donc la non-régression reste verte ;
- DQN et PPO devront être réentraînés ;
- les composantes constantes de T17 disparaissent avec le bloc par camion.

**Décision :** **variables agrégées** (option 1, recommandée par le plan), décidée par moi le 15 octobre 2026, **à confirmer avec la directrice**. Justification écrite :
1. la taille ne dépend plus du nombre de camions, donc le transfert sans réentraînement devient testable dans les trois régimes, ce qui est l'affirmation centrale de l'article ;
2. l'option « remplissage à 24 camions » aurait laissé des emplacements (13 à 24) jamais actifs pendant l'entraînement nominal : un réseau ne sait rien de ce qu'il n'a jamais vu ;
3. l'observation devient petite (18 composantes au lieu de 147 ou 213) et sans composante constante, ce qui règle T17 du même coup ;
4. argument de coût : un seul entraînement nominal par algorithme peut servir aux trois régimes, et la campagne peut passer de 30 à 10 entraînements.

En même temps : **T15, T16 et T16 bis sont reportés** jusqu'à l'avis de la directrice. Si elle retient T16 bis (décision à la verse), le bloc `position` devra inclure les verses.

**Mise à jour du 16 octobre 2026 :** choix **confirmé** (variables agrégées), à présenter à la directrice. T16 bis a été fait : l'observation passe à **17 composantes** (position {parc, verse 1, verse 2}, camions engagés vers chaque pelle), toujours indépendante du nombre de camions. Voir T16 bis.

**Ce que j'ai changé :**
- `env/observation_builder.py`, réécrit : **18 composantes** (4S + D + 4), dans cet ordre :

| Bloc | Taille | Contenu | Normalisation |
|---|---|---|---|
| `shovel_wait` | 3 | temps avant que chaque pelle soit libre | ÷ 480 min |
| `dump_wait` | 2 | temps avant que chaque verse soit libre | ÷ 480 min |
| `shovel_trucks` | 3 | camions rattachés à chaque pelle (en route, en file ou en service) | ÷ 24 |
| `distance_to_shovel` | 3 | distance du camion qui décide à chaque pelle | ÷ 10 km |
| `position` | 4 | one-hot : parc ou pelle 1, 2, 3 | |
| `tonnage` | 1 | cumul du camion qui décide | ÷ 3 500 t (25 cycles × 140 t) |
| `fuel` | 1 | cumul du camion qui décide | ÷ 1 085 L (135,6 L/h × 8 h, Cat 785D usage intensif) |
| `time` | 1 | progression du poste | ÷ 480 min |

  Supprimés : le bloc par camion (statut, disponibilité, tonnage, carburant, position one-hot : 11 × N valeurs) et les 6 distances pelle–verse (constantes).
- `observation_layout(S, D)` : la position de chaque bloc **par son nom**. `observation_size(S, D)` ne prend plus le nombre de camions.
- `rl/discretize.py` : il lit les blocs par leur nom, et non plus « depuis la fin du vecteur ».
- `simulation/constants.py` : `FUEL_OBS_MAX_L` = 135,6 × 8 (plafond publié au lieu de 500 × facteur).
- `env/mine_env.py` : `observation_size(S, D)`. Le statut des camions n'est plus passé à l'observation.
- `experiments/scenarios.py` : **`unseen_medium_load` (15 camions) réintroduit**.
- Tests mis à jour :
  - `test_model_loading.py` : le test qui documentait le défaut (147 contre 180) est remplacé par « un modèle nominal se charge avec 15 camions » ;
  - `test_small_fixes.py` : `unseen_medium_load` est de retour ;
  - `test_fuel.py` : les tests « observation identique avant et après T14 » sont retirés, puisque T18 change l'observation ;
  - `test_behavior.py` : le test lent des modèles v1 est marqué **historique** (réussi au commit `d1cec48`), car les DQN et PPO v1 (147 entrées) ne peuvent plus être rejoués.

**Pourquoi :** un réseau entraîné avec 12 camions (147 entrées) ne pouvait pas recevoir l'état de 18 camions (213 entrées). Le transfert sans réentraînement n'était donc testable qu'en fortes pannes, et même `unseen_medium_load` plantait (M1).

**Test prévu par le plan :** `observation_space.shape` est identique pour 12 et 18 camions ; un modèle PPO entraîné brièvement avec 12 camions s'exécute sans erreur dans un environnement à 18 camions.

**Ce que le test vérifie :** `tests/test_observation.py`, partie T18 :
- **test du plan** : même taille (18,) pour 12, 15, 18 et 24 camions ;
- la disposition couvre tout le vecteur, sans trou ni chevauchement (4S + D + 4, y compris avec 5 pelles et 3 verses) ;
- contenu au début du poste (tout au parc, rien de produit) et après un cycle (1/24 sur la pelle choisie) ;
- l'état tabulaire est inchangé ;
- **test du plan** : un PPO entraîné 1 024 pas avec 12 camions joue un épisode complet avec 18 camions, chargé avec `trained_on = nominal`.

Suite complète : **158 passed** (+ 1 historique ignoré). Non-régression verte. Ruff ne signale rien.

**Résultat comparé à ma prédiction :** confirmée.
- La simulation est inchangée (non-régression verte).
- **Q-Learning et SARSA** : leur état discret lit les mêmes valeurs (attentes et pelle la plus proche), donc **leurs Q-tables restent valables**.
- **DQN et PPO** : nouvelle observation, **réentraînement obligatoire** (phase 7).
- Sur 30 épisodes par régime, les valeurs maximales restent sous 1 : tonnage 0,92, carburant 0,71, camions par pelle 0,54. Les nouveaux plafonds ne saturent pas.
- Limite remarquée : les attentes n'utilisent qu'une petite partie de [0, 1] (maximum ≈ 0,05, car divisées par 480 min). Ce n'est pas faux, mais une normalisation plus serrée pourrait aider l'apprentissage. Je la laisse telle quelle, parce qu'elle sert aussi à l'état discret des agents tabulaires.

**Ce que j'ai appris :**
- *La forme de l'entrée décide de ce qu'on peut affirmer* : sans observation de taille fixe, « transfert sans réentraînement » était une affirmation intestable en forte charge.
- *Agréger, c'est choisir ce que l'agent peut savoir* : on garde ce qui sert à décider (charge de chaque pelle, état du camion qui décide) et on renonce au détail par camion.
- *Lire un vecteur par le nom de ses blocs* (`observation_layout`), pas par des positions comptées depuis la fin : la discrétisation ne casse plus si l'observation change.
- *Normaliser avec des plafonds justifiés* : 25 cycles × 140 t (au-dessus du maximum observé), consommation maximale publiée.
- *Un test historique se garde, en disant pourquoi il ne peut plus tourner*, au lieu d'être supprimé en silence.

**Commit :** `d778e30` (15 octobre 2026), commun à T17 et T18.

**Questions « Savoir expliquer » :**
- [x] pourquoi ce choix conditionne toute l'affirmation de transfert sans réentraînement de l'article : l'article affirme qu'un agent entraîné en conditions nominales peut être évalué dans un régime perturbé sans être réentraîné. Avec l'ancienne observation, c'était **impossible** pour DQN et PPO dès que le nombre de camions changeait : un réseau à 147 entrées ne peut pas recevoir 213 valeurs. Le transfert n'était testable qu'en fortes pannes (même nombre de camions). Avec une observation de taille fixe, le même réseau peut recevoir l'état de 12, 15 ou 18 camions. Le transfert devient testable dans **tous** les régimes, et le scénario `unseen_medium_load` (15 camions) redevient possible. Sans cette tâche, l'article ne pourrait affirmer le transfert que pour les pannes.
- [x] ce que l'agent ne voit plus avec des variables agrégées :
  - la position, le statut et la disponibilité de **chaque** camion individuellement ;
  - le tonnage et le carburant des **autres** camions ;
  - **quel** camion va arriver à quelle pelle, et **quand**.

  Il garde la charge de chaque pelle (temps avant disponibilité et nombre de camions rattachés), les attentes aux verses, sa propre position, ses distances aux pelles, son tonnage, son carburant et l'heure. Une stratégie qui consisterait à anticiper l'arrivée d'un camion précis n'est donc plus apprenable : c'est une **limite à écrire dans l'article**, en contrepartie du transfert.

## T19 — Masque d'actions et action ATTENDRE
*Recommandé · [MESURE] pour la journalisation, [RÉENTR] si le masque est utilisé · M16*

**Date :** 15 octobre 2026 (méthode 1 : code et tests écrits par l'IA, relus par moi)

**Décision :** **mesurer seulement** (mon choix, parmi « mesurer seulement », « masque avec MaskablePPO » ou « règle dans l'environnement »). Raisons :
1. la cause du problème est dans la **récompense**, pas dans l'ensemble des actions permises ;
2. DQN ne sait pas utiliser de masque dans Stable-Baselines3 : masquer PPO seul rendrait la comparaison PPO/DQN inégale ;
3. la correction de fond (T25) s'appliquera à tous les agents de la même façon.

T19 reste donc une tâche **[MESURE]**, sans nouvelle dépendance.

**Prédiction (avant de modifier) :** pas de prédiction chiffrée écrite. Attendu :
- la simulation ne change pas (non-régression verte) ;
- l'Affectation fixe n'utilise jamais ATTENDRE ;
- le comptage retrouve ce que mesurait déjà le journal des cycles.

**Ce que j'ai changé :**
- `env/action_mask.py` **supprimé** : il autorisait toutes les actions et n'était appelé nulle part.
- `env/mine_env.py` :
  - deux compteurs par épisode, `decision_count` et `wait_action_count`, remis à zéro à chaque `reset` et exposés dans `info["decisions"]` et `info["wait_actions"]` ;
  - la docstring de `_apply_wait_action` explique pourquoi la récompense rend ATTENDRE attrayante, et renvoie à T25.
- `experiments/run_benchmark.py` : chaque épisode rapporte `decisions`, `wait_actions` et `wait_share_pct`. Le prochain benchmark contiendra donc la part d'ATTENDRE, sans passer par le journal des cycles.

**Pourquoi :** en forte charge, PPO choisissait ATTENDRE dans 87 % de ses décisions (T12 bis), et rien dans les résultats du benchmark ne le montrait. Le masque d'actions, présenté comme existant, ne faisait rien (M16).

**Ce que le test vérifie :** `tests/test_wait_action.py`, 5 tests :
- l'Affectation fixe n'attend jamais (0 %), et fait une décision par entrée du journal ;
- **cohérence** : avec une politique aléatoire, le compteur d'`info` = le nombre d'entrées « wait » du journal (T12), et la part d'ATTENDRE = celle de l'analyse du comportement (T12 bis) ;
- les compteurs sont remis à zéro à chaque `reset` ;
- **le mécanisme de la récompense** : au début du poste, une décision ATTENDRE de 5 min vaut exactement −w4 × 5/30 = −0,05, **sans coût de distance**, et une ATTENDRE de 0,2 min ne coûterait que −0,002 ;
- à l'inverse, un cycle vers la pelle la plus éloignée est moins bien récompensé qu'un cycle vers la plus proche : il paie sa distance.

Suite complète : **163 passed**. Non-régression verte. Ruff ne signale rien.

**Résultat comparé à ma prédiction :** confirmée. Simulation inchangée, Affectation fixe à 0 % d'ATTENDRE, comptage identique au journal des cycles.

**Ce que j'ai appris :**
- *Un agent optimise exactement ce qu'on lui demande* : s'il « triche » en multipliant les attentes, c'est que la récompense le permet. Il faut corriger la récompense, pas punir l'agent.
- *Supprimer le code qui ne fait rien* : un masque qui autorise tout et qui n'est appelé nulle part laissait croire qu'une contrainte existait.
- *Corriger à la source, pour tous* : un masque pour PPO seul aurait faussé la comparaison avec DQN.
- *Mesurer avant de corriger* : le compteur dans `info` et dans les KPI permettra de vérifier en T25 et en T31 si la correction de la récompense fait baisser la part d'ATTENDRE.

**Commit :** `9eb2035` (15 octobre 2026).

**Questions « Savoir expliquer » :**
- [x] pourquoi la récompense actuelle peut rendre ATTENDRE intéressante pour l'agent : la récompense d'une décision vaut R = w1 · tonnage − w2 · variance des files − w3 · distance / 5 − w4 · min(attente / 30, 1), avec w1 = 1, w2 = 0,1, w3 = 0,05 et w4 = 0,3.

| | Cycle de transport | Décision ATTENDRE |
|---|---|---|
| Tonnage | +1 si le cycle livre avant la fin du poste | 0 |
| Distance | **payée** : par exemple −0,15 pour 15 km | **0** : `_compute_reward` est appelée avec `route_distance_km = 0` |
| Attente | la file réelle : par exemple −0,1 pour 10 min | la durée de l'ATTENDRE : **−0,002** pour 0,2 min |

  En forte charge, partir tout de suite expose souvent à une file, qui coûte en attente et retarde le cycle. Une ATTENDRE très courte ne coûte presque rien, n'a aucun coût de distance, et laisse parfois passer le moment où la file se forme. L'agent apprend donc à enchaîner de très nombreuses petites attentes : 4 794 décisions par épisode pour PPO, dont 87 % d'ATTENDRE et 54 min d'attente volontaire par camion et par poste.

  Le défaut vient de la **récompense, pas de l'agent**. La correction (T25) doit donner à ATTENDRE un coût à la hauteur du temps perdu, par exemple proportionnel à sa durée, ou au moins égal au coût d'une file équivalente. On vérifiera l'effet avec le compteur `wait_share_pct` ajouté ici.

## T20 — Représentation tabulaire documentée
*Recommandé · [MESURE] · M8*

**Date :** 15 octobre 2026 (méthode 1 : code et tests écrits par l'IA, relus par moi)

**Prédiction (avant de modifier) :** pas de prédiction chiffrée écrite. Attendu :
- nommer les constantes sans changer leurs valeurs ne change pas l'état discret, donc pas de réentraînement ;
- l'action par défaut sert sans doute un peu, surtout quand on transfère un modèle nominal en forte charge.

**Ce que j'ai changé :**
- `rl/discretize.py` :
  - constantes nommées et commentées, **valeurs inchangées** : `WAIT_SCALE = 50` (une attente sature à `WAIT_SATURATION_MIN` = 480 / 50 = 9,6 min) et `DEFAULT_N_BINS = 8` (intervalles de 1,2 min) ;
  - nouvelle fonction `state_space_size(n_bins, S, D)` = n_bins^(S+D) × S ;
  - docstring du module : ce que contient l'état, pourquoi il ne dépend pas du nombre de camions, et l'avertissement « modifier ces valeurs oblige à réentraîner ».
- `rl/tabular_training.py` :
  - `DEFAULT_ACTION = 0` (pelle 1, verse 1), l'action choisie sur un état jamais vu, nommée et documentée ;
  - `TabularPolicy` compte ses décisions et ses états inconnus : `reset_coverage()`, et `coverage()` qui renvoie `known_state_pct` et `default_action_pct`.
- `experiments/policies.py` + `experiments/run_benchmark.py` : pour Q-Learning et SARSA, chaque épisode du benchmark remet le compteur à zéro, puis rapporte `known_state_pct` et `default_action_pct`.
- `run_benchmark.py` et `rl/diagnostics/*` : le `n_bins=8` écrit en dur est remplacé par `DEFAULT_N_BINS`.

**Pourquoi :** la représentation tabulaire reposait sur des nombres sans nom (× 50, 8 intervalles, action 0). On ne savait ni ce qu'ils signifiaient, ni combien de fois l'agent tombait sur un état jamais vu, auquel cas il jouait l'action 0 sans rien avoir appris (M8).

**Ce que le test vérifie :** `tests/test_tabular.py`, 8 tests :
- valeurs inchangées (50, 8, action 0, saturation 9,6 min) ;
- `state_space_size()` = 8⁵ × 3 = 98 304 ;
- **discrétisation identique à l'ancienne formule** sur 200 observations aléatoires, ce qui prouve qu'il n'y a pas besoin de réentraîner ;
- état identique avec 12 et 18 camions au début du poste ;
- comptage : état connu → action apprise, état inconnu → action par défaut, 50 % / 50 %, et remise à zéro ;
- chaque épisode du benchmark rapporte sa couverture, remise à zéro entre deux épisodes ;
- les heuristiques n'ont pas ces colonnes.

Suite complète : **171 passed**. Non-régression verte. Ruff ne signale rien.

**Résultat comparé à ma prédiction :** la première partie est confirmée (état inchangé, pas de réentraînement). La seconde est **fausse** : l'action par défaut ne sert **jamais**. Mesuré avec les Q-tables v1, sur 10 graines :

| Évaluation | Modèle | États déjà vus (Q-Learning et SARSA) |
|---|---|---|
| nominal | nominal | 100 % |
| forte charge | forte charge | 100 % |
| fortes pannes | fortes pannes | 100 % |
| forte charge (18 camions) | **nominal** (transféré) | **100 %** |
| fortes pannes | **nominal** (transféré) | **100 %** |

Chaque Q-table v1 n'a appris qu'environ **17 000 états sur 98 304** (16 599 à 17 519, soit 17 %). Mais l'évaluation reste toujours dans cette région, même transférée à 18 camions. **Pour l'article :** les résultats des agents tabulaires ne reposent pas sur l'action par défaut.

Au passage, avec les Q-tables v1 et 10 graines, Q-Learning nominal transféré en forte charge produit 4 611 t/h, contre 4 797 t/h pour le modèle entraîné en forte charge. Cela donne une première idée du coût du transfert ; les vrais chiffres viendront de la campagne finale.

**Ce que j'ai appris :**
- *Nommer une constante, c'est la documenter* : `WAIT_SCALE = 50` dit qu'au-delà de 9,6 min, toutes les attentes se ressemblent pour l'agent.
- *Refactoriser sans changer le comportement se prouve* : on compare la nouvelle fonction à l'ancienne formule sur de nombreuses entrées.
- *Mesurer une hypothèse au lieu de la supposer* : je pensais que l'action par défaut servirait en transfert, et elle ne sert jamais.
- *Une petite partie de l'espace d'états suffit* : l'agent ne visite qu'environ 17 % des états possibles, ceux que la dynamique de la mine produit réellement.

**Commit :** `56920b7` (15 octobre 2026).

**Questions « Savoir expliquer » :**
- [x] pourquoi la table compte 98 304 états possibles (8⁵ × 3) : l'état discret a 6 composantes :
  - 5 attentes (aux 3 pelles et aux 2 verses), chacune découpée en **8 intervalles** de 1,2 min, de 0 à 9,6 min (au-delà, tout tombe dans le dernier intervalle). Cela fait 8⁵ = 32 768 combinaisons ;
  - l'indice de la **pelle la plus proche** du camion qui décide, avec 3 valeurs possibles.

  D'où 8⁵ × 3 = **98 304** états possibles. En pratique, une Q-table n'en apprend qu'environ 17 000, ceux que la simulation produit vraiment.
- [x] pourquoi l'agent tabulaire peut, lui, être transféré à 18 camions : **aucune de ses 6 composantes ne dépend du nombre de camions**. Ce sont des attentes aux pelles et aux verses, et une pelle la plus proche. Avec 18 camions, les files sont plus longues, donc les attentes tombent dans d'autres intervalles, mais l'état a **la même forme et le même nombre de valeurs**. La Q-table s'applique telle quelle. Mesuré : en forte charge, le modèle nominal ne rencontre que des états déjà vus (100 %). C'était déjà vrai **avant T18**, alors que DQN et PPO ne pouvaient pas être transférés : leur observation contenait un bloc par camion (147 entrées pour 12 camions, 213 pour 18).

---

# Phase 5. Heuristiques de référence (2 à 4 jours)

## T21 — Pelle la plus proche et Plus court chemin
*Indispensable · Impact selon l'option · M16, M25*

**Date :** 16 octobre 2026

**Prédiction (avant de modifier) :** avec l'option (b), les deux règles répartissent leurs cycles entre les pelles.

**Décision :** **option (b)**, réalisée par T16 bis (avec T15), prise par moi le 16 octobre 2026, à partir des recommandations de `docs/note_directrice_decisions.md`, **à présenter à la directrice**. Les règles elles-mêmes ne changent pas (sauf le coût de Plus court chemin, ramené à deux segments).

**Ce que j'ai changé :** voir T16 bis et T15.

**Pourquoi :** voir T16 bis : la décision à la verse supprime le détour qui pénalisait toutes les politiques qui changent de pelle.

**Test prévu par le plan :** compter la part des cycles par pelle sur 30 graines. Avec l'option (b), aucune pelle ne doit dépasser, disons, 60 %.

**Ce que le test vérifie :** `tests/test_behavior.py::test_heuristiques_sur_les_trois_regimes` : Pelle la plus proche à 100 % sur la pelle 1 dans les trois régimes ; Plus court chemin entre 60 et 100 % ; aucune heuristique n'utilise ATTENDRE.

**Résultat comparé à ma prédiction :** **infirmée.** Pelle la plus proche reste à **100 %** sur la pelle 1 : c'est la plus proche du parc et des deux verses, et la règle ignore les files. Plus court chemin descend à **≈ 72 %** (73 % en nominal, 70 % en forte charge, 72 % en fortes pannes), parce que son coût compte aussi le trajet pelle → verse. Productivité en nominal : 3 893 et 4 035 t/h. Dans l'article, ces deux règles restent des **bornes basses**, avec la bonne explication : règles statiques et géométrie, et non plus « le camion revient à sa pelle ».

**Ce que j'ai appris :** *une règle qui ignore les files dégénère dès qu'un choix domine géométriquement* ; ce n'est pas un défaut du simulateur mais une propriété de la règle.

**Commit :** `157f97b` (16 octobre 2026).

**Questions « Savoir expliquer » :**
- [x] pourquoi la productivité de ces règles plafonne vers 3 900 t/h quel que soit le régime (capacité d'une seule pelle, environ 4 200 t/h) : une pelle charge un camion en 2 min en moyenne, soit au plus 30 chargements par heure, donc **30 × 140 t = 4 200 t/h**. Pelle la plus proche envoie tous les camions à la pelle 1 : sa production ne peut pas dépasser cette capacité, quel que soit le nombre de camions. On reste un peu en dessous (3 893 t/h en nominal) à cause de l'aléa des chargements et des trous dans l'arrivée des camions. Ajouter des camions (forte charge : 4 023 t/h) n'aide presque pas : ils allongent seulement la file (188 min par camion et par poste en forte charge). Plus court chemin envoie environ 28 % de ses cycles ailleurs, d'où un plafond un peu plus haut (4 035 et 4 675 t/h).

## T22 — Docstrings de FIFO et de l'Affectation fixe
*Indispensable · Aucun impact (ou [RÉÉVAL] si tu ajoutes une réaffectation) · Dir. corr. 1*

**Date :** 15 octobre 2026 (méthode 1 : code et tests écrits par l'IA, relus par moi)

**Prédiction (avant de modifier) :** aucun résultat ne change, puisque seules les docstrings sont modifiées. Les tests doivent confirmer que les règles font exactement ce que disent les nouvelles docstrings.

**Décision (avec la directrice) :** deux décisions restent **à confirmer avec elle** :
1. **Garder le nom « FIFO »** dans le code et les CSV, pour la continuité avec la v1, en écrivant clairement que la règle n'est pas un « premier arrivé, premier servi ». Il faudra choisir avec elle le nom à donner dans l'article (par exemple « Pelle libre la plus tôt »).
2. **Pas de réaffectation de secours** pour l'Affectation fixe : elle n'aurait de sens qu'avec des pannes de pelles (T16, reporté).

**Mise à jour du 16 octobre 2026 :** les deux choix sont **confirmés**, à présenter à la directrice : le nom « FIFO » reste dans le code et les CSV, et la règle sera **renommée dans l'article** (par exemple « Pelle libre la plus tôt ») en T31 ; pas de réaffectation, puisque T16 n'est pas fait.

**Ce que j'ai changé :**
- `baselines/fifo_policy.py`, docstrings du module, de la classe et de `predict` :
  - **règle exacte** : la pelle qui sera libre le plus tôt (`shovel_available_at` minimal) et, indépendamment, la verse libre le plus tôt ; en cas d'égalité, le plus petit indice ; sans `info`, l'action 0 ;
  - **ce qu'elle ignore** : les temps de trajet, la position du camion et l'ordre d'arrivée des camions. Les camions changent souvent de pelle, d'où des cycles plus longs (≈ 29,8 min contre ≈ 24,1 min pour l'Affectation fixe en nominal) ;
  - pourquoi ce n'est **pas** un FIFO, et pourquoi le nom est gardé ;
  - un renvoi vers MTWT (T23), la règle qui combine files et trajets.
- `baselines/fixed_policy.py`, docstrings du module, de la classe et de `predict` :
  - **règle exacte** : le camion i va à la pelle i mod S et à la verse i mod D, pendant tout le poste, **sans aucune réaffectation**. L'état n'est jamais lu, seul l'indice du camion compte ;
  - **conséquences** : 4 camions par pelle avec 12 camions, 6 avec 18, 5/4/4 avec 13 (scénario « nominal_asymétrique »), et aucun trajet entre pelles après le premier cycle ;
  - la réaffectation de secours n'est pas implémentée, et le texte dit pourquoi.
- Aucune modification du comportement : le code des règles est inchangé.

**Pourquoi :** les docstrings décrivaient d'autres règles que le code (Dir. corr. 1). Pour FIFO : « file la plus courte », « pelle inactive depuis le plus longtemps », alors que le code choisit la pelle libre le plus tôt, sans trajets. Pour l'Affectation fixe : « round-robin », alors que c'est un modulo fixe, sans réaffectation. L'article doit décrire les références exactement comme elles sont codées.

**Ce que le test vérifie :** `tests/test_baselines.py`, 19 tests :
- **FIFO** :
  - pelle libre à 2 min et verse libre à 1 min → action (pelle 2, verse 2) ;
  - **la position du camion ne change rien** (parc ou n'importe quelle pelle) ;
  - égalité → plus petit indice ; sans `info` → action 0.
- **Affectation fixe** :
  - pour chacun des 12 camions, action (i mod 3, i mod 2), **quelles que soient l'observation et les files** (12 tests) ;
  - sur un épisode complet avec 12, 13 et 18 camions : **une seule paire (pelle, verse) par camion** pendant tout le poste, et la répartition attendue (3 tests) ;
  - après le premier cycle (depuis le parc), **aucun trajet vers la pelle**.

Suite complète : **190 passed**. Non-régression verte. Ruff ne signale rien.

**Résultat comparé à ma prédiction :** confirmée. Les tests montrent que les nouvelles docstrings décrivent exactement le code, et la non-régression prouve que rien n'a changé dans les résultats.

**Ce que j'ai appris :**
- *Une docstring est une affirmation qui se teste* : pour une règle déterministe, chaque phrase (« ignore les trajets », « sans réaffectation ») peut devenir un test.
- *Un nom peut tromper* : « FIFO » suggère une file servie dans l'ordre d'arrivée, alors que la règle regarde la disponibilité des pelles. Garder un nom pour la continuité est acceptable, à condition de l'écrire clairement.
- *Décrire les conséquences, pas seulement la règle* : « i mod 3 » ne dit pas que les camions ne changent jamais de pelle ni qu'ils n'ont jamais de trajet entre pelles. C'est pourtant ce qui explique les résultats.
- *Une référence simple peut être très forte* : l'Affectation fixe, qui ne lit rien de l'état, est la meilleure heuristique dans les trois régimes.

**Commit :** `3d9b608` (15 octobre 2026).

**Questions « Savoir expliquer » :**
- [x] pourquoi l'Affectation fixe est si difficile à battre en forte charge :
  1. **Répartition équilibrée d'office** : 6 camions par pelle avec 18 camions (35 / 33 / 32 % des cycles). Aucune pelle n'est surchargée pendant qu'une autre attend, contrairement à Pelle la plus proche, qui met 100 % des cycles sur la pelle 1.
  2. **Aucun trajet inutile** : le camion revient toujours à sa propre pelle et ne traverse jamais la mine pour en changer. Ses cycles sont les plus courts (24,1 min en nominal, contre 29,8 pour FIFO, qui change souvent de pelle).
  3. **Pas de goulot à mieux gérer** : même avec 18 camions, le facteur d'appariement vaut 0,53 et les pelles ne sont occupées que 49 % du temps (T11). Il n'y a pas de capacité de chargement rare qu'une politique plus « intelligente » pourrait mieux exploiter : répartir à parts égales suffit.

  Résultat : en forte charge, la meilleure productivité (≈ 5 870 t/h) et le meilleur ratio de robustesse avec FIFO (0,962). Pour la battre, un agent doit faire mieux qu'une répartition équilibrée sans trajets inutiles, ce que ni DQN (−222 t/h) ni PPO (−725 t/h) ne réussissent dans la v1.

## T23 — Règle du temps d'attente minimal des camions (MTWT)
*Recommandé · [RÉÉVAL] · M25*

**Date :** 15 octobre 2026 (méthode 1 : code et tests écrits par l'IA, relus par moi)

**Prédiction (avant de modifier) :** pas de prédiction chiffrée écrite. Attendu :
- une règle qui combine files et trajets devrait réduire l'attente ;
- elle devrait se placer entre FIFO et l'Affectation fixe.

**Ce que j'ai changé :**
- `baselines/mtwt_policy.py` (nouveau), `MTWTPolicy(graph, pelles, verses, load_time_min=2)`, règle exacte :
  1. **pelle** : pour chaque pelle j, arrivée prévue a_j = maintenant + temps de trajet moyen à vide jusqu'à j ; attente prévue w_j = max(0, disponibilité de j − a_j). On choisit le plus petit w_j, puis le plus petit a_j en cas d'égalité ;
  2. **verse** : même logique depuis la pelle choisie, avec un départ estimé = max(a_j, disponibilité de j) + 2 min de chargement, et le trajet en charge.

  La docstring décrit aussi la différence avec FIFO et les approximations : les camions affectés entre-temps ne sont pas anticipés, et le trajet vers sa propre pelle vaut 0 dans le simulateur actuel (lien avec T16 bis).
- `simulation/graph_model.py` : `mean_route_time_minutes(src, dst, loaded)`, le temps moyen d'un trajet complet le long du plus court chemin, celui que suit le simulateur ; 0 si départ = arrivée.
- MTWT devient la **5ᵉ heuristique** :
  - `experiments/policies.py` : `HEURISTICS`, `heuristic_policies`, avec le graphe synchronisé comme pour Pelle la plus proche, puisque l'état des pistes change ;
  - `baselines/__init__.py` ;
  - `plots/common.py` : ordre et libellé « Temps d'attente minimal ».

  Le benchmark et l'analyse du comportement la jouent automatiquement.

**Pourquoi :** les quatre heuristiques de la v1 ne combinaient jamais files et trajets : FIFO regarde les files sans les trajets, Pelle la plus proche et Plus court chemin les trajets sans les files, l'Affectation fixe ni l'un ni l'autre. Les systèmes commerciaux de gestion de flotte utilisent des règles comme MTWT (Zeng et al., 2022). Sans elle, l'article comparait les agents à des références plus faibles que la pratique (M25).

**Test prévu par le plan :** sur un état construit à la main (pelle 1 libre dans 10 min, pelle 2 libre maintenant mais à 3 min de trajet de plus), vérifier que la règle choisit la pelle 2.

**Ce que le test vérifie :** `tests/test_mtwt.py`, 7 tests :
- le temps d'un trajet est la somme des arêtes du plus court chemin, et vaut 0 vers soi-même ;
- **test du plan** : depuis le parc, la pelle 1 donne 10 min d'attente à l'arrivée, la pelle 2 est libre mais plus loin (5,57 contre 4,79 min de trajet) : MTWT choisit la **pelle 2** ;
- **MTWT diffère de FIFO** : le camion est à la pelle 1, qui est libre dans 5 min, et la pelle 2 est libre dans 6 min mais à 6,96 min de trajet. FIFO choisit la pelle 1, MTWT la pelle 2 ;
- égalité : l'arrivée la plus tôt ;
- choix de la verse de plus petite attente ;
- sans information : action 0 ;
- **sur des épisodes réels** (5 graines × 3 régimes) : MTWT n'utilise jamais ATTENDRE, a la plus petite file des trois règles, et produit plus que FIFO mais moins que l'Affectation fixe.

Suite complète : **197 passed**. Non-régression verte. Ruff ne signale rien.

**Résultat comparé à ma prédiction :** confirmée, avec une nuance importante. Moyennes sur 10 graines :

| Régime | Politique | Productivité (t/h) | File par camion (min/poste) | Durée d'un cycle (min) |
|---|---|---|---|---|
| Nominal | Affectation fixe | 4 074 | 21,8 | 24,1 |
| | FIFO | 3 306 | 24,8 | 29,6 |
| | **MTWT** | **3 500** | **8,3** | 28,0 |
| Forte charge | Affectation fixe | 5 852 | 36,3 | 25,1 |
| | FIFO | 4 760 | 38,7 | 30,8 |
| | **MTWT** | **4 776** | **13,0** | 30,7 |
| Fortes pannes | Affectation fixe | 3 869 | 20,0 | 25,3 |
| | FIFO | 3 166 | 24,0 | 30,9 |
| | **MTWT** | **3 365** | **8,1** | 29,1 |

- MTWT **divise la file par environ trois**, mais elle **produit moins que l'Affectation fixe** : pour éviter une attente, elle envoie le camion vers une autre pelle, et le trajet supplémentaire allonge le cycle (28 min contre 24).
- **Pour l'article :** la meilleure heuristique reste l'Affectation fixe. MTWT montre que réduire l'attente ne suffit pas à produire plus, et sert de référence « industrielle » plus crédible que FIFO.

**Ce que j'ai appris :**
- *Minimiser l'attente n'est pas maximiser la production* : le critère d'une règle (ici l'attente) n'est pas forcément l'objectif (le tonnage). Il faut regarder le cycle entier.
- *Une estimation suffit pour décider* : MTWT utilise des temps moyens, sans aléa, et ignore les camions affectés entre-temps. C'est une approximation honnête si elle est écrite.
- *Un test qui montre une différence de comportement* (MTWT contre FIFO dans la même situation) explique mieux une règle qu'une description.
- *Le simulateur influence les règles* : le trajet vers sa propre pelle vaut 0 tant que la décision se prend à la pelle (T16 bis).

**Commit :** `199ad46` (15 octobre 2026).

**Questions « Savoir expliquer » :**
- [x] en quoi cette règle diffère de ton FIFO actuel :
  - **FIFO** (la règle appelée ainsi) regarde seulement **quand chaque pelle sera libre**, et choisit la plus tôt. Il ignore complètement le trajet.
  - **MTWT** regarde **combien de temps le camion attendra en arrivant** : elle ajoute le temps de trajet à l'heure de départ, puis compare cette heure d'arrivée à la disponibilité de chaque pelle.

  **Exemple** : le camion est à la pelle 1, la pelle 1 est libre dans 5 min, la pelle 2 dans 6 min mais à 7 min de trajet. FIFO choisit la pelle 1 (libre le plus tôt), et le camion attend 5 min. MTWT choisit la pelle 2, où le camion arrive quand elle est déjà libre, donc 0 min d'attente. MTWT combine files et trajets ; FIFO ne regarde que les files.

  **Conséquence mesurée** : MTWT a environ trois fois moins de file que FIFO et produit 6 % de plus (3 500 contre 3 306 t/h en nominal), mais reste derrière l'Affectation fixe, qui évite à la fois la file **et** les trajets entre pelles.

## T24 — Politique fondée sur un programme linéaire
*Optionnel · [RÉÉVAL] · M25*

**Date :** 16 octobre 2026 (méthode 1 : code et tests écrits par l'IA, relus par moi)

**Prédiction (avant de modifier) :** une règle de type DISPATCH, qui planifie les flux, devait faire au moins aussi bien que MTWT, puisqu'elle connaît la capacité des pelles et le nombre de camions.

**Décision :** **faire T24 avant la campagne**, prise par moi le 16 octobre 2026, à partir des recommandations de `docs/note_directrice_decisions.md`, **à présenter à la directrice**. La note recommandait d'abord de ne pas la faire ; après T15 et T16 bis, MTWT est devenue une référence forte, et une heuristique issue de la recherche opérationnelle rend la comparaison avec les agents plus solide. Elle ne demande aucun réentraînement.

**Ce que j'ai changé :**
- `baselines/lp_policy.py` (nouveau), deux étages :
  - **planification** (`LPPlan`, une fois par poste) : un programme linéaire (`scipy.optimize.linprog`, HiGHS) fixe des taux de cycles par heure x_jk sur chaque route pelle → verse, et les flux à vide y_kj. Il maximise la production sous les contraintes : capacité des pelles et des verses (× u), conservation des camions à chaque pelle et à chaque verse, heures-camions disponibles (chargement, trajets moyens, déversement, réparation moyenne p × 20 min). Un second programme garde cette production (à 0,1 % près) et minimise les trajets ;
  - **affectation en temps réel** (`LPDispatchPolicy`) : le camion part sur la route la **plus en retard sur son plan**, en tenant compte du temps pour atteindre la pelle : retard = (x_jk × (t + trajet)/60 − cycles déjà lancés) ÷ (x_jk/60), en minutes de plan ;
  - le compteur de cycles lancés est remis à zéro au début de chaque épisode (`info["decisions"] == 0`) : un épisode ne dépend que de sa graine (évaluation parallèle, T29).
- **Plafond d'utilisation u** : sans plafond (u = 1), le plan sature la pelle 1 (30 chargements par heure) ; c'est optimal sans aléa, mais avec des durées aléatoires la file explose (34 min par camion) et la règle fait **moins bien que l'Affectation fixe** (4 227 contre 4 246 t/h). u est choisi par `select_max_utilization` parmi {0,6 ; 0,7 ; 0,75 ; 0,8 ; 0,85 ; 0,9 ; 1}, sur la productivité moyenne des trois régimes sur les **graines de sélection** (20 000 à 20 009), une seule valeur pour tous les régimes : **u = 0,7**.
- `configs/experiment_config.yaml` : section `heuristics` (`lp_max_utilization: 0.7` et les candidats) ; `config.py` la valide strictement.
- `experiments/policies.py` : 6ᵉ heuristique « LP » (`lp_plan`) ; `plots/common.py` : libellé « Programme linéaire (DISPATCH) ».
- `baselines/mtwt_policy.py` : docstring mise à jour (elle décrivait encore la décision à la pelle).
- Tests qui comptaient 5 heuristiques : ils lisent maintenant le nombre dans le code.

**Pourquoi :** DISPATCH (Soumis et al., 1989) est la méthode classique de recherche opérationnelle du dispatching minier ; un relecteur s'attend à ce que des agents RL y soient comparés, pas seulement à des règles simples (M25).

**Ce que le test vérifie :** `tests/test_lp_policy.py`, 15 tests :
- le plan respecte toutes ses contraintes, dans les trois régimes, avec u = 0,7 et u = 1 ;
- sans plafond, la pelle 1 est saturée (35,88 cycles par heure planifiés en nominal) ; plus de camions donnent plus de production planifiée ; un plafond invalide est refusé ;
- la règle choisit la route la plus en retard ; le compteur est remis à zéro au début d'un épisode ; un épisode rejoué après un autre redonne le même résultat ;
- sur un poste, la part des cycles de chaque route suit le plan à ± 5 points ;
- **refaire la sélection sur les graines de sélection redonne u = 0,7**, la valeur de la configuration ;
- la règle n'utilise jamais ATTENDRE et produit plus que l'Affectation fixe dans les trois régimes.

Les autres fichiers de tests passent (vérifiés fichier par fichier pendant le réglage de T25 bis, qui occupait le processeur).

**Résultat comparé à ma prédiction :** partiellement infirmée. Sur les 30 graines d'évaluation, en t/h :

| Régime | Affectation fixe | **LP (u = 0,7)** | MTWT |
|---|---|---|---|
| Nominal | 4 246 | **4 404** | 4 437 |
| Forte charge | 6 205 | **6 351** | 6 365 |
| Fortes pannes | 3 999 | **4 150** | 4 184 |

La règle LP dépasse l'Affectation fixe partout, mais reste juste derrière MTWT. Sa file est plus longue (≈ 18 min par camion en nominal contre 12 pour MTWT) : le plan fixe des flux moyens et réagit moins vite aux files du moment.

**Ce que j'ai appris :**
- *L'optimum d'un modèle déterministe peut être mauvais dans un monde aléatoire* : saturer une ressource à 100 % crée une file sans limite.
- *Un paramètre de référence se règle comme un hyperparamètre d'agent* : sur des graines à part, une seule valeur pour tous les régimes.
- *Une règle qui garde un compteur doit le remettre à zéro à chaque épisode*, sinon l'évaluation parallèle donnerait d'autres résultats que l'évaluation séquentielle.

**Commit :** `6639c6b` (16 octobre 2026).

**Questions « Savoir expliquer » :**
- [x] la différence entre l'étage de planification (programme linéaire) et l'étage d'affectation en temps réel : la **planification** répond à « combien de cycles par heure envoyer sur chaque route pelle–verse, en moyenne, sur le poste ? ». Elle se fait une fois, avec des valeurs moyennes (trajets, chargements, pannes), sous les contraintes de capacité et de flotte : c'est un problème d'optimisation de flux, résolu par un programme linéaire. L'**affectation en temps réel** répond à « où envoyer **ce** camion, **maintenant** ? ». Elle se fait à chaque décision, avec l'état réel du moment (heure, position du camion, cycles déjà lancés), et cherche à suivre le plan : on envoie le camion sur la route la plus en retard. La planification donne la cible, l'affectation corrige au fil de l'eau les écarts dus aux aléas.

---

# Phase 6. Récompense et protocole d'expérience (3 jours)

## T25 — Module de récompense séparé
*Indispensable · [MESURE] · M10, M9*

**Date :** 15 octobre 2026 (méthode 1 : code et tests écrits par l'IA, relus par moi)

**Prédiction (avant de modifier) :** c'est un déplacement de code, donc la récompense ne change pas : le test de non-régression reste vert sur `total_reward`. Le terme de distance doit peser environ −19 sur un poste d'Affectation fixe en nominal (valeur du plan).

**Ce que j'ai changé :**
- `simulation/reward.py` (nouveau), fonctions pures :
  - `reward_terms(tonnage, capacité, distance_km, attentes_aux_pelles, attente_min)` renvoie les 4 termes **non pondérés** (`RewardTerms`) ;
  - `compute_reward(termes, poids)` renvoie la récompense ;
  - `RewardTerms.weighted(poids)` renvoie la contribution de chaque terme ;
  - `RewardWeights` (production, équilibre, distance, attente) remplace w1 à w4 ; `from_tuple` garde w4 = 0,3 par défaut, comme dans la v1 ;
  - constantes nommées : `DISTANCE_SCALE_KM = 5`, `WAIT_SCALE_MIN = 30`, `VARIANCE_SCALE_MIN2 = 100` ;
  - la docstring du module donne la formule de chaque terme, avec son unité et sa normalisation.
- `env/mine_env.py` :
  - `_compute_reward` délègue au module ;
  - `info["reward_terms"]` (contributions de la dernière décision) et `info["episode_reward_terms"]` (cumul du poste, remis à zéro au `reset`) ;
  - les attributs `w1` à `w4` sont conservés, car `EnvConfig.from_env` et des tests les lisent.
- `experiments/run_benchmark.py` : chaque épisode rapporte `reward_production`, `reward_balance`, `reward_distance` et `reward_wait`, dont la somme vaut `total_reward`.

**Pourquoi :** la récompense était enfouie dans l'environnement, sans unité ni normalisation documentées, et ses termes n'étaient visibles nulle part. On ne pouvait ni vérifier la §5.5 de l'article ni tracer la contribution de chaque terme (M10, M9).

**Test prévu par le plan :** la somme pondérée des quatre termes égale la récompense renvoyée par `step`. Sur un poste d'Affectation fixe en nominal, le terme de distance pèse environ −19.

**Ce que le test vérifie :** `tests/test_reward.py`, 8 tests :
- **mêmes valeurs que la v1** : sur 500 décisions aléatoires, le module redonne exactement l'ancienne formule, recopiée dans le test ;
- un cycle type (+1 de production, −3 de distance non pondérée, soit 0,85 au total) et une ATTENDRE type (−0,2/30) ;
- les plafonds (attente plafonnée à −1) et la variance ;
- `from_tuple` avec 3 poids : w4 = 0,3 ;
- **test du plan, partie 1** : avec une politique aléatoire (cycles et ATTENDRE), à chaque décision la somme des contributions de `info` égale la récompense de `step`, et le cumul égale la récompense cumulée ;
- **test du plan, partie 2** : Affectation fixe, nominal, 30 graines. Les quatre termes ont pour somme `total_reward`, le terme de production égale le nombre de cycles, le terme de distance vaut −19,1 et le terme d'attente −2,71 ;
- l'intervalle entre deux décisions : 1,97 min avec 12 camions, 1,36 avec 18.

Suite complète : **205 passed**. Non-régression verte (`total_reward` inchangé). Ruff ne signale rien.

**Résultat comparé à ma prédiction :** confirmée. Décomposition de la récompense, Affectation fixe en nominal, moyenne sur 30 graines :

| | Ce code | Article v2 (§5.5) |
|---|---|---|
| Récompense cumulée | 210,1 | 210,7 |
| Production | +232,1 (= 232,1 cycles) | 233 cycles |
| Distance | **−19,1** | −19,2 |
| Attente | −2,7 | −2,7 |
| Équilibre | −0,1 | −0,1 |

  Les petits écarts viennent de T13 (nouveau tirage aléatoire) : à recalculer dans l'article en T31.
- **Ce que T25 n'a pas corrigé :** le coût d'ATTENDRE (T19) reste tel quel, puisque la tâche est un déplacement sans changement de calcul. Trois options pour le corriger, à décider avec la directrice, et qui obligeraient à réentraîner :
  1. un coût d'opportunité (une minute d'attente coûte la production d'un camion pendant une minute, environ 1/24 de cycle). C'est ma recommandation ;
  2. un coût minimal par ATTENDRE ;
  3. ne rien changer et l'écrire comme limite.

**Ce que j'ai appris :**
- *Séparer un calcul pour pouvoir le tester* : une fonction pure (entrées → termes → récompense) se teste sur des centaines de cas, sans simulateur.
- *Prouver qu'un déplacement de code ne change rien* : on recopie l'ancienne formule dans le test et on compare sur des entrées aléatoires, en plus de la non-régression.
- *Exposer les termes rend la récompense explicable* : on voit que la récompense, c'est surtout « un point par cycle, moins la distance ».
- *Un déplacement n'est pas une correction* : le défaut d'ATTENDRE demande une décision de modélisation à part, avec son impact (réentraînement).

**Commit :** `586daba` (15 octobre 2026).

**Suite : correction du coût d'ATTENDRE (16 octobre 2026).**
- **Décision :** option 1, **coût d'opportunité**, prise par moi le 16 octobre 2026, à partir des recommandations de `docs/note_directrice_decisions.md`, **à présenter à la directrice**.
- **Ce que j'ai changé :** `simulation/reward.py` ajoute un 5ᵉ terme, `opportunity` = − (durée de l'ATTENDRE comprise dans le poste) ÷ `OPPORTUNITY_CYCLE_MIN`, la production perdue en cycles. Son poids est **w1**, celui de la production (même unité) : aucun nouveau poids à régler, et l'ablation de w4 n'est pas touchée. La durée de référence est celle d'un cycle de l'Affectation fixe en nominal : 24 min au moment du changement, **23,2 min** depuis T15 et T16 bis. `env/mine_env.py` passe la durée de l'attente (bornée au poste) au calcul.
- **Effet :** attendre 5 min rapportait −0,05 ; cela coûte maintenant −0,266, environ un cinquième de cycle ; attendre la durée d'un cycle coûte un cycle. Les **heuristiques n'attendent jamais** (vérifié dans les trois régimes) : leur récompense est inchangée, et les références de non-régression passaient sans changement.
- **Tests :** `tests/test_reward.py` (valeur du terme, égalité avec la formule v1 moins le coût, durée de référence mesurée, attente coupée en fin de poste, heuristiques inchangées) ; `tests/test_wait_action.py::test_cout_d_attendre` vérifie les valeurs v1 et v2.
- **Commit :** `be95977` (16 octobre 2026). Durée de référence mise à jour dans `157f97b`.
- **À écrire dans l'article (T31) :** le terme d'opportunité dans la formule de la récompense (section 5.5), et pourquoi il a été ajouté (87 % d'ATTENDRE pour PPO v1 en forte charge).

**Questions « Savoir expliquer » :**
- [x] les formules de la section 5.5 de l'article v2 sans les regarder :

  R = w1 · (tonnage / 140) − w2 · (variance des attentes aux pelles / 100) − w3 · (distance du cycle / 5) − w4 · min(attente / 30, 1), avec w1 = 1, w2 = 0,1, w3 = 0,05, w4 = 0,3.
  - **production** : 1 par cycle terminé avant la fin du poste, 0 sinon et pour ATTENDRE ;
  - **équilibre** : variance des attentes aux 3 pelles au moment de la décision, en min², ÷ 100 ;
  - **distance** : trajet vers la pelle + pelle → verse + retour, en km, ÷ 5. Elle vaut 0 pour ATTENDRE ;
  - **attente** : file à la pelle + file à la verse, ou durée de l'ATTENDRE, ÷ 30 min, plafonnée à 1.
- [x] pourquoi la récompense cumulée vaut à peu près le nombre de cycles : chaque cycle terminé rapporte exactement +1 (140 t / 140 t), et c'est le terme dominant. Les trois pénalités sont petites : sur un poste d'Affectation fixe, la distance coûte −19,1 en tout (environ −0,08 par cycle), l'attente −2,7 et l'équilibre −0,1. Donc 232,1 cycles donnent 210,1, environ 90 % du nombre de cycles. La récompense est essentiellement « un point par cycle, moins la distance parcourue ».
- [x] combien de minutes représente γ = 0,99 par décision avec 12 puis 18 camions : l'horizon effectif vaut 1/(1 − γ) = **100 décisions**. Avec l'Affectation fixe :
  - **12 camions** : 244 décisions par poste de 480 min, soit **1,97 min** par décision, donc environ **197 min** d'horizon ;
  - **18 camions** : 353 décisions, soit **1,36 min** par décision, donc environ **136 min**.

  Avec plus de camions, les décisions sont plus rapprochées, donc l'agent « voit » moins loin dans le temps alors que γ n'a pas changé. Une actualisation en temps continu, de la forme γ^Δt (SMDP), rendrait l'horizon indépendant de la taille de la flotte. C'est une limite à écrire.

## T25 bis — Réglage des hyperparamètres documenté
*Indispensable (au moins la décision) · [RÉENTR] si tu fais le balayage · M12*

**Date :** 15 octobre 2026 (méthode 1 : code et tests écrits par l'IA, relus par moi)

**Prédiction (avant de modifier) :** pas de prédiction chiffrée : la tâche prépare un outil, elle ne produit pas encore de résultat. Attendu : le script est reproductible, puisque les graines sont fixées.

**Décision (avec la directrice) :** **pas encore prise.** Mon choix provisoire est intermédiaire : écrire et tester **maintenant** le script de l'option (a), le balayage documenté, et décider **avec la directrice**, avant la phase 7, entre :
- **(a)** lancer le balayage réel : 8 configurations × 200 000 pas, environ 2 h de calcul, puis reporter le tableau dans l'article ;
- **(b)** ne pas le lancer : retirer la mention des « essais préliminaires » de l'article, garder les valeurs usuelles et écrire l'inégalité de réglage dans les limites.

Le choix doit être fait **avant la phase 7** : l'option (a) peut changer les hyperparamètres de la campagne finale.

**Mise à jour du 16 octobre 2026 :** **option (a)**, prise par moi le 16 octobre 2026, à partir des recommandations de `docs/note_directrice_decisions.md`, **à présenter à la directrice**. Le balayage a été lancé le 16 octobre à 10 h 38, **après** les changements de la récompense (coût d'ATTENDRE) et du moteur (T15, T16 bis), puisqu'ils peuvent changer les meilleurs hyperparamètres. Il tourne depuis une copie de travail figée sur le commit `157f97b`, pour que les modèles ne soient pas marqués « code modifié » pendant que T24 était écrit (premier modèle : `git_dirty: false`). Sorties : `data/results_v2/tuning/`. Il sert aussi de premier contrôle : PPO et DQN apprennent-ils avec la récompense versée à la fin du cycle ? Résultats à reporter ici.

**Ce que j'ai changé :**
- `rl/train_ppo.py` : paramètres `learning_rate` (3e-4) et `ent_coef` (0,02), plus `progress_bar`. Les valeurs par défaut sont celles de la v1, donc un entraînement normal ne change pas.
- `rl/train_dqn.py` : paramètres `learning_rate` (3e-4) et `exploration_fraction` (0,3), plus `progress_bar`, avec les mêmes valeurs par défaut que la v1.
- Les deux enregistrent les valeurs utilisées dans `metadata.json` (rubrique `training`).
- `experiments/tune_hyperparams.py` (nouveau) :
  - **grilles de même taille** : PPO, `ent_coef` ∈ {0 ; 0,02} × `learning_rate` ∈ {1e-4 ; 3e-4} ; DQN, `exploration_fraction` ∈ {0,1 ; 0,3} × `learning_rate` ∈ {1e-4 ; 3e-4}. Soit 4 configurations chacun, qui incluent les valeurs de la v1 ;
  - **même graine d'entraînement** (42) et **même budget** pour tous (200 000 pas par défaut, 10 % de la campagne finale, annoncé dans le tableau) ;
  - **critère** : productivité moyenne en nominal sur les **graines 20 000 à 20 009**, disjointes des graines d'évaluation (42 à 71) ;
  - sorties : un dossier par configuration (modèle + `metadata.json`) et `tuning_results.csv` (valeurs testées, budget, graines de sélection, moyenne ± écart-type, meilleure configuration par algorithme) ;
  - reprise : une configuration déjà entraînée est réévaluée sans être réentraînée.

**Pourquoi :** l'article v2 affirme que des « essais préliminaires » ont fixé le coefficient d'entropie de PPO à 0,02 (0 ; 0,01 ; 0,02 ; 0,05 testés), mais aucun script du projet ne les contient, et DQN tourne avec ses valeurs par défaut. Un relecteur peut objecter que PPO a été mieux réglé que DQN (M12).

**Ce que le test vérifie :** `tests/test_tune_hyperparams.py`, 5 tests, sur un budget minuscule (1 024 pas) :
- même nombre de configurations pour PPO et DQN, valeurs de la v1 incluses ;
- les 10 graines de sélection sont disjointes des graines d'évaluation de **tous** les scénarios ;
- identifiant de configuration stable, indépendant de l'ordre des paramètres ;
- **test du plan** : relancer une configuration (PPO et DQN) dans un autre dossier redonne **exactement** le même critère ;
- les valeurs testées sont écrites dans `metadata.json`, et une configuration déjà entraînée n'est pas réentraînée.

Suite complète : **210 passed**. Non-régression verte. Ruff ne signale rien.

**Résultat comparé à ma prédiction :** confirmée. Le script est reproductible et prêt. **Aucun réglage réel n'a été lancé** : il n'y a donc pas encore de tableau chiffré pour l'article.

**Ce que j'ai appris :**
- *Séparer les données de réglage et d'évaluation* : choisir une configuration sur les graines de l'évaluation, c'est choisir en partie celle qui a eu de la chance sur ces épisodes.
- *L'équité d'une comparaison se joue avant l'évaluation* : même nombre d'essais, même budget, même graine d'entraînement pour les deux algorithmes.
- *Un budget réduit est acceptable s'il est écrit* : on règle à 10 % du budget final, et on le dit.
- *Écrire l'outil avant de décider de le lancer* : le code et les tests ne coûtent presque rien, alors que les heures de calcul se décident en connaissance de cause.

**Commit :** `62a4be3` (15 octobre 2026), pour le script. La décision reste à prendre.

**Questions « Savoir expliquer » :**
- [x] pourquoi un réglage fait sur les graines d'évaluation biaise la comparaison : la productivité d'un modèle sur 30 épisodes contient une part de hasard. Si l'on teste plusieurs configurations et qu'on garde la meilleure **sur ces mêmes 30 graines**, on retient en partie celle dont le hasard a été favorable sur ces épisodes. Sa performance rapportée sur ces graines est donc **surestimée** : c'est le principe d'un jeu de test qui a servi à choisir le modèle. Il faut choisir sur des graines **à part** (20 000 à 20 009), puis mesurer le résultat sur les graines d'évaluation (42 à 71), que le réglage n'a jamais vues.
- [x] pourquoi un budget de réglage inégal fausse « PPO contre DQN » : un algorithme essayé avec plusieurs configurations a plus de chances d'en trouver une bonne qu'un algorithme laissé avec ses valeurs par défaut. Si PPO a eu plusieurs essais sur son coefficient d'entropie et DQN aucun, un écart en faveur de PPO peut venir du **réglage**, et non de l'**algorithme**. La comparaison n'est équitable qu'avec le même nombre d'essais, le même budget et le même critère de sélection pour les deux. Le script les impose : 4 configurations chacun, 200 000 pas, graines 20 000 à 20 009.

## T26 — Graines d'entraînement et d'évaluation séparées
*Indispensable · [RÉENTR] · M7*

**Date :** 15 octobre 2026 (méthode 1 : code et tests écrits par l'IA, relus par moi)

**Prédiction (avant de modifier) :** aucun chiffre de l'article ne change tant que la campagne finale n'est pas relancée : les heuristiques ne s'entraînent pas, et les modèles existants restent lisibles. Le test du plan devrait passer dès que les entraîneurs reçoivent leurs nouvelles graines.

**Écart volontaire avec le plan :** le plan proposait de déplacer l'**évaluation** vers les graines 10 000 à 10 029. J'ai déplacé l'**entraînement** à la place. Le résultat est le même (trois ensembles disjoints, vérifiés par les tests), et les graines d'évaluation 42 à 71 utilisées par les tâches précédentes (références de non-régression, T13, T23…) ne changent pas.

**Ce que j'ai changé :**
- `seeds.py` (nouveau, à la racine du paquet pour que `rl/` ne dépende pas d'`experiments/`) : un seul endroit pour toutes les graines.
  - **évaluation** : `EVAL_SEEDS` = 42 à 71 ;
  - **sélection des hyperparamètres** (T25 bis) : `TUNING_SEEDS` = 20 000 à 20 009 ;
  - **entraînement** : `TRAIN_SEEDS` = 1 à 5. Un agent de graine *s* joue ses épisodes sur les graines *s* × 1 000 000, + 1, + 2… (`training_episode_seed`). Elles ne croisent jamais les deux autres ensembles, ni celles d'une autre graine d'entraînement.
- `rl/train_q_learning.py`, `rl/train_sarsa.py` : `reset(seed=training_episode_seed(seed, ep))` au lieu de `seed + ep`.
- `rl/train_ppo.py`, `rl/train_dqn.py` : l'environnement démarre à `training_episode_seed(seed)` ; la graine du modèle (réseau, exploration) reste `seed`.
- Graine par défaut des quatre entraîneurs : `TRAIN_SEEDS[0]` = 1 (avant : 42). Dossier par défaut : `<algo>_nominal_seed1`.
- `rl/model_paths.py` : `model_dir(..., train_seed)` ajoute `_seed<s>` (`ppo_nominal_seed3`) ; sans graine, on garde le chemin des modèles v1 (`ppo_nominal`).
- `env/mine_env.py` : `last_episode_seed`, la graine réellement utilisée par le dernier `reset`, compteur interne compris. Elle sert au test.
- `experiments/run_benchmark.py` :
  - paramètre `train_seeds` (par défaut 1 à 5) : un modèle par (algorithme, régime, graine d'entraînement), entraîné s'il manque, chacun évalué sur les 30 graines d'évaluation ;
  - colonne **`train_seed`** dans le CSV (vide pour les heuristiques) ;
  - l'entraînement est regroupé dans une fonction `_train`, au lieu de quatre blocs presque identiques ;
  - `write_results` : un doublon est maintenant (régime, politique, graine d'entraînement, graine d'évaluation).
- `analysis/statistical_tests.py` :
  - `load_results` met `train_seed` dans ses clés quand la colonne existe ; les heuristiques (graine vide) forment un groupe ;
  - `paired_table` **refuse** une politique qui a plusieurs graines d'entraînement : le tableau apparié ne sait pas encore les combiner (agrégation en T30).
- `rl/diagnostics/` (3 scripts) : ils s'entraînaient sur `scenario.seeds[0]` = 42, une graine d'évaluation ; ils utilisent maintenant `TRAIN_SEEDS[0]`.
- `experiments/tune_hyperparams.py` : graine d'entraînement `TRAIN_SEEDS[0]` = 1, et non plus 42 comme écrit dans le journal de T25 bis.

**Deux défauts trouvés par les nouveaux tests, corrigés dans le même commit :**
1. **PPO et DQN jouaient leur premier épisode sur la graine du modèle.** Stable-Baselines3 ré-ensemence l'environnement avec la graine donnée au modèle (`set_random_seed` → `env.seed(seed)`), ce qui écrase celle de `make_vec_env`. Le test a vu passer la graine 1 pendant l'entraînement de DQN. En v1, cette graine était 42 : **la première graine d'évaluation était aussi le premier épisode d'entraînement de PPO et DQN.** Correction : `env.seed(training_episode_seed(seed))` juste après la création du modèle.
2. **`write_results` aurait planté sur le benchmark complet.** Il prenait les colonnes de la première ligne, qui est une heuristique ; les colonnes des agents tabulaires ajoutées en T20 (`known_state_pct`, `default_action_pct`) provoquaient une erreur. Correction : on écrit l'union des colonnes, dans l'ordre d'apparition ; les cases manquantes restent vides.

**Pourquoi :** avant T26, il n'y avait qu'une graine d'entraînement (42), et les graines d'évaluation 42 à 71 étaient exactement les premiers épisodes d'entraînement. L'évaluation n'était donc pas faite sur des épisodes nouveaux. De plus, un seul modèle par algorithme ne permet pas de mesurer la variabilité d'entraînement (M7).

**Ce que le test vérifie :**
- `tests/test_seeds.py`, 11 tests :
  - les trois ensembles de graines sont disjoints, et les scénarios n'utilisent que des graines d'évaluation ;
  - deux graines d'entraînement ne partagent aucun épisode ; un numéro d'épisode hors plage est refusé ;
  - **test du plan**, pour Q-Learning, SARSA, DQN et PPO : pendant un petit entraînement (épisodes de 60 min, pour avoir plusieurs remises à zéro), toutes les graines passées à `reset`, compteur interne compris, sont des graines d'entraînement, jamais d'évaluation ni de sélection ;
  - un dossier distinct par graine d'entraînement ;
  - `run_benchmark` avec deux graines d'entraînement : chaque agent est entraîné une fois par graine, dans son dossier, et chaque ligne porte sa `train_seed` (vide pour les heuristiques) ; le CSV s'écrit ;
  - `load_results` : deux graines d'entraînement ne sont pas des doublons, et un modèle incomplet est signalé ;
  - `paired_table` accepte une seule graine d'entraînement et en refuse plusieurs, en citant T30.
- `tests/test_results_writer.py`, 2 tests ajoutés : colonnes absentes de la première ligne ; même politique avec deux graines d'entraînement (pas un doublon) ou deux fois la même (doublon).

Suite complète : **223 passed, 1 skipped**. Ruff ne signale rien.

**Résultat comparé à ma prédiction :** confirmée pour les chiffres : rien ne change tant que la campagne n'est pas relancée. **Mais le test du plan n'est pas passé du premier coup** : il a révélé que Stable-Baselines3 réutilisait la graine du modèle pour le premier épisode. Sans ce test, la fuite serait restée (graine 1, inoffensive maintenant, mais 42 en v1). Il a aussi révélé le défaut de `write_results`, qui aurait fait échouer la campagne finale après des heures de calcul.

**Conséquence pour la campagne finale :** 5 entraînements par algorithme et par régime au lieu d'un, soit environ 5 fois le temps d'entraînement de la v1. À prévoir avec la directrice, en même temps que la décision de T25 bis.

**Mise à jour du 16 octobre 2026 :** **5 graines confirmées**, à présenter à la directrice. Reste à décider avant T31 : la portée de l'ablation de w4 (PPO seul ou tous les agents) et le lancement de plusieurs entraînements en parallèle.

**Ce que j'ai appris :**
- *Une bibliothèque peut agir sur des objets qu'on ne lui a pas confiés explicitement* : donner une graine au modèle ré-ensemence aussi l'environnement. Seul un test qui **observe** les graines réellement utilisées pouvait le voir ; relire le code n'aurait pas suffi.
- *Un seul endroit pour une décision* : toutes les graines sont dans `seeds.py`, et un test vérifie qu'elles ne se recouvrent pas, au lieu de 42 éparpillés dans le code.
- *Un test de bout en bout trouve des défauts que les tests unitaires ne voient pas* : `write_results` était testé, mais jamais avec des lignes d'heuristiques et d'agents tabulaires mélangées.
- *Refuser plutôt que se tromper en silence* : `paired_table` lève une erreur avec plusieurs graines d'entraînement au lieu de produire un tableau faux.

**Commit :** `afcc713` (15 octobre 2026)

**Questions « Savoir expliquer » :**
- [x] la différence entre variabilité d'entraînement et variabilité d'évaluation : la **variabilité d'évaluation** vient du poste de travail. Pannes, temps de trajet et temps de chargement changent d'une graine d'évaluation à l'autre, pour un modèle **fixé**. La **variabilité d'entraînement** vient de l'agent lui-même. Initialisation du réseau, exploration et épisodes vus changent avec la graine d'entraînement, et donnent des **modèles différents**. Évaluer un seul modèle sur 30 graines mesure la première, jamais la seconde.
- [x] pourquoi on ne peut pas dire « PPO est meilleur que DQN » avec un seul entraînement chacun : avec une seule graine d'entraînement, on compare **un** PPO et **un** DQN, deux tirages parmi tous les modèles que chaque algorithme aurait pu produire. Un autre tirage pourrait inverser l'ordre. Les 30 graines d'évaluation rendent très précise la comparaison de **ces deux modèles-là**, mais elles ne disent rien sur l'**algorithme**. Pour conclure sur l'algorithme, il faut plusieurs graines d'entraînement (ici 5) et agréger les résultats (T30).

## T27 — Évaluation sans réentraînement
*Indispensable · [MESURE] · M1*

**Date :** 15 et 16 octobre 2026 (méthode 1 : code et tests écrits par l'IA, relus par moi)

**Prédiction (avant de modifier) :** avec les modèles v1 sur le code d'avant T13, PPO nominal en fortes pannes doit redonner environ 3 926 t/h (tableau 14 de l'article v2). Je m'attendais aussi à ce qu'un agent transféré soit **moins bon** qu'un agent spécialisé, puisqu'il n'a jamais vu le régime.

**Ce que j'ai changé :**
- `experiments/scenarios.py` :
  - deux modes : `SPECIALISED` = « spécialisé » (agent entraîné dans le régime évalué) et `TRANSFERRED` = « transféré » (agent entraîné en nominal, évalué tel quel) ;
  - `training_regimes(nom)` décide quels agents évaluer : en régime perturbé (forte charge, fortes pannes), les agents de ce régime **et** les agents nominaux ; pour un scénario non vu, seulement les nominaux ; sinon (nominal, ablations), seulement les siens ;
  - `evaluation_mode(scénario, régime d'entraînement)` donne le mode.
- `experiments/run_benchmark.py` :
  - `model_scenario` (un seul régime) devient `model_scenarios` (une liste, par défaut `training_regimes(...)`) ;
  - chaque agent est évalué une fois par régime d'entraînement ; les heuristiques une seule fois ;
  - nouvelle colonne **`mode`** dans le CSV (vide pour les heuristiques), ajoutée à la clé de doublon de `write_results` ;
  - `train_seeds=(None,)` relit les modèles d'avant T26 (dossiers sans suffixe, modèles v1) ; s'ils manquent, erreur, jamais d'entraînement ;
  - `run_all_benchmarks` ne fait plus de cas particulier pour les scénarios non vus : tout passe par `training_regimes`.
- `analysis/statistical_tests.py` :
  - `select_mode(data, mode)` garde les lignes d'un mode, plus les heuristiques ;
  - `load_results(..., mode="spécialisé")` par défaut : les tableaux principaux de l'article (Friedman, Wilcoxon, interaction, figures) ne mélangent pas les deux modes ; `mode=None` garde tout, et le mode devient une clé ;
  - un CSV sans colonne `mode` (v1) est lu comme avant.
- `analysis/transfer.py` (nouveau) : `transfer_table` calcule, par régime et par agent, le **coût du transfert** (transféré − spécialisé) sur les **mêmes** graines d'évaluation : moyennes, écart, IC 95 % bootstrap apparié, écart en %. Avec plusieurs graines d'entraînement, les écarts sont moyennés par graine d'évaluation (la variabilité d'entraînement est pour T30). Lancement : `python -m memoire_master_rl_logistique.analysis.transfer RESULTATS.csv`.

**Pourquoi :** l'article v1 évaluait chaque agent dans le régime où il avait été entraîné. Il ne mesurait donc pas ce qu'il annonçait : la robustesse d'un agent face à des conditions qu'il n'a pas vues (M1). Le tableau `tab:transfer` de l'article v2 avait été produit hors de mes scripts.

**Test prévu par le plan, et pourquoi il est fait autrement :** le test (PPO v1 nominal en fortes pannes ≈ 3 926 t/h) **ne peut plus tourner sur le code actuel** : depuis T18, l'observation a 18 composantes et PPO v1 en attend 147 ; depuis T13, les tirages aléatoires ont changé. Je l'ai donc fait **une fois**, sur le code d'avant T13 (commit `d1cec48`), dans un dossier de travail git séparé, supprimé ensuite. Modèles v1, fortes pannes, 30 graines d'évaluation (42 à 71), productivité moyenne en t/h :

| Agent | Transféré (entraîné en nominal) | Spécialisé (entraîné en fortes pannes) | Écart |
|---|---|---|---|
| Q-Learning | **3 410,8** | 3 180,9 | + 229,9 |
| SARSA | 2 923,1 | 2 980,8 | − 57,7 |
| DQN | 3 882,1 | 3 949,2 | − 67,1 |
| PPO | **3 926,4** | 3 899,6 | + 26,8 |

PPO nominal transféré = **3 926,4 t/h** : le tableau 14 de l'article v2 est retrouvé. Pas de test de significativité sur ces écarts : ce sont des chiffres v1, à recalculer en T31.

**Ce que le test vérifie :** `tests/test_transfer.py`, 13 tests :
- `training_regimes` et `evaluation_mode` pour un régime perturbé, un scénario non vu, le nominal et une ablation ;
- en forte charge **et** en fortes pannes : chaque agent est entraîné dans les deux régimes (dossiers `<algo>_<régime>_seed1` et `<algo>_nominal_seed1`), évalué dans les deux modes **sur les mêmes graines**, les heuristiques une seule fois sans mode, et le CSV s'écrit (même agent dans deux modes ≠ doublon). La forte charge ne marche que grâce à T18 ;
- scénario non vu : seulement le mode transféré ;
- un modèle d'avant T26 absent lève une erreur et n'est jamais entraîné ;
- les Q-tables nominales v1 s'évaluent en fortes pannes sans réentraînement (ignoré si `data/results_v1_article` est absent) ;
- `select_mode` et `load_results` : par défaut, pas d'agent transféré ; avec `mode=None`, spécialisé et transféré ne sont pas des doublons ; CSV v1 intact ;
- `transfer_table` sur des données dont l'écart est connu (−10 t/h sur chaque graine → écart −10, IC [−10 ; −10]) ; deux graines d'entraînement moyennées ; graine évaluée dans un seul mode ignorée ; CSV sans mode refusé.

Suite complète : **236 passed, 1 skipped**. Ruff ne signale rien.

**Résultat comparé à ma prédiction :** la première partie est **confirmée** : 3 926,4 t/h. La seconde est **infirmée** : en fortes pannes, l'agent transféré n'est pas systématiquement moins bon. PPO et Q-Learning nominaux font même mieux que leurs versions spécialisées ; SARSA et DQN perdent un peu (moins de 2 %). Pour les fortes pannes au moins, l'entraînement spécialisé n'apporte donc pas grand-chose. Reste à voir la forte charge (18 camions), testable seulement depuis T18, avec la campagne finale.

**Ce que j'ai appris :**
- *Une affirmation se mesure avec le protocole qui lui correspond* : « robuste » demande une politique figée face à un régime inconnu, pas une politique réentraînée.
- *Séparer les modes dans les données, filtrer à la lecture* : une seule colonne `mode` et un filtre par défaut dans `load_results` évitent de mélanger les deux dans les tableaux principaux.
- *Un résultat ancien se vérifie sur le code ancien* : un dossier de travail git sur un ancien commit permet de rejouer une mesure sans toucher à la branche.
- *Une prédiction infirmée est un résultat* : un agent spécialisé n'est pas forcément meilleur ; le réentraînement a aussi sa variabilité (une seule graine d'entraînement en v1, voir T26).

**Commit :** `da45df5` (16 octobre 2026)

**Questions « Savoir expliquer » :**
- [x] pourquoi l'article v1 mesurait l'adaptation et non la robustesse : en v1, chaque agent était **réentraîné** dans chaque régime avant d'y être évalué. Une bonne performance en fortes pannes montrait seulement que l'algorithme sait **apprendre** une bonne politique quand on lui fournit ce régime : c'est de l'**adaptation**. La **robustesse**, c'est qu'une politique **figée**, entraînée en nominal, garde ses performances quand les conditions changent sans qu'on la réentraîne, comme dans une vraie mine où l'on ne réentraîne pas l'agent à chaque panne. Seule l'évaluation des agents **transférés** mesure cela ; depuis T27, le benchmark évalue les deux et l'écart entre eux est le coût du transfert.

## T28 — Fichiers de configuration réellement utilisés
*Recommandé · [MESURE] · cohérence code/article*

**Date :** 16 octobre 2026 (méthode 1 : code et tests écrits par l'IA, relus par moi)

**Prédiction (avant de modifier) :** aucun chiffre ne change. On déplace des valeurs du code vers les fichiers YAML sans les modifier, donc les scénarios, les graines, les hyperparamètres et les références de non-régression doivent rester identiques.

**Ce que j'ai changé :**
- `configs/experiment_config.yaml` (réécrit) :
  - les graines : évaluation (42, 30 graines), sélection (20 000, 10 graines), entraînement (1 à 5) et l'écart de 1 000 000 entre les épisodes de deux graines d'entraînement ;
  - `scenario_defaults` (12 camions, 3 pelles, 2 verses, 480 min, pannes 2 %, poids de la récompense, 2 000 000 pas), puis les **9 scénarios**, chacun avec seulement ce qui change ;
  - `main_scenarios` : les 3 régimes de l'article, le premier étant la référence dont les agents sont transférés (T27) ;
  - `tuning` : budget et grilles du réglage de T25 bis ;
  - `reproducibility` : versions attendues (informatif).
- `configs/ppo_hyperparameters.yaml`, `configs/dqn_hyperparameters.yaml` : valeurs v1 inchangées ; la clé `seed: 42` est retirée (depuis T26, un agent par graine d'entraînement).
- `configs/tabular_hyperparameters.yaml` (nouveau) : Q-Learning et SARSA, 30 000 épisodes, alpha 0,2, gamma 0,99, ε de 1,0 à 0,01, 8 intervalles.
- `config.py` (nouveau) :
  - `experiment_config()` et `hyperparameters(algorithme, **remplacements)` lisent et **valident strictement** les fichiers : une clé manquante ou inconnue (faute de frappe) lève une erreur ;
  - un remplacement à `None` est ignoré (argument non fourni) ; un remplacement inconnu est refusé ;
  - les valeurs renvoyées sont des copies : les modifier ne change pas la configuration ;
  - `snapshot_configs(dossier)` archive les fichiers utilisés ; une archive existante doit être identique ;
  - `MINE_RL_CONFIG_DIR` permet de lire un autre dossier que `configs/`.
- `seeds.py` : les graines viennent du YAML.
- `experiments/scenarios.py` : `SCENARIOS` et `MAIN_SCENARIOS` sont construits à partir du YAML (`_build_scenarios`). La classe `Scenario` n'a plus de valeurs par défaut en dur (sauf les graines), pour qu'elles ne puissent plus diverger du fichier.
- `rl/train_ppo.py`, `rl/train_dqn.py`, `rl/train_q_learning.py`, `rl/train_sarsa.py` : hyperparamètres lus dans le YAML ; un argument donné les remplace (le réglage de T25 bis s'en sert). `metadata.json` contient maintenant **tous** les hyperparamètres utilisés, plus seulement ceux qui sont réglés. Leurs `main()` n'ont plus de valeurs en dur.
- `experiments/run_benchmark.py` :
  - `run_all_benchmarks` archive la configuration dans `<dossier>/configs_<run_id>/` **avant** le calcul, refuse un scénario inconnu et renvoie le chemin du CSV ;
  - commande : `python -m memoire_master_rl_logistique.experiments.run_benchmark DOSSIER [--scenarios ...]` (par défaut, les régimes principaux de la configuration) ;
  - `_train` ne passe plus d'hyperparamètres tabulaires en dur.
- `experiments/tune_hyperparams.py` : grilles et budget lus dans le YAML ; configuration archivée dans `<dossier>/configs` ; une reprise avec une configuration modifiée est refusée.
- `main.py` (`--episodes`, `--timesteps`) et `rl/diagnostics/` : plus de 30 000 épisodes ni de 2 000 000 de pas écrits en dur.
- `pyproject.toml` : `pyyaml` ajouté aux dépendances.

**Pourquoi :** le dossier `configs/` existait, et l'article v1 l'annonçait, mais **aucun script ne le lisait**. Les valeurs étaient recopiées en dur dans `scenarios.py` et `train_*.py`, et elles avaient déjà divergé :
- la graine 42 des YAML n'était plus celle de l'entraînement depuis T26 ;
- les valeurs par défaut des fonctions tabulaires (alpha 0,1, 10 000 épisodes) n'étaient pas celles de la campagne (alpha 0,2, 30 000 épisodes, passées par `run_benchmark`) ;
- la version de Python était déjà fausse (T3).

Un relecteur qui lisait `configs/` croyait voir les réglages de l'expérience, sans garantie que ce soient les bons.

**Ce que le test vérifie :**
- `tests/test_config.py`, 16 tests :
  - **les valeurs n'ont pas changé** : les 9 scénarios, les régimes principaux, les graines et les hyperparamètres PPO, DQN et tabulaires sont exactement ceux d'avant T28 ;
  - **les valeurs restées dans le code sont cohérentes** : les valeurs par défaut des 4 fonctions d'entraînement (environnement, graine, budget) et `DEFAULT_N_BINS` sont égales à la configuration ;
  - **les scripts lisent vraiment la configuration** : le `metadata.json` de Q-Learning contient les hyperparamètres du YAML ; les modèles PPO et DQN construits ont les valeurs du YAML (`n_steps`, `batch_size`, `buffer_size`, architecture…), et un argument donné les remplace ;
  - **une autre configuration change vraiment le résultat** : avec `MINE_RL_CONFIG_DIR`, forte charge à 20 camions et évaluation à partir de la graine 100, vérifié dans un processus séparé ;
  - **validation stricte** : faute de frappe, clé manquante, fichier d'un autre algorithme, scénario mal décrit et régime principal inconnu sont refusés ; les valeurs renvoyées sont des copies ;
  - **archivage** : les fichiers sont copiés à l'identique, la reprise est acceptée, une archive modifiée est refusée ; le benchmark archive `configs_<run_id>/` à côté de `benchmark_<run_id>.csv` ; la commande évalue les régimes principaux par défaut.
- `tests/test_tune_hyperparams.py` adapté : `metadata.json` contient tous les hyperparamètres, et la configuration est archivée.

Avant de modifier `scenarios.py`, j'ai aussi enregistré tous les scénarios et toutes les graines en JSON, puis comparé après modification : identiques, champ par champ et dans le même ordre.

Suite complète : **252 passed, 1 skipped**. Non-régression verte. Ruff ne signale rien.

**Résultat comparé à ma prédiction :** confirmée : aucun scénario, aucune graine, aucun hyperparamètre de la campagne n'a changé, et les références de non-régression passent. Seule exception, voulue : appelées **sans argument**, les fonctions d'entraînement tabulaires utilisent maintenant les valeurs de la campagne (alpha 0,2, 30 000 épisodes) au lieu d'alpha 0,1 et 10 000 épisodes. Les campagnes, qui passaient déjà ces valeurs, ne changent pas.

**Ce que j'ai appris :**
- *Une configuration non lue est pire que pas de configuration* : elle donne une fausse assurance à celui qui la lit.
- *Une seule source pour chaque valeur* : quand une valeur doit rester dans le code (valeur par défaut d'une fonction), un test vérifie qu'elle est égale à la configuration.
- *Valider strictement* : une faute de frappe dans un YAML ne doit pas être ignorée en silence, sinon on lance des heures de calcul avec la valeur par défaut sans le savoir.
- *Archiver la configuration avant le calcul, pas après* : même un lancement interrompu garde la trace de ses réglages.
- *Tester qu'une autre configuration change vraiment le résultat* : vérifier que le fichier est lu ne suffit pas, il faut vérifier qu'il **agit**.

**Commit :** `0ab2031` (16 octobre 2026)

**Questions « Savoir expliquer » :**
- [x] comment un relecteur relancerait une expérience précise avec une seule commande : chaque campagne archive ses fichiers de configuration à côté de ses résultats (`configs_<run_id>/`, même identifiant que `benchmark_<run_id>.csv`), et chaque modèle garde ses hyperparamètres et le commit Git dans son `metadata.json`. Le relecteur reprend le commit indiqué, puis lance une seule commande : `MINE_RL_CONFIG_DIR=<dossier>/configs_<run_id> python -m memoire_master_rl_logistique.experiments.run_benchmark <nouveau dossier>` (sous PowerShell : `$env:MINE_RL_CONFIG_DIR="..."` d'abord). Régimes, graines et hyperparamètres sont alors exactement ceux du lancement d'origine, et la validation stricte garantit qu'aucune valeur n'est ignorée en silence.

## T29 — Parallélisation et mesure du temps de calcul
*Recommandé · [MESURE] pour l'évaluation, [RÉENTR] pour PPO à plusieurs environnements · M28*

**Date :** 16 octobre 2026 (méthode 1 : code et tests écrits par l'IA, relus par moi)

**Prédiction (avant de modifier) :**
- l'évaluation en parallèle donne **exactement** les mêmes résultats qu'en séquentiel, graine par graine : chaque épisode ne dépend que de sa politique et de sa graine (flux aléatoires propres depuis T13) ;
- seules les durées mesurées changent ;
- avec un seul environnement, PPO et DQN restent identiques à T28 ;
- avec plusieurs environnements, PPO donne un **autre** modèle ;
- la latence par décision reste inférieure à la milliseconde et dépend peu du nombre de camions, puisque l'observation a une taille fixe depuis T18.

**Ce que j'ai changé :**
- `experiments/run_benchmark.py` :
  - **évaluation parallèle**, avec le paramètre `n_workers` et l'option `--workers` (1 par défaut, donc rien ne change si on ne la donne pas) ;
  - chaque épisode (politique, graine) est une tâche indépendante, exécutée par `_evaluate` ; `_run_tasks` les exécute en séquence ou avec `ProcessPoolExecutor`, et garde l'ordre des lignes ;
  - une politique chargée (une fonction qui contient le modèle) ne peut pas être envoyée à un autre processus. On envoie donc sa **description** (`PolicySpec` : nom, mode, graine d'entraînement, dossier du modèle, date du fichier, environnement d'entraînement), et chaque processus recharge le modèle **une seule fois** grâce à un cache ;
  - tous les modèles sont chargés **avant** les épisodes : un modèle incompatible arrête tout de suite, pas au milieu d'un calcul parallèle ;
  - le séquentiel et le parallèle passent par le même code ;
  - nouvelle colonne `episode_time_s` (durée réelle de l'épisode).
- `analysis/statistical_tests.py` : `episode_time_s` est ajouté aux colonnes dynamiques. Deux lancements identiques qui ne diffèrent que par les durées ne sont pas des doublons.
- `env/mine_env.py` :
  - nouveau paramètre `seed_step`, le pas du compteur de graines d'épisodes (1 par défaut) ;
  - une graine explicite repositionne le compteur ;
  - avec un seul environnement, la suite des graines est exactement celle d'avant.
- `rl/env_factory.py` :
  - `make_training_vec_env(graine, n_envs)` construit les environnements d'entraînement : `DummyVecEnv` pour un seul environnement (comme avant), `SubprocVecEnv` au-delà ;
  - l'environnement i joue les épisodes base + i, base + i + n, base + i + 2n…, donc **aucun épisode n'est joué deux fois** ;
  - `reseed_training_vec_env` remplace le ré-ensemencement écrit à la main dans chaque script.
- `configs/ppo_hyperparameters.yaml` : `n_envs: 1` (protocole v1), avec un commentaire qui dit que changer cette valeur change le modèle.
- `rl/train_ppo.py` : `n_envs` est lu dans le YAML, ou passé en argument. Il est enregistré dans `metadata.json`.
- `rl/train_ppo.py`, `train_dqn.py`, `train_q_learning.py`, `train_sarsa.py`, `model_metadata.py` : la **durée d'entraînement** (`training_time_s`) est enregistrée dans `metadata.json`. Les anciens fichiers, qui n'ont pas ce champ, se relisent toujours.
- `experiments/timing.py` (nouveau) :
  - mesure la latence par décision (appel de la politique + construction de l'observation) et la durée d'un épisode, pour 12, 18 et 24 camions ;
  - les agents sont les modèles **nominaux**, évalués tels quels avec plus de camions ;
  - toujours en séquentiel, pour que des processus concurrents ne faussent pas les durées ;
  - ne réécrit jamais un fichier existant ;
  - commande : `python -m memoire_master_rl_logistique.experiments.timing SORTIE.csv [--models-dir DOSSIER --train-seed 1] [--n-seeds 5]`.
- `tests/test_config.py` : les hyperparamètres PPO attendus contiennent `n_envs`.

**Pourquoi :**
- **Les réplications sont indépendantes.** Les exécuter l'une après l'autre gaspille 7 cœurs sur 8. La campagne T31 en compte des milliers : 9 scénarios × 30 graines × (5 heuristiques + 4 agents × 5 graines d'entraînement × 1 ou 2 modes).
- **PPO est l'entraînement le plus long** (environ 34 h prévues pour la campagne). Il fallait savoir combien plusieurs environnements feraient gagner, et à quel prix pour la reproductibilité.
- **L'article annonce une latence (environ 0,9 ms) et un temps de calcul** (`sec:compute`), mais aucun script ne les mesurait de façon reproductible, ni en fonction de la taille de la flotte.

**Test prévu par le plan :** les résultats parallèles sont identiques aux résultats séquentiels, graine par graine.

**Ce que le test vérifie :**
- `tests/test_parallel.py`, 14 tests :
  - **test du plan** : les 8 politiques (5 heuristiques, Q-Learning, SARSA, DQN et PPO), spécialisées et transférées, sur 3 graines en fortes pannes, donnent un tableau **identique** en séquentiel et sur 3 processus. Seules les colonnes de durée sont exclues de la comparaison ;
  - un épisode rejoué après un autre redonne la même ligne : c'est ce qui rend le parallélisme exact ;
  - une tâche se transmet bien à un autre processus (sérialisation) ;
  - un nombre de processus invalide est refusé ;
  - `episode_time_s` est une colonne dynamique ;
  - le compteur avec pas (`seed_step`) et la graine explicite qui le repositionne ;
  - avec un seul environnement, la suite des graines est celle d'avant T29 ;
  - avec **3 environnements `SubprocVecEnv`**, chaque environnement joue base + i + 3k, aucun épisode n'est joué deux fois, et toutes les graines restent dans la plage de la graine d'entraînement ;
  - **Savoir expliquer** : même graine et même budget avec 1 puis 2 environnements, PPO donne deux modèles différents ; deux entraînements avec 1 environnement donnent des modèles identiques ;
  - `training_time_s` est présent dans `metadata.json`, et un ancien `metadata.json` sans ce champ se relit ;
  - la table de latence a une ligne par (camions, politique), plus de camions donnent plus de décisions, les agents absents sont ignorés ;
  - la commande de latence refuse de réécrire un fichier existant.

Vérification hors tests : j'ai entraîné un petit PPO et un petit DQN sur le code de T28 (une copie de travail séparée), puis sur celui de T29 avec un seul environnement. Les **poids sont identiques** (26 tableaux comparés), et les graines d'épisodes aussi.

Suite complète : **266 passed, 1 skipped**. Non-régression verte. Ruff ne signale rien.

**Mesures sur ma machine** (8 cœurs, sans autre calcul en même temps ; une seule mesure par chiffre) :

| Mesure | Résultat |
|---|---|
| PPO, 65 536 pas en nominal, 1 environnement | 164,6 s |
| PPO, 65 536 pas en nominal, 4 environnements (`SubprocVecEnv`) | 112,0 s (**× 1,47**) |
| Évaluation de 150 épisodes (5 heuristiques × 30 graines, nominal), 1 / 4 / 8 processus | 19,0 s / **9,7 s** / 12,9 s |

Latence par décision (décision + observation, en ms, 5 graines) :

| Politique | 12 camions | 18 camions | 24 camions |
|---|---|---|---|
| Fixed | 0,043 | 0,047 | 0,046 |
| FIFO | 0,058 | 0,062 | 0,057 |
| Q-Learning | 0,092 | 0,078 | 0,080 |
| Nearest | 0,095 | 0,098 | 0,097 |
| MTWT | 0,16 | 0,15 | 0,14 |
| DQN | 0,36 | 0,48 | 0,49 |
| ShortestPath | 0,41 | 0,38 | 0,38 |
| PPO | 0,48 | 0,71 | 0,71 |

Une première mesure de PPO (178,6 s contre 160,5 s) a été écartée : un pytest tournait en même temps et faussait les durées.

**Résultat comparé à ma prédiction :** confirmée sur tous les points.
- Parallèle = séquentiel, graine par graine.
- Avec un seul environnement, PPO et DQN sont identiques à T28.
- Avec plusieurs environnements, PPO donne un autre modèle.
- La latence reste sous 1 ms partout et dépend peu de la taille de la flotte.

Ce que je n'avais pas prévu :
- **8 processus sont plus lents que 4** pour l'évaluation : chaque processus doit d'abord charger PyTorch, ce qui coûte plus que le gain sur un petit calcul. Pour la campagne, 4 processus est le bon réglage.
- **Le gain de PPO à 4 environnements reste modeste** (× 1,47, pas × 4). Les environnements sont rapides, et le calcul du réseau ainsi que les échanges entre processus ne sont pas parallélisés.

Limites :
- une seule mesure par chiffre : le gain est un ordre de grandeur ;
- la latence a été mesurée avec des agents peu entraînés. Elle dépend de l'architecture du réseau, pas de ses poids, donc elle est juste ; mais le nombre de décisions et la durée d'un épisode sont à remesurer avec les modèles finaux en T31 ;
- je n'ai pas vérifié qu'un PPO à 4 environnements est **aussi bon**, seulement qu'il est plus rapide et différent.

**Ce que j'ai appris :**
- *Paralléliser n'est exact que si les tâches sont indépendantes* : ici, chaque épisode ne dépend que de sa politique et de sa graine. C'est le travail de T13 (flux aléatoires propres) qui rend ce parallélisme sûr.
- *Envoyer la description d'un objet, pas l'objet* : un modèle chargé ne passe pas d'un processus à l'autre, sa description si. Chaque processus le recharge une fois.
- *Échouer tôt* : charger tous les modèles avant le calcul évite de découvrir une erreur au bout d'une heure de calcul parallèle.
- *Plus de processus n'est pas toujours plus rapide* : le coût de démarrage (ici, charger PyTorch) peut dépasser le gain.
- *Une accélération de l'entraînement n'est pas neutre* : changer `n_envs` change les données collectées, donc le modèle. C'est une décision scientifique, pas seulement technique.
- *Mesurer les durées sans rien d'autre en parallèle* : un pytest concurrent suffisait à fausser la mesure de 10 %.

**Commit :** `90ff1ea` (16 octobre 2026)

**Décision à prendre avec la directrice :** passer PPO à 4 environnements (`n_envs: 4` dans `configs/ppo_hyperparameters.yaml`) ferait gagner environ 30 % du temps d'entraînement, mais **changerait les modèles** (4 × 1 024 pas par mise à jour au lieu de 1 024), ce qui devrait être annoncé dans l'article. Laissé à 1 (protocole v1) en attendant. L'évaluation parallèle (`--workers 4`), elle, est utilisable tout de suite sans rien changer aux résultats.

**Mise à jour du 16 octobre 2026 :** **1 environnement gardé** (protocole v1), à présenter à la directrice : un gain d'environ 30 % ne justifie pas un changement de protocole dont la qualité n'est pas vérifiée.

**Questions « Savoir expliquer » :**
- [x] pourquoi changer le nombre d'environnements de PPO change le modèle obtenu, alors que paralléliser l'évaluation ne change rien : pendant l'entraînement, PPO apprend des épisodes qu'il collecte. Avec 4 environnements, il collecte d'autres épisodes, dans un autre ordre, et fait une mise à jour tous les 4 × 1 024 pas au lieu de 1 024 (`n_steps × n_envs`). Il suit donc un autre chemin d'apprentissage et aboutit à un autre réseau : le test le montre avec 1 et 2 environnements. Pendant l'évaluation, au contraire, le modèle est **figé**, et chaque épisode ne dépend que de sa politique et de sa graine (flux aléatoires propres depuis T13, politiques sans mémoire d'un épisode à l'autre). Qu'un épisode soit calculé par tel ou tel processus, avant ou après un autre, ne change rien à son résultat : le test du plan le vérifie graine par graine.

## T30 — Agrégation sur plusieurs graines et courbes d'apprentissage
*Recommandé · [MESURE] · M7, N7, Dir. plan*

**Date :** 16 octobre 2026 (méthode 1 : code et tests écrits par l'IA, relus par moi)

**Prédiction (avant de modifier) :**
- aucun modèle ne change : on ajoute des colonnes aux journaux d'entraînement et on change la façon de résumer les résultats, pas l'apprentissage ;
- les tests appariés de `statistical_tests`, qui refusaient plusieurs graines d'entraînement depuis T26, fonctionnent sur une campagne à 5 graines ;
- les figures de pertes, d'erreur TD et d'exploration n'ont plus besoin de réentraînements séparés, puisque les entraînements de la campagne écrivent déjà ces journaux ;
- la bibliothèque `rliable` s'installe et s'utilise telle quelle.

**Ce que j'ai changé :**
- `analysis/aggregate_seeds.py` (nouveau), pour résumer les runs (un agent × une graine d'entraînement) :
  - le score d'un run est sa moyenne sur les 30 graines d'évaluation ; les heuristiques, sans graine d'entraînement, sont exclues ;
  - `aggregate_table` : par régime, **IQM** des 5 runs, son **IC 95 % par bootstrap stratifié**, moyenne, médiane, plus mauvais et meilleur run ;
  - `aggregate_across_scenarios` : IQM sur les 3 régimes à la fois. Les scores sont divisés par la moyenne de l'Affectation fixe du même régime, comme le score normalisé par l'humain sur Atari. Le bootstrap tire les runs **à l'intérieur de chaque régime** ;
  - `improvement_table` : **probabilité d'amélioration** P(X meilleur que Y), par régime et sur l'ensemble, pour PPO > DQN, PPO > Q-Learning, DQN > Q-Learning et Q-Learning > SARSA. « Meilleur » tient compte du sens de la métrique (moins de carburant, c'est mieux) ;
  - un run évalué sur d'autres graines que les autres, ou un nombre de runs différent selon le régime, est refusé ;
  - commande : `python -m memoire_master_rl_logistique.analysis.aggregate_seeds RESULTATS.csv`.
- `analysis/statistical_tests.py` : `average_over_training_seeds`. Pour chaque graine d'évaluation, un agent est remplacé par la moyenne de ses 5 modèles ; il faut toutes les graines d'entraînement sur chaque épisode. `main` l'applique, et `paired_table` renvoie vers cette fonction au lieu de « prévu en T30 ».
- `analysis/learning_curves.py` (nouveau) :
  - chaque run est résumé sur **100 intervalles égaux de son budget** (épisodes ou pas d'entraînement), puis on prend la moyenne des runs et son IC bootstrap, point par point ;
  - 8 courbes : productivité, récompense, erreur TD, ε, perte de DQN, pertes de la politique et de la valeur de PPO, entropie de PPO ;
  - le budget et la durée du poste sont lus dans le `metadata.json` de chaque run, et des budgets différents sont refusés ;
  - un journal d'avant T30 donne un message clair ;
  - export CSV : `python -m memoire_master_rl_logistique.analysis.learning_curves DOSSIER_MODELES SORTIE.csv`.
- `rl/callbacks.py` : `training_kpis.csv` (DQN, PPO) gagne `timesteps`, le pas atteint à la fin de l'épisode, et `reward`, la récompense de l'épisode.
- `rl/tabular_training.py`, `train_q_learning.py`, `train_sarsa.py` : `training_diagnostics.csv` gagne `tonnage_t`, pour tracer la productivité des 4 agents.
- `analysis/sample_efficiency.py` : le point de convergence (premier moment où la courbe reste à ± 5 % de sa valeur finale) est calculé **pour chaque graine** ; on rapporte la moyenne, l'IC, la plus rapide et la plus lente.
- 5 figures réécrites dans `plots/` : `convergence_deeprl`, `convergence_qlearning_sarsa`, `policy_value_loss`, `td_error`, `exploration_exploitation`.
  - Elles tracent la moyenne ± IC sur les graines, lue dans les dossiers des modèles **évalués**.
  - Le dossier de sortie est obligatoire, et aucun fichier existant n'est écrasé : avant, elles écrivaient directement dans `reports/figures`, sans protection.
  - Pour ε, la courbe tabulaire est lue dans le journal, et non plus recalculée avec une formule recopiée.
  - Outils communs dans `plots/common.py` : `plot_band`, couleurs des agents et arguments de la commande.
- **N7, suppression** de `rl/diagnostics/` : `train_diagnostics.py`, `train_diagnostics_nominal.py`, `train_losscurve.py` et l'export Power BI `export_convergence_qlearning_sarsa.py`.
- `Makefile` : les 9 cibles `train-*` et `plot-diagnostics` sont remplacées par une seule cible, `training-figures` (`RESULTS ?= data/results_v2`, `FIGURES ?= reports/figures_v2`).
- Tests existants : `test_behavior.py` (docstring du test historique corrigée), `test_observation.py` (`progress_bar=False`) et `test_transfer.py` (l'avertissement `allow_legacy` des Q-tables v1 est maintenant vérifié). La suite passe sans aucun avertissement.

**Pourquoi :**
- **M7** : depuis T26, chaque agent a 5 modèles. Il fallait un résumé qui montre la variabilité d'entraînement : une seule graine ne dit rien sur l'algorithme (voir T26, Savoir expliquer).
- **Les tests appariés étaient bloqués** : `statistical_tests.main` ne pouvait plus analyser une campagne à plusieurs graines.
- **N7** : les figures de pertes, d'erreur TD et d'exploration de l'article v1 venaient de réentraînements séparés (`*_losscurve`, `*_diag`), **pas des modèles évalués**. Un lecteur pouvait croire qu'elles décrivaient l'apprentissage des agents du tableau de résultats, ce qui n'était pas le cas.
- **Les courbes d'une seule graine** ne permettent pas de distinguer une tendance d'un hasard d'entraînement.

**Ce que le test vérifie :**
- `tests/test_aggregate_seeds.py`, 18 tests :
  - **définitions de rliable** : l'IQM de [1, 2, 3, 4, 100] vaut 3 (moyenne : 22) ; l'IQM est la moyenne tronquée à 25 % ; P(X > Y) est le U de Mann-Whitney divisé par n_x × n_y ;
  - **valeurs de référence calculées avec rliable lui-même**, installé à part : IQM, son IC, P(X > Y) et son IC ;
  - le bootstrap stratifié ne mélange jamais deux régimes ; l'IC est reproductible ; un seul run ne donne pas d'IC ;
  - score d'un run = sa moyenne sur l'évaluation ; un run évalué sur d'autres graines est refusé ; normalisation par l'Affectation fixe ; un nombre de runs différent selon le régime est refusé ;
  - P(X > Y) dans le bon sens pour la productivité et pour le carburant ;
  - `average_over_training_seeds` : bonnes moyennes, graine d'entraînement manquante refusée, CSV d'avant T26 inchangé ;
  - **`statistical_tests.main` fonctionne sur un CSV à 5 graines**, alors qu'il échouait avant.
- `tests/test_learning_curves.py`, 23 tests, sur une **vraie mini-campagne** (4 agents × 2 graines, dans les dossiers `model_dir` de `run_benchmark`) :
  - nouvelles colonnes des journaux ; pas croissants et inférieurs au budget ;
  - les courbes sont lues dans **les mêmes dossiers que ceux évalués** (N7) ;
  - la moyenne par intervalle est vérifiée à la main (bornes et intervalles vides compris), ainsi que la productivité (tonnage ÷ durée du poste) ;
  - les 8 courbes pour chacun de leurs algorithmes ; ε part bien de 1,0 et décroît ; l'entropie est positive ;
  - erreurs claires pour une courbe inexistante, un run manquant, des budgets différents ou un journal d'avant T30 ;
  - point de convergence calculé à la main, puis par graine ;
  - les 5 figures sont créées et refusent d'écraser ; un régime absent n'écrit aucune figure ; un intervalle vide ne coupe pas le trait ;
  - l'export CSV refuse d'écraser **avant** de calculer ;
  - `rl.diagnostics` n'existe plus, et le `Makefile` n'y fait plus référence.

Vérifications hors tests :
- **Modèles identiques à T29** : Q-Learning, SARSA, DQN et PPO entraînés sur le code de T29 (copie de travail de `90ff1ea`) puis sur celui de T30 : **173 tableaux identiques** (Q-tables et poids). Une première comparaison avait importé deux fois le même code ; je l'ai écartée et refaite en forçant le chemin d'import, et l'en-tête des journaux confirme que chaque exécution a utilisé son propre code.
- **Figures regardées** sur une mini-campagne de 3 graines.

Suite complète : **307 passed, 1 skipped, 0 avertissement**. Non-régression verte. Ruff ne signale rien sur le code ; sa configuration ne vérifie pas `tests/`.

**Résultat comparé à ma prédiction :**
- **Confirmée** : aucun modèle ne change, les tests appariés fonctionnent sur 5 graines, et les figures se tracent à partir des modèles évalués.
- **Infirmée sur `rliable`.** Elle ne fonctionne dans aucune configuration de mon environnement :
  - avec `arch` 7, l'import plante à cause de pandas 3 ;
  - avec `arch` 8, le bootstrap plante parce que `arch` a renommé un argument (`random_state` est devenu `seed`).

  J'ai donc recodé ses trois définitions (une trentaine de lignes) et vérifié contre la vraie bibliothèque, installée à part sans toucher à mon environnement. Les valeurs ponctuelles sont identiques, l'IC de l'IQM aussi à 3 décimales. L'IC de P(X > Y) différait au départ : rliable applique **les mêmes indices de runs à X et à Y**, alors que je les tirais séparément. Je me suis aligné sur rliable.

Deux défauts trouvés en vérifiant :
- la commande d'export calculait tout avant de refuser d'écrire : la vérification se fait maintenant au départ ;
- sur les vraies figures, un intervalle sans valeur coupait le trait. L'entropie de PPO (une ligne de journal tous les 1 024 pas) était même invisible, et ε de SARSA cachait celui de Q-Learning. Le trait relie maintenant les points, et SARSA est en tirets.

Limites :
- avec 5 graines, l'IC bootstrap est un peu trop étroit : les quantiles portent sur peu de valeurs différentes. Agarwal et al. le reconnaissent aussi ;
- les p-values des tests appariés portent sur la performance **moyenne** d'un algorithme (5 modèles moyennés) ; elles ne tiennent pas compte de la variabilité d'entraînement, que rapportent l'IQM et son IC ;
- les modèles v1 n'ont pas les nouvelles colonnes : les courbes ne se tracent qu'avec la campagne T31 ;
- je me suis écarté du plan sur deux points : le calcul est dans `analysis/` et non dans `experiments/` (convention de la réorganisation), et `rliable` est recodé au lieu d'être importé.

**Ce que j'ai appris :**
- *Une figure doit venir du modèle qu'elle décrit* : des pertes tracées sur un réentraînement séparé décrivent un autre modèle, même avec les mêmes réglages.
- *Une grille commune avant de moyenner des courbes* : les épisodes n'ont pas la même longueur d'une graine à l'autre ; on moyenne donc par intervalle du budget, pas par numéro d'épisode.
- *Recoder une bibliothèque se vérifie contre elle* : sans la comparaison, je n'aurais pas vu que rliable tire les mêmes indices pour X et Y.
- *Vérifier ce qu'une comparaison compare vraiment* : ma première comparaison avec T29 avait importé deux fois le même code ; un détail (l'en-tête d'un CSV) l'a révélé.
- *Regarder les figures, pas seulement les tester* : un test vérifiait que le fichier existait, mais pas que la courbe était visible.
- *Un avertissement attendu doit être vérifié* : avec `pytest.warns`, il devient une garantie au lieu d'un bruit.

**Commit :** `c064b3d` (16 octobre 2026)

**Questions « Savoir expliquer » :**
- [x] ce qu'est la moyenne interquartile et pourquoi Agarwal et al. (2021) la recommandent : la moyenne interquartile (IQM) est la moyenne des 50 % de scores centraux. On retire le quart le plus bas et le quart le plus haut ; avec 5 runs, c'est la moyenne des 3 du milieu. Agarwal et al. (2021, *Deep Reinforcement Learning at the Edge of the Statistical Precipice*) partent d'un constat : en RL, on n'a que quelques runs, souvent 3 à 10, parce que chaque entraînement coûte cher. Avec si peu de runs :
  - la **moyenne** est tirée par un seul run aberrant, par exemple une graine qui diverge : [1, 2, 3, 4, 100] donne 22 ;
  - la **médiane** n'utilise qu'une ou deux valeurs, donc elle varie beaucoup d'une série de runs à l'autre ;
  - l'**IQM** est entre les deux : robuste aux runs aberrants (elle donne 3), mais elle utilise la moitié des scores, donc elle est bien plus stable que la médiane.

  Ils recommandent de l'accompagner d'un **IC par bootstrap stratifié**, qui mesure ce que d'autres graines auraient donné, et de la **probabilité d'amélioration** P(X > Y), qui dit si un algorithme bat l'autre de façon régulière et pas seulement en moyenne. C'est la réponse à la question de T26 : « PPO est-il meilleur que DQN ? ».

---

## Mise au point des agents avant la campagne (16 et 17 octobre 2026)
*Hors plan · protocole écrit le 17 octobre 2026 à 5 h 54, **avant** de connaître le résultat de la piste 2*

**Pourquoi cette section :** après les corrections du simulateur (T15, T16 bis), le réglage de T25 bis a montré des agents moins bons qu'une politique au hasard. J'ai alors fait plusieurs essais pour comprendre et améliorer l'apprentissage. Multiplier les essais jusqu'à ce que le RL batte les heuristiques serait un biais : chaque essai est une chance de tomber par hasard sur une version gagnante. Cette section fixe la règle et note **tous** les essais, y compris ceux qui ont échoué, pour qu'ils puissent être déclarés dans l'article.

**Conditions communes à tous les essais :** régime nominal, graine d'entraînement 1, critère = productivité moyenne de la politique sans exploration sur les **graines de sélection 20 000 à 20 009**. Les graines d'évaluation de l'article (42 à 71) n'ont **jamais** servi. Références sur ces mêmes graines : MTWT 4 443 t/h, programme linéaire 4 412, FIFO 4 246, Affectation fixe 4 226, politique au hasard 4 167, Pelle la plus proche 3 915.

**Tous les essais, dans l'ordre :**

| # | Essai | Budget | PPO | DQN | Décision |
|---|---|---|---|---|---|
| 0 | Réglage T25 bis (récompense versée à la fin du cycle, observation de 17 composantes), meilleure configuration | 200 000 pas | 3 953 t/h (99 % à la pelle 1) | 3 992 t/h | point de départ |
| 1 | Récompense estimée et versée à la décision (option 1b) | 200 000 pas | 3 915 | 4 191 (ATTENDRE 52 %) | rejeté (voir essai 4) |
| 2 | 1b + coût d'opportunité des minutes en file | 200 000 pas | 3 981 (ATTENDRE 21 %) | 4 106 (ATTENDRE 65 %) | rejeté |
| 3 | Budget complet, récompense commitée (1a) | 2 000 000 pas | **4 260** | — | 1a gardée |
| 4 | Budget complet, récompense à la décision (1b) | 2 000 000 pas | 3 943 | — | 1b rejetée ; code d'essai retiré |
| 5 | **Piste 1** : attente prévue par route dans l'observation (23 composantes) | 2 000 000 pas | **4 345** | — | **retenue** |
| 6 | **Piste 2** : piste 1 + récompense normalisée (VecNormalize) + γ = 0,995 | 2 000 000 pas | **4 405** (ATTENDRE 36 %) | — | **retenue** (> piste 1) |
| 7 | **Contrôle de DQN** (points 11 et 12) : piste 1 + γ = 0,995, sans normalisation, exploration 0,1, apprentissage 0,0003 | 2 000 000 pas | — | **3 593** (99 % à la pelle 3) | version de campagne (point 12) |
| 8 | **Chronométrage de Q-Learning** (point 7), configuration actuelle (α = 0,2, 8 intervalles) | 30 000 épisodes, 163,6 min | — | Q-Learning : **3 955** (62 % à la pelle 3) | point de départ du réglage (point 8) |
| 9 | **Chronométrage de SARSA** (point 7), configuration actuelle (α = 0,2, 8 intervalles) | 30 000 épisodes, 159,2 min | — | SARSA : **3 535** (98 % à la pelle 3) | point de départ du réglage (point 8) |

Les essais 1, 2 et 4 ont été faits sur une copie non commitée du simulateur ; le patch est conservé hors du dépôt. Les essais 3 à 6 utilisent la récompense commitée.

**Règle fixée avant le résultat de la piste 2 :**
1. **Les essais s'arrêtent après la piste 2, quel que soit son résultat.** Pas de piste 3, même si le RL reste sous MTWT.
2. **Choix de la version** : la piste 2 n'est retenue que si elle fait mieux que la piste 1 sur les graines de sélection ; sinon on garde la piste 1 seule.
3. **Équité entre les agents** : tout ce qui est retenu pour PPO s'applique aussi à DQN quand cela a un sens (la piste 1 s'applique déjà à tous, puisque l'observation est commune ; si la piste 2 est retenue, DQN reçoit le même γ, et la normalisation de la récompense seulement si elle s'applique de la même façon). Sinon, la piste 2 n'est pas retenue.
4. **Évaluation finale une seule fois** : les graines 42 à 71 ne servent qu'à la campagne T31, et rien n'est modifié après avoir vu leurs résultats.
5. **Tout est déclaré dans l'article** : un paragraphe « mise au point des agents » avec ce tableau (essais, résultats sur les graines de sélection, ce qui a été retenu).
6. **Le résultat est présenté tel qu'il sortira**, même si les heuristiques MTWT et programme linéaire restent devant le RL.

**Ajout du 17 octobre 2026 à 6 h 00, avant le résultat de la piste 2 et avant tout essai des agents tabulaires — même protocole pour Q-Learning et SARSA :**

*Constat :* le réglage de T25 bis ne portait que sur PPO et DQN, la piste 1 ne profite pas aux agents tabulaires (leur état discret, 5 attentes et la pelle la plus proche, ne lit pas le bloc d'attente prévue par route), et ils n'ont pas encore été entraînés sur le simulateur de T15 et T16 bis. Les laisser dans leur état d'origine rendrait la comparaison entre agents inéquitable.

7. **Chronométrer d'abord** un entraînement complet de Q-Learning et de SARSA (30 000 épisodes, jamais mesuré), sans rien lancer de lourd pendant l'essai de la piste 2.
8. **Réglage de même taille que pour PPO et DQN** : 4 configurations par algorithme, α ∈ {0,1 ; 0,2} × nombre d'intervalles de discrétisation ∈ {6 ; 8}, qui incluent les valeurs actuelles (α = 0,2, 8 intervalles). Même graine d'entraînement (1), même critère (productivité, nominal, graines de sélection 20 000 à 20 009), budget complet (30 000 épisodes) si le chronométrage le permet, sinon un budget réduit annoncé, le même pour les deux algorithmes.
9. **Équivalent de la piste 1 pour l'état discret** : la variable « pelle la plus proche » est remplacée par « route de plus petite attente prévue » (indice de 0 à 5, calculé comme MTWT). Ajouter les 6 attentes prévues elles-mêmes est exclu : la table exploserait (8⁶ combinaisons de plus). Cette variante est essayée **une seule fois**, avec la meilleure configuration du réglage, et retenue seulement si elle fait mieux sur les graines de sélection.
10. **Même règle d'arrêt et de déclaration** : pas d'autre variante après celle-ci ; les essais, réussis ou non, vont dans le tableau ci-dessus et dans l'article ; les graines 42 à 71 ne servent qu'à la campagne T31.

**Ajout du 17 octobre 2026 à 6 h 04, avant le résultat de la piste 2 et avant tout essai de DQN au budget complet — contrôle de DQN :**

*Constat :* DQN a eu le réglage de T25 bis (200 000 pas ; meilleure configuration : exploration 0,1, apprentissage 0,0003, 3 992 t/h), et la piste 1 s'applique à lui puisque l'observation est commune. Mais les trois essais au budget complet (essais 3, 5 et 6) n'ont porté que sur PPO : DQN n'a jamais été entraîné avec la piste 1 ni au budget complet. L'effort de mise au point serait inégal entre les deux agents profonds.

11. **Un essai de contrôle pour DQN au budget complet** (2 000 000 pas), avec sa meilleure configuration du réglage et tout ce qui a été retenu pour PPO : la piste 1 ; le γ de la piste 2 si elle est retenue ; la normalisation de la récompense **seulement si elle s'applique de la même façon**. Pour DQN, elle est moins standard : sa mémoire de rejeu garde des récompenses normalisées avec des statistiques qui changent au fil de l'entraînement. Si elle n'est pas appliquée à DQN, c'est déclaré dans l'article. Même graine (1), même critère (graines de sélection 20 000 à 20 009).
12. **Cet essai vérifie et se déclare ; il ne prolonge pas la mise au point** : quel que soit son résultat, DQN part en campagne avec cette version, sans autre variante.

**Ce qui rend cette mise au point défendable :** elle se fait sur des graines séparées de l'évaluation ; chaque changement retenu se justifie indépendamment du résultat (la piste 1 donne à l'agent la même information que MTWT ; la piste 2 applique des pratiques standard de PPO) ; et les références ont été renforcées en même temps (prévision des camions en route pour FIFO et MTWT, programme linéaire), ce qui a rendu la comparaison plus difficile pour le RL.

**Résultat de la piste 2 et décision (17 octobre 2026, 9 h 37) :**
- Une première tentative a été arrêtée à 59 % (7 h 39) par Claude Code, faute de mémoire sur la machine ; sans modèle sauvegardé, elle n'a pas pu être évaluée. Ses journaux sont gardés dans `data/results_v2/piste2_interrompu_59pct`. L'essai a été relancé à l'identique à 7 h 40.
- **Résultat : 4 404,8 ± 50,2 t/h** sur les graines de sélection, contre **4 345,2** pour la piste 1 (+60 t/h). Cycles : 62 % pelle 1, 33 % pelle 2, 5 % pelle 3. Productivité pendant l'entraînement, par dixième du budget : 4 324, 4 390, 4 405, 4 391, 4 380, 4 409, 4 408, 4 406, 4 392, 4 411 t/h (la piste 2 atteint son plateau dès 10 à 20 % du budget).
- **Décision, par application de la règle (point 2) : la piste 2 est retenue**, puisqu'elle fait mieux que la piste 1. PPO reste sous MTWT (4 443, −38 t/h) et à égalité avec le programme linéaire (4 412).
- **Point à surveiller, sans changer la décision :** cet agent choisit ATTENDRE dans **36 %** de ses décisions (1,7 % pour la piste 1), malgré le coût d'opportunité. La règle ne prévoit pas d'autre variante : ce comportement sera mesuré et décrit dans l'analyse comportementale de la campagne.
- **Fin de la mise au point de PPO.** Suite prévue par la règle : contrôle de DQN au budget complet (points 11 et 12, avec γ = 0,995 ; normalisation selon le point 11), puis réglage et variante de la piste 1 pour Q-Learning et SARSA (points 7 à 10).

**Résultat du contrôle de DQN (essai 7, 19 octobre 2026, 0 h 40) :**
- Une première tentative a été arrêtée à 97 % le 18 octobre par Claude Code, faute de mémoire ; sans modèle sauvegardé, elle n'a pas pu être évaluée (journaux gardés dans `data/results_v2/agent_check_interrompu`). Relancé à l'identique le 18 octobre à 22 h 40, en même temps que le chronométrage de Q-Learning et de SARSA.
- Entraînement : **120,5 min** pour 2 000 000 pas.
- **Résultat : 3 592,8 ± 59,5 t/h** sur les graines de sélection, ATTENDRE 0 %, cycles : pelle 1 0 %, pelle 2 1 %, **pelle 3 99 %**. C'est moins que toutes les heuristiques (Pelle la plus proche 3 915) et moins que DQN au réglage à 200 000 pas (3 992).
- Productivité pendant l'entraînement, par dixième du budget : 4 076, 3 896, 3 901, 3 909, 3 913, 3 932, 3 917, 3 921, 3 911, 3 905 t/h. Le premier dixième contient la phase d'exploration (ε passe de 1 à 0,02 sur les 200 000 premiers pas), donc des actions presque au hasard ; ensuite l'agent se fixe sur une politique presque constante et ne progresse plus.
- **Ma lecture :** DQN s'est effondré sur une seule route. La piste 1 et γ = 0,995, qui ont aidé PPO, n'ont pas suffi à DQN sans normalisation de la récompense. Analyse ci-dessous.

- **Décision, par application de la règle (point 12) : DQN part en campagne avec cette version, sans autre variante.** Ce résultat est déclaré dans l'article (tableau ci-dessus) et l'effondrement sera décrit dans l'analyse comportementale, avec ses hypothèses présentées comme telles. *(Valable si la récompense est gardée ; voir la décision en attente à la fin de cette section.)*

**Analyse de l'échec de DQN (19 octobre 2026), sans rien modifier ni réentraîner.** Deux mesures en lecture seule sur les graines de sélection (scripts `dqn_qvalues.py` et `reward_per_decision.py`, hors du dépôt, à y ranger avec les autres scripts d'essai).

*Mesuré :*
- **Valeurs Q du modèle entraîné** (graine 20 000, 215 décisions) : environ 174 pour toutes les actions. Écart moyen entre la meilleure et la deuxième : **0,05** ; entre la meilleure et la pire : 0,56. Erreur d'apprentissage (`train/loss`) entre 0,14 et 0,23 pendant tout l'entraînement, donc plus grande que l'écart entre les deux meilleures actions. L'action 4 (pelle 3 → verse 1) est choisie 214 fois sur 215.
- **Récompense par décision de plusieurs politiques** (5 graines de sélection) :

| Politique | t/h | Décisions | Récompense par décision |
|---|---|---|---|
| MTWT | 4 448 | 266 | 0,871 |
| Programme linéaire | 4 410 | 264 | 0,863 |
| Affectation fixe | 4 221 | 253 | 0,862 |
| Pelle la plus proche (= toujours action 0) | 3 913 | 235 | 0,820 |
| Toujours action 2 (pelle 2 → verse 1) | 3 790 | 228 | 0,822 |
| Toujours action 4 (pelle 3 → verse 1, ce que fait DQN) | 3 566 | 216 | 0,813 |

  La récompense par décision **classe les politiques dans le bon ordre** (MTWT en tête, la politique de DQN en dernier), mais l'écart est faible : environ 6 % entre la meilleure et la pire. La longueur de la route y apparaît très peu : entre toujours la pelle 1 et toujours la pelle 3, −350 t/h mais seulement −0,007 par décision. L'écart vient surtout des pénalités d'attente.
- **Vérifié dans le code :** Stable-Baselines3 continue d'estimer la valeur au-delà de la coupure en fin de poste, pour DQN (`handle_timeout_termination`, mémoire de rejeu) comme pour PPO (`collect_rollouts`). γ s'applique par décision, et chaque cycle rapporte 1 quelle que soit sa durée (`simulation/reward.py`). Avec ε = 0,02 dès 200 000 pas, la mémoire de rejeu (200 000 transitions, environ 840 postes) ne contient ensuite presque que des transitions de la politique figée.

*Interprétation (hypothèses, non prouvées par un réentraînement) :*
- La valeur d'un état est grande (≈ 0,82 / (1 − 0,995) ≈ 164) et l'écart utile entre actions est très petit en comparaison. Sans récompense normalisée, l'erreur du réseau cache cet écart. L'exploration courte a ensuite figé le choix : les autres actions ne sont presque plus essayées, donc plus corrigées.
- ~~Le choix de la pelle 3 vient du bruit.~~ **Corrigé le 19 octobre à 1 h 30** : Q-Learning et SARSA, avec un autre algorithme et un état discret, dérivent eux aussi vers la pelle 3 (essais 8 et 9). Ce n'est donc pas du bruit propre à DQN, mais un biais de la récompense par décision (voir « Résultat des agents tabulaires » ci-dessous).
- **DQN n'a pas su optimiser son propre objectif** : selon sa récompense, il a trouvé presque le plus mauvais choix. Contrairement à ce que j'avais d'abord écrit, l'objectif n'est pas aveugle à la productivité ; il la reflète faiblement.
- PPO s'en sort sans doute parce qu'il apprend à partir des différences entre actions (avantages) et que sa récompense est normalisée (piste 2) ; c'est un raisonnement, pas une mesure.
- Le faible poids de la durée du cycle dans la récompense par décision touche aussi PPO, et peut expliquer en partie pourquoi il reste sous MTWT.

*Décision :* rien n'est changé (point 12). DQN part en campagne avec cette version. Son échec est déclaré dans l'article, avec ces mesures, en séparant ce qui est mesuré de ce qui est supposé.

*Perspectives à écrire dans l'article :* une actualisation en temps réel (γ par minute, formulation semi-markovienne), qui ferait compter la durée des cycles ; une récompense par unité de temps ; la normalisation de la récompense ou une architecture *dueling* pour DQN. Ne pas les essayer maintenant : ce serait modifier le problème ou l'agent après avoir vu le résultat.

**Résultat des agents tabulaires (essais 8 et 9, 19 octobre 2026, 1 h 22 et 1 h 26) :**
- Lancés le 18 octobre à 22 h 40, en parallèle avec DQN (une première tentative avait été arrêtée par manque de mémoire ; journaux dans `data/results_v2/agent_check_interrompu`).
- **Durée (point 7) :** 163,6 min pour Q-Learning, 159,2 min pour SARSA, pour 30 000 épisodes, avec trois entraînements en même temps sur la machine.
- **Q-Learning : 3 955,0 ± 36,9 t/h**, ATTENDRE 1,9 %, cycles : pelle 1 1 %, pelle 2 36 %, pelle 3 62 %. Courbe par dixième : 4 162, 4 149, 4 130, 4 109, 4 082, 4 060, 4 038, 4 011, 3 992, 3 949.
- **SARSA : 3 535,0 ± 42,9 t/h**, ATTENDRE 0 %, cycles : pelle 1 1 %, pelle 2 1 %, pelle 3 98 %. Courbe par dixième : 4 163, 4 153, 4 136, 4 109, 4 077, 4 034, 3 968, 3 878, 3 748, 3 626.
- **Constat :** pour les deux, la productivité **baisse à mesure que l'exploration diminue** : la politique apprise est moins bonne que des actions au hasard (4 167 t/h). Avec DQN, trois agents à valeurs sur trois dérivent vers la pelle 3, la plus éloignée.

**Mesure complémentaire (lecture seule, script `reward_terms_by_policy.py`) — termes de la récompense par décision, 5 graines de sélection :**

| Politique | t/h | production | équilibre | distance | attente |
|---|---|---|---|---|---|
| MTWT | 4 448 | +0,956 | −0,009 | −0,072 | −0,005 |
| Toujours pelle 1 → verse 1 | 3 913 | +0,950 | −0,029 | −0,068 | −0,033 |
| Toujours pelle 3 → verse 1 | 3 566 | +0,945 | −0,030 | −0,079 | −0,023 |

*Ce que cela montre (mesuré) :* aller toujours à la pelle 3 plutôt qu'à la pelle 1 fait perdre 9 % de productivité, mais la récompense par décision ne baisse presque pas : −0,011 de distance, compensé par +0,010 d'attente en moins.

*Interprétation (hypothèse, cohérente avec les trois agents) :* la récompense paie **un cycle**, quelle que soit sa durée, et γ s'applique **par décision**. Une minute en file coûte (terme d'attente, 0,3 / 30 = 0,01 par minute), mais une minute de trajet ne coûte presque rien (seulement la distance, poids 0,05). Pour un agent qui maximise la récompense par décision, échanger de l'attente contre du trajet est donc avantageux, alors que cela fait perdre de la production par heure. Pendant l'exploration, la pelle 3, peu utilisée, a une file courte : ses valeurs apprises montent, et l'agent glisse vers elle. PPO y échappe en grande partie (5 % à la pelle 3 pour la piste 2), sans doute grâce à la normalisation et à son retour sur plusieurs pas ; c'est une supposition.

**Conséquence importante pour la suite :** le défaut n'est pas dans les agents, il est dans **la définition de la récompense** (héritée de la v1 : un cycle = 1, sans tenir compte du temps). Le réglage de α et du nombre d'intervalles (point 8) ne s'attaque pas à cette cause. Changer la récompense maintenant modifierait le problème pour tous les agents après avoir vu les résultats : c'est une décision à prendre **avec la directrice**, et à déclarer, pas à faire seul. Options à lui présenter :
1. **Garder la règle telle quelle** : réglage des tabulaires (point 8), variante de l'état (point 9), puis campagne ; déclarer et expliquer le biais de la récompense dans l'article, avec la récompense par unité de temps en perspective.
2. **Corriger la récompense** (par exemple production par minute, ou γ par minute), le déclarer comme une correction de la définition du problème trouvée pendant la mise au point, puis **refaire toute la mise au point** (PPO, DQN, tabulaires) sur les graines de sélection avec la nouvelle récompense, avant la campagne. Plus long, et à justifier comme une correction de défaut et non comme une recherche d'un meilleur score.

**Ma recommandation : option 2**, si la directrice l'accepte. Une récompense qui pousse les agents à faire baisser la productivité est un défaut de conception, pas un réglage. La règle (points 1 et 12) interdisait d'autres variantes **des agents** ; elle ne prévoyait pas un défaut de **la définition du problème**. Si l'option 2 est choisie, il faudra l'écrire comme un ajout à la règle, daté, avec la raison, avant de lancer le moindre essai avec la nouvelle récompense.

**Note à la directrice mise à jour (19 octobre 2026, vers 1 h 45)** : `docs/note_directrice_decisions.md`, nouvelle section 2 « Mise au point des agents » (règle, tableau des essais, cause mesurée) et section 4, point 1 (options A et B, ma recommandation). **Rien n'est lancé avant sa réponse.**

**État au 19 octobre 2026 :**
- Fait : réglage T25 bis ; pistes 1 et 2 pour PPO (retenues, commit `ddc99a9`) ; configuration de la campagne (commit `1fd04de`) ; contrôle de DQN (point 11) ; chronométrage de Q-Learning et de SARSA (point 7).
- Pas fait, en attente de la décision sur la récompense : réglage des agents tabulaires (point 8) et variante « route de plus petite attente prévue » (point 9).
- À faire avant la campagne, quelle que soit la décision :
  - ~~ranger dans le dépôt les scripts d'essai et de mesure, aujourd'hui dans le dossier temporaire de Claude Code (`agent_check.py`, `piste1.py`, `piste2.py`, `diag_full.py`, `tuning_diag.py`, `dqn_qvalues.py`, `reward_per_decision.py`, `reward_terms_by_policy.py`), pour que chaque essai déclaré puisse être refait~~ **fait le 20 octobre** (voir ci-dessous) ;
  - ~~sauvegarder les modèles en cours d'entraînement (deux essais perdus par manque de mémoire), avec un test~~ **fait le 20 octobre** ;
  - ~~supprimer le worktree `scratchpad/wt_tuning`~~ **fait le 20 octobre** ;
  - lancer la campagne dans une fenêtre PowerShell séparée, mise en veille désactivée, PyCharm fermé.

**Ce que j'ai appris de cette mise au point :**
- Un agent qui fait moins bien que le hasard n'est pas forcément mal réglé : il peut optimiser correctement une récompense qui ne mesure pas ce qu'on veut. Il faut vérifier que **la récompense classe les politiques comme l'objectif réel** (ici, les tonnes par heure) avant d'accuser l'algorithme.
- Une mesure simple (récompense par décision de politiques fixes) a suffi à trouver la cause, sans aucun réentraînement.
- Mon premier diagnostic de DQN (« du bruit ») était faux. C'est la comparaison avec deux autres algorithmes, arrivés au même choix, qui l'a montré : un résultat isolé ne suffit pas pour conclure sur une cause.

## Préparation de la campagne : sauvegardes, essais refaisables, nettoyage (20 octobre 2026)
*Hors plan · utile quelle que soit la décision sur la récompense*

**Date :** 20 octobre 2026

**Prédiction (avant de modifier) :**
- La sauvegarde ne doit rien changer à un entraînement qui n'est pas interrompu.
- La reprise peut être exacte pour Q-Learning et SARSA : chaque épisode part de sa propre graine, et tout le reste (Q-table, générateur aléatoire, ε) se sauvegarde.
- Pour PPO et DQN, je ne m'attendais pas à une reprise exacte : l'épisode en cours et l'état aléatoire de PyTorch sont difficiles à restaurer.
- Les mesures du 19 octobre, une fois rangées dans le code, doivent redonner exactement les valeurs notées dans ce journal.

**Ce que j'ai changé :**
1. **Worktree `scratchpad/wt_tuning` supprimé.** Il était propre, et le réglage écrivait ses résultats dans le dépôt principal (`data/results_v2/tuning`) : rien n'est perdu.
2. **Sauvegarde en cours d'entraînement et reprise** (`rl/checkpoints.py`). Les sauvegardes vont dans `<dossier du modèle>/checkpoint/`, et ce dossier est supprimé à la fin d'un entraînement complet.
   - **Q-Learning et SARSA :** une sauvegarde tous les 1 000 épisodes, contenant :
     - la Q-table ;
     - l'état du générateur aléatoire ;
     - ε ;
     - les journaux.

     La reprise est **exacte**. Le code commun est `TabularProgress` dans `rl/tabular_training.py`.
   - **PPO et DQN :** une sauvegarde tous les 100 000 pas, à la fin d'une collecte, contenant :
     - le modèle ;
     - la mémoire de rejeu (DQN) ;
     - les statistiques de normalisation de la récompense (PPO) ;
     - les journaux des épisodes terminés.

     La reprise est **proche mais pas identique** : l'épisode en cours est rejoué depuis sa graine, et l'état aléatoire de PyTorch n'est pas restauré. Chaque reprise est écrite dans `metadata.json` (`resumed_from_timesteps` ; `resumed_from_episodes` pour les tabulaires), pour être déclarée. Elle n'est possible qu'avec un seul environnement (`n_envs = 1`, valeur de la campagne).
   - **Garde-fous :**
     - fichiers écrits sous un nom temporaire puis renommés, `state.json` en dernier ;
     - reprise refusée si la configuration diffère (algorithme, graine, budget, hyperparamètres, environnement), en nommant le champ qui diffère ;
     - nouvel entraînement refusé s'il trouve la sauvegarde d'un entraînement interrompu, sauf `overwrite`.
   - **Utilisation :**
     - relancer la campagne reprend tout seul les entraînements interrompus (`run_benchmark._train`) ;
     - option `--resume` dans `main.py`.
3. **Essais de la mise au point refaisables :**
   - `experiments/agent_trials.py` refait les essais 5 à 9 sur le code actuel (`python -m memoire_master_rl_logistique.experiments.agent_trials 6 DOSSIER` ; `--list` affiche les 10 essais déclarés). Il écrit `trial_result.json` et les mêmes lignes DUREE, COURBE, RESULT que les scripts d'origine.
   - `archives/mise_au_point_agents/` garde les essais 0 à 4, qui demandent le commit `157f97b` : le patch d'essai de la récompense, les scripts d'origine (seule la docstring a été corrigée) et un README qui explique comment les refaire.
4. **Mesures de l'analyse de la récompense** dans `analysis/reward_diagnostics.py` (récompense par décision et par terme selon la politique, écarts des valeurs Q d'un modèle DQN). Elles remplacent `dqn_qvalues.py`, `reward_per_decision.py` et `reward_terms_by_policy.py`.
5. **`pyproject.toml` :** `testpaths = ["tests"]`. Sans cela, pytest prenait les scripts archivés nommés `*_test.py` pour des tests et s'arrêtait dès la collecte.

**Pourquoi :**
- Deux essais ont été perdus par manque de mémoire, à 59 % et à 97 %. La campagne compte environ 75 entraînements de 2 à 3 h chacun.
- La règle de la mise au point exige de déclarer tous les essais. Un essai déclaré qu'on ne peut pas refaire ne se vérifie pas, et ses scripts n'existaient que dans un dossier temporaire.

**Ce que le test vérifie :**
- *`tests/test_checkpoints.py`, 7 tests :*
  - PPO interrompu après 2 048 pas repart de sa sauvegarde, avec ses statistiques de normalisation, et finit son budget ; le journal des épisodes d'avant la coupure est conservé ;
  - DQN reprend avec sa mémoire de rejeu (800 transitions reprises, puis 400 nouvelles) ;
  - sauvegarder ne change pas un entraînement : mêmes poids avec ou sans sauvegarde ;
  - les garde-fous fonctionnent ;
  - **pour Q-Learning et SARSA, un entraînement interrompu puis repris donne exactement la même Q-table et les mêmes journaux** qu'un entraînement sans interruption.
- *`tests/test_agent_trials.py`, 8 tests :*
  - les essais 0 à 9 sont tous déclarés ;
  - les essais 6 à 9 n'écrasent aucun hyperparamètre de la campagne ;
  - l'essai 5 ne diffère de la campagne que par γ et la normalisation ;
  - un essai écrit son résultat et ne se réentraîne pas ;
  - **la mesure du 19 octobre redonne les valeurs du journal** (MTWT 4 448 t/h et 0,871 par décision ; toujours pelle 3 → verse 1 : 3 566 t/h et 0,813), et la somme des termes est égale à la récompense par décision.
- Suite complète : **365 réussis, 1 ignoré** (le test historique de la v1).

**Résultat comparé à ma prédiction :**
- Conforme : la sauvegarde ne change rien, la reprise tabulaire est exacte, et les mesures retrouvent les valeurs du journal.
- Imprévu : pytest collectait les scripts archivés dès que la recherche des tests n'était plus limitée à `tests/`.
- Non vérifié : je n'ai pas relancé les essais 0 à 9 au budget complet. Seul leur code est vérifié (le patch s'applique sur `157f97b` ; les essais 5 à 9 tournent sur un petit budget).

**Ce que j'ai appris :**
- Une reprise exacte n'est possible que si tout l'aléa vient de graines qu'on peut retrouver. C'est vrai pour les agents tabulaires, mais pas pour PyTorch en cours d'entraînement. Il faut donc dire honnêtement que la reprise n'est pas identique et l'écrire dans les métadonnées, plutôt que la présenter comme transparente.
- Une valeur notée à la main dans le journal peut devenir un test : si le code change, le test le dira.

**Commit :** `2461c7c`

**Questions « Savoir expliquer » :**
- [ ] pourquoi la reprise de Q-Learning est exacte et pas celle de PPO :
- [ ] que se passe-t-il si on relance la campagne après une coupure de courant :
- [ ] pourquoi une reprise doit être déclarée dans l'article :

## Option B sur le papier : coût du temps pour tous les cycles (20 octobre 2026)
*Hors plan · mesure en lecture seule, aucun entraînement · à déclarer comme les essais*

**Date :** 20 octobre 2026

**Pourquoi :** préparer une proposition chiffrée pour la décision de la directrice sur la récompense (option A : la garder ; option B : la corriger).

**Proposition :** étendre à tous les cycles le coût d'opportunité d'ATTENDRE. Chaque cycle coûte sa durée dans le poste / 23,2 min, en plus des termes actuels. Aucun nouveau paramètre.

**Prédiction (avant la mesure) :**
- Le temps total des camions sur un poste devrait être fixe (12 × 480 min). Le coût ajouté serait alors le même pour toutes les politiques, et le classement des politiques sur un poste ne changerait pas.
- Par décision, la nouvelle récompense devrait classer les politiques comme la productivité.

**Mesure** (script `option_b.py` dans le dossier temporaire, à ranger dans `analysis/reward_diagnostics.py` si l'option B est retenue) : 12 politiques fixées (6 heuristiques + 6 actions constantes), graines de sélection 20 000 à 20 004, 3 régimes. La nouvelle récompense est calculée après coup à partir du journal des cycles, sans modifier le simulateur.

**Résultat :**
- **Minutes-camion par poste : exactement 5 760 (nominal, fortes pannes) et 8 640 (forte charge) pour les 12 politiques.** La prédiction est confirmée : le coût ajouté est une constante (248,3 en nominal), et l'objectif du poste ne change pas.
- Corrélation de Spearman entre la productivité et la récompense par décision :

| Régime | Récompense actuelle | Option B |
|---|---|---|
| Nominal | 0,965 | 1,000 |
| Fortes pannes | 0,972 | 1,000 |
| Forte charge | 0,654 | 0,767 |

- En forte charge, l'ordre reste imparfait, mais seulement entre 7 politiques mauvaises presque à égalité (3 980 à 4 040 t/h : la pelle 1 est saturée). Les 5 meilleures sont bien classées.
- Écart par décision en nominal :
  - entre MTWT et « toujours action 4 » (le choix de DQN) : 0,058 → 0,275 ;
  - entre toujours pelle 1 et toujours pelle 3 : 0,007 → 0,103.
- Tableaux complets dans `docs/option_b_recompense.md`.

**Ce que cela ne montre pas :** que les agents apprendront mieux. Seul un entraînement le dira.

**Suite :** la proposition est dans `docs/option_b_recompense.md` (correction, justification, code à changer, protocole en 5 points, phrase pour l'article). La section 4 de la note à la directrice a été mise à jour. **Rien n'est modifié ni lancé avant sa décision.**

**Ce que j'ai appris :** une correction de la récompense se défend mieux si on peut montrer qu'elle **ne change pas l'objectif** (ici, une constante par poste) et qu'elle change seulement **l'attribution du mérite** à chaque décision. La vérification (temps total fixe) tient en une colonne de tableau.

## Décision de la directrice : option B, et ajout à la règle (20 octobre 2026, 18 h 00)
*Écrit **avant** toute modification du code de la récompense et avant tout entraînement avec elle*

**Décision :** Dr. Ferchichi a retenu **l'option B** : à chaque cycle, coût d'opportunité de sa durée dans le poste / 23,2 min, comme pour ATTENDRE (`docs/option_b_recompense.md`).

**Ajout à la règle de la mise au point (points 13 à 18), qui remplace pour la suite les points 8 à 12 :**

13. **La mise au point n'est pas refaite, elle est rejouée.** Tout ce qui a été retenu est gardé tel quel :
    - piste 1 (observation de 23 composantes) ;
    - piste 2 pour PPO (γ = 0,995, récompense normalisée) ;
    - hyperparamètres du réglage T25 bis pour PPO et DQN ;
    - γ = 0,995 sans normalisation pour DQN ;
    - configuration actuelle pour Q-Learning et SARSA.

    Seule la récompense change.
14. **Quatre essais de contrôle au budget complet** avec la nouvelle récompense et la configuration de la campagne (essais 10 à 13 : PPO, DQN, Q-Learning, SARSA). Graine d'entraînement 1, critère = productivité sur les graines de sélection 20 000 à 20 009. Ils sont lancés **après le commit** du changement de récompense, pour que chaque modèle porte le commit qui l'a produit.
15. **Puis le réglage des agents tabulaires** déjà prévu, avec la nouvelle récompense et sur les mêmes graines :
    - point 8 : α ∈ {0,1 ; 0,2} × {6 ; 8} intervalles, pour Q-Learning et SARSA ;
    - point 9 : la variante « route de plus petite attente prévue », une seule fois.

    La configuration actuelle (α = 0,2, 8 intervalles) fait partie de la grille ; son essai de contrôle (point 14) compte comme l'une des 4 configurations.
16. **Aucune autre variante**, ni de la récompense ni des agents, quel que soit le résultat. En particulier, si PPO fait moins bien avec la nouvelle récompense qu'avec l'ancienne, on garde la nouvelle : elle a été choisie pour sa justification, pas pour son score, et elle s'applique à tous les agents.
17. **Les graines 42 à 71 ne servent qu'à la campagne T31**, une seule fois.
18. **Tout est déclaré** :
    - les essais 0 à 9 avec l'ancienne récompense ;
    - la raison du changement (mesures des 19 et 20 octobre) ;
    - les essais à partir de 10 ;
    - la phrase de la section 6 de `docs/option_b_recompense.md`.

**Prédiction (avant le changement) :**
- Les heuristiques gardent exactement les mêmes KPI. Leur récompense par poste baisse d'une constante : 5 760 / 23,2 = 248,3 en nominal.
- Les agents à valeurs devraient cesser de dériver vers la pelle 3. Je ne prédis pas leur niveau.
- PPO pourrait gagner un peu ou rester stable. C'est incertain, puisqu'il ne dérivait déjà presque pas.

## Option B dans le code (20 octobre 2026)

**Ce que j'ai changé :**
- `simulation/reward.py` : le terme `opportunity` vaut − minutes-camion de la décision dans le poste / 23,2. La durée prise en compte est celle de l'ATTENDRE, comme avant, ou désormais celle du cycle entier (trajet à vide, files, chargement, trajet chargé, déversement, panne). Le paramètre `idle_min` devient `truck_min`. La docstring explique le choix et pourquoi l'objectif d'un poste ne change pas.
- `env/mine_env.py` : à l'arrivée à la verse, `truck_min` = durée du cycle dans le poste, depuis la décision.
- `analysis/reward_diagnostics.py` :
  - nouvelles colonnes `reward_per_decision_before_b` (la récompense d'avant, recalculée à partir du journal des cycles) et `truck_min_per_shift` ;
  - nouvelle mesure `option_b_rankings` (tableau 2 de `docs/option_b_recompense.md`), qui remplace le script temporaire `option_b.py`.

  Les mesures des 19 et 20 octobre se refont donc toutes sur le code actuel.
- `experiments/agent_trials.py` :
  - les essais 5 à 9 portent le commit `2461c7c` (ancienne récompense), et `run_trial` refuse de les lancer sur le code actuel ;
  - ajout des essais de contrôle 10 à 13 (PPO, DQN, Q-Learning, SARSA, configuration de la campagne).
- Référence de non-régression régénérée (`tests/fixtures/`), avec la raison écrite dans l'historique du générateur.

**Résultat comparé à ma prédiction :** conforme.
- Avant de régénérer la référence, j'ai vérifié sur les 10 postes qu'elle contient : productivité identique, attente identique (écart < 3·10⁻¹⁴), récompense plus basse d'exactement 248,276 sur chaque poste.
- L'environnement redonne exactement les valeurs mesurées sur le papier (MTWT −16,874 et Affectation fixe −29,914 par poste en moyenne sur les graines de sélection 20 000 à 20 004).

**Ce que le test vérifie :**
- `test_reward.py` :
  - un cycle de 30 min coûte 30 / 23,2 ;
  - dans l'environnement, le coût d'un poste égale la somme des durées des cycles du journal, bornées à 480 min, soit 12 × 480 ;
  - pour les 6 heuristiques dans les 3 régimes, le coût d'opportunité est la même constante (camions × 480 / 23,2) ;
  - la décomposition d'un poste de l'Affectation fixe est mise à jour.
- `test_agent_trials.py` :
  - les mesures du journal avant et après l'option B (tableau 1) sont retrouvées ;
  - le tableau 2 (corrélations de Spearman 0,965 → 1,000, 0,654 → 0,767, 0,972 → 1,000) est retrouvé ;
  - les essais 0 à 13 sont tous déclarés ;
  - les contrôles 10 à 13 utilisent les mêmes agents que les essais 6 à 9 ;
  - l'essai 6 est refusé sur le code actuel.
- Suite complète : **368 réussis, 1 ignoré**.

**Commit :** `78ed5f5`

**Suite (règle, point 14) :** essais 10 à 13 lancés le 20 octobre vers 18 h 20, après le commit, dans 4 fenêtres PowerShell séparées (sortie `data/results_v2/option_b`, journaux `data/results_v2/option_b_essai_<n>.log`).

**Questions « Savoir expliquer » :**
- [ ] pourquoi la récompense d'un poste baisse de la même constante pour toutes les heuristiques :
- [ ] pourquoi cette constante ne change pas l'objectif, mais change ce que voit l'agent à chaque décision :
- [ ] pourquoi les essais 5 à 9 ne se refont plus sur le code actuel :

## Résultat des essais de contrôle de l'option B (essais 10 à 13, 21 octobre 2026)

Lancés le 20 octobre vers 18 h 20 sur le commit `78ed5f5`, sans interruption ni reprise. Graines de sélection 20 000 à 20 009, régime nominal.

| Essai | Agent | t/h | ATTENDRE | Pelles 1 / 2 / 3 | Durée | Avant l'option B (essais 6 à 9) |
|---|---|---|---|---|---|---|
| 10 | PPO | **210,0 ± 0,0** | **99,8 %** | 49 / 0 / 51 % | 406,7 min | 4 404,8 |
| 11 | DQN | 3 531,5 ± 42,7 | 94,8 % | 33 / 36 / 31 % | 374,8 min | 3 592,8 |
| 12 | Q-Learning | 4 098,5 ± 42,7 | 42,5 % | 35 / 33 / 33 % | 461,1 min | 3 955,0 |
| 13 | SARSA | 4 133,5 ± 55,2 | 37,2 % | 35 / 38 / 27 % | 457,8 min | 3 535,0 |

Courbes d'entraînement (t/h par dixième du budget) :
- PPO : 3 700, 220, 218, 191, 218, 212, 212, 212, 210, 210. Seulement 169 épisodes en 2 000 000 pas, soit environ 11 800 décisions par poste, presque toutes des ATTENDRE.
- DQN : 4 135, 3 795, 3 924, 3 837, 3 756, 3 829, 3 847, 3 896, 3 874, 3 764.
- Q-Learning : 4 170, 4 173, 4 173, 4 171, 4 167, 4 165, 4 159, 4 155, 4 148, 4 138.
- SARSA : 4 172, 4 176, 4 178, 4 176, 4 175, 4 170, 4 164, 4 157, 4 151, 4 130.

**Constat :**
- **La dérive vers la pelle 3 a disparu** pour les quatre agents : les parts par pelle sont équilibrées. Q-Learning et SARSA progressent de +143 et +598 t/h.
- **Mais PPO et DQN ont appris à ATTENDRE presque tout le temps.** PPO s'effondre de 4 405 à 210 t/h. Aucun agent n'atteint le hasard (4 167). Les courbes des agents tabulaires baissent toujours quand l'exploration diminue.

**Cause (mesurée en lecture seule, graines de sélection 20 000 à 20 004) :**

| Politique | t/h | Décisions par poste | Récompense par poste | Par décision |
|---|---|---|---|---|
| MTWT | 4 449 | 266 | −16,9 | −0,063 |
| MTWT une fois sur deux, ATTENDRE sinon | 4 298 | 524 | −28,0 | **−0,054** |
| Toujours ATTENDRE | 0 | 1 152 | −305,9 | −0,266 |

- Avec l'option B, un cycle rapporte **en moyenne une valeur négative** (1 − durée / 23,2 − les autres pénalités), alors qu'une ATTENDRE courte coûte presque rien.
- Le facteur d'actualisation s'applique **par décision**. Repousser une récompense négative de quelques décisions la rend donc moins coûteuse, et ATTENDRE est justement le moyen de la repousser. Intercaler des ATTENDRE améliore la récompense par décision (−0,054 contre −0,063), alors que sur le poste entier c'est bien pire (−28,0 contre −16,9).
- Avant l'option B, un cycle rapportait environ +0,87 par décision : l'agent avait intérêt à le toucher **le plus tôt possible**, et cette incitation disparaît dès que la récompense moyenne d'un cycle devient négative.
- En résumé : l'option B a corrigé l'attribution du coût du temps entre les routes, mais elle a créé une incitation à **remettre à plus tard**. Les deux défauts viennent de la même cause, notée dès le 19 octobre : **γ s'applique par décision, et les décisions n'ont pas toutes la même durée**.

**Mon erreur :**
- La mesure sur le papier (`docs/option_b_recompense.md`) ne comprenait **aucune politique qui attend** : les 12 politiques testées n'attendent jamais. Elle comparait aussi des récompenses par décision, sans regarder l'effet de l'actualisation sur une récompense devenue négative.
- J'aurais dû inclure ATTENDRE dans la mesure, puisque c'était justement le défaut corrigé en T19. La conclusion « l'option B pousse dans la bonne direction » était vraie pour le choix de la route, fausse pour le choix d'attendre.

**La règle (point 16) dit : aucune autre variante, on part en campagne avec l'option B.** Mais une campagne où PPO produit 210 t/h en attendant tout le temps mesurerait le défaut de la récompense, pas la qualité des agents. **Décision à reprendre avec la directrice**, sans rien lancer d'ici là. Options :
1. **Appliquer la règle telle quelle** : campagne avec l'option B, et l'effondrement présenté dans l'article.
2. **Revenir à la récompense d'avant l'option B** (commit `2461c7c`, essais 6 à 9). On déclare que l'option B a été essayée selon un protocole fixé à l'avance, qu'elle a échoué et pourquoi. Puis on fait le réglage tabulaire prévu (points 8 et 9) et la campagne. PPO y est au niveau du programme linéaire. Les autres agents dérivent vers la pelle 3 : c'est expliqué par les mesures du 19 octobre, et l'actualisation par minute est citée en perspective.
3. **Corriger la cause** : actualisation par minute (γ^durée) dans les quatre algorithmes. C'est un travail de 1 à 2 jours sur les algorithmes de Stable-Baselines3, puis un nouveau cycle d'essais, et un deuxième changement fait après avoir vu les résultats. C'est le plus juste sur le fond, mais le plus long et le plus exposé au reproche de « forcer » le résultat.

**Recommandation :** l'option 2.

**Message à la directrice rédigé (21 octobre 2026)** : `docs/message_directrice_option_b.md`. Il contient les résultats, la cause mesurée, mon erreur, la proposition de retour en arrière et l'autre voie (actualisation par minute).

## Retour à la récompense d'avant l'option B : code préparé (21 octobre 2026)
*Préparé en attendant la décision de la directrice : **ni commité, ni lancé***

**Ce que j'ai changé (non commité) :**
- `env/mine_env.py` : un cycle ne transmet plus sa durée au coût d'opportunité. Seule ATTENDRE le paie, comme depuis T19. Par rapport à `2461c7c`, le simulateur ne diffère plus que par le nom du paramètre (`idle_min` → `truck_min`).
- `simulation/reward.py` : la docstring revient à la description d'avant et ajoute un paragraphe « Option B, essayée puis abandonnée », avec la cause.
- `tests/fixtures/regression_reference.csv` : la référence de `2461c7c` est restaurée **telle quelle**, et le code la retrouve exactement (10 postes).
- `tests/test_reward.py` : version de `2461c7c`, au nom du paramètre près.
- `analysis/reward_diagnostics.py` :
  - `reward_per_decision` est de nouveau la récompense actuelle ;
  - l'option B est recalculée dans `reward_per_decision_option_b` ;
  - nouvelle mesure `option_b_wait_incentive`, qui retrouve exactement le tableau du 21 octobre (−0,063 / −0,054 par décision, −16,874 / −28,037 par poste).
- `experiments/agent_trials.py` :
  - les essais 5 à 9 se refont de nouveau sur le code actuel ;
  - les essais 10 à 13 sont rattachés au commit `78ed5f5`, avec leurs résultats.
- `docs/option_b_recompense.md` : avertissement en tête (option abandonnée, et pourquoi).

**Projet d'ajout à la règle, à dater et à compléter quand la directrice aura répondu (points 19 à 21) :**

19. Retour à la récompense des essais 0 à 9 : l'option B est abandonnée pour la cause mesurée le 21 octobre (incitation à ATTENDRE). Plus aucune modification de la récompense, quel que soit le résultat.
20. PPO et DQN partent en campagne avec les versions des essais 6 et 7, sans nouvel essai. Q-Learning et SARSA suivent les points 8 et 9 de la règle (réglage, puis variante de l'état), avec cette récompense.
21. Tout est déclaré dans l'article :
    - les essais 0 à 13 ;
    - l'option B, sa justification, son échec et sa cause ;
    - l'actualisation par unité de temps, en perspective.

*Ce projet de points 19 à 21 n'a pas été retenu : la directrice a choisi l'actualisation par minute (ci-dessous).*

## Décision de la directrice : actualisation par minute, et ajout à la règle (21 octobre 2026)
*Écrit **avant** de modifier les algorithmes et avant tout entraînement*

**Décision :** Dr. Ferchichi a retenu la correction de la cause : **actualisation par minute** (formulation semi-markovienne) dans les quatre algorithmes, avec la récompense d'avant l'option B.

**Ajout à la règle (points 19 à 25) :**

19. **La récompense est celle des essais 0 à 9** : celle d'avant l'option B, avec le coût d'opportunité d'ATTENDRE seulement. Elle ne sera plus modifiée, quel que soit le résultat.
20. **Seule la façon de compter le temps change.** Une récompense reçue après une décision qui a duré Δt minutes est actualisée par γ_min^Δt au lieu de γ.
21. **Aucun nouveau paramètre réglé sur les résultats.** On calcule γ_min = γ^(1 / 1,89), où γ est la valeur actuelle de chaque algorithme et 1,89 min l'intervalle moyen entre deux décisions de l'Affectation fixe en nominal (déjà mesuré et testé, `test_intervalle_entre_decisions`). En nominal, l'horizon reste donc le même en moyenne. Il ne dépend plus de la taille de la flotte : c'était une limite notée en T31.
22. **Tout le reste est gardé** :
    - observation de 23 composantes ;
    - normalisation de la récompense pour PPO ;
    - hyperparamètres du réglage ;
    - configuration actuelle de Q-Learning et de SARSA.
23. **Quatre essais de contrôle au budget complet** (essais 14 à 17 : PPO, DQN, Q-Learning, SARSA). Configuration de la campagne avec l'actualisation par minute, graine d'entraînement 1, graines de sélection 20 000 à 20 009. Ils sont lancés **après le commit**.
24. **Puis, pour Q-Learning et SARSA, le réglage prévu** (points 8 et 9 de la règle) avec l'actualisation par minute. Ensuite, la campagne, **quel que soit le résultat**. Aucune autre variante, ni de la récompense, ni de l'actualisation, ni des agents. Si les résultats sont moins bons qu'avant (par exemple PPO sous 4 405 t/h), on garde quand même l'actualisation par minute : elle est retenue parce que c'est la formulation correcte, pas pour son score.
25. **Tout est déclaré** :
    - les essais 0 à 17 ;
    - l'option B et son échec ;
    - la raison de l'actualisation par minute ;
    - les approximations de l'implémentation, écrites dans le code et dans l'article.

**Prédiction (avant le code) :**
- Les heuristiques ne changent pas, puisqu'elles ne dépendent pas de γ.
- L'incitation à ATTENDRE et la dérive vers la pelle 3 devraient disparaître ou diminuer, car attendre ou rouler longtemps retarde la prochaine production en minutes, pas seulement en décisions.
- Je ne prédis pas le niveau des agents. Pour PPO, je m'attends à un résultat proche de 4 405 t/h, puisqu'il ne dérivait déjà presque pas.

## Actualisation par minute dans le code (21 octobre 2026)

**Ce que j'ai changé :**
- **Base :** le retour à la récompense d'avant l'option B, préparé plus haut (même simulateur qu'en `2461c7c`, aux noms près).
- `env/mine_env.py` : `info["elapsed_min"]` donne la durée de chaque décision (jusqu'à la décision suivante ou à la fin du poste ; 0 après reset). Aucun KPI ne change.
- **Nouveau module `rl/time_discount.py`** :
  - γ_min = γ^(1 / 1,89) et facteur γ_min^Δt par décision ;
  - `TimeDiscountReplayBuffer` (DQN) : la mémoire de rejeu garde le facteur de chaque transition, que Stable-Baselines3 2.8 utilise dans la cible (champ `discounts`) ;
  - `TimeDiscountRolloutBuffer` et `TimeDiscountCallback` (PPO) : avantages GAE avec un facteur par pas, prolongement de la valeur à la coupure avec le même facteur, γ de la configuration remis dans le modèle à la fin ;
  - pour Q-Learning et SARSA : facteur γ_min^Δt dans la cible.
- **Approximations déclarées dans le module :**
  - la récompense d'un pas est comptée au début du pas ;
  - la normalisation de la récompense de PPO garde le γ par décision pour estimer l'échelle.
- `configs/*.yaml` : `time_discount: true` pour les quatre algorithmes, avec un commentaire. `config.py` accepte la clé.
- Entraînements : option `time_discount`, déclarée dans `metadata.json` avec γ_min. Elle fait partie de la signature des sauvegardes, donc une reprise refuse un autre choix.
- `experiments/agent_trials.py` :
  - les essais 5 à 9 précisent `time_discount: false` et restent refaisables ;
  - nouveaux essais de contrôle 14 à 17.

**Ce que le test vérifie** (`tests/test_time_discount.py`, 12 tests) :
- **Test central : si toutes les décisions durent 1,89 min, l'actualisation par minute donne le même apprentissage que l'actualisation par décision, pour les quatre algorithmes** : mêmes poids pour PPO et DQN (à 10⁻⁴ près), mêmes Q-tables pour Q-Learning et SARSA. Seule la façon de compter le temps change.
- Avec les vraies durées, l'apprentissage change.
- Les durées couvrent exactement les 480 min d'un poste, et 1,89 min est bien l'intervalle moyen de l'Affectation fixe en nominal.
- Le GAE du nouveau tampon est égal au GAE usuel quand les durées valent la référence.
- La mémoire de rejeu rend le bon facteur pour chaque transition.
- Le modèle enregistré garde γ = 0,995, et plusieurs environnements sont refusés.
- Deux tests existants ont été adaptés : la métadonnée γ_min est déclarée en plus, et le test des environnements parallèles (T29) précise l'actualisation par décision.
- Suite complète : **382 réussis, 1 ignoré** (380 au premier passage, puis les 2 tests adaptés, relancés).

**Commit :** `498f37b`

**Suite (règle, point 23) :** essais 14 à 17 lancés le 21 octobre vers 7 h 45, **après** le commit, dans 4 fenêtres PowerShell (sortie `data/results_v2/smdp`, journaux `data/results_v2/smdp_essai_<n>.log`).

## Résultat des essais de contrôle de l'actualisation par minute (essais 14 à 17, 21 octobre 2026)

Commit `498f37b`, sans interruption ni reprise. Graines de sélection 20 000 à 20 009, régime nominal.

| Agent | Par décision (essais 6 à 9) | Option B (10 à 13) | **Par minute (14 à 17)** | ATTENDRE | Pelles 1 / 2 / 3 | Durée |
|---|---|---|---|---|---|---|
| PPO | 4 404,8 | 210,0 | **4 413,5 ± 67,9** | 0,0 % | 62 / 36 / 2 % | 222,8 min |
| DQN | 3 592,8 | 3 531,5 | **120,8 ± 5,5** | 99,8 % | 29 / 71 / 0 % | 167,7 min |
| Q-Learning | 3 955,0 | 4 098,5 | **3 974,2 ± 46,3** | 30,0 % | 16 / 35 / 49 % | 253,4 min |
| SARSA | 3 535,0 | 4 133,5 | **3 601,5 ± 39,4** | 0,0 % | 0 / 0 / 100 % | 241,4 min |

Repères (mêmes graines) : MTWT 4 443, programme linéaire 4 412, hasard 4 167.

Courbes d'entraînement (t/h par dixième du budget) :
- PPO : 4 346, 4 406, 4 409, 4 404, 4 397, 4 414, 4 405, 4 415, 4 410, 4 414.
- DQN : 4 098, 4 004, 3 923, 3 757, 3 653, 3 576, 3 554, 3 481, 3 538, 3 488.
- Q-Learning : 4 162, 4 146, 4 124, 4 094, 4 059, 4 014, 3 957, 3 935, 3 941, 3 962.
- SARSA : 4 163, 4 154, 4 136, 4 105, 4 068, 4 014, 3 948, 3 837, 3 726, 3 589.

**Constat :**
- **PPO : 4 413,5 t/h**, son meilleur résultat, à égalité avec le programme linéaire (4 412) et à 30 t/h de MTWT. Surtout, il **n'attend plus jamais** (0 % contre 36 % avec l'actualisation par décision), et ses parts par pelle sont proches de celles de MTWT.
- **DQN s'effondre en attendant** (99,8 % d'ATTENDRE, 120,8 t/h).
- **Q-Learning et SARSA** retrouvent à peu près leurs résultats de l'actualisation par décision. SARSA dérive de nouveau vers la pelle 3 (100 %), et leurs courbes baissent toujours quand l'exploration diminue.

**Mesure sur DQN (lecture seule, graine de sélection 20 000 et une graine d'entraînement) :**

| Politique jouée | Décisions par poste | Tonnage | Part d'ATTENDRE |
|---|---|---|---|
| Gloutonne, graine de sélection | 1 251 | 840 t | 99,5 % |
| Gloutonne, graine d'entraînement | 1 258 | 1 120 t | 99,4 % |
| ε = 0,02, graine de sélection | 5 302 | 15 120 t | 97,9 % |

Les derniers postes d'entraînement ont produit de 25 620 à 30 660 t : les mises à jour continuelles et l'exploration masquent pendant l'entraînement ce que fait le modèle final.

**Interprétation (hypothèse, cohérente avec la mesure) :** avec l'actualisation par minute, une ATTENDRE de 0,2 min ne coûte presque rien en valeur : la production suivante n'est repoussée que de 0,2 min. C'est juste, puisqu'attendre 0,2 min coûte réellement très peu. Mais la valeur d'ATTENDRE est donc presque égale à celle de la meilleure action, et l'écart est plus petit que l'erreur du réseau de DQN. Dès que l'erreur la place en tête, le modèle attend, et une attente en entraîne d'autres. PPO apprend une politique, pas des valeurs à comparer : il n'est pas touché. Ce n'est pas un défaut du code ; le test central (durées égales à 1,89 min = apprentissage identique) le confirme.

**Décision, par application de la règle (point 24) : aucune autre variante.**
- PPO et DQN partent en campagne avec ces versions (essais 14 et 15), DQN compris.
- Pour Q-Learning et SARSA, on fait maintenant le réglage prévu (point 8 : α ∈ {0,1 ; 0,2} × {6 ; 8} intervalles ; les essais 16 et 17 comptent pour α = 0,2 et 8 intervalles), puis la variante de l'état (point 9), une seule fois, avec la meilleure configuration.
- L'effondrement de DQN sera décrit dans l'article avec cette mesure et cette hypothèse.

## Réglage des agents tabulaires et variante de l'état : code (21 octobre 2026)

**Ce que j'ai changé :**
- `rl/discretize.py` : la dernière variable de l'état discret est maintenant au choix (`state_feature`) :
  - `nearest_shovel` (valeur d'origine, configuration actuelle) : la pelle la plus proche ;
  - `best_route` (variante du point 9) : la route de plus petite attente prévue (0 à 5, dans l'ordre des actions). Elle est lue dans le bloc `route_wait` de l'observation, c'est-à-dire la même prévision que MTWT ; à égalité, la première route. La table peut alors avoir 8⁵ × 6 = 196 608 états.
- `configs/tabular_hyperparameters.yaml` : `state_feature: nearest_shovel`. La Q-table enregistrée garde sa représentation ; une Q-table plus ancienne (sans cette clé) est lue avec la pelle la plus proche.
- `experiments/agent_trials.py` :
  - essais 18 à 23 = réglage : Q-Learning (18 à 20) et SARSA (21 à 23), pour α ∈ {0,1 ; 0,2} × {6 ; 8} intervalles, sans (0,2 ; 8), déjà couvert par les essais 16 et 17 ;
  - résultats des essais 14 à 17 inscrits.
- **Écart au point 9, déclaré :** la règle disait « calculé comme MTWT ». MTWT choisit d'abord la pelle de plus petite attente, puis la verse. La variante prend la route de plus petite attente **totale** prévue, qui se lit directement dans l'observation. C'est la même prévision, mais pas exactement la même règle de choix.

**Ce que le test vérifie :**
- `test_tabular.py` :
  - la variable de la variante est l'argmin des attentes prévues par route de `MineEnv.route_waits()` ;
  - elle change au cours du poste ;
  - les 5 attentes discrétisées ne changent pas ;
  - la Q-table enregistre sa représentation, et une ancienne Q-table est relue avec la pelle la plus proche.
- `test_agent_trials.py` : la grille, avec les essais 16 et 17, couvre exactement {0,1 ; 0,2} × {6 ; 8}, et seuls α et le nombre d'intervalles changent.
- Suite complète : **385 réussis, 1 ignoré**.

**Commit :** `8a7cd84`

## Résultat du réglage des agents tabulaires (essais 16 à 23, 22 octobre 2026)

Lancés le 21 octobre vers 20 h 40 sur le commit `8a7cd84` (6 en même temps), sans interruption. Actualisation par minute, graines de sélection 20 000 à 20 009.

| Essai | Agent | α | Intervalles | t/h | ATTENDRE | Pelles 1 / 2 / 3 |
|---|---|---|---|---|---|---|
| 16 | Q-Learning | 0,2 | 8 | 3 974,2 ± 46,3 | 30,0 % | 16 / 35 / 49 % |
| 18 | Q-Learning | 0,1 | 6 | 3 928,8 ± 51,0 | 30,5 % | 19 / 23 / 58 % |
| 19 | Q-Learning | 0,1 | 8 | 3 368,8 ± 78,4 | 74,9 % | 2 / 9 / 89 % |
| **20** | **Q-Learning** | **0,2** | **6** | **4 180,8 ± 48,4** | 33,3 % | 35 / 46 / 18 % |
| 17 | SARSA | 0,2 | 8 | 3 601,5 ± 39,4 | 0 % | 0 / 0 / 100 % |
| **21** | **SARSA** | **0,1** | **6** | **3 626,0 ± 58,8** | 0,8 % | 1 / 1 / 98 % |
| 22 | SARSA | 0,1 | 8 | 3 596,2 ± 30,0 | 0 % | 0 / 0 / 100 % |
| 23 | SARSA | 0,2 | 6 | 3 480,8 ± 47,7 | 0,5 % | 0 / 0 / 100 % |

Durées : 357 à 381 min, avec 6 entraînements en même temps (contre 241 à 253 min à 4).

**Constat :**
- **Q-Learning, α = 0,2 et 6 intervalles (essai 20) : 4 180,8 t/h.** C'est le premier agent tabulaire au-dessus du hasard (4 167). Ses parts par pelle sont équilibrées, et sa courbe remonte en fin d'entraînement (4 052 → 4 088).
- **SARSA reste sur la pelle 3** quelle que soit la configuration : de 98 à 100 % des cycles, de 3 481 à 3 626 t/h. L'écart entre ses configurations est faible, à peine plus grand que les écarts-types.
- Le choix se fait sur une seule graine d'entraînement. L'écart entre les configurations de SARSA (145 t/h au plus) ne permet pas de dire qu'elles diffèrent vraiment.

**Décision, par application de la règle (points 8, 9 et 24) :**
- Meilleures configurations : **Q-Learning α = 0,2, 6 intervalles** (essai 20) ; **SARSA α = 0,1, 6 intervalles** (essai 21).
- Variante de l'état (point 9), une seule fois, avec ces configurations : **essais 24 (Q-Learning) et 25 (SARSA)**, inscrits dans `experiments/agent_trials.py` (test : ce sont bien les configurations de meilleure productivité). La variante n'est retenue que si elle fait mieux que les essais 20 et 21.
- Ensuite, Q-Learning et SARSA sont figés pour la campagne. Ils n'ont plus la même configuration : `configs/tabular_hyperparameters.yaml` devra donner des valeurs par algorithme (à faire avant la campagne, avec un test).

**Suite (règle, points 8, 9 et 24) :**
1. Après le commit, lancer les essais 18 à 23 (6 entraînements, en même temps).
2. Avec les essais 16 et 17, choisir pour chaque algorithme la configuration de meilleure productivité sur les graines de sélection.
3. Inscrire les essais 24 (Q-Learning) et 25 (SARSA) : variante `best_route` avec cette configuration, une seule fois. Elle n'est retenue que si elle fait mieux.
4. Puis la campagne.

**Commit :** `7ef41e6` (résultats des essais 18 à 23, essais 24 et 25 inscrits). Essais 24 et 25 lancés le 22 octobre **après** ce commit, dans 2 fenêtres PowerShell (sortie `data/results_v2/smdp`, journaux `data/results_v2/smdp_essai_<n>.log`).

## Résultat de la variante de l'état (essais 24 et 25, 22 octobre 2026)

Lancés le 22 octobre vers 5 h 58 sur le commit `7ef41e6` (2 en même temps), sans interruption. Durées : 192,1 min (Q-Learning) et 172,6 min (SARSA). Graines de sélection 20 000 à 20 009.

| Essai | Agent | Dernière variable | t/h | ATTENDRE | Pelles 1 / 2 / 3 | Courbe d'entraînement (t/h, 1er → dernier dixième) |
|---|---|---|---|---|---|---|
| 20 | Q-Learning | pelle la plus proche | **4 180,8 ± 48,4** | 33,3 % | 35 / 46 / 18 % | 4 162 → 4 088 |
| 24 | Q-Learning | route de plus petite attente | 4 061,8 ± 44,8 | 32,2 % | 26 / 32 / 42 % | 4 161 → 4 068 |
| 21 | SARSA | pelle la plus proche | **3 626,0 ± 58,8** | 0,8 % | 1 / 1 / 98 % | 4 162 → 3 614 |
| 25 | SARSA | route de plus petite attente | 3 505,2 ± 59,5 | 9,2 % | 2 / 3 / 95 % | 4 160 → 3 543 |

(Q-Learning et SARSA avec α = 0,2 / 0,1 et 6 intervalles dans les deux cas. États visités en fin d'entraînement : 9 807 et 9 591, sur 46 656 possibles.)

**Décision, par application de la règle (point 9) :** la variante fait moins bien pour les deux algorithmes (−119 et −121 t/h). Elle est **rejetée**. La dernière variable reste la pelle la plus proche. La mise au point est **terminée** : les agents tabulaires sont figés pour la campagne, avec Q-Learning α = 0,2 et SARSA α = 0,1, 6 intervalles, γ = 0,99, 30 000 épisodes, actualisation par minute.

**Constat :**
- Savoir quelle route MTWT choisirait ne corrige pas la dérive de SARSA vers la pelle 3 (95 % des cycles au lieu de 98 %).
- **Les deux agents se dégradent pendant l'apprentissage, avec ou sans la variante.** La courbe part d'environ 4 160 t/h, quand l'exploration est presque aléatoire (la politique aléatoire fait 4 167), puis baisse. Le meilleur agent tabulaire (essai 20) égale à peine le hasard. Ce n'est pas un problème de réglage : les 10 essais tabulaires faits avec l'actualisation par minute (16 à 25) le montrent tous. Il faudra le dire tel quel dans l'article, comme un défaut de convergence de l'approche tabulaire (état discrétisé, 30 000 épisodes), sans le masquer.
- Comme au réglage, le choix repose sur une seule graine d'entraînement. Un écart de 120 t/h fait environ 2,5 écarts-types entre graines d'évaluation : la décision est nette, mais elle ne dit rien de la variabilité entre graines d'entraînement.

**Ce que j'ai changé :**
- `configs/tabular_hyperparameters.yaml` : une valeur peut maintenant dépendre de l'algorithme (`alpha: {Q-Learning: 0.2, SARSA: 0.1}`), `n_bins: 6`, `state_feature: nearest_shovel`. Un fichier archivé avant le réglage (valeurs communes) se lit toujours.
- `config.py` : `_tabular_value` choisit la valeur de l'algorithme. Une valeur par algorithme doit nommer Q-Learning **et** SARSA, sinon `ValueError` (une faute de frappe comme « Sarsa » est refusée).
- `rl/discretize.py` : `DEFAULT_N_BINS` passe de 8 à 6, égal à la configuration. Le nombre d'intervalles est enregistré avec chaque table : les tables à 8 intervalles se relisent correctement.
- `experiments/agent_trials.py` :
  - résultats des essais 24 et 25 ;
  - `PRE_TUNING = {alpha: 0,2, n_bins: 8}`, fixé explicitement pour les essais 8, 9, 16 et 17. Avant, ces essais prenaient α et les intervalles dans `configs/`, qui a changé : sans cela, les refaire aurait donné un autre essai.

**Ce que le test vérifie :**
- `test_config.py` :
  - Q-Learning reçoit α = 0,2, SARSA α = 0,1, les deux 6 intervalles et la pelle la plus proche ;
  - une valeur par algorithme est lue pour chacun, un argument la remplace, une valeur commune reste acceptée ;
  - un nom d'algorithme manquant ou mal écrit est refusé.
- `test_tabular.py` : 6⁵ × 3 = 23 328 états (98 304 avec 8 intervalles), 6⁵ × 6 = 46 656 avec la variante ; discrétisation identique à la formule écrite en dur, pour 6 et 8 intervalles.
- `test_agent_trials.py` :
  - les essais 8, 9, 16 et 17 fixent la configuration d'avant le réglage ;
  - la grille du réglage est complète (16 et 17 compris) ;
  - **la configuration de la campagne est celle que donne la règle** : pour chaque algorithme, l'essai gardé est le meilleur entre le réglage et la variante, et `configs/` contient ses valeurs.
- Suite complète : **389 réussis, 1 ignoré**.

## Modèle de carburant revu (22 octobre 2026, avant la campagne, option A)

**Déclencheur :** des documents ajoutés dans `references/` (rangés dans `references/7-donnees-equipements/`, avec une fiche chacun) :
- le Caterpillar Performance Handbook, édition 49 ;
- la fiche technique du Cat 785D ;
- le mémoire de Kubler (2015), qui a **mesuré** sur un Cat 785D en exploitation un ralenti à 11,3 L/h et la consommation en rampe en fonction de la pente.

Notre ralenti valait 64,9 L/h (6 fois la mesure), et l'effet de la pente était bien plus faible que mesuré. Le calibrage du T14 multipliait tout par 6,49 pour atteindre 95 L/h en moyenne : la moyenne était juste, mais la répartition fausse.

**Vérifié :** les plages du Handbook citées dans le code (785D : 54,2-81,4 / 81,4-108,5 / 108,5-135,6 L/h) sont exactes. En revanche :
- l'édition était fausse dans le code (« 2022 », alors que c'est l'édition 49 de 2019) ;
- la charge utile aussi (« ≈ 136 t », alors que la charge cible est de 133 t et la charge nominale de 145 t).

**Mesures (prototypes dans le scratchpad, nominal, heuristiques) :**

| Modèle | L/h par camion (Affectation fixe) | Remarque |
|---|---|---|
| Actuel (T14) | 95,7 | ralenti 64,9 L/h |
| Kubler tel quel | **211** | au-delà du maximum publié (135,6 L/h) ; ralenti 0,9 % du carburant |
| A1 : ralenti Kubler + ancien roulage remis à l'échelle | 95,0 | pente : +3 % par point (arbitraire) |
| **A2 : ralenti Kubler + courbes de Kubler × 0,4466** | **95,0** | retenu |

Kubler tel quel dépasse le maximum publié, parce que toutes les pistes du simulateur montent, retour compris (1 à 8 %), et qu'il ne représente ni descente ni roue libre.

**Décision (option A, choisie par l'utilisateur ; variante A2) :**
- ralenti à 11,3 L/h (Kubler) ;
- roulage = courbes de Kubler selon la pente, multipliées par 0,4466 ;
- le facteur est choisi pour que l'Affectation fixe consomme 95 L/h **sur les graines de sélection** (pas sur celles de l'évaluation, qui donnent 95,4 L/h).

A2 plutôt que A1 : la répartition entre charge, vide et pente vient de mesures, et non d'une hypothèse.

Avec A2, la consommation par tonne de l'Affectation fixe change peu (0,271 → 0,268 L/t), mais **le classement des politiques change** :

| Politique | L/t, ancien modèle | L/t, A2 |
|---|---|---|
| Affectation fixe | 0,271 | 0,268 |
| MTWT | 0,256 (la plus sobre) | 0,243 |
| Plus court chemin | 0,272 | 0,238 |
| Pelle la plus proche | 0,273 (la moins sobre) | **0,234 (la plus sobre, avec LP)** |

L'ancien modèle faisait payer cher l'attente en file (ralenti à 64,9 L/h), ce qui pénalisait Pelle la plus proche (beaucoup de file à la pelle 1). Le nouveau modèle fait payer la distance et la pente, ce qui favorise les routes courtes. **Les conclusions sur le carburant de la v2 (« DQN consomme le moins », §7.x) ne tiennent plus telles quelles** : elles seront refaites avec la campagne.

**Effets sur le reste :**
- Le carburant **n'entre pas dans la récompense** : la productivité et les cycles ne changent pas.
- Il entre dans **une composante sur 23 de l'état** (carburant du camion qui décide) : la mise au point (essais 0 à 25) a été faite avec l'ancien modèle. À déclarer ; c'est un changement venu d'une donnée externe, pas d'un résultat des agents.
- **Plus court chemin** : γ garde sa valeur numérique (0,2 / 6,49). Quelques décisions changent, car le coût en carburant d'un trajet dépend plus de la pente : 4 034,9 → 4 027,3 t/h en nominal.
- La borne de normalisation de l'observation est inchangée (135,6 L/h × 8 h).

**Ce que j'ai changé :**
- **`simulation/constants.py`** : sources exactes (Handbook éd. 49 section 25, Kubler § 5.1.3 et § 5.3), courbes, `FUEL_IDLE_L_PER_H = 11.3`, `FUEL_TRAVEL_SCALE = 0.4466`. L'ancien facteur de calibrage est supprimé.
- **`simulation/fuel_model.py`** : `l_per_km` (polynôme borné au domaine mesuré, une descente comptant comme du plat), et `FuelParams` avec les nouveaux champs.
- **`baselines/shortest_path_policy.py`** : γ = 0,2 / (95 / 14,635), valeur de T14 écrite explicitement.
- **`experiments/scenarios.py`** : précise 785C/785D.
- **Article v3** :
  - §5.3 : module carburant et ligne de `tab:simparams` ;
  - §8.5 : limite sur le carburant ;
  - note M20 remplacée ;
  - trois entrées `.bib` ajoutées (Caterpillar2019, Caterpillar785D2025, Kubler2015).

**Ce que le test vérifie (`tests/test_fuel.py`, réécrit) :**
- les courbes redonnent les valeurs des figures de Kubler ;
- la pente est bornée au domaine mesuré ;
- trajet et ralenti suivent le modèle ;
- les paramètres sont modifiables ;
- le facteur donne 95 L/h sur les graines de sélection ;
- les graines d'évaluation restent dans la plage « usage moyen » ;
- sans facteur, le simulateur dépasse le maximum publié ;
- γ de Plus court chemin est inchangé.

Suite complète : **397 réussis, 1 ignoré**. Les références de non-régression ne contiennent pas le carburant et passent sans être régénérées.

## Lanceur de la campagne et répétition (22 octobre 2026)

**Constat :** il n'existait pas de lanceur pour la campagne. `run_benchmark` entraîne les modèles manquants un par un, dans un seul processus : environ 13 jours pour 90 entraînements. Lancé dans plusieurs fenêtres, il pouvait aussi entraîner deux fois un modèle nominal, que les régimes perturbés utilisent pour le transfert.

**Ce que j'ai fait :** `experiments/campaign.py`, en trois étapes et une répétition :
- `train` : les 90 entraînements, 4 à la fois (60 dans les régimes principaux, 30 pour l'ablation). Les plus longs passent d'abord ; un modèle déjà fait est sauté ; un entraînement interrompu reprend ; un journal par modèle ; la configuration est archivée.
- `evaluate` : une seule évaluation, dans les 8 régimes. Elle refuse de démarrer s'il manque un modèle.
- `analyze` : agrégation sur les graines, tests appariés, transfert, ablation, courbes ; une sortie par analyse dans `analyses/`.
- `rehearse` : les trois étapes avec une configuration réduite (4 096 pas, 40 épisodes), et des **graines d'évaluation 30 000 à 30 029, disjointes de 42-71**. Chaque étape tourne dans un processus séparé, avec `MINE_RL_CONFIG_DIR`.

Tests : `tests/test_campaign.py` (8). Ils vérifient :
- les 90 entraînements, sans doublon ;
- les régimes évalués ;
- qu'un modèle déjà fait est sauté ;
- qu'un échec n'arrête pas les autres entraînements ;
- que l'évaluation refuse de démarrer s'il manque un modèle ;
- qu'il n'y a qu'un fichier de résultats ;
- que la configuration de répétition a des graines disjointes et les mêmes hyperparamètres.

**Répétition** (`data/results_v2/repetition`, 22/10, 4 processus) :
- durée totale 27 min : entraînement 8,4 min, sans échec ; évaluation 18,2 min ; analyses 0,2 min ;
- 6 540 lignes, le nombre attendu ;
- 37 Mo sur le disque ;
- aucune valeur manquante dans les analyses.

**Défaut trouvé :** dans `statistical_tests.main`, le test d'interaction politique × régime prenait tous les régimes du fichier, non vus et ablation compris, et donnait F(49, 1682). Corrigé : l'analyse est restreinte aux trois régimes principaux, ce qui donne **F(18, 783)**. Le test ajouté vérifie que des lignes d'ablation ajoutées au CSV v1 ne changent rien à la sortie.

**Point à décider avant la campagne :** les comparaisons par paires de `statistical_tests` sont encore celles de la v1 (PPO contre Fixed, DQN contre Fixed, PPO contre DQN). Avec dix politiques, et MTWT comme meilleure heuristique, il faut fixer la liste des comparaisons **avant** de voir les résultats.

**Décision (22/10/2026, avant la campagne, sur ma proposition) :** chacun des quatre agents est comparé à **MTWT** (meilleure heuristique sur les graines de sélection) et à l'**Affectation fixe** (référence de la v1). Cela fait 8 comparaisons par régime et par indicateur, avec correction de Holm sur ces 8.
- `statistical_tests.PAIRS` contient ces 8 comparaisons. `V1_PAIRS` garde celles de la v1, pour relire son CSV, et `main(csv, pairs)` accepte la liste.
- Article §6.5 : les comparaisons y sont déclarées comme fixées avant la campagne.
- Test : `test_comparaisons_fixees_avant_la_campagne` vérifie la liste. Les tests sur le CSV v1 et sur le CSV synthétique utilisent `V1_PAIRS`.
- Vérifié sur le fichier de la répétition : les 8 comparaisons sont produites dans chaque régime.
- Suite complète : **407 réussis, 1 ignoré**.

**Article v3 :** les cases rouges qui attendaient ces essais sont remplies :
- §5.6 : sixième variable = pelle la plus proche, 23 328 états ;
- `tab:tuning` : ligne 24–25 ;
- §6.3 : paragraphe « Les agents tabulaires » ;
- §8.5 : la limite sur la convergence tabulaire décrit ce qu'on a observé, et la confirmation par la campagne reste à faire.

Le PDF fait 45 pages, sans erreur.

---

# Phase 7. Relancer, vérifier, nettoyer (1 à 2 semaines, surtout du calcul)

## T31 — Campagne finale et mise à jour de l'article
*Indispensable · nouveaux résultats*

**Date :** 

**Prédiction (avant de modifier) :** 

**Décision (avec la directrice) :** décisions prises le 22 octobre 2026, avant la campagne, sur ma recommandation. Les deux points avaient été posés à la directrice dans le message du même jour (`docs/message_directrice_fin_mise_au_point.md`). **Accord de la directrice reçu le 22 octobre 2026** : il porte sur le modèle de carburant revu (`docs/message_directrice_carburant.md`), et la campagne est lancée avec les choix préparés (DQN tel quel, ablation sur PPO et DQN, 8 comparaisons).
1. **DQN part en campagne tel quel**, dans la version qui s'effondre sur ATTENDRE (essai 15).
   - C'est ce que prévoit la règle.
   - La conclusion ne change pas : même avant l'actualisation par minute, DQN restait loin sous les heuristiques (essai 7 : 3 593 t/h, 99 % pelle 3).
   - L'article le présente comme le résultat de cette configuration, avec la mesure et l'hypothèse.
2. **Ablation de w4 : PPO et DQN seulement**, 5 graines d'entraînement, w4 ∈ {0 ; 0,1 ; 0,5}, soit 30 entraînements de plus (environ 1,5 jour). Le point w4 = 0,3 est le nominal de la campagne.
   - PPO, parce que c'est l'agent principal.
   - DQN, parce qu'il attend dans 99,8 % de ses décisions et que w4 pénalise justement l'attente. L'ablation teste donc l'hypothèse de l'article.
   - Q-Learning et SARSA en sont exclus : ils convergent mal.
   - La v1 faisait déjà l'ablation sur PPO et DQN, mais avec une seule graine.

**Ce que j'ai changé :**
- **`configs/experiment_config.yaml`** : nouvelle clé `algorithms`, qui donne les agents entraînés et évalués dans chaque scénario. Les quatre par défaut, `[DQN, PPO]` pour les trois scénarios `ablation_w4_*`.
- **`experiments/scenarios.py`** : `Scenario.algorithms`. Un nom inconnu ou une liste vide est refusé. Sans la clé (configuration archivée avant ce jour), les quatre agents, comme avant.
- **`experiments/run_benchmark.py`** : seuls les algorithmes du scénario évalué sont entraînés et évalués. Les heuristiques le sont toujours.
- **`experiments/ablation_study.py`** réécrit :
  - il se lance dans le **dossier de la campagne**, avec 5 graines. Avant, il réentraînait les quatre agents avec une seule graine dans un dossier à part, y compris le nominal ;
  - `ablation_table` construit le tableau (w4 lu dans la configuration) à partir des CSV de la campagne et de l'ablation.
- **Article v3** :
  - §6.2 : nouveau paragraphe « Étude d'ablation » ;
  - §7.9 : la note orange dit que l'ablation est prévue (réécriture au lot 2) ;
  - résumé réécrit pour la partie méthode (dix politiques, processus semi-markovien, règle fixée à l'avance, 5 graines, transfert). Les résultats et la conclusion sont en `\pending`. L'ancien résumé de la v2 est gardé dans le `.tex`, mis en commentaire.
- **Tests** : `test_config.py` (agents par scénario, refus d'un nom inconnu ou d'une liste vide, clé absente = les quatre), `test_transfer.py` (seuls les algorithmes du scénario sont entraînés), `test_ablation_study.py` (lancement dans le dossier de la campagne, tableau par w4). Suite complète : **395 réussis, 1 ignoré**.

**Pourquoi :** 

**Ce que le test vérifie :** 

**Résultat comparé à ma prédiction :** 

**Ce que j'ai appris :** 

**Commit :** 

**Questions « Savoir expliquer » :**
- [ ] pour chaque tableau de l'article, quel script et quel fichier l'ont produit :


## Nom de la règle FIFO : vérifié correct (22 octobre 2026)

**Constat :** la docstring de `fifo_policy.py` (T22) et la v3 disaient que la règle n'était « pas un premier arrivé, premier servi ». La décision du 16 octobre de la renommer dans l'article en découlait, et la question avait été reposée à la directrice (message du lot 1, point 4).

**Vérification dans le code :** pour une pelle déjà libre, `MineEnv.forecast_free_at` donne l'instant, passé, où elle s'est libérée. Cet instant n'est pas ramené à l'heure courante. Quand plusieurs pelles attendent, `argmin` choisit donc celle qui attend un camion **depuis le plus longtemps**. C'est la définition FIFO du dispatching donnée par l'article (§2, « sert en priorité l'unité de chargement qui attend un camion depuis le plus longtemps »). Si toutes les pelles sont occupées, la règle choisit celle qui se libérera le plus tôt. Le « premier arrivé, premier servi » des files d'attente (ordre de service des camions à une pelle) est respecté par le simulateur pour toutes les politiques : ce n'est pas une règle de dispatching.

**Décision :** garder le nom FIFO, sans renommage. Le point 4 du message du lot 1 est réglé.

**Ce que j'ai changé :**
- `baselines/fifo_policy.py` : docstring corrigée.
- Article v3 : §5.7 et `tab:baselines`, avec la définition précise.
- Tests (`tests/test_baselines.py`) :
  - parmi deux pelles libres, la règle choisit celle qui attend depuis le plus longtemps ;
  - dans le simulateur, sur un poste complet, à chaque décision où au moins deux pelles attendent, la règle choisit celle qui attend depuis le plus longtemps.


## Campagne : déroulement, et carburant par tonne d'un poste sans tonne (28 octobre 2026)

**Déroulement :**
- entraînement lancé le 23 octobre, 90 modèles sur 90, sans échec, fini le 27 octobre à 23 h 34. Quatre DQN de l'ablation ont pris environ 7 h 30 au lieu de 3 h (machine ralentie l'après-midi du 27) : cela ne touche que la durée ;
- évaluation **unique** (graines 42 à 71) le 28 octobre, de 1 h 10 à 5 h 28, sans erreur : `campagne/benchmark_20261028_011004.csv`, 6 540 lignes, soit le nombre attendu ;
- analyses le 28 octobre à 5 h 40 : `campagne/analyses_20261028_0540_carburant_1e12/` (gardées pour la trace).

**Constat :** dans l'ablation w4 = 0,5, deux DQN (graines d'entraînement 2 et 4) choisissent ATTENDRE dans 100 % des décisions et ne transportent **aucune tonne** : 60 épisodes à 0 t. `simulation/kpi.py` divisait le carburant par `max(tonnage, 1e-9)`, d'où 1,08e12 L/t pour ces épisodes, et une moyenne DQN de 4,3e11 L/t dans le tableau d'agrégation.

**Décision :** le carburant par tonne d'un poste sans tonne **n'est pas défini**. Il vaut NaN. Ce n'est pas une décision prise sur les résultats : la valeur 1e12 est un artefact de calcul, pas une mesure.

**Ce que j'ai changé :**
- `simulation/kpi.py` : `specific_fuel_l_per_ton` = NaN si le tonnage est nul (pour les évaluations futures) ;
- `analysis/statistical_tests.py` : `undefined_fuel_as_nan`, appelée par `load_results`, remet à NaN les valeurs des CSV écrits avant ce jour, dont celui de la campagne. Le CSV de la campagne n'est pas modifié ;
- `analysis/aggregate_seeds.py` : un run sans valeur définie est exclu du résumé et compté dans une nouvelle colonne `n_undefined` ;
- `statistical_tests.main` : refuse une valeur non définie dans les régimes principaux, au lieu de la laisser ignorer en silence par la moyenne sur les graines d'entraînement (aucune dans la campagne) ;
- tests : `test_wait_decomposition.py` (NaN sans tonne, valeur exacte sinon), `test_aggregate_seeds.py` (cas de l'ablation : 3 runs résumés, 2 exclus ; aucun run défini ; refus des tests appariés). Suite complète : **413 réussis, 1 ignoré**.

**Analyses relancées** (`campagne/analyses/`, sans rejouer aucune graine) : tests appariés, transfert, ablation et courbes **identiques octet pour octet**. Agrégation : seule la ligne DQN de `ablation_w4_05` change pour le carburant (3 runs résumés, 2 non définis, IQM 0,271 L/t au lieu de 3,6e11) ; la colonne `n_undefined` est ajoutée (0 partout ailleurs).

**Pour l'article :** dire que 2 des 5 DQN de l'ablation w4 = 0,5 ne transportent rien, et que leur carburant par tonne n'est pas défini.


## T31 lot 2 : article réécrit avec les résultats de la campagne (28 octobre 2026)

**Correction du déroulement noté plus haut :** la campagne a été lancée le **22 octobre à 14 h 35** (création de `configs_campagne`), et non le 23 ; fin le 27 octobre à 23 h 34, soit 5 jours et 9 heures, 515 heures d'entraînement cumulées.

**Ce que j'ai changé dans le code :**
- `experiments/run_benchmark.py` : `policy_specs` (les politiques d'un scénario, extraites de `run_benchmark`, avec `train_missing=False` pour ne jamais entraîner) ; `policy_from_spec` (ex-`_policy_fn`) ; `_run_tasks` accepte un autre traitement d'épisode (`worker`).
- `analysis/behavior.py` : rejeu des épisodes de la campagne avec le journal des cycles (`campaign_runs`, `campaign_behaviour`), et `check_replay`, qui compare chaque épisode rejoué au CSV de l'évaluation (productivité, attente en file, carburant ; égalité exacte) et n'écrit rien sinon. CSV lu avec `float_precision="round_trip"` : le lecteur rapide de pandas s'écartait d'un bit (écarts de 1e-16 en relatif sur la répétition). `behaviour_summary` résume aussi par mode.
- `experiments/campaign.py` : étape `behaviour` (sortie `analyses/comportement/`), ajoutée à la répétition.
- `plots/campaign_curves.py` (nouveau) : courbes d'apprentissage de la campagne, un panneau par régime, moyenne ± IC sur 5 modèles.
- `plots/benchmark_comparisons.py` : moyenne sur les graines d'entraînement avant l'IC, régimes principaux seulement. `plots/behaviour.py` : agents spécialisés seulement. `plots/common.py` : libellés MTWT et LP, comme dans l'article.
- Tests : rejeu identique à l'évaluation, refus d'un rejeu différent ou incomplet, refus sans modèle (`test_transfer.py`) ; courbes de la campagne et figures avec 5 graines (`test_benchmark_plots.py`). Suite complète : **416 réussis, 1 ignoré** (avant les tests de figures).
- Vérification réelle : rejeu de la répétition du 22/10 (graines 30 000-30 029), **3 540 épisodes identiques** à son évaluation.

**Article v3 (`reports/article/RL_article_FR_v3.tex`) :**
- §7 réécrit en entier avec la campagne : productivité (10 politiques), cycles et récompense, attente (file, ATTENDRE, totale), Pareto (attente et carburant), robustesse, transfert (avec régimes non vus), 8 comparaisons, variabilité entre modèles (`tab:runs`, nouveau), convergence sur 5 modèles, ablation sur 5 modèles ;
- §8 : réponses aux QR, hypothèses (H1 et H2 réfutées, H3 non validée), interprétation, implications, durabilité (`tab:fuel` avec L/h par camion), limites (DQN, agents tabulaires, rejeu) ; §9 conclusion réécrite ;
- résumé complété ; « huit politiques » → dix (objectif, revendications, C2, cadre) ; phrase de l'état de l'art qui disait MTWT et LP « non inclus » corrigée ; cellules rouges de la §6 (durée de la campagne, taille des modèles) ;
- figures de la campagne dans `reports/article/v3/figures` (rien dans `reports/figures`) ; figures de décomposition de l'attente (aucun code) et du front de Pareto (illisible avec dix politiques) retirées au profit des tableaux ;
- reste : `tab:behaviour`, son commentaire et sa figure, **après le rejeu**.

**Ce que j'ai appris en écrivant :** la tête du classement ne change pas d'un régime à l'autre (MTWT, puis PPO et LP) ; l'interaction significative vient du bas du classement. H2 est donc réfutée telle qu'elle était écrite, même si l'interaction est significative.


## Rejeu du comportement, figures et version anglaise (28 octobre 2026)

**Rejeu :** `campaign behaviour`, du 28/10 7 h 40 à 10 h 16 : 3 540 épisodes rejoués (nominal 780, forte charge 1 380, fortes pannes 1 380), **tous identiques à l'évaluation** (productivité, file, carburant ; égalité exacte). Sorties : `campagne/analyses/comportement/`.

**Constats nouveaux :**
- le seul DQN nominal qui fonctionne (graine 2, 3 932 t/h) envoie 99 % de ses cycles à la pelle 1 : il reproduit la Pelle la plus proche (3 893 t/h) ;
- PPO suit le profil de MTWT mais utilise moins la pelle 3 (4 % contre 15 % en nominal ; 7 % contre 14 % en fortes pannes), d'où sa consommation plus faible par tonne ;
- SARSA envoie 99 % de ses cycles à la pelle 3 (nominal, fortes pannes) ; Plus court chemin 71 à 74 % à la pelle 1 ; FIFO et l'Affectation fixe environ un tiers à la pelle 3.

**Article :** `tab:behaviour`, son commentaire et sa figure remplis dans les deux versions ; trois phrases reliées (variabilité, interprétation PPO et DQN). Plus aucune case « à compléter ».

**Figures :**
- `plots/scenarios_schematic.py` : la figure des régimes ne dessine plus de camions « en panne » (10 % est une probabilité par cycle) ; elle ajoute le facteur d'appariement et reprend les noms de l'article ; le dossier de sortie est obligatoire. **Incident :** l'ancienne version a réécrit une fois `reports/figures/fig3_scenarios.png` (lancée par erreur après l'échec d'une modification) ; le fichier a été restauré depuis git aussitôt.
- `plots/common.py` : `set_language` / `tr` et l'option `--lang en` dans les scripts de figures. Les figures de la version anglaise sont dans `reports/article/v3/en/figures/` (graphicspath anglais : `v3/en/` d'abord).
- `fig:architecture` supprimée (doublon de `fig:mdp`).
- Les textes des figures draw.io à refaire (`fig:simulation`, `fig:mdp`) et de la figure à créer (`fig:framework`) sont dans `docs/textes_figures_drawio.md`.

**Version anglaise :** `RL_article.tex` est la traduction complète de la v3. C'est la version que la directrice corrige sur Overleaf (`docs/overleaf_RL_article.zip`) et la version soumise. Bibliographie réunie dans un seul `RL_references.bib`. Marques de révision coupées dans les deux versions (`\revmarksfalse`), à la demande de l'étudiant ; les changements restent recensés dans ce journal.


## Rangement après la campagne (28 octobre 2026)

- **Journaux d'entraînement** (`experiments/campaign.py`) : le fichier de journal est ouvert ligne par ligne (`buffering=1`). Avec le tampon par défaut (8 Ko), la sortie de Stable-Baselines3 n'apparaissait qu'à la fin de chaque modèle pendant la campagne. Test : une ligne écrite sans flush (stdout et stderr) est lisible avant la fin de l'entraînement ; il échoue sans la correction.
- **README du dépôt** : résultat de la campagne (hypothèse réfutée), tableau des résultats, commandes pour reproduire (environnement, campagne, figures), organisation du dépôt.
- **Annexes du mémoire** (`reports/memoire/rl_thesis.tex`) : versions exactes, modules, commandes, hyperparamètres, régimes (MF 0,35 / 0,52 au lieu de 4,0 / 6,0, régimes non vus et ablation), structure du code, sortie de `--check-env` (23 composantes), format du CSV. L'annexe des tests de Tukey (résultats v1) est marquée en commentaire « à refaire avec le chapitre 6 ». Le mémoire compile sans erreur ; le PDF du mémoire n'a pas été régénéré (corps encore en v1).
