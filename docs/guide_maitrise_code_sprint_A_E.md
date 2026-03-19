# Guide de maitrise totale du code (Sprint A -> Sprint E)

Ce document est ton compagnon de travail pour maitriser en profondeur chaque fichier, chaque variable, chaque constante et chaque logique du projet, sprint par sprint.

Objectif final:
- Etre capable d'expliquer et defendre chaque ligne de code a la soutenance.
- Etre capable de modifier le systeme sans casser la coherence globale.
- Etre capable de justifier tous les choix techniques et metier.

---

## 1. Regle d'or de maitrise

Pour dire que tu maitrises une ligne de code, tu dois pouvoir repondre a ces 4 questions:
1. Que fait cette ligne ?
2. Pourquoi cette ligne existe ?
3. Que se passe-t-il si je la supprime ou la change ?
4. Quelle alternative correcte puis-je proposer ?

Si une de ces reponses manque, la ligne n'est pas encore maitrisee.

---

## 2. Methode unique a appliquer a chaque fichier

A appliquer pour tous les sprints.

### Etape 1: Cartographie du fichier
- Nom du fichier.
- Role du fichier en 1 phrase.
- Fonctions/classes presentes.
- Entrees/sorties principales.

### Etape 2: Lecture active par blocs
- Lire 20 a 40 lignes maximum.
- Reformuler chaque bloc en francais simple dans ton carnet.
- Identifier:
  - variables d'etat,
  - constantes,
  - conditions critiques,
  - hypothese cachee.

### Etape 3: Execution et observation
- Lancer le code.
- Observer les valeurs cles a chaque etape (print/debugger).
- Noter les transitions importantes.

### Etape 4: Micro-modifications de comprehension
- Modifier une constante.
- Modifier une condition.
- Supprimer une ligne non critique.
- Observer l'impact et expliquer le pourquoi.

### Etape 5: Validation par mini-tests
- Test nominal.
- Test limite.
- Test erreur (entree invalide).

### Etape 6: Restitution sans ecran
- Expliquer le fichier a voix haute sans regarder le code.
- Verifier ensuite les oublis.

### Etape 7: Fiche finale du fichier
- Resume de logique.
- Variables critiques.
- Constantes et justification.
- Risques/limites.
- Points a surveiller si refactor.

---

## 3. Routine de travail quotidienne

Temps recommande par session (2h30 environ):
1. 45 min: lecture active d'un fichier.
2. 35 min: execution + debug.
3. 30 min: micro-modifications.
4. 25 min: tests.
5. 15 min: synthese ecrite.

Regle: 1 a 2 fichiers max par session pour garder de la profondeur.

---

## 4. Plan de maitrise par sprint

## Sprint A (MVP environnement)
Fichiers cibles:
- simulation/graph_model.py
- simulation/entities.py
- simulation/fuel_model.py
- simulation/events.py
- simulation/kpi.py
- simulation/run_mvp.py

Ce que tu dois maitriser a 100%:
- modelisation graphe (noeuds/arcs/attributs).
- cycle camion complet (affectation -> chargement -> trajet -> dechargement -> retour).
- stochasticite minimale.
- calcul des KPI et generation CSV.

Exercice de validation:
- Predire l'effet de +10% sur pente moyenne avant execution.
- Predire l'effet de panne = 0% puis 10%.
- Expliquer pourquoi la productivite varie.

---

## Sprint B (interface RL + validation environnement)
Fichiers cibles (a ajouter/maitriser):
- env/mine_env.py
- env/observation_builder.py
- env/action_mask.py
- baselines/random_policy.py
- baselines/shortest_path_policy.py

Ce que tu dois maitriser a 100%:
- reset()/step() et dynamique d'episode.
- observation_space et action_space.
- logique de masquage des actions invalides.
- baseline random vs shortest-path.

Exercice de validation:
- Expliquer un step complet, entrees et sorties exactes.
- Montrer un cas d'action invalide et son masquage.

---

## Sprint C (MDP stabilise + baselines metier)
Fichiers cibles:
- baselines/queue_aware_policy.py
- reward module (si separe)
- scenarios de reference

Ce que tu dois maitriser a 100%:
- chaque terme de la fonction de recompense et ses poids.
- compromis productivite / attente / carburant / congestion.
- comportements differents des baselines.

Exercice de validation:
- Justifier mathematiquement la recompense.
- Defendre pourquoi un poids augmente ou diminue.

---

## Sprint D (agent RL)
Fichiers cibles:
- rl/train_ppo.py
- rl/evaluate_agent.py
- rl/callbacks.py
- config d'entrainement

Ce que tu dois maitriser a 100%:
- pipeline d'entrainement.
- hyperparametres critiques.
- interpretation des learning curves.
- causes classiques d'instabilite.

Exercice de validation:
- Expliquer pourquoi tu choisis PPO (ou autre).
- Expliquer l'impact d'un changement de learning rate.

---

## Sprint E (experimentation et analyse)
Fichiers cibles:
- experiments/scenarios.py
- experiments/run_benchmark.py
- experiments/stats_report.py
- notebooks/figures de resultat

Ce que tu dois maitriser a 100%:
- protocole experimental complet.
- reproductibilite (seed, config, logs).
- construction des tableaux comparatifs.
- analyse statistique minimale.

Exercice de validation:
- Defendre la validite de tes resultats face au jury.
- Expliquer les limites et menaces a la validite.

---

## 5. Template de fiche a remplir pour chaque fichier

Copie-colle ce bloc pour chaque fichier.

## Fiche fichier: <nom_fichier>
- Role du fichier:
- Dependances:
- Fonctions/classes:

### Fonction/Classe: <nom>
- Entrees:
- Sorties:
- Variables critiques:
- Constantes:
- Hypotheses:
- Cas limites:
- Risque de bug:

### Ligne sensible (si applicable)
- Ligne:
- Pourquoi elle est critique:
- Effet si modifiee:

### Validation personnelle
- [ ] Je peux l'expliquer sans ecran.
- [ ] Je peux predire l'effet d'une modification.
- [ ] J'ai teste cas nominal + limite + erreur.

---

## 6. Checklist de maitrise finale (avant soutenance)

### Niveau fichier
- [ ] Chaque fichier a une fiche complete.
- [ ] Chaque fonction est reformulee en francais simple.
- [ ] Chaque constante est justifiee.

### Niveau sprint
- [ ] Je peux expliquer tout le flux du sprint de memoire.
- [ ] Je peux reproduire un run complet sans aide.
- [ ] Je peux diagnostiquer une anomalie typique.

### Niveau projet
- [ ] Je peux relier code -> KPI -> conclusion memoire.
- [ ] Je peux justifier les choix devant un jury technique.
- [ ] Je peux vulgariser pour un jury non specialiste.

---

## 7. Strategie de progression (recommandee)

Ordre de maitrise:
1. Sprint A complet.
2. Sprint B complet.
3. Sprint C complet.
4. Sprint D complet.
5. Sprint E complet.

Ne passe au sprint suivant que si la checklist du sprint courant est validee.

---

## 8. Regle anti-oubli

A la fin de chaque session, ecris 5 lignes:
1. Ce que j'ai compris aujourd'hui.
2. Ce qui reste flou.
3. Une question technique pour demain.
4. Un test que je dois encore faire.
5. Une phrase de soutenance que je sais maintenant dire clairement.

Ce rituel transforme la lecture en maitrise durable.
