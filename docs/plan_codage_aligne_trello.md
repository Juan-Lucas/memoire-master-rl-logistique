# Plan de codage aligne Trello + memoire

Contexte actuel:
- Chapitres 1 et 2: en correction.
- Chapitre 3: redaction en cours.
- Objectif: demarrer le codage maintenant avec une trajectoire claire et defendable dans le memoire.

References de pilotage:
- Trello: phases 2, 3, 4, 5.
- Memoire: chapitres 3, 4, 5, 6.

---

## 1. Alignement global (Trello -> Code -> Memoire)

### Phase 2 Trello (Simulation & Environnement)
- Cartes cibles:
  - Developpement du graphe routier
  - Modelisation physique et evenements dynamiques
  - Creation de l'environnement Gymnasium
  - Validation de l'environnement
- Livrable code attendu:
  - environnement simulable minimal (MVP) executable.
- Sections memoire alimentees:
  - Chapitre 3 (modelisation)
  - Chapitre 4.2 (conception de l'environnement)
  - Chapitre 5.2 (details implementation environnement)

### Phase 3 Trello (Modelisation & Baselines)
- Cartes cibles:
  - Formalisation RL (MDP)
  - Implementation agent RL
  - Implementation baselines
  - Chapitre 3 (methodologie)
- Livrable code attendu:
  - MDP operationnel + baselines + premier entrainement RL.
- Sections memoire alimentees:
  - Chapitre 4.3, 4.4, 4.5
  - Chapitre 5.3

### Phase 4 Trello (Experimentation & Evaluation)
- Cartes cibles:
  - Protocole experimental
  - Simulations comparatives
  - Analyse statistique et visualisation
- Livrable code attendu:
  - pipeline de tests reproductibles + graphes + tableaux KPI.
- Sections memoire alimentees:
  - Chapitre 5.4, 5.5
  - Chapitre 6.1, 6.2, 6.3, 6.4

### Phase 5 Trello (Redaction & Synthese)
- Cartes cibles:
  - Demonstrateur
  - Relecture finale
  - Preparation soutenance
- Livrable code attendu:
  - script demo (ou Streamlit simple) + figures finales.
- Sections memoire alimentees:
  - Conclusion + annexes + slides soutenance.

---

## 2. Architecture de code cible (simple et robuste)

## 2.1 Dossiers proposes
- memoire_master_rl_logistique/simulation/
  - entities.py
  - graph_model.py
  - events.py
  - fuel_model.py
  - kpi.py
- memoire_master_rl_logistique/env/
  - mine_env.py
  - observation_builder.py
  - action_mask.py
- memoire_master_rl_logistique/baselines/
  - random_policy.py
  - shortest_path_policy.py
  - queue_aware_policy.py
- memoire_master_rl_logistique/rl/
  - train_ppo.py
  - evaluate_agent.py
  - callbacks.py
- memoire_master_rl_logistique/experiments/
  - scenarios.py
  - run_benchmark.py
  - stats_report.py
- memoire_master_rl_logistique/utils/
  - config.py
  - seed.py
  - io.py

## 2.2 Principes techniques
- Tout scenario est configurable via fichier (yaml/json).
- Toutes les executions journalisent:
  - seed,
  - config,
  - version code,
  - metriques.
- Les sorties vont vers un dossier unique par run:
  - logs,
  - csv,
  - figures.

---

## 3. Plan de codage operationnel (ordre recommande)

## Sprint A (demarrage immediat - MVP environnement)
Objectif: obtenir un environnement qui tourne de bout en bout.

Taches:
1. Implementer le graphe routier (noeuds + arcs + attributs distance/pente/etat).
2. Implementer les entites minimales (camion, pelle, point de dechargement).
3. Implementer cycle operationnel minimal:
   - affectation,
   - chargement,
   - trajet,
   - dechargement,
   - retour.
4. Ajouter stochasticite minimale:
   - temps de trajet,
   - temps de chargement/dechargement,
   - panne simple.
5. Ajouter calcul KPI de base:
   - productivite,
   - attente,
   - utilisation,
   - consommation.

Definition de fini:
- un script lance 1 episode complet sans erreur.
- un csv de KPI est genere.
- tu peux decrire ce MVP dans chapitre 3/4.

## Sprint B (API Gymnasium + validation)
Objectif: rendre l'environnement utilisable par RL.

Taches:
1. Implementer reset() et step().
2. Definir observation_space et action_space.
3. Implementer masquage des actions invalides.
4. Ajouter agent random et policy naive shortest-path.
5. Valider la logique avec tests simples de coherence.

Definition de fini:
- un agent random peut interagir N episodes.
- la baseline shortest-path tourne et produit des KPI comparables.

## Sprint C (baselines metier + MDP final)
Objectif: verrouiller le referentiel avant RL avance.

Taches:
1. Implementer baseline queue-aware (minimum attente estimee).
2. Finaliser la recompense ponderee (production, carburant, attente, congestion).
3. Versionner 2-3 variantes de reward pour comparaison.
4. Fixer scenarios experimentaux de reference.

Definition de fini:
- tableau comparatif baselines disponible.
- version MDP stabilisee pour chapitre 4.

## Sprint D (agent RL)
Objectif: premier entrainement exploitable.

Taches:
1. Integrer PPO (SB3) avec seed fixe.
2. Lancer entrainement court pour verifier apprentissage.
3. Sauvegarder modele + courbes d'apprentissage.
4. Evaluer sur scenarios identiques aux baselines.

Definition de fini:
- learning curve exploitable.
- evaluation RL vs baselines sur les memes KPI.

## Sprint E (experimentation rigoureuse)
Objectif: produire les resultats pour chapitre 5/6.

Taches:
1. Definir protocole officiel (scenarios, repetitions, seeds).
2. Executer campagne comparative complete.
3. Produire tableaux + figures finales.
4. Faire analyse statistique minimale:
   - moyenne,
   - ecart-type,
   - intervalle de confiance.

Definition de fini:
- pack resultats pret pour chapitre 6.
- figures de soutenance pretes.

---

## 4. Couplage redaction/codage (important)

Regle de travail:
- Quand tu termines un sprint, tu rediges immediatement la sous-section memoire correspondante.

Mapping direct:
1. Sprint A-B -> Chapitre 4.2 + 5.2
2. Sprint C -> Chapitre 4.3 + 4.5
3. Sprint D -> Chapitre 4.4 + 5.3
4. Sprint E -> Chapitre 5.4 + 5.5 + Chapitre 6

Avantage:
- evite d'accumuler du retard de redaction.
- garde une coherence parfaite entre ce qui est code et ce qui est ecrit.

---

## 5. Checklist de qualite (pour viser une tres bonne note)

## 5.1 Qualite code
- [ ] Code organise par modules clairs.
- [ ] Parametres centralises dans config.
- [ ] Seeds fixes et tracees.
- [ ] Aucune comparaison unfair entre methodes.

## 5.2 Qualite experimentale
- [ ] Meme protocole pour RL et baselines.
- [ ] KPI calcules de la meme maniere.
- [ ] Plusieurs runs par scenario.
- [ ] Resultats sauvegardes et reproductibles.

## 5.3 Qualite memoire
- [ ] Chaque choix de modelisation est justifie.
- [ ] Chaque figure a un message clair.
- [ ] Limites discutees honnetement.
- [ ] Lien explicite entre gains KPI et valeur metier.

---

## 6. Prochaine action concrete (maintenant)

Demarrer Sprint A avec ces 3 premieres taches:
1. coder graph_model.py,
2. coder entities.py,
3. coder un script run_mvp.py qui execute un episode et exporte les KPI.

Une fois ces 3 taches faites, enchaîner avec reset()/step() pour basculer rapidement vers Sprint B.