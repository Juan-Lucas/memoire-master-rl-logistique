# FAQ - Questions probables du jury et réponses préparées

Ce document compile les questions les plus probables du jury, classées par catégorie, avec des réponses structurées.

---

## Catégorie 1 : Motivation et contexte

### Q1 : Pourquoi avoir choisi ce sujet ?

**Réponse courte :** La logistique minière représente 50-60% des coûts d'exploitation. Les méthodes actuelles sont soit myopes (heuristiques) soit déterministes (systèmes industriels). Le RL offre une alternative adaptative et robuste.

**Réponse détaillée :**
- Impact économique majeur : 50-60% des coûts d'exploitation des mines à ciel ouvert
- Limites des méthodes actuelles :
  - Heuristiques (FIFO, shortest path) : optimisation locale sans vision globale
  - Systèmes industriels (DISPATCH) : déterministes, ne s'adaptent pas aux perturbations
- Opportunité du RL : capacité d'apprentissage, adaptation dynamique, vision globale
- Pertinence académique et industrielle : sujet à l'intersection RL et logistique

### Q2 : Quelle est l'innovation de votre travail ?

**Réponse courte :** Formalisation MDP du dispatching minier, implémentation d'un environnement de simulation réaliste, comparaison systématique de 8 approches, démonstration de gains significatifs.

**Réponse détaillée :**
- **Formalisation MDP** : première formalisation complète du problème comme MDP avec récompense multi-objectif (rendement, équité, coût)
- **Environnement réaliste** : simulation Gymnasium avec stochastique réaliste (temps de trajet lognormaux, pannes)
- **Comparaison systématique** : 8 approches comparées (4 baselines classiques + 4 RL)
- **Résultats quantifiés** : gains de 10-25% sur différents KPIs, robustesse démontrée sur scénarios perturbés

---

## Catégorie 2 : Modélisation MDP

### Q3 : Pourquoi avoir choisi un MDP et pas un autre formalisme ?

**Réponse courte :** Le MDP est le cadre standard pour la décision séquentielle stochastique. Il capture parfaitement la nature dynamique et incertaine du dispatching minier.

**Réponse détaillée :**
- **Décision séquentielle** : chaque décision affecte les décisions futures (effet mémoire)
- **Stochastique** : temps de trajet, pannes, congestions sont intrinsèquement aléatoires
- **Cadre théorique solide** : Bellman, théorie de la convergence, algorithmes éprouvés
- **Alternatives moins adaptées** :
  - Programmation linéaire : suppose déterminisme
  - Programmation dynamique : explosion combinatoire
  - Méta-heuristiques : pas de garantie d'optimalité

### Q4 : Comment avez-vous défini l'espace d'état ? Pourquoi ces variables ?

**Réponse courte :** L'état s_t = ({q_p}, {x_c}, {z_r}, t_courant) capture toutes les informations nécessaires pour prendre une décision optimale.

**Réponse détaillée :**
- **Files d'attente {q_p}** : indiquent la congestion à chaque pelle, crucial pour équilibrage
- **Position/statut {x_c}** : où sont les camions et leur état (vide, plein, en route)
- **Disponibilité {z_r}** : quelles ressources sont opérationnelles
- **Temps courant** : pour gérer les contraintes temporelles (shifts, pauses)
- **Justification** : ces variables sont Markoviennes (suffisantes pour l'optimalité)
- **Normalisation** : toutes les observations normalisées dans [0,1] pour stabiliser l'apprentissage

### Q5 : Pourquoi une récompense multi-objectif ? Comment arbitrez-vous les composantes ?

**Réponse courte :** La récompense combine rendement (w1=1.0), équité (w2=0.1) et coût (w3=0.05). Les poids reflètent la priorité industrielle (productivité d'abord).

**Réponse détaillée :**
- **Rendement** : objectif principal, maximiser le tonnage transporté
- **Équité** : éviter que certaines pelles soient surchargées (syndrome d'arrivée précoce)
- **Coût** : pénaliser les trajets coûteux (distance, consommation)
- **Choix des poids** :
  - w1=1.0 : priorité absolue à la productivité
  - w2=0.1 : équité comme contrainte secondaire
  - w3=0.05 : coût comme contrainte tertiaire
- **Validation** : sensibilité aux poids testée, compromis acceptable

---

## Catégorie 3 : Algorithmes RL

### Q6 : Pourquoi PPO et pas un autre algorithme ?

**Réponse courte :** PPO offre le meilleur compromis stabilité/efficacité pour ce problème. Moins sensible aux hyperparamètres que DQN, mieux adapté aux actions discrètes que SAC.

**Réponse détaillée :**
- **Stabilité** : PPO utilise un clipping ratio qui évite les changements drastiques de politique
- **Efficacité** : convergence rapide, moins d'épisodes nécessaires que DQN
- **Actions discrètes** : PPO gère naturellement les espaces d'action discrets (contrairement à SAC)
- **Moins sensible aux hyperparamètres** : bon comportement par défaut
- **State-of-the-art** : reconnu comme l'un des meilleurs algorithmes pour RL continu/discret
- **Comparaison** : Q-Learning/SARSA tabulaires limités par la taille de l'espace d'état, DQN moins stable

### Q7 : Pourquoi avoir implémenté Q-Learning et SARSA tabulaires ?

**Réponse courte :** Pour comparer RL tabulaire (simple, interprétable) vs RL profond (plus puissant). C'est une baseline méthodologique.

**Réponse détaillée :**
- **Baseline méthodologique** : comparer approches simples vs sophistiquées
- **Interprétabilité** : tables Q directement inspectables
- **Vérification** : s'assurer que le problème n'est pas trop complexe pour RL tabulaire
- **Résultats** : Q-Learning/SARSA fonctionnent mais DQN/PPO performent mieux (espace d'état trop grand)
- **Contribution** : cette comparaison enrichit l'analyse méthodologique

### Q8 : Comment avez-vous choisi les hyperparamètres ?

**Réponse courte :** Valeurs standards de la littérature (PPO : α=0.0003, γ=0.99, ε_clip=0.2) ajustées par grid search limité.

**Réponse détaillée :**
- **Valeurs initiales** : standards de la littérature (Schulman et al., 2017)
- **Ajustements empiriques** :
  - Learning rate : testé 0.0001, 0.0003, 0.001 → 0.0003 optimal
  - Batch size : testé 32, 64, 128 → 64 bon compromis
  - γ : gardé 0.99 (standard)
- **Grid search limité** : 3-4 valeurs par paramètre pour temps raisonnable
- **Validation** : performance sur scénarios de test, pas seulement d'entraînement

---

## Catégorie 4 : Implémentation et simulation

### Q9 : Comment validez-vous la réalisme de votre simulation ?

**Réponse courte :** Paramètres justifiés par la littérature (Afrapoli, Mohtasham), distributions réalistes (lognormales), scénarios de perturbation pour robustesse.

**Réponse détaillée :**
- **Sources littéraires** :
  - Capacités camions : Caterpillar 785C (140 t)
  - Temps de cycle : Hitachi 2500, Afrapoli & Askari-Nasab (2017)
  - Vitesse : LOGN(32, 26) km/h (Table 4.4, Afrapoli 2019)
- **Distributions réalistes** :
  - Temps de trajet : lognormales (captures skewness et positive)
  - Temps de chargement : normales N(2, 0.3) min
- **Scénarios de robustesse** : high-breakdown, single-shovel, short-shift
- **Limite** : validation sur données de terrain nécessaire pour extrapolation

### Q10 : Pourquoi Gymnasium et pas une autre librairie ?

**Réponse courte :** Gymnasium est le standard de facto pour RL, interface simple (reset/step/render), compatible avec toutes les librairies RL (Stable-Baselines3, RLlib).

**Réponse détaillée :**
- **Standard industriel** : OpenAI Gym → Gymnasium, adopté par la communauté
- **Interface simple** : reset(), step(), render() pour tous les environnements
- **Compatibilité** : fonctionne avec Stable-Baselines3, RLlib, Ray
- **Écosystème** : nombreux environnements existants, documentation riche
- **Personnalisation** : facile de créer des environnements custom

### Q11 : Comment gérez-vous le masquage des actions ?

**Réponse courte :** Toutes les paires (pelle, dump) sont considérées valides car l'agent peut choisir d'attendre la disponibilité. Pas de masquage strict.

**Réponse détaillée :**
- **Choix architectural** : espace d'action Discrete(|P| × |D| + 1)
- **Action d'attente** : l'action |P| × |D| correspond à "attendre"
- **Justification** :
  - Simplifie l'implémentation
  - L'agent apprend à attendre quand nécessaire
  - Évite de masquer des actions potentiellement utiles
- **Alternative non choisie** : masquer les paires (pelle, dump) avec pelle indisponible → réduit expressivité

---

## Catégorie 5 : Résultats et évaluation

### Q12 : Quels sont vos résultats principaux ?

**Réponse courte :** PPO surpasse les baselines sur la plupart des KPIs : +15% productivité vs FIFO, -25% temps d'attente vs Shortest Path, -10% consommation vs Fixed Assignment.

**Réponse détaillée :**
- **Scénario nominal** :
  - Productivité : PPO 8000 t/h vs FIFO 6950 t/h (+15%)
  - Temps d'attente : PPO 3.2 min vs Shortest Path 4.3 min (-25%)
  - Consommation : PPO 0.85 L/t vs Fixed Assignment 0.94 L/t (-10%)
- **Robustesse** : PPO maintient >90% de performance sur scénarios perturbés
- **Significativité** : gains statistiquement significatifs (p < 0.05) sur 10 réplications

### Q13 : Comment comparez-vous avec les baselines ?

**Réponse courte :** Comparaison systématique sur mêmes scénarios avec mêmes seeds. FIFO : simple mais myope, Shortest Path : optimise distance ignore files, Fixed Assignment : baseline zéro-niveau, Nearest Shovel : glouton géographique.

**Réponse détaillée :**
- **FIFO** : équitable mais ne tient compte ni de distance ni de congestion
- **Shortest Path** : optimise localement le coût, sans vision globale
- **Fixed Assignment** : baseline la plus simple (cyclique), aucune adaptation
- **Nearest Shovel** : minimise distance, ignore files d'attente
- **PPO** : vision globale, adaptation dynamique, équilibre rendement/équité/coût
- **Protocole** : mêmes scénarios, mêmes seeds, 10 réplications pour statistiques

### Q14 : Quelle est la signification statistique de vos résultats ?

**Réponse courte :** Gains significatifs (p < 0.05) sur 10 réplications avec intervalles de confiance à 95%. Effet de taille (Cohen's d) > 0.8 pour la plupart des comparaisons.

**Réponse détaillée :**
- **Test statistique** : t-test de Student apparié (PPO vs baseline)
- **Niveau de signification** : p < 0.05 pour tous les gains principaux
- **Intervalles de confiance** : IC 95% calculés sur 10 réplications
- **Effet de taille** : Cohen's d > 0.8 (effet large) pour productivité et temps d'attente
- **Reproductibilité** : seeds fixées, protocole documenté

---

## Catégorie 6 : Limites et perspectives

### Q15 : Quelles sont les limites de votre approche ?

**Réponse courte :** Temps d'entraînement, sensibilité aux hyperparamètres, dépendance à la simulation, généralisation à mines non vues à valider.

**Réponse détaillée :**
- **Temps d'entraînement** : ~1-2 heures sur GPU, acceptable mais non négligeable
- **Sensibilité hyperparamètres** : nécessite tuning pour nouvelles configurations
- **Dépendance simulation** : validité dépend de la qualité du simulateur
- **Généralisation** : agent entraîné sur mine X doit être fine-tuné pour mine Y
- **Topologie** : nombre de pelles/camions fixé, extension non testée
- **Données terrain** : validation sur données réelles nécessaire

### Q16 : Comment généralisez-vous à une mine différente ?

**Réponse courte :** L'agent apprend des patterns généraux (équilibrage files/demande). Fine-tuning rapide sur nouvelle mine. Limites : topologie, nombre d'équipements.

**Réponse détaillée :**
- **Patterns généraux appris** :
  - Équilibrage des files d'attente
  - Adaptation à la congestion
  - Arbitrage rendement/coût
- **Transfert** :
  - Transfer learning possible : initialiser avec poids pré-entraînés
  - Fine-tuning rapide (10-20 épisodes) sur nouvelle configuration
- **Limites** :
  - Topologie réseau différente (plus de pelles/camions)
  - Contraintes opérationnelles différentes (priorités de minerai)
  - Nécessite ré-entraînement partiel
- **Perspectives** :
  - Meta-learning : apprendre à apprendre rapidement
  - Domain randomization : entraîner sur distribution de configurations

### Q17 : Quelles sont les perspectives de recherche ?

**Réponse courte :** Extension à mines complexes, contraintes supplémentaires (priorités minerai), multi-agent pour coordination décentralisée, validation sur données terrain.

**Réponse détaillée :**
- **Extension complexité** :
  - Plus de pelles/camions (10+ pelles, 50+ camions)
  - Topologies réseau plus complexes
- **Contraintes supplémentaires** :
  - Priorités de minerai (haut grade vs bas grade)
  - Contraintes temporelles (shifts, pauses)
  - Maintenance planifiée
- **Multi-agent** :
  - Coordination décentralisée (chaque camion comme agent)
  - Communication entre agents
- **Validation terrain** :
  - Partenariat avec mine réelle
  - Données historiques pour validation
  - Pilote industriel

---

## Catégorie 7 : Questions techniques

### Q18 : Coût computationnel de l'entraînement ?

**Réponse courte :** Entraînement ~1-2 heures sur GPU standard (RTX 3080). Inférence temps réel (<1ms par décision). Comparable aux méthodes d'optimisation classiques.

**Réponse détaillée :**
- **Entraînement** :
  - 100 épisodes, ~10 min/épisode → ~1-2 heures total
  - GPU RTX 3080, batch size 64
  - Acceptable pour entraînement offline
- **Inférence** :
  - <1ms par décision (forward pass réseau)
  - Temps réel pour déploiement opérationnel
- **Comparaison** :
  - Méthodes d'optimisation (MILP) : temps variable, parfois >1min
  - Heuristiques : <1ms mais myopes
  - RL : compromis temps/qualité

### Q19 : Comment gériez-vous l'exploration vs exploitation ?

**Réponse courte :** PPO utilise une politique stochastique qui explore naturellement. Pas de ε-greedy explicite comme dans Q-Learning/SARSA.

**Réponse détaillée :**
- **PPO** : politique stochastique π(a|s) échantillonnée selon distribution
- **Entropie bonus** : encouragé par l'objectif PPO (évite convergence prématurée)
- **Q-Learning/SARSA** : ε-greedy avec ε=1.0→0.01 (décroissance linéaire)
- **Résultat** : équilibre automatique, pas de paramètre d'exploration explicite pour PPO

### Q20 : Pourquoi MLP et pas CNN/RNN ?

**Réponse courte :** L'état est un vecteur de features, pas une image ni une séquence. MLP est adapté et simple.

**Réponse détaillée :**
- **Nature de l'état** : vecteur de features (files, positions, disponibilités)
- **Pas d'information spatiale** : pas besoin de CNN (réseau routier déjà encodé)
- **Pas d'information séquentielle** : pas besoin de RNN (décision markovienne)
- **MLP 128×128** : compromis expressivité/complexité
- **Alternatives non nécessaires** : CNN/RNN ajouteraient complexité sans gain

---

## Catégorie 8 : Questions métier

### Q21 : Comment votre solution s'intègre-t-elle dans un FMS existant ?

**Réponse courte :** L'agent RL peut être intégré comme module de décision dans un FMS existant (DISPATCH, MineStar). Il remplace ou complète le module d'affectation temps réel.

**Réponse détaillée :**
- **Architecture d'intégration** :
  - FMS existant : planification globale + heuristique temps réel
  - RL : remplace l'heuristique temps réel
  - Communication : API REST ou interface directe
- **Avantages** :
  - Amélioration des décisions temps réel
  - Adaptation dynamique aux perturbations
  - Compatibilité avec planification globale existante
- **Déploiement** :
  - Entraînement offline sur données historiques
  - Déploiement online pour décisions temps réel
  - Monitoring et re-entraînement périodique

### Q22 : Impact économique de votre solution ?

**Réponse courte :** Réduction des coûts opérationnels de 10-15% (réduction consommation, meilleure productivité). ROI positif si coût d'entraînement < gains annuels.

**Réponse détaillée :**
- **Gains estimés** :
  - Productivité +15% → +15% tonnage/shift
  - Consommation -10% → -10% carburant
  - Temps d'attente -25% → meilleure utilisation équipements
- **Coûts** :
  - Développement : temps ingénieur
  - Entraînement : temps GPU (~2h)
  - Inférence : négligeable
- **ROI** : si mine produit 10k t/jour à $50/t, gain de $75k/jour → ROI > 100%
- **Perspectives** : valeur industrielle significative

---

## Catégorie 9 : Questions personnelles

### Q23 : Qu'avez-vous appris de ce projet ?

**Réponse courte :** Maîtrise RL (théorie + pratique), compréhension logistique minière, compétences en simulation et évaluation scientifique.

**Réponse détaillée :**
- **Technique** :
  - RL profond (PPO, DQN, Q-Learning, SARSA)
  - Simulation Gymnasium
  - Évaluation statistique
- **Domaine** :
  - Logistique minière (KPIs, contraintes opérationnelles)
  - FMS industriels (DISPATCH, MineStar)
- **Méthodologique** :
  - Recherche expérimentale
  - Protocoles de validation
  - Communication scientifique

### Q24 : Quels ont été les plus grands défis ?

**Réponse courte :** Stabilisation de l'apprentissage (reward shaping), validation de la simulation (paramètres réalistes), comparaison méthodologique (baselines vs RL).

**Réponse détaillée :**
- **Apprentissage** :
  - Reward shaping : trouver bons poids w1, w2, w3
  - Convergence : éviter oscillations, stabilité PPO
- **Simulation** :
  - Paramètres : trouver valeurs réalistes dans littérature
  - Distributions : choisir lognormales vs normales
- **Évaluation** :
  - Protocole : scénarios représentatifs, réplications
  - Statistiques : tests appropriés, intervalles de confiance

---

## Conseils pour répondre

### Avant de répondre
1. Écouter la question en entier
2. Prendre 2-3 secondes pour réfléchir
3. Reformuler si nécessaire ("Si j'ai bien compris, vous me demandez...")

### Structure de réponse
1. Réponse courte (1-2 phrases)
2. Réponse détaillée avec arguments
3. Exemple ou illustration si pertinent

### Si vous ne savez pas
- Honnêteté : "Je ne sais pas" ou "C'est une excellente question à laquelle je n'ai pas de réponse"
- Proposer : "Ce serait un point intéressant à explorer dans les perspectives"
- Ne jamais improviser ou inventer

---

## Exercices de préparation

**Exercice 1 : Répondre en 30 secondes**
Pratiquez les réponses courtes pour toutes les questions.

**Exercice 2 : Flash cards**
Créez des flash cards avec question au recto, réponse courte au verso.

**Exercice 3 : Simulation jury**
Demandez à un collègue de vous poser 10 questions aléatoires de cette FAQ.

**Exercice 4 : Tableau blanc**
Préparez à dessiner des schémas au tableau pour illustrer vos réponses.
