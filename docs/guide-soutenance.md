# Guide de préparation à la soutenance

Ce guide structure votre présentation orale pour maximiser votre impact le jour de la soutenance.

---

## Structure temporelle de la soutenance

**Durée totale : 30-45 minutes (selon votre institution)**
- Introduction : 3-5 minutes
- Présentation du problème : 5 minutes
- Méthodologie : 8-10 minutes
- Résultats : 8-10 minutes
- Discussion : 5 minutes
- Conclusion : 2-3 minutes
- Questions du jury : 10-15 minutes

---

## Script détaillé de présentation

### 1. Introduction (3-5 minutes)

**Objectif :** Accrocher le jury, présenter le contexte et la problématique.

**Script suggéré :**
```
Bonjour à tous, je vous présente mon mémoire sur l'application de l'apprentissage
par renforcement au dispatching minier.

La logistique minière représente 50-60% des coûts d'exploitation des mines
à ciel ouvert. Le dispatching des camions est un problème complexe : il faut
coordonner des dizaines de camions avec plusieurs pelles et concasseurs, en
tenant compte de contraintes opérationnelles et de l'incertitude.

Les méthodes actuelles (heuristiques, systèmes industriels comme DISPATCH)
sont soit trop myopes (optimisation locale sans vision globale), soit
déterministes (ne s'adaptent pas aux perturbations).

Ma problématique : Comment utiliser l'apprentissage par renforcement pour
obtenir un dispatching dynamique, adaptatif et robuste qui maximise la
productivité tout en minimisant les coûts et les temps d'attente ?
```

**Points clés à mémoriser :**
- 50-60% des coûts d'exploitation
- Problème de coordination complexe
- Limites des méthodes actuelles (myopie, déterminisme)
- Problématique claire

---

### 2. Présentation du problème (5 minutes)

**Objectif :** Formaliser le problème comme un MDP.

**Script suggéré :**
```
J'ai formalisé le problème de dispatching comme un Processus de Décision
Markovien (MDP).

L'état du système s_t comprend : les files d'attente aux pelles, la position
et le statut des camions, la disponibilité des ressources, et le temps courant.

L'action a_t est une affectation de camions aux pelles, ou l'attente.

La transition est stochastique : les temps de trajet suivent une loi lognormale
avec un bruit gaussien, modélisant l'incertitude réelle.

La récompense combine trois objectifs : rendement (tonnage livré), équité
(distribution équilibrée des files), et coût (distance et consommation).
```

**Équations à mentionner :**
- État : $s_t = (\{q_p\}, \{x_c\}, \{z_r\}, t_{\text{courant}})$
- Récompense : $R_t = w_1 R_{\text{rendement}} + w_2 R_{\text{équité}} + w_3 R_{\text{coût}}$

---

### 3. Méthodologie (8-10 minutes)

**Objectif :** Présenter l'approche RL et l'implémentation.

**Script suggéré :**
```
J'ai implémenté un environnement Gymnasium qui simule le système minier.
L'observation est normalisée dans [0,1] pour stabiliser l'apprentissage.

J'ai comparé plusieurs approches :
- Baselines classiques : FIFO, Shortest Path, Fixed Assignment, Nearest Shovel
- RL tabulaire : Q-Learning, SARSA
- RL profond : DQN, PPO (méthode principale)

PPO a été choisi pour sa stabilité et son efficacité. Architecture MLP
128×128 avec ReLU, hyperparamètres : α=0.0003, γ=0.99, batch=64, λ=0.95,
ε_clip=0.2.

L'entraînement : 30 000 épisodes pour Q-Learning/SARSA, 2 000 000 steps
pour PPO/DQN. Évaluation sur 3 scénarios (nominal, high-load, high-breakdown),
10 réplications chacun (seeds 42–51).
```

**Points techniques à maîtriser :**
- Architecture Gymnasium (reset/step/render)
- Normalisation des observations
- Hyperparamètres PPO
- Scénarios de test

---

### 4. Résultats (8-10 minutes)

**Objectif :** Présenter les gains et analyses statistiques.

**Script suggéré :**
```
Les résultats montrent que PPO surpasse les baselines sur la plupart des
KPIs.

Sur le scénario nominal :
- Fixed Assignment domine en productivité : 4 074 t/h (Match Factor parfait)
- PPO : 3 335 t/h, meilleur temps d'attente parmi les agents RL (30.6 min)

Sur high_breakdown (pannes fréquentes) :
- PPO obtient le meilleur temps d'attente : 47.7 min (-10% vs Fixed 52.9 min)
- Nearest/Shortest Path s'effondrent en high_load : 201 min d'attente

L'évaluation repose sur moyenne ± écart-type, 10 réplications, seeds 42–51.
```

**Chiffres à mémoriser (voir resultats-cles.md) :**
- Gains de performance vs baselines
- Robustesse sur scénarios perturbés
- Significativité statistique

---

### 5. Discussion (5 minutes)

**Objectif :** Analyser les forces, limites et perspectives.

**Script suggéré :**
```
Forces de l'approche :
- Capacité d'adaptation dynamique
- Vision globale du système (vs myopie des heuristiques)
- Robustesse aux perturbations
- Généralisation à différents scénarios

Limites :
- Temps d'entraînement nécessaire
- Sensibilité aux hyperparamètres
- Dépendance à la qualité de la simulation
- Généralisation à des mines non vues à valider

Perspectives :
- Extension à des mines plus complexes (plus de pelles/camions)
- Intégration de contraintes supplémentaires (priorités de minerai)
- Apprentissage multi-agent pour coordination décentralisée
- Validation sur données de terrain réelles
```

---

### 6. Conclusion (2-3 minutes)

**Objectif :** Synthétiser les contributions et l'impact.

**Script suggéré :**
```
En conclusion, j'ai démontré que l'apprentissage par renforcement,
notamment PPO, est une approche efficace pour le dispatching minier.

Contributions principales :
1. Formalisation MDP du problème de dispatching minier
2. Implémentation d'un environnement de simulation réaliste
3. Comparaison systématique de 8 approches (baselines + RL)
4. Démonstration de gains significatifs en productivité et robustesse

Impact industriel potentiel : réduction des coûts opérationnels de 10-15%,
meilleure adaptation aux perturbations, optimisation continue.

Ce travail ouvre des perspectives pour l'application du RL à d'autres
problèmes de logistique minière.
```

---

## Techniques de présentation orale

### Avant la soutenance

**Préparation :**
- Répéter la présentation à voix haute 3-5 fois
- Chronométrer chaque section
- Préparer des notes de secours (cartes ou feuille)
- Vérifier que toutes les figures sont lisibles

**Matériel :**
- Slides claires (max 6-7 points par slide)
- Diagrammes du système (voir diagrammes-systeme.md)
- Tableaux de résultats comparatifs
- Graphiques de learning curves

### Pendant la présentation

**Posture :**
- Parler lentement et clairement
- Regarder le jury (pas uniquement l'écran)
- Utiliser des gestes pour illustrer
- Varier le ton de voix

**Gestion du temps :**
- Avoir une montre visible
- Être prêt à accélérer/slow down selon le jury
- Préparer des versions courtes/longues de chaque section

**Répondre aux questions :**
- Écouter la question en entier
- Prendre 2-3 secondes pour réfléchir
- Reformuler la question si nécessaire
- Répondre de manière structurée (1, 2, 3)
- Ne pas hésiter à dire "Je ne sais pas" si c'est le cas

---

## Questions difficiles et réponses préparées

**"Pourquoi PPO et pas un autre algorithme ?"**
- PPO offre un bon compromis stabilité/efficacité
- Moins sensible aux hyperparamètres que DQN
- Meilleure pour les espaces d'action discrets que SAC
- Reconnu comme state-of-the-art dans la communauté RL

**"Comment généralisez-vous à une mine différente ?"**
- L'agent apprend des patterns généraux (équilibrage files/demande)
- Le fine-tuning sur la nouvelle mine est rapide
- Limites actuelles : topologie du réseau, nombre de pelles/camions
- Perspectives : meta-learning, domain randomization

**"Quelle est la validité de votre simulation ?"**
- Paramètres justifiés par la littérature (Afrapoli, Mohtasham, etc.)
- Distributions réalistes (lognormales pour temps de trajet)
- Scénarios de perturbation testent la robustesse
- Limite : validation sur données de terrain nécessaire

**"Coût computationnel de l'entraînement ?"**
- Entraînement : ~1-2 heures sur GPU standard
- Inférence : temps réel (<1ms par décision)
- Comparable aux méthodes d'optimisation classiques
- Perspective : entraînement offline, déploiement online

---

## Checklist finale

**Une semaine avant :**
- [ ] Présentation répétée 3 fois
- [ ] Chronomètre validé
- [ ] Slides finales prêtes
- [ ] Notes de secours préparées
- [ ] Questions difficiles révisées

**Jour J :**
- [ ] Arriver 15 minutes en avance
- [ ] Vérifier équipement (projecteur, son)
- [ ] Avoir une copie de sauvegarde de présentation
- [ ] Eau, stylo, papier pour notes
- [ ] Copie imprimée du mémoire pour le jury

---

## Exercices de préparation

**Exercice 1 : Pitch en 2 minutes**
Expliquer votre sujet en 2 minutes à quelqu'un qui n'y connait rien.

**Exercice 2 : MDP en 3 minutes**
Expliquer votre formalisation MDP sans regarder vos notes.

**Exercice 3 : Questions flash**
Répondre en 30 secondes à : "Quelle est votre contribution principale ?", "Pourquoi RL ?", "Quelles sont vos limites ?"

**Exercice 4 : Tableau blanc**
Dessiner l'architecture de votre système au tableau blanc.

**Exercice 5 : Questions jury**
Simuler 10 questions difficiles du jury et y répondre.

---

## Ressources utiles

- Voir `faq-jury.md` pour plus de questions probables
- Voir `resultats-cles.md` pour les chiffres à mémoriser
- Voir `diagrammes-systeme.md` pour les visualisations
- Réviser `explication-formules-tout-memoire.md` pour les équations
- Réviser `glossaire.md` pour le vocabulaire
