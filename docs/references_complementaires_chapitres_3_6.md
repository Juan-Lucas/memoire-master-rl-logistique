# Références complémentaires recommandées (Chapitres 3 à 6)

Objectif: compléter les 22 articles déjà utilisés par un noyau de références méthodologiques et expérimentales très solides pour viser un mémoire de niveau supérieur.

## Principes de sélection
- Prioriser les références qui renforcent la rigueur scientifique (modélisation, protocoles, reproductibilité, statistiques).
- Ajouter peu mais bien: environ 8 à 12 références de haute valeur.
- Éviter les doublons avec les 22 articles déjà dans la revue.

---

## Chapitre 3 — Modélisation du problème (MDP, hypothèses, fonction de récompense)

### Références à ajouter en priorité
1. Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction (2nd ed.).
- Pourquoi: base théorique incontournable pour MDP, politique, valeur, retour, exploration/exploitation.
- Où citer: section MDP, espace d'états/actions, fonction de récompense.

2. Puterman, M. L. (1994). Markov Decision Processes: Discrete Stochastic Dynamic Programming.
- Pourquoi: formalisme mathématique rigoureux des MDP et transitions stochastiques.
- Où citer: formulation probabiliste de la transition et justification de l'approximation markovienne.

3. Law, A. M. (2015). Simulation Modeling and Analysis (5th ed.).
- Pourquoi: référence standard pour la simulation à événements discrets, validation et vérification de simulateur.
- Où citer: modélisation du système, hypothèses stochastiques, crédibilité du simulateur.

4. Banks, J., Carson, J. S., Nelson, B. L., & Nicol, D. M. (2010). Discrete-Event System Simulation (5th ed.).
- Pourquoi: référence forte pour la construction d'un environnement DES réaliste.
- Où citer: choix de la simulation événementielle et limitations.

### Références optionnelles utiles
5. Bellman, R. (1957). Dynamic Programming.
- Pourquoi: fondement historique du contrôle séquentiel et principe d'optimalité.
- Où citer: introduction théorique du MDP.

---

## Chapitre 4 — Méthodologie et conception de la solution RL

### Références à ajouter en priorité
1. Schulman, J., et al. (2017). Proximal Policy Optimization Algorithms.
- Pourquoi: si PPO est l'algorithme principal, c'est la citation méthodologique centrale.
- Où citer: choix d'algorithme RL, stabilité et efficacité d'entraînement.

2. Haarnoja, T., et al. (2018). Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor.
- Pourquoi: baseline moderne (surtout actions continues) pour comparaison avec PPO.
- Où citer: section choix d'algorithmes et discussion des alternatives.

3. Mnih, V., et al. (2015). Human-level control through deep reinforcement learning.
- Pourquoi: papier fondateur DQN, utile pour contextualiser les approches valeur.
- Où citer: panorama des familles d'algorithmes (value-based vs policy-based).

4. Van Hasselt, H., Guez, A., & Silver, D. (2016). Deep Reinforcement Learning with Double Q-learning.
- Pourquoi: référence pour limiter le biais d'optimisme des approches Q-learning.
- Où citer: si baseline DQN/Double DQN est discutée.

### Références optionnelles utiles
5. Costa, C., & Ontañón, S. (2020). A Closer Look at Invalid Action Masking in Policy Gradient Algorithms.
- Pourquoi: justifie scientifiquement le masquage des actions invalides dans l'environnement minier.
- Où citer: espace d'actions et implémentation de contraintes.

---

## Chapitre 5 — Implémentation et expérimentation

### Références à ajouter en priorité
1. Brockman, G., et al. (2016). OpenAI Gym.
- Pourquoi: référence standard pour la structure d'un environnement RL (reset, step, spaces).
- Où citer: architecture logicielle et interface d'environnement.

2. Raffin, A., et al. (2021). Stable-Baselines3: Reliable Reinforcement Learning Implementations.
- Pourquoi: crédibilise l'implémentation des algorithmes et la reproductibilité logicielle.
- Où citer: stack technique, protocole d'entraînement.

3. Henderson, P., et al. (2018). Deep Reinforcement Learning that Matters.
- Pourquoi: référence critique pour bonnes pratiques expérimentales (variance, seeds, reporting).
- Où citer: protocole expérimental, nombre de répétitions, robustesse des résultats.

### Références optionnelles utiles
4. Agarwal, R., et al. (2021). Deep Reinforcement Learning at the Edge of the Statistical Precipice.
- Pourquoi: métriques robustes et bonnes pratiques de comparaison en RL.
- Où citer: définition des indicateurs de performance et analyse des distributions de résultats.

---

## Chapitre 6 — Résultats, analyse et validité scientifique

### Références à ajouter en priorité
1. Demšar, J. (2006). Statistical Comparisons of Classifiers over Multiple Data Sets.
- Pourquoi: cadre classique pour comparer plusieurs méthodes (baselines vs RL) avec tests non paramétriques.
- Où citer: comparaison inter-méthodes.

2. Dror, R., et al. (2018). The Hitchhiker's Guide to Testing Statistical Significance in NLP.
- Pourquoi: bonnes pratiques de tests de significativité applicables aux comparaisons d'algorithmes.
- Où citer: section validité statistique et p-values/intervalles.

3. Efron, B., & Tibshirani, R. J. (1994). An Introduction to the Bootstrap.
- Pourquoi: intervalles de confiance robustes sur les métriques simulées.
- Où citer: estimation d'incertitude sur KPI (attente, productivité, coût).

### Références optionnelles utiles
4. Colas, C., et al. (2019). How Many Random Seeds? Statistical Power Analysis in Deep Reinforcement Learning Experiments.
- Pourquoi: justifie le nombre de réplications/seeds et la puissance statistique.
- Où citer: protocole expérimental et robustesse des conclusions.

---

## Recommandation pratique (pour rester efficace)

- Cible minimale très forte: 8 références
  - Chapitre 3: Sutton & Barto, Puterman
  - Chapitre 4: PPO, SAC
  - Chapitre 5: Gym, SB3
  - Chapitre 6: Demšar, Bootstrap

- Cible premium (si temps): 12 références
  - Ajouter Law/Banks, Henderson, Agarwal, Colas

---

## Comment les intégrer vite dans ton mémoire

1. Ajouter les entrées BibTeX dans reports/export.bib.
2. Citer explicitement ces références dans les sections correspondantes (3.4, 4.4, 5.3, 6.2/6.3).
3. Dans le chapitre 6, rendre explicite le protocole statistique:
- nombre de runs par méthode,
- intervalle de confiance,
- test de significativité,
- interprétation des effets.

Ce bloc de références améliorera fortement la perception de rigueur scientifique lors de l'évaluation.