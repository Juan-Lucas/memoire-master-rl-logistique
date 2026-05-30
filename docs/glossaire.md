# Glossaire logistique minière, optimisation et RL

Ce glossaire est aligné sur les termes utilisés dans le projet et les chapitres 4 à 6 du mémoire. Chaque entrée comprend :
- Définition claire
- Explication simple
- Exemple concret

---

## Apprentissage par renforcement (RL)
**Définition :** Méthode d’intelligence artificielle où un agent apprend à prendre des décisions en interagissant avec un environnement pour maximiser une récompense cumulative.
**Explication :** L’agent essaie différentes actions, observe les résultats et ajuste sa politique pour améliorer progressivement sa performance.
**Exemple :** Un agent RL apprend à dispatcher des camions dans une mine pour réduire les temps d’attente et augmenter la productivité.

---

## Environnement Gymnasium
**Définition :** Interface standardisée pour les environnements d’apprentissage par renforcement en Python.
**Explication :** Gymnasium définit les méthodes `reset()`, `step()`, `render()` et les espaces d’observation/action.
**Exemple :** L’environnement minier du projet est implémenté avec Gymnasium pour être compatible avec Stable-Baselines3.

---

## Stable-Baselines3
**Définition :** Bibliothèque Python proposant des implémentations d’algorithmes d’apprentissage par renforcement profond.
**Explication :** Elle fournit PPO, DQN, A2C et d’autres algorithmes, avec gestion des callbacks et du logging.
**Exemple :** PPO et DQN sont entraînés avec Stable-Baselines3 dans le projet.

---

## Q-Learning
**Définition :** Algorithme tabulaire d’apprentissage par renforcement off-policy qui apprend la qualité d’un état-action.
**Explication :** Il met à jour une Q-table en utilisant la récompense et la valeur estimée du prochain état.
**Exemple :** Q-Learning choisit la prochaine pelle d’un camion en se basant sur les Q-values apprises.

---

## SARSA
**Définition :** Algorithme d’apprentissage par renforcement on-policy.
**Explication :** La mise à jour utilise l’action effectivement choisie dans l’état suivant.
**Exemple :** SARSA est plus conservateur lorsque l’environnement est stochastique.

---

## DQN (Deep Q-Network)
**Définition :** Algorithme RL profond qui approxime la fonction Q par un réseau de neurones.
**Explication :** DQN utilise un replay buffer et un réseau cible pour stabiliser l’apprentissage.
**Exemple :** DQN apprend une politique sur les observations continues de l’environnement minier.

---

## PPO (Proximal Policy Optimization)
**Définition :** Méthode de policy gradient profond qui limite les mises à jour trop importantes de la politique.
**Explication :** PPO utilise un ratio de probabilités et un clipping pour stabiliser l’entraînement.
**Exemple :** PPO est entraîné pendant 2 millions de timesteps dans le projet.

---

## Baseline
**Définition :** Méthode de référence utilisée pour comparer les performances d’une nouvelle stratégie.
**Explication :** Une baseline permet de vérifier qu’une nouvelle approche apporte un réel gain.
**Exemple :** FIFO, Fixed Assignment, Nearest Shovel et Shortest Path servent de baselines.

---

## Heuristique
**Définition :** Règle de décision simple et rapide pour produire une solution approximative.
**Explication :** Les heuristiques sont faciles à implémenter mais ne garantissent pas l’optimalité.
**Exemple :** Envoyer un camion à la pelle la plus proche sans analyser l’état global.

---

## Méta-heuristique
**Définition :** Algorithme d’optimisation inspiré de phénomènes naturels ou d’intelligence collective.
**Explication :** Il cherche des solutions de bonne qualité sans tester toutes les possibilités.
**Exemple :** Algorithme génétique ou recuit simulé appliqué au dispatching minier.

---

## Politique (en RL)
**Définition :** Fonction qui indique à l’agent quelle action choisir dans chaque état.
**Explication :** La politique se met à jour avec l’entraînement et peut être déterministe ou stochastique.
**Exemple :** « Si un camion est vide et une pelle est libre, aller vers la pelle la plus proche. »

---

## Fonction de récompense
**Définition :** Fonction qui attribue un score à chaque action ou séquence d’actions.
**Explication :** Elle traduit les objectifs opérationnels comme réduire l’attente ou maximiser le rendement.
**Exemple :** Ajouter une récompense positive pour chaque tonne transportée et une pénalité pour l’attente.

---

## Reward Shaping
**Définition :** Technique consistant à enrichir la fonction de récompense pour guider l’apprentissage.
**Explication :** On ajoute des termes intermédiaires pour encourager des comportements souhaités.
**Exemple :** Pénaliser l’attente et récompenser la ponctualité pour accélérer la formation de la politique.

---

## Exploration vs Exploitation
**Définition :** Dilemme entre tester de nouvelles actions (exploration) et utiliser les meilleures actions connues (exploitation).
**Explication :** Un bon équilibre est nécessaire pour apprendre sans rester bloqué dans une solution sous-optimale.
**Exemple :** Choisir parfois une action aléatoire pour découvrir une meilleure affectation de camion.

---

## Replay buffer
**Définition :** Mémoire circulaire qui stocke les transitions observées pour réutilisation lors de l’entraînement.
**Explication :** Il améliore la stabilité de l’apprentissage en entraînant plusieurs fois sur les mêmes expériences.
**Exemple :** DQN conserve les transitions `état, action, récompense, état suivant` dans un replay buffer.

---

## On-policy / Off-policy
**Définition :** On-policy apprend à partir des actions générées par la politique actuelle ; off-policy peut apprendre à partir de données générées par une autre politique.
**Explication :** PPO est on-policy, tandis que Q-Learning et DQN sont off-policy.
**Exemple :** Un agent DQN peut apprendre d’anciennes données stockées dans un replay buffer.

---

## Temporal difference
**Définition :** Méthode d’estimation de la valeur en utilisant la différence entre les prédictions successives.
**Explication :** Elle combine l’apprentissage par simulation et le bootstrap pour mettre à jour les valeurs.
**Exemple :** La mise à jour Q de Q-Learning repose sur un terme de différence temporelle.

---

## Discount factor (γ)
**Définition :** Coefficient qui pondère l’importance des récompenses futures.
**Explication :** Une valeur proche de 1 privilégie les performances à long terme, une valeur plus faible privilégie le court terme.
**Exemple :** Dans un environnement minier, γ = 0,99 permet de planifier des cycles complets.

---

## Learning rate (α)
**Définition :** Taux de mise à jour des paramètres ou des valeurs pendant l’entraînement.
**Explication :** Un learning rate trop élevé peut déstabiliser l’apprentissage, trop faible le ralentit.
**Exemple :** Un taux de 0,0003 est souvent utilisé pour PPO et DQN.

---

## Fonction d’approximation
**Définition :** Modèle utilisé pour estimer les valeurs ou la politique dans les environnements de grande dimension.
**Explication :** Il remplace les tables dans les problèmes à état élevé.
**Exemple :** Un réseau de neurones approximant la Q-function de DQN.

---

## Acteur-critique
**Définition :** Architecture composée d’un acteur qui choisit les actions et d’un critique qui évalue les états.
**Explication :** Le critique guide l’acteur vers de meilleures décisions.
**Exemple :** PPO utilise une architecture acteur-critique.

---

## Taux d’utilisation
**Définition :** Pourcentage de temps où un équipement est actif.
**Explication :** Indicateur de disponibilité et performance.
**Exemple :** Si une pelle charge 45 minutes sur 60, son taux d’utilisation est de 75 %.

---

## Productivité
**Définition :** Quantité de minerai transportée par unité de temps.
**Explication :** Mesure la performance opérationnelle globale.
**Exemple :** Tonnes transportées par heure.

---

## Temps d’attente
**Définition :** Durée pendant laquelle un camion reste inactif en file ou en zone tampon.
**Explication :** Indicateur principal de fluidité.
**Exemple :** Temps moyen d’attente par camion.

---

## Consommation spécifique
**Définition :** Quantité de carburant consommée par tonne transportée.
**Explication :** Mesure l’efficacité énergétique.
**Exemple :** Litres par tonne.

---

## Scénario
**Définition :** Configuration expérimentale paramétrée par le nombre de camions, pelles, dumps et le taux de pannes.
**Explication :** Permet de tester les méthodes dans des contextes différents.
**Exemple :** nominal, high_load, high_breakdown.

---

## Robustesse
**Définition :** Capacité d’une méthode à conserver ses performances malgré des perturbations.
**Explication :** Mesurée sur des scénarios de surcharge ou de pannes.
**Exemple :** PPO garde un bon temps d’attente en high_breakdown.

---

## Congestion
**Définition :** Situation où plusieurs camions attendent en file, réduisant la productivité.
**Explication :** Indique un déséquilibre entre flux et capacité.
**Exemple :** Plusieurs camions bloqués devant une pelle occupée.

---

## Zone tampon
**Définition :** Espace d’attente entre les pelles et les dumps.
**Explication :** Régule le flux des camions pour éviter la congestion aux pelles.
**Exemple :** Camions en attente avant d’être affectés à une pelle.

---

## Stérile
**Définition :** Matériau sans valeur économique extrait pour accéder au minerai.
**Explication :** L’extraction du stérile est nécessaire mais augmente les coûts et le ratio de décapage.
**Exemple :** Roche stérile extraite avant d’atteindre le minerai.
