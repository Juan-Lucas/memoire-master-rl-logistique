# Glossaire logistique minière, optimisation et RL

Ce glossaire sera enrichi au fil de la rédaction. Chaque entrée comprend :
- Définition claire
- Explication simple
- Exemple concret

---

## Apprentissage par renforcement (RL)
**Définition :** Méthode d’intelligence artificielle où un agent apprend à prendre des décisions en interagissant avec un environnement pour maximiser une récompense cumulative.
**Explication :** L’agent essaie différentes actions, observe les résultats, et adapte sa stratégie pour obtenir la meilleure performance possible.
**Exemple :** Un agent RL apprend à dispatcher des camions dans une mine pour réduire les temps d’attente et augmenter la productivité.

---

## Heuristique
**Définition :** Règle ou méthode simple permettant de trouver rapidement une solution approximative à un problème complexe.
**Explication :** Les heuristiques ne garantissent pas la solution optimale mais sont faciles à mettre en œuvre.
**Exemple :** Attribuer chaque camion à la pelle la plus proche sans tenir compte de l’état global de la flotte.

---

## KPI (Key Performance Indicator)
**Définition :** Indicateur clé de performance utilisé pour mesurer l’efficacité d’un système ou d’un processus.
**Explication :** Les KPIs permettent d’évaluer et de comparer différentes stratégies d’optimisation.
**Exemple :** Tonnage transporté par heure, temps d’attente moyen, taux d’utilisation des camions.

---

## Système de gestion de flotte (FMS)
**Définition :** Logiciel ou ensemble d’outils permettant de planifier, suivre et optimiser les opérations d’une flotte de véhicules (camions, pelles, etc.) dans une mine.
**Explication :** Le FMS collecte des données en temps réel et aide à prendre des décisions pour améliorer la productivité.
**Exemple :** DISPATCH, MineStar, Jigsaw sont des FMS industriels utilisés dans les mines à ciel ouvert.

---

## Méta-heuristique
**Définition :** Algorithme d’optimisation avancé, souvent inspiré de phénomènes naturels, utilisé pour résoudre des problèmes complexes.
**Explication :** Les méta-heuristiques explorent intelligemment l’espace des solutions pour trouver de bonnes réponses sans tout tester.
**Exemple :** Algorithme génétique, colonies de fourmis, recuit simulé appliqués à l’optimisation du dispatching minier.

---

## MILP (Mixed Integer Linear Programming)
**Définition :** Programmation linéaire en nombres entiers mixtes, méthode mathématique d’optimisation où certaines variables doivent être entières.
**Explication :** Utilisée pour modéliser des problèmes de planification ou d’affectation avec contraintes.
**Exemple :** Optimiser l’affectation des camions aux pelles en respectant des contraintes de capacité et de temps.

---

## Politique (en RL)
**Définition :** Règle ou fonction qui indique à l’agent RL quelle action choisir dans chaque situation.
**Explication :** La politique évolue au fil de l’apprentissage pour maximiser la récompense.
**Exemple :** « Si un camion est vide et une pelle est libre, aller vers la pelle la plus proche. »

---

## Fonction de récompense
**Définition :** Fonction qui attribue un score à chaque action ou séquence d’actions de l’agent RL, guidant ainsi son apprentissage.
**Explication :** Elle traduit les objectifs du problème (ex : minimiser le temps d’attente, maximiser la production).
**Exemple :** Récompenser l’agent à chaque cycle où le temps d’attente total diminue.

---

## Simulation
**Définition :** Reproduction informatique du fonctionnement d’un système réel pour tester des stratégies ou analyser des performances.
**Explication :** Permet d’évaluer des approches sans risque ni coût réel.
**Exemple :** Simuler le dispatching de camions pour comparer heuristique, RL et optimisation mathématique.

---

## Cycle camion-pelle
**Définition :** Suite d’opérations effectuées par un camion : chargement, transport, déchargement, retour.
**Explication :** Un cycle complet permet de mesurer la productivité et l’efficacité du système.
**Exemple :** Un camion charge 100 tonnes, les transporte à l’usine, décharge, puis revient à la pelle.

---

## Match Factor (MF)
**Définition :** Rapport entre le nombre de camions et le nombre de pelles, indicateur d’équilibre du système.
**Explication :** Un MF optimal maximise l’utilisation des équipements.
**Exemple :** 10 camions pour 2 pelles → MF = 5.

---

## Robustesse
**Définition :** Capacité d’un système à maintenir ses performances malgré des perturbations ou des incertitudes.
**Explication :** Un système robuste s’adapte aux pannes, retards ou variations de demande.
**Exemple :** Un algorithme de dispatching qui continue de bien fonctionner même si un camion tombe en panne.

---

## Adaptation
**Définition :** Capacité d’un système à ajuster son comportement en fonction des changements de l’environnement.
**Explication :** L’adaptation permet d’améliorer la performance face à l’imprévu.
**Exemple :** Modifier la politique de dispatching si la demande de transport augmente soudainement.

---

## Congestion
**Définition :** Situation où plusieurs camions attendent en file, causant des retards et une baisse de productivité.
**Explication :** La congestion est un indicateur de mauvaise synchronisation ou de sous-dimensionnement.
**Exemple :** Trois camions attendent devant une pelle occupée, augmentant le temps d’attente moyen.

---

*À compléter et enrichir au fil de la rédaction...*

---

## Processus de Décision Markovien (MDP)
**Définition :** Modèle mathématique pour la prise de décision séquentielle où l’état du système évolue selon des probabilités dépendant de l’action choisie.
**Explication :** Un MDP est défini par un ensemble d’états, d’actions, une fonction de transition (probabilité de passer d’un état à un autre) et une fonction de récompense.
**Exemple :** Dans le dispatching minier, l’état peut être la position des camions, l’action le choix de la prochaine destination, la récompense le temps ou le coût économisé.

---

## DISPATCH (Système)
**Définition :** Système industriel de dispatching minier basé sur une planification en deux étapes (planification globale puis affectation temps réel).
**Explication :** Utilise un modèle de programmation linéaire pour planifier les flux, puis une heuristique pour affecter les camions en temps réel.
**Exemple :** DISPATCH attribue les camions pour respecter les flux optimaux tout en minimisant les temps d’attente.

---

## Vehicle Routing Problem (VRP)
**Définition :** Problème d’optimisation consistant à déterminer les tournées optimales d’une flotte de véhicules pour desservir un ensemble de clients.
**Explication :** Le VRP est un problème classique de la logistique, généralisé dans le dispatching minier.
**Exemple :** Trouver l’ordre optimal pour qu’une flotte de camions livre du minerai à différents points de déchargement.

---

## Policy Gradient (REINFORCE)
**Définition :** Méthode d’apprentissage par renforcement qui ajuste directement les paramètres d’une politique pour maximiser la récompense attendue.
**Explication :** L’algorithme REINFORCE met à jour la politique en fonction des retours obtenus lors des épisodes d’entraînement.
**Exemple :** Utilisé pour entraîner un agent RL à résoudre le VRP sans heuristique humaine.

---

## Pointer Network
**Définition :** Architecture de réseau de neurones permettant de générer des séquences d’indices (pointeurs) sur des entrées de taille variable.
**Explication :** Utilisé pour résoudre des problèmes de type VRP où la taille de l’entrée (nombre de clients) varie.
**Exemple :** Un Pointer Network choisit l’ordre de visite des clients pour optimiser la tournée d’un véhicule.

---

## Planification multi-étapes (Upper/Lower Stage)
**Définition :** Approche où une première étape planifie globalement (upper stage) et une seconde affecte les ressources en temps réel (lower stage).
**Explication :** Permet de combiner vision stratégique et réactivité opérationnelle.
**Exemple :** DISPATCH utilise une planification linéaire pour fixer les flux, puis une heuristique pour l’affectation instantanée des camions.

---

## Heuristique myope
**Définition :** Heuristique qui prend des décisions optimales localement sans anticiper les conséquences globales.
**Explication :** Peut conduire à des congestions ou à une sous-optimisation globale.
**Exemple :** Envoyer un camion à la pelle la plus proche sans vérifier si elle sera disponible à son arrivée.

---

## Affectation contrainte (Hauck)
**Définition :** Méthode d’affectation des camions prenant en compte collectivement les contraintes de la flotte.
**Explication :** Utilise la programmation dynamique et linéaire pour optimiser l’affectation globale.
**Exemple :** Répartir les camions pour respecter à la fois la productivité et les contraintes de mélange de minerai.

---

## Simulation à événements discrets
**Définition :** Modélisation où l’état du système évolue à chaque événement (arrivée, départ, panne, etc.).
**Explication :** Permet de simuler précisément les opérations minières et d’évaluer différentes stratégies.
**Exemple :** Un événement « fin de chargement » déclenche l’affectation d’un nouveau camion à la pelle.

---

## Taux d’utilisation
**Définition :** Pourcentage de temps où un équipement (camion, pelle) est effectivement utilisé.
**Explication :** Indique l’efficacité d’utilisation des ressources.
**Exemple :** Si une pelle charge 45 minutes sur 60, son taux d’utilisation est de 75%.

---

## Temps de cycle
**Définition :** Durée totale pour qu’un camion effectue un cycle complet (chargement, transport, déchargement, retour).
**Explication :** Mesure clé de la performance logistique.
**Exemple :** Un cycle de 30 minutes inclut 5 min de chargement, 10 min de transport, 5 min de déchargement, 10 min de retour.

---

## Ratio de décapage
**Définition :** Rapport entre le volume de stérile extrait et le volume de minerai extrait.
**Explication :** Indicateur de la qualité de l’exploitation minière.
**Exemple :** Un ratio de 3:1 signifie qu’il faut extraire 3 tonnes de stérile pour 1 tonne de minerai.

---

## Baseline
**Définition :** Méthode de référence utilisée pour comparer les performances d’un nouvel algorithme.
**Explication :** Sert de point de comparaison pour valider l’intérêt d’une nouvelle approche.
**Exemple :** Comparer un agent RL à une heuristique classique ou à DISPATCH.

---

## Hyperparamètre
**Définition :** Paramètre fixé avant l’entraînement d’un modèle d’apprentissage automatique, non appris par le modèle.
**Explication :** Influence la vitesse, la stabilité ou la performance de l’apprentissage.
**Exemple :** Taux d’apprentissage, nombre d’épisodes, taille du batch.

---

## État (en RL)
**Définition :** Représentation de la situation actuelle de l’environnement observée par l’agent.
**Explication :** L’état contient toutes les informations nécessaires pour prendre une décision.
**Exemple :** Positions des camions, files d’attente, état des routes.

---

## Action (en RL)
**Définition :** Décision prise par l’agent à chaque étape.
**Explication :** Peut être discrète (choix d’une destination) ou continue (vitesse, accélération).
**Exemple :** Choisir d’envoyer un camion à une pelle ou à une zone tampon.

---

## Fonction de transition
**Définition :** Fonction qui décrit comment l’état du système évolue après chaque action.
**Explication :** Peut être déterministe ou stochastique.
**Exemple :** Après l’action « charger », l’état passe de « camion vide » à « camion plein ».

---

## Fonction de valeur
**Définition :** Fonction qui estime la récompense totale attendue à partir d’un état donné.
**Explication :** Utilisée pour évaluer la qualité d’une politique.
**Exemple :** La valeur d’un état où tous les camions sont en file d’attente est faible.

---

## Reward Shaping
**Définition :** Technique consistant à modifier la fonction de récompense pour faciliter l’apprentissage de l’agent RL.
**Explication :** Permet de guider l’agent vers des comportements souhaités plus rapidement.
**Exemple :** Ajouter une petite pénalité à chaque étape pour encourager des cycles plus courts.

---

## Congestion
**Définition :** Situation où plusieurs camions attendent en file, causant des retards et une baisse de productivité.
**Explication :** La congestion est un indicateur de mauvaise synchronisation ou de sous-dimensionnement.
**Exemple :** Trois camions attendent devant une pelle occupée, augmentant le temps d’attente moyen.

---

## Zone tampon
**Définition :** Espace où les camions attendent avant d’être affectés à une pelle ou à un point de chargement.
**Explication :** Permet de réguler le flux et d’éviter la congestion directe aux pelles.
**Exemple :** Une zone tampon peut contenir 5 camions en attente avant d’accéder à la pelle.

---

## Stérile
**Définition :** Matériau sans valeur économique extrait pour accéder au minerai.
**Explication :** L’extraction du stérile est nécessaire mais augmente le ratio de décapage.
**Exemple :** Déblayer la terre pour atteindre une veine de cuivre.

---

## Pseudo-coût
**Définition :** Coût artificiel utilisé dans la planification pour guider l’optimisation sans refléter un coût réel.
**Explication :** Sert à équilibrer les flux ou à respecter des contraintes dans les modèles linéaires.
**Exemple :** Attribuer un pseudo-coût élevé à une route pour éviter la congestion.

---

## Policy (Politique)
**Définition :** Règle ou fonction qui indique à l’agent RL quelle action choisir dans chaque situation.
**Explication :** La politique évolue au fil de l’apprentissage pour maximiser la récompense.
**Exemple :** « Si un camion est vide et une pelle est libre, aller vers la pelle la plus proche. »

---

## Policy Network
**Définition :** Réseau de neurones qui approxime la politique d’un agent RL.
**Explication :** Prend l’état en entrée et prédit la probabilité de chaque action.
**Exemple :** Un policy network décide si un camion doit aller à la pelle A ou B selon l’état du système.

---

## Baseline (en RL)
**Définition :** Valeur de référence utilisée pour réduire la variance lors de la mise à jour des paramètres d’un agent RL.
**Explication :** Permet d’améliorer la stabilité de l’apprentissage.
**Exemple :** Utiliser la moyenne des récompenses passées comme baseline dans REINFORCE.

---

## Exploration vs Exploitation
**Définition :** Dilemme fondamental en RL entre essayer de nouvelles actions (exploration) et utiliser les meilleures actions connues (exploitation).
**Explication :** Un bon équilibre est nécessaire pour apprendre efficacement.
**Exemple :** Essayer une nouvelle route pour un camion même si la route habituelle semble meilleure.

---

## Épisode (en RL)
**Définition :** Séquence complète d’actions et d’états, du début à la fin d’une simulation ou d’un problème.
**Explication :** Permet de mesurer la performance globale d’une politique.
**Exemple :** Un épisode correspond à une journée complète de dispatching simulée.

---

## Généralisabilité
**Définition :** Capacité d’un modèle ou d’un agent à bien fonctionner sur des situations non vues pendant l’entraînement.
**Explication :** Un agent RL généralisable s’adapte à de nouveaux scénarios miniers.
**Exemple :** Un agent entraîné sur une mine A qui réussit aussi sur une mine B.

---

## Planification opérationnelle
**Définition :** Élaboration de plans détaillés pour l’utilisation optimale des ressources à court terme.
**Explication :** Vise à maximiser la productivité et à minimiser les coûts au quotidien.
**Exemple :** Planifier l’affectation des camions pour la journée selon la demande et les contraintes.

---

## Planification stratégique
**Définition :** Définition des grandes orientations et objectifs à long terme pour l’exploitation minière.
**Explication :** Prend en compte les investissements, la durée de vie de la mine, les réserves, etc.
**Exemple :** Décider d’ouvrir une nouvelle fosse ou d’investir dans de nouveaux camions.

---

## Simulation hybride
**Définition :** Combinaison de simulation et d’optimisation mathématique pour évaluer des stratégies complexes.
**Explication :** Permet d’intégrer incertitude, contraintes et dynamique réelle.
**Exemple :** Simuler le dispatching avec optimisation mathématique pour chaque scénario.

---

## Réseau encodeur-décodeur (LSTM, attention)
**Définition :** Architecture de réseau de neurones pour traiter des séquences, utilisée pour modéliser des problèmes de décision séquentielle.
**Explication :** L’encodeur résume l’état, le décodeur génère les actions.
**Exemple :** Utilisé dans le RL pour le VRP avec des entrées de taille variable.

---

## Masquage dynamique des actions
**Définition :** Technique consistant à rendre certaines actions impossibles selon l’état courant.
**Explication :** Permet d’éviter que l’agent RL ne choisisse des actions non valides.
**Exemple :** Masquer l’action « aller à une pelle pleine » si aucune place n’est disponible.

---

## Récompense négative
**Définition :** Pénalité attribuée à l’agent RL pour encourager l’évitement de comportements indésirables.
**Explication :** Utilisée pour guider l’agent vers des solutions plus efficaces.
**Exemple :** Pénaliser chaque minute d’attente d’un camion.

---

## Production supplémentaire
**Définition :** Gain de production obtenu grâce à une optimisation ou une nouvelle stratégie.
**Explication :** Permet de mesurer l’impact d’une innovation sur la performance.
**Exemple :** Passer de 60 000 à 66 000 tonnes/shift grâce à un nouvel algorithme.

---

## Capacité résiduelle
**Définition :** Quantité de charge qu’un camion peut encore transporter avant d’atteindre sa capacité maximale.
**Explication :** Sert à déterminer la prochaine affectation ou le retour au dépôt.
**Exemple :** Un camion de 100 t ayant déjà chargé 60 t a une capacité résiduelle de 40 t.

---

## Période de planification
**Définition :** Intervalle de temps sur lequel les décisions de planification sont prises.
**Explication :** Peut être une heure, un shift, une journée, etc.
**Exemple :** Planifier les affectations de camions pour chaque shift de 8h.

---

## Panne (en simulation)
**Définition :** Événement où un équipement devient indisponible temporairement.
**Explication :** Les pannes sont modélisées pour tester la robustesse des stratégies.
**Exemple :** Un camion tombe en panne et doit être réparé avant de reprendre le service.

---

## Événement stochastique
**Définition :** Événement dont l’occurrence est aléatoire, modélisé par une probabilité.
**Explication :** Permet de simuler l’incertitude dans l’environnement minier.
**Exemple :** Une route peut se dégrader de façon imprévisible pendant la simulation.