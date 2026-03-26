# Au-delà des heuristiques statiques : Conception et mise en œuvre d'un agent d'apprentissage par renforcement pour l'optimisation holistique et dynamique de la logistique minière

Présenté par **Jean-Luc MUPASA KALUNGA**

Supervisé par **Dr. Olfa FERCHICHI**

Université Don Bosco de Lubumbashi (UDBL) — Master 2 Data Science, spécialité Logistique — Année académique 2025-2026

---

## Introduction

Dans un contexte minier où l'optimisation des coûts opérationnels et la réduction de l'empreinte environnementale sont devenues des impératifs stratégiques, les systèmes de gestion de flotte (Fleet Management Systems - FMS) traditionnels montrent leurs limites. Bien que des solutions industrielles comme Caterpillar MineStar Dispatch et Komatsu Jigsaw aient révolutionné la coordination des flottes, ces systèmes reposent principalement sur des heuristiques statiques, des règles prédéfinies et des zones tampons pour gérer les flux de véhicules.

Ces approches, bien qu'efficaces dans des conditions stables, peinent à s'adapter dynamiquement aux micro-conditions évolutives : état dégradé des routes, variations stochastiques du trafic, conditions météorologiques imprévisibles ou encore style de conduite et état mécanique des véhicules.

Ce projet propose de dépasser ces limites en développant un système d'aide à la décision basé sur l'Apprentissage par Renforcement (Reinforcement Learning - RL), capable d'apprendre une politique de conduite holistique optimisant simultanément l'itinéraire et le rythme de déplacement des véhicules.

---

## Problématique

Les FMS actuels optimisent localement (affectation camion-pelle, gestion des files d'attente), mais ne parviennent pas à optimiser globalement l'ensemble de la flotte en temps réel, ni à anticiper et s'adapter aux perturbations dynamiques telles que la dégradation des routes ou les congestions imprévues.

Un camion suivant une règle statique peut sprinter vers sa destination pour finalement attendre dans une zone tampon, gaspillant carburant et temps. Une approche adaptative pourrait ajuster le rythme de déplacement pour arriver "juste à temps".

---

## Hypothèse

L'utilisation d'un agent autonome basé sur l'Apprentissage par Renforcement permettra de dépasser les performances des systèmes heuristiques actuels en apprenant une politique de navigation adaptative qui minimise la consommation de carburant, réduit le temps d'attente improductif et s'adapte en temps réel aux conditions changeantes de l'environnement minier.

Cette hypothèse sera validée par une comparaison rigoureuse entre l'agent d'apprentissage par renforcement et des baselines représentatives des approches actuelles (heuristiques statiques et logique "Dispatch" simplifiée).

---

## Questions de recherche

1. Comment modéliser l'environnement minier complexe (réseau routier, zones tampons, événements stochastiques) dans un cadre d'apprentissage par renforcement compatible avec les standards industriels ?

2. Quelle architecture d'agent d'apprentissage par renforcement est la plus adaptée pour apprendre une politique de navigation holistique intégrant à la fois le choix d'itinéraire et le contrôle de vitesse ?

3. Comment concevoir une fonction de récompense qui capture fidèlement les coûts réels (carburant, usure, temps) tout en pénalisant les comportements non souhaitables ?

4. Quelles métriques et protocoles d'évaluation permettront de démontrer quantitativement la supériorité de l'approche d'apprentissage par renforcement par rapport aux approches classiques, notamment dans des scénarios de perturbation ?

5. Comment intégrer un tel système d'apprentissage par renforcement comme couche d'intelligence additionnelle dans l'écosystème des FMS existants, sans nécessiter un remplacement complet de l'infrastructure ?

