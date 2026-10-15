"""Package principal du projet Memoire RL Logistique.

Ce package regroupe les sous-modules principaux :
- `env` : environnement Gymnasium de la mine (MineEnv)
- `simulation` : entités, modèle de graphe, fonctions liées à la simulation
- `rl` : algorithmes d'apprentissage (SARSA, Q-Learning, DQN, PPO)
- `baselines` : heuristiques de référence (FIFO, Fixed, Nearest, ShortestPath)
- `experiments` : scénarios et lancement des expériences (benchmark, ablation)
- `analysis` : calculs sur les résultats (statistiques, robustesse), sans figure
- `plots` : figures de l'article et du mémoire
"""
