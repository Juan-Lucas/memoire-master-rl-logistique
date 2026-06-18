# Glossaire RL, Deep RL, Machine Learning et Statistiques

Ce glossaire couvre les termes techniques RL, Deep RL, ML, DL et statistiques utilisés dans le mémoire. Chaque entrée comprend :
- Définition très simple
- Explication avec une analogie du quotidien
- Exemple concret tiré du projet
- Pour aller plus loin : formules et chiffres précis pour les questions du jury

---

## Apprentissage par renforcement (RL)
**Définition :** C'est une façon d'apprendre en essayant des choses, en se trompant, et en gardant ce qui marche le mieux — un peu comme on apprend à faire du vélo.
**Explication :** Un agent (le « cerveau » du programme) se retrouve dans une situation, doit choisir une action, et reçoit ensuite une récompense (bonne ou mauvaise) selon le résultat. Personne ne lui dit à l'avance quoi faire : il essaie, observe ce qui se passe, et ajuste petit à petit sa façon de choisir pour obtenir le plus de récompenses possible sur la durée — pas seulement tout de suite, mais sur l'ensemble de la « partie ».
**Exemple :** Dans le projet, l'agent regarde l'état du site minier (files d'attente, position des camions, heure de la journée) et choisit où envoyer un camion. Au début, ses choix sont presque au hasard. Après des milliers d'essais simulés, il a appris une bonne stratégie : avec PPO, la production moyenne atteint 4 208,75 t/h.
**Pour aller plus loin :** Formellement, à chaque pas $t$, l'agent observe $s_t$, choisit $a_t\sim\pi(\cdot|s_t)$, reçoit $r_t$ et passe à $s_{t+1}$ ; l'objectif est de maximiser le retour actualisé $G_t=\sum_{k=0}^{\infty} \gamma^k r_{t+k}$.

---

## Retour G_t (Return)
**Définition :** La somme de toutes les récompenses futures actualisées à partir d'un instant t — c'est exactement ce que l'agent cherche à maximiser, pas la récompense immédiate.
**Explication :** C'est la différence entre un joueur myope (qui veut juste marquer au coup suivant) et un joueur stratège (qui pense à toute la partie). G_t additionne toutes les récompenses à venir, mais en « dépréciant » les récompenses lointaines via le facteur γ — une récompense dans 10 coups compte moins qu'une récompense maintenant. C'est ce retour total que Q-Learning, SARSA, DQN et PPO cherchent tous à maximiser, chacun à sa façon.
**Exemple :** Si un camion livre dans 2 pas (r₁ = 0 ; r₂ = 1,0) avec γ = 0,99, alors G_t = 0 + 0,99 × 1,0 = 0,99. Si la livraison se fait en 5 pas, G_t = 0,99⁴ × 1,0 ≈ 0,96. L'agent apprend donc à préférer les affectations qui amènent à une livraison rapide.
**Pour aller plus loin :** $G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k} = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \ldots$ Relation avec V et Q : $V^\pi(s) = \mathbb{E}_\pi[G_t \mid s_t = s]$, $Q^\pi(s,a) = \mathbb{E}_\pi[G_t \mid s_t=s, a_t=a]$. Horizon fini dans ce projet : $T = 480$ min, donc $G_t = \sum_{k=0}^{T-t} \gamma^k r_{t+k}$.

---

## Programmation dynamique
> *📌 Contexte de base — non nommé dans le mémoire, mais le jury peut demander le lien entre DP et Q-Learning.*

**Définition :** Une technique algorithmique qui résout des problèmes complexes en les décomposant en sous-problèmes plus simples, en mémorisant les solutions déjà calculées — la base mathématique de l'équation de Bellman et de tous les algorithmes RL basés sur la valeur.
**Explication :** L'idée est que la valeur optimale d'une situation peut être calculée à partir de la valeur optimale des situations suivantes — sans refaire les calculs depuis zéro. C'est le « principe d'optimalité de Bellman ». La différence entre programmation dynamique classique et RL : la DP exige de connaître le modèle complet de l'environnement (toutes les transitions $P(s'|s,a)$ et récompenses $R(s,a)$), tandis que le RL apprend en interagissant avec l'environnement sans ce modèle.
**Exemple :** Value Iteration et Policy Iteration sont des algorithmes de DP : ils parcourent tous les états possibles et appliquent l'équation de Bellman à répétition jusqu'à convergence. Impossible sur ce projet (147 dimensions continues = espace infini). Q-Learning est une version approchée qui n'a pas besoin du modèle complet : il apprend $Q(s,a)$ directement par essais.
**Pour aller plus loin :** DP : $V^*(s) = \max_a \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma V^*(s')]$ — nécessite $P$ explicite. Q-Learning approche la même solution sans $P$, en observant les transitions réelles. Complexité DP : $O(|\mathcal{S}|^2 \cdot |\mathcal{A}|)$ par itération — infaisable pour $|\mathcal{S}| = \infty$.

---

## Itération de valeur (Value Iteration)
> *📌 Contexte de base — non nommé dans le mémoire, mais utile pour expliquer pourquoi on ne peut pas faire de DP sur cet environnement continu.*

**Définition :** Un algorithme de planification par programmation dynamique qui calcule la fonction de valeur optimale $V^*$ en appliquant l'équation de Bellman à tous les états, de façon répétée, jusqu'à convergence.
**Explication :** L'algorithme parcourt tous les états, met à jour chaque valeur $V(s)$ avec la meilleure action possible, et répète jusqu'à ce que les valeurs ne changent plus. C'est l'algorithme théoriquement exact pour résoudre un MDP — mais il nécessite de connaître tout le modèle ($P$, $R$) et de pouvoir énumérer tous les états. Q-Learning est son équivalent model-free, qui apprend $Q(s,a)$ directement par expérience.
**Exemple :** Sur un MDP avec 5 états et 3 actions, Value Iteration converge en quelques dizaines d'itérations. Sur ce projet (état continu à 147 dimensions) : impossible directement — c'est pourquoi on utilise Q-Learning (avec discrétisation) ou DQN/PPO (avec approximation neuronale).
**Pour aller plus loin :** Mise à jour : $V_{k+1}(s) = \max_a \sum_{s'} P(s'|s,a)[R + \gamma V_k(s')]$. Convergence garantie si $\gamma < 1$. Point fixe : $V^* = TV^*$ où $T$ est l'opérateur de Bellman. Q-Learning = Value Iteration stochastique sans modèle.

---

## Itération de politique (Policy Iteration)
> *📌 Contexte de base — non nommé dans le mémoire, mais PPO peut être vu comme une Policy Iteration approchée et model-free.*

**Définition :** Un algorithme de planification alternant entre évaluation de la politique courante (calculer $V^\pi$) et amélioration de la politique (choisir la meilleure action selon $V^\pi$) — converge vers la politique optimale $\pi^*$.
**Explication :** Contrairement à Value Iteration qui optimise directement les valeurs, Policy Iteration part d'une politique initiale, l'évalue complètement, puis la met à jour pas à pas vers une meilleure politique. Il converge souvent en moins d'itérations que Value Iteration. Même limitation : nécessite le modèle complet et l'énumération des états.
**Exemple :** Policy Iteration dans ce projet serait impossible (espace continu). PPO peut être vu comme une Policy Iteration approchée et model-free : il évalue sa politique courante via le critique (V(s)) puis améliore via le gradient de politique, sans jamais connaître $P(s'|s,a)$.
**Pour aller plus loin :** Évaluation : résoudre $V^\pi(s) = \sum_a \pi(a|s)\sum_{s'} P(s'|s,a)[R + \gamma V^\pi(s')]$. Amélioration : $\pi'(s) = \arg\max_a \sum_{s'}P(s'|s,a)[R+\gamma V^\pi(s')]$. PPO ≈ Policy Iteration approchée : évaluation via GAE, amélioration via $L^\text{CLIP}$.

---

## Équation de Bellman
**Définition :** La relation mathématique fondamentale qui dit que la valeur d'une situation aujourd'hui = récompense immédiate + valeur (actualisée) de la meilleure situation qui peut suivre — le principe derrière tous les algorithmes RL basés sur la valeur.
**Explication :** C'est comme évaluer une position aux échecs : la valeur de ma position actuelle = ce que je gagne au coup suivant + la valeur de la position résultante si je joue parfaitement ensuite. Cette définition est récursive — elle se répète à l'infini — mais c'est précisément ça qui permet à Q-Learning, SARSA et DQN de converger vers la politique optimale en appliquant cette relation encore et encore jusqu'à ce que les estimations se stabilisent.
**Exemple :** Q-Learning applique l'équation de Bellman à chaque pas : "la valeur d'envoyer ce camion vers la pelle 2 = récompense reçue + 0,99 × valeur de la meilleure action dans la situation suivante". La règle de mise à jour Q-Learning est une descente de gradient stochastique vers cette équation.
**Pour aller plus loin :** Pour Q : $Q^*(s,a) = \mathbb{E}[r + \gamma \max_{a'} Q^*(s', a')]$. Pour V : $V^*(s) = \max_a \mathbb{E}[r + \gamma V^*(s')]$. Convergence de Q-Learning garantie sous hypothèses de visite suffisante (théorème de Watkins, 1989).

---

## MDP (Processus de Décision Markovien)
**Définition :** C'est la façon « officielle » de décrire un problème de décision : on dit clairement ce que l'agent voit, ce qu'il peut faire, comment le monde change, et ce qui lui rapporte des points.
**Explication :** Pense à un jeu de société : à chaque tour, tout ce qui compte pour décider du prochain coup, c'est la position actuelle du plateau — pas l'historique complet de la partie. Un MDP, c'est exactement cette idée : ce qui se passe ensuite ne dépend que de la situation présente et de l'action choisie maintenant, jamais de ce qui s'est passé avant. Grâce à cette règle, l'agent n'a pas besoin de « se souvenir » de tout son passé : la situation actuelle lui suffit pour décider.
**Exemple :** Dans le projet, la « situation actuelle » regroupe 147 chiffres : longueur des files d'attente, positions des camions, heure de la journée. L'agent choisit entre 7 actions. La journée simulée dure 8h (480 min), sans « fin de partie » — on regarde le tonnage total.
**Pour aller plus loin :** Formalisation par le 4-tuple $(\mathcal{S},\mathcal{A},\mathcal{P},\mathcal{R})$ ; $s_t=(\{q_p\},\{x_c\},\{z_r\},t_{\text{courant}})$, 147 valeurs normalisées ; $\mathcal{A}=\text{Discrete}(7)$ ; propriété markovienne $\mathcal{P}(s_{t+1}|s_t,a_t)$ ; horizon fini $T=480$ min sans état terminal.

---

## Épisode
**Définition :** Une « partie » complète dans la simulation : l'agent démarre dans un état initial, prend des décisions jusqu'à la fin du poste, mesure le résultat — puis recommence de zéro.
**Explication :** C'est exactement comme une partie de jeu de société : on distribue les cartes (état initial), on joue toute la partie (des centaines de décisions), on regarde le score final (tonnage transporté), puis on remet tout en place et on rejoue — en s'améliorant à chaque fois. Un épisode = un poste de travail de 8 heures simulées.
**Exemple :** Au début de chaque épisode, tous les camions sont au yard, les pelles sont libres, l'heure est à 0. Après 480 min simulées, on mesure la productivité (ex : 4 208 t/h pour PPO en nominal). Q-Learning s'entraîne sur 30 000 épisodes ; PPO sur ≈ 9 677 épisodes (2 M pas au total).
**Pour aller plus loin :** Horizon fixe $T = 480$ min, pas d'état terminal. $30\,000$ épisodes tabulaires ≈ 15–20 M pas de décision. À chaque épisode, `reset(seed)` remet le simulateur dans un état initial reproductible.

---

## Espace d'observation (vecteur d'état)
**Définition :** Les 147 chiffres que l'agent reçoit à chaque instant pour décrire la situation complète de la mine — tout ce qu'il « voit » pour prendre sa décision.
**Explication :** C'est comme un tableau de bord complet : position et statut de chaque camion, longueur des files d'attente, distances aux ressources, heure dans le poste. Ces 147 chiffres sont normalisés entre 0 et 1 pour que le réseau de neurones puisse les traiter efficacement.
**Exemple :** Les 147 dimensions : 3 files pelles + 2 files dumps + 12 statuts + 12 disponibilités + 12 tonnages + 12 carburants + 84 positions one-hot (12×7 nœuds) + 3 distances camion→pelles + 6 distances pelles→dumps + 1 temps courant = 147. Les 9 dimensions de distance ont été ajoutées pour que l'agent ait la même information que les heuristiques.
**Pour aller plus loin :** `observation_space = Box(0, 1, shape=(147,), dtype=float32)`. Les méthodes tabulaires extraient 6 features clés et les discrétisent en 8 bins. DQN et PPO consomment les 147 valeurs continues brutes via le MLP 128×128.

---

## Normalisation des observations
**Définition :** La transformation qui ramène toutes les valeurs du vecteur d'état dans l'intervalle [0, 1] — indispensable pour que les réseaux de neurones apprennent correctement.
**Explication :** Sans normalisation, une jauge qui varie de 0 à 480 minutes écrase complètement une autre qui varie de 0 à 3 km dans les calculs du réseau. En ramenant tout entre 0 et 1, chaque dimension a un poids initial équivalent, ce qui stabilise l'apprentissage et évite les gradients explosifs lors de la rétropropagation.
**Exemple :** File d'attente (0–30 camions) ÷ 30 → [0, 1]. Temps courant (0–480 min) ÷ 480 → [0, 1]. Distance (0–10 km) ÷ 10 → [0, 1]. Position camion : encodée en one-hot → valeur 0 ou 1.
**Pour aller plus loin :** Requis par Stable-Baselines3 (`observation_space = Box(0, 1, ...)`). Évite les gradients explosifs : $\|\nabla_\theta L\| \ll 1$ si toutes les entrées sont dans $[0, 1]$. Les méthodes tabulaires n'ont pas ce besoin car elles discrétisent l'état.

---

## Discrétisation de l'espace d'état
**Définition :** La technique qui transforme un espace d'état continu en un nombre fini de cases — nécessaire pour que Q-Learning et SARSA puissent fonctionner avec une table Q.
**Explication :** Q-Learning tient un carnet avec une ligne par situation. Mais les situations continues peuvent prendre une infinité de valeurs. La discrétisation découpe chaque dimension en un petit nombre de « cases » (bins) : « file de 0 à 5 camions → case 1 », « 6 à 10 → case 2 », etc. L'état complet devient un tuple de numéros de cases utilisable comme clé dans le carnet Q.
**Exemple :** Dans le projet, 6 features décisionnelles sont discrétisées en 8 bins chacune. L'espace théorique est $8^5 \times 3 = 98\,304$ combinaisons. En pratique, 17 034 états visités sur 30 000 épisodes.
**Pour aller plus loin :** `n_bins = 8`. Discrétisation : $\text{bin}(x) = \min(\lfloor x \times n\_\text{bins} \rfloor, n\_\text{bins}-1)$. Si on discrétisait les 147 dimensions : $8^{147} \approx 10^{133}$ — inaccessible. D'où le Deep RL qui approxime sans discrétisation.

---

## Complexité algorithmique
**Définition :** Une mesure de la quantité de ressources (temps, mémoire) nécessaires à un algorithme en fonction de la taille de son entrée — notée en « grand O » (O(n), O(n²), etc.).
**Explication :** O(n) signifie que si la taille des données double, le temps de calcul double. O(n²) : le temps quadruple. O(2ⁿ) : explosion exponentielle — l'algorithme devient rapidement impraticable. En RL, la complexité de la table Q est directement liée à la taille de l'espace d'état : avec des états continus à 147 dimensions, cette taille est infinie, ce qui rend les méthodes tabulaires impossibles sans discrétisation.
**Exemple :** Table Q tabulaire : $O(|\mathcal{S}| \times |\mathcal{A}|)$ en mémoire. Avec $|\mathcal{S}| = 8^5 \times 3 = 98\,304$ et $|\mathcal{A}| = 7$ : 688 128 entrées — gérable. Avec les 147 dimensions continues discrétisées en 8 bins chacune : $8^{147} \approx 10^{133}$ — impossible. D'où le MLP 128×128 qui n'a que ≈ 35 700 paramètres pour couvrir tout l'espace continu.
**Pour aller plus loin :** Complexités clés : inférence MLP = $O(p)$ avec $p$ = nb de paramètres. Dijkstra sur graphe = $O((V+E)\log V)$. Recherche dans table Q = $O(1)$ (hash map). Entraînement DQN/PPO = $O(T \times p)$ avec $T$ = nb de steps. La complexité spatiale de la table Q est la raison fondamentale du passage au Deep RL.

---

## Environnement Gymnasium
**Définition :** Un cadre standard qui permet de brancher n'importe quel agent sur n'importe quelle simulation — comme une prise électrique universelle pour le RL.
**Explication :** Gymnasium donne à chaque simulation les mêmes boutons : `reset()` (recommencer à zéro) et `step(action)` (jouer un coup et voir ce qui se passe). Un agent qui sait utiliser ces deux boutons peut s'entraîner sur n'importe quel environnement conforme — mine, jeu vidéo, robot — sans rien changer à son code.
**Exemple :** La mine simulée du projet est un environnement Gymnasium. `step(3)` signifie « envoie le camion courant vers la pelle 2 et le dump 1 » : la simulation avance et renvoie le nouvel état et la récompense.
**Pour aller plus loin :** Contrat : `reset()` → $s_0$, `step(action)` → `(observation, reward, terminated, truncated, info)` ; `observation_space = Box(0,1,shape=(147,))`, `action_space = Discrete(7)`.

---

## Stable-Baselines3
**Définition :** Une bibliothèque Python qui fournit des implémentations prêtes à l'emploi des algorithmes RL les plus connus — DQN, PPO, SAC, A2C, etc.
**Explication :** C'est un livre de recettes de cuisine : au lieu d'implémenter soi-même chaque algorithme (ce qui prend des semaines et introduit des bugs), on prend la recette déjà testée, on lui donne son environnement et ses hyperparamètres, et on lance l'entraînement. La bibliothèque gère la rétropropagation, l'optimiseur, le replay buffer, etc.
**Exemple :** Dans le projet : `PPO("MlpPolicy", env, learning_rate=3e-4, gamma=0.99, gae_lambda=0.95, clip_range=0.2, batch_size=64).learn(total_timesteps=2_000_000)`. Entraînement en ≈ 2h sur CPU.
**Pour aller plus loin :** Version 2.8.0 utilisée. Implémente aussi DQN avec replay buffer et target network. Basée sur PyTorch. Interface unifiée : `model.learn()`, `model.predict()`, `model.save()`, `model.load()`.

---

## Hyperparamètre
**Définition :** Un réglage que le chercheur fixe AVANT l'entraînement — à distinguer des paramètres (poids du réseau) que l'algorithme apprend lui-même.
**Explication :** C'est comme régler une machine à laver avant de lancer : tu choisis la température, la durée, la vitesse d'essorage — ces réglages ne changent pas pendant le lavage. La machine fait son travail toute seule. De même, le chercheur fixe learning rate, discount, taille du batch AVANT l'entraînement — puis l'algorithme ajuste ses propres poids tout seul.
**Exemple :** Hyperparamètres PPO dans le projet : $\alpha = 3 \times 10^{-4}$, $\gamma = 0{,}99$, $\lambda_\text{GAE} = 0{,}95$, $\epsilon_\text{clip} = 0{,}2$, batch = 64, MLP 128×128. Valeurs par défaut SB3, éprouvées sur une large gamme d'environnements.
**Pour aller plus loin :** Hyperparamètres $\neq$ paramètres appris. Paramètres appris : $\theta$ (≈ 35 700 poids du MLP) — mis à jour à chaque step. Hyperparamètres : fixés une fois. La recherche d'hyperparamètres (grid search, Bayesian optimization) est une piste d'amélioration citée en perspectives.

---

## Q-Learning
**Définition :** Une méthode qui tient un grand carnet où, pour chaque situation et chaque action possible, on note « combien de points ça vaut à peu près ».
**Explication :** Chaque page du carnet = une situation. Chaque ligne = une action. Au début, toutes les notes valent zéro. Chaque fois que l'agent essaie une action, il corrige la note dans son carnet — en se disant « si je joue ensuite le mieux possible, combien ça vaudra ». Petit à petit, le carnet devient fiable.
**Exemple :** Si une action avait une note de 50, rapporte 10 points, et que la meilleure action suivante vaut 55, la note passe à 51,45. Dans le projet, ce carnet contient 17 034 situations différentes.
**Pour aller plus loin :** $Q(s_t,a_t) \leftarrow Q(s_t,a_t) + \alpha[r_t + \gamma \max_{a'} Q(s_{t+1},a') - Q(s_t,a_t)]$ ; exemple : $Q=50, r=10, \gamma=0{,}99, \max Q'=55, \alpha=0{,}1 \Rightarrow 51{,}45$. Off-policy ; $\epsilon$ décroît de 1,0 à 0,01.

---

## Stratégie ε-greedy
**Définition :** La règle de compromis exploration/exploitation : avec probabilité ε, l'agent essaie une action aléatoire ; sinon, il choisit la meilleure action connue.
**Explication :** C'est comme un touriste qui découvre une nouvelle ville : parfois il essaie un restaurant au hasard (exploration), le reste du temps il retourne dans son préféré (exploitation). ε contrôle la fréquence des visites aléatoires. Au début, ε ≈ 1 (tout aléatoire) ; à la fin, ε ≈ 0 (quasi-exploitation totale).
**Exemple :** Q-Learning et SARSA : ε de 1,0 à 0,01 sur 30 000 épisodes. DQN : de 1,0 à 0,02 sur 2M steps. PPO n'utilise pas ε-greedy — il régule l'exploration via l'entropie de sa distribution de probabilités.
**Pour aller plus loin :** Décroissance linéaire : $\epsilon_k = \max(\epsilon_{\min}, \epsilon_0 - k \cdot \Delta\epsilon)$. À $\epsilon = 0{,}01$ : 99 % exploitation ($\arg\max_a Q$) + 1 % exploration aléatoire uniforme sur les 7 actions.

---

## SARSA
**Définition :** Une cousine de Q-Learning, mais plus prudente : elle note la valeur de l'action qu'on va *vraiment* faire ensuite, pas la meilleure action imaginable.
**Explication :** Q-Learning corrige son carnet en imaginant « si je jouais ensuite le coup parfait ». SARSA regarde ce que l'agent fait *réellement* au coup suivant — même si ce n'est pas le meilleur coup possible. Résultat : SARSA est plus réaliste et prudente, mais souvent moins performante que Q-Learning.
**Exemple :** Avec les mêmes chiffres (note 50, récompense 10), si le coup vraiment joué ensuite vaut 52 (au lieu du meilleur possible, 55), la note ne monte qu'à 51,15 au lieu de 51,45. Dans le projet, SARSA est systématiquement moins bon que Q-Learning.
**Pour aller plus loin :** $Q(s_t,a_t) \leftarrow Q(s_t,a_t) + \alpha[r_t + \gamma Q(s_{t+1},a_{t+1}) - Q(s_t,a_t)]$ ; $Q(s',a_{t+1})=52 \Rightarrow 51{,}15$. On-policy (utilise l'action réellement prise, pas l'optimale).

---

## DQN (Deep Q-Network)
**Définition :** Comme Q-Learning, mais au lieu d'un carnet, on utilise un réseau de neurones capable de « deviner » la valeur même pour des situations jamais vues.
**Explication :** Q-Learning ne peut pas stocker une ligne par situation quand il y a 147 variables continues — l'espace est infini. DQN remplace le carnet par un réseau de neurones : il reçoit la situation et calcule directement une note pour chaque action, même pour des situations jamais vues exactement. Deux astuces stabilisent l'apprentissage : Experience Replay (replay buffer) et Target Network (réseau cible).
**Exemple :** Dans le projet, DQN atteint 4 168,5 t/h en nominal et reste très stable même en high_breakdown (3 991,8 t/h), avec faible variance ($\sigma \approx 65$ vs PPO $\sigma = 350$ en high_load).
**Pour aller plus loin :** Réseau MLP 128×128, ReLU ; replay buffer 200 000 transitions, batch=64 ; réseau cible $Q_{\theta^-}$ pour cible $r+\gamma\max_{a'}Q_{\theta^-}(s',a')$ ; $\epsilon$ décroît de 1,0 à 0,02 sur 2M steps.

---

## Réseau cible (Target Network)
**Définition :** Une copie « figée » du réseau de DQN, utilisée comme référence stable pour calculer les cibles d'apprentissage — sans ça, l'agent courrait après ses propres prédictions changeantes.
**Explication :** Si la cible d'apprentissage changeait à chaque pas (parce qu'elle est calculée par le même réseau qu'on est en train de modifier), l'apprentissage devient instable — on court après une cible qui bouge. Le réseau cible est une copie mise à jour seulement périodiquement, servant de référence fixe pendant un certain nombre de pas.
**Exemple :** DQN calcule la cible $y = r + \gamma \max_{a'} Q_{\theta^-}(s', a')$ avec $\theta^-$ figé. Sans ce mécanisme, les oscillations peuvent déstabiliser tout l'apprentissage.
**Pour aller plus loin :** Mise à jour dure : $\theta^- \leftarrow \theta$ toutes les $N$ steps. Ou douce : $\theta^- \leftarrow \tau\theta + (1-\tau)\theta^-$. PPO n'a pas besoin de réseau cible car c'est une méthode on-policy.

---

## PPO (Proximal Policy Optimization)
**Définition :** Une méthode d'apprentissage qui améliore la politique de l'agent petit pas par petit pas, avec une « ceinture de sécurité » pour ne jamais changer trop brusquement.
**Explication :** Si une action s'est révélée bonne, PPO augmente un peu sa probabilité d'être choisie — mais une « ceinture de sécurité » (clipping) empêche cette probabilité de changer trop violemment en une seule fois. Cela évite de tout casser si l'estimation était légèrement fausse. C'est pourquoi PPO est plus stable que DQN dans les environnements bruités.
**Exemple :** Avec un MLP 128×128 entraîné sur 2M pas (≈ 9 677 épisodes), PPO atteint 4 208,75 t/h en nominal — la meilleure productivité de toutes les méthodes testées.
**Pour aller plus loin :** Ratio $r_t(\theta)=\pi_\theta(a|s)/\pi_{\theta_\text{old}}(a|s)$ ; objectif clippé $L^{CLIP}=\mathbb{E}[\min(r_t\hat A_t, \text{clip}(r_t,1-\epsilon,1+\epsilon)\hat A_t)]$, $\epsilon=0{,}2$ → ratio limité à $[0{,}8 ; 1{,}2]$. On-policy, pas de replay buffer.

---

## Méthodes basées sur la valeur vs basées sur la politique
**Définition :** Les deux grandes familles d'algorithmes RL — les premières apprennent à évaluer les situations/actions, les secondes apprennent directement à choisir les meilleures actions.
**Explication :** **Basées sur la valeur** (Q-Learning, SARSA, DQN) : l'agent apprend Q(s,a) qui note chaque action, puis choisit la meilleure note. **Basées sur la politique** (PPO) : l'agent apprend directement une règle de décision (la politique $\pi$), sans passer par une table de valeurs. Plus stable dans les environnements bruités et scalable à plus d'actions.
**Exemple :** DQN calcule Q(état, action 1), Q(état, action 2)... et prend l'argmax. PPO calcule directement les probabilités de chaque action sans Q-valeur. En high_load, PPO reste plus stable car la politique n'oscille pas autant.
**Pour aller plus loin :** Value-based : $\pi^* = \arg\max_a Q^*(s,a)$. Policy-based : optimise $J(\theta) = \mathbb{E}_{\pi_\theta}[G_t]$ directement. Acteur-critique (PPO) : hybride — acteur pour la politique, critique pour évaluer les situations. DQN = value-based off-policy. PPO = policy-based on-policy.

---

## Modèle-libre (Model-Free)
**Définition :** Une approche RL où l'agent apprend par essais-erreurs, sans jamais construire un modèle explicite de l'environnement — c'est l'approche de TOUS les agents de ce mémoire.
**Explication :** Un agent modèle-monde (model-based) essaierait d'apprendre comment l'environnement fonctionne : « si j'envoie un camion vers la pelle 1 avec 3 camions en file, il attendra environ X minutes ». Un agent modèle-libre ne fait pas ça : il interagit directement et apprend par les récompenses reçues, sans modéliser la dynamique.
**Exemple :** Q-Learning, SARSA, DQN et PPO sont tous model-free : ils ne savent pas que les temps de trajet suivent une loi log-normale. Ils l'apprennent implicitement via les récompenses. Avantage : pas besoin de modéliser l'environnement (souvent impossible en pratique).
**Pour aller plus loin :** Model-free : $Q(s,a)$ ou $\pi(a|s)$ appris par expérience directe. Model-based : apprend $\hat{\mathcal{P}}(s'|s,a)$ et $\hat{\mathcal{R}}(s,a)$, puis planifie (ex : AlphaZero, Dreamer). Les 4 agents du projet sont model-free via Stable-Baselines3.

---

## Méthode de Monte Carlo (RL)
**Définition :** Une approche RL qui attend la FIN complète d'un épisode avant de mettre à jour ses estimations — à l'opposé des méthodes TD qui mettent à jour à chaque pas.
**Explication :** Monte Carlo attend d'avoir joué toute la partie avant de juger si ses décisions étaient bonnes. Problème : avec T = 480 minutes et ≈ 200 décisions par épisode, attribuer la récompense finale à la décision n°3 est très imprécis. Les méthodes TD (Q-Learning, SARSA, PPO via GAE) apprennent à chaque pas sans attendre la fin.
**Exemple :** Pourquoi ce mémoire utilise TD et non Monte Carlo : attendre 480 min simulées avant d'apprendre quoi que ce soit ralentirait considérablement la convergence. L'erreur TD δt permet des corrections immédiates.
**Pour aller plus loin :** Monte Carlo : $G_t = \sum_{k=0}^{T-t} \gamma^k r_{t+k}$ calculé après l'épisode. TD : $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ calculé à chaque pas. GAE est un compromis entre les deux via $\lambda \in [0,1]$.

---

## Bootstrap (en RL)
**Définition :** Utiliser ses propres estimations actuelles pour calculer les cibles d'apprentissage, au lieu d'attendre les vraies valeurs finales.
**Explication :** Au lieu d'attendre de savoir combien la situation suivante vaut vraiment (ce qui nécessiterait la fin de l'épisode), l'agent utilise son estimation actuelle de V(s') comme cible. C'est plus rapide que Monte Carlo, mais introduit un biais car on apprend à partir d'estimations imparfaites.
**Exemple :** Dans Q-Learning : cible $= r_t + \gamma \max_{a'} Q(s_{t+1}, a')$. Le terme $Q(s_{t+1}, a')$ est une estimation (bootstrap). Au début, mauvaise (table Q à 0) ; elle s'améliore progressivement. PPO via GAE bootstrap sur $V_\phi(s_{t+1})$.
**Pour aller plus loin :** Toutes les méthodes TD bootstrappent : Q-Learning, SARSA, DQN, PPO. Monte Carlo : pas de bootstrap (attend le vrai $G_t$). Compromis bootstrap : biais (estimation imparfaite) + faible variance. GAE contrôle ce biais-variance via $\lambda \in [0,1]$.

---

## Gradient de politique (Policy Gradient)
**Définition :** La famille d'algorithmes RL qui améliorent directement la politique en calculant dans quelle direction modifier ses paramètres pour augmenter le retour espéré — PPO appartient à cette famille.
**Explication :** Au lieu d'apprendre Q(s,a) (comme DQN), les méthodes de gradient de politique demandent : « si j'avais légèrement modifié mes paramètres θ, aurais-je obtenu plus de récompenses ? ». Elles calculent ce gradient et font un pas dans la bonne direction. PPO ajoute une sécurité : le pas ne peut pas être trop grand.
**Exemple :** PPO calcule $\nabla_\theta J(\theta)$ : si l'avantage $\hat{A} > 0$, on augmente la probabilité de cette action ; si $\hat{A} < 0$, on la diminue. DQN n'a pas de gradient de politique — il dérive sa politique de la Q-valeur. C'est la différence fondamentale.
**Pour aller plus loin :** Policy Gradient Theorem : $\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}[\nabla_\theta \log \pi_\theta(a|s) \cdot Q^\pi(s,a)]$. PPO = version stabilisée via $L^{CLIP}$. REINFORCE = version Monte Carlo la plus simple. L'objectif tronqué limite $|r_t(\theta) - 1| \leq \epsilon_\text{clip} = 0{,}2$.

---

## GAE (Generalized Advantage Estimation)
**Définition :** Une façon équilibrée de juger « est-ce que cette décision était bonne ? » — ni trop à court terme (TD pur), ni trop à long terme (Monte Carlo pur).
**Explication :** Pour juger si une décision était bonne, on pourrait regarder seulement ce qui se passe juste après (rapide mais peu fiable), ou attendre la fin de la journée (fiable mais lent). GAE fait un compromis : il regarde les conséquences sur les prochains pas, en donnant de moins en moins d'importance à ce qui est loin dans le temps. Le paramètre λ contrôle cet horizon.
**Exemple :** Dans le projet, GAE avec γ=0,99 et λ=0,95 permet de juger une décision en regardant l'effet sur ≈ 10 pas en avant — assez pour voir si ça a désengorgé une file, sans attendre la fin des 8h.
**Pour aller plus loin :** $\hat{A}_t = \sum_{l=0}^{\infty}(\gamma\lambda)^l \delta_{t+l}$ où $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$. Avec $\gamma=0{,}99$, $\lambda=0{,}95$ : $(\gamma\lambda)^{10} \approx 0{,}54$. $\lambda=0$ → TD pur ; $\lambda=1$ → Monte Carlo.

---

## CRISP-RL
**Définition :** Le cadre méthodologique en 6 étapes suivi dans ce mémoire pour structurer le projet RL, de la compréhension du problème jusqu'à l'évaluation comparative.
**Explication :** Contrairement à un projet de data science classique (données fixes → modèle), le RL exige de construire d'abord un environnement de simulation, puis de formuler le MDP, d'entraîner les agents, et enfin d'évaluer — chaque étape dépendant de la précédente. Si les résultats révèlent un problème, on retourne à l'étape correspondante et on itère.
**Exemple :** Dans ce projet : (1) comprendre la logistique minière → (2) modéliser l'environnement → (3) formuler le MDP → (4) développer Q-Learning/SARSA/DQN/PPO → (5) entraîner sur 3 scénarios → (6) comparer 8 politiques. La découverte que $\Delta R < 6 \times 10^{-4}$ a déclenché un retour itératif à l'étape 3.
**Pour aller plus loin :** CRISP-RL = CRoss-Industry Standard Process for Reinforcement Learning. Adaptation du CRISP-DM proposée par Szatmári (2025). 6 étapes : compréhension métier → environnement → MDP → agents → entraînement → évaluation.

---

## Baseline
**Définition :** Une méthode « témoin », simple et fixe, à laquelle on compare l'agent pour savoir s'il fait vraiment mieux.
**Explication :** Sans référence, impossible de savoir si 4 208 t/h c'est bien ou pas. Une baseline applique une règle simple sans apprentissage. Si l'agent ne fait pas mieux que la meilleure baseline, tout l'apprentissage n'a servi à rien. C'est une exigence de rigueur scientifique fondamentale en RL.
**Exemple :** Dans le projet, 4 baselines (FIFO, Fixed Assignment, Nearest Shovel, Shortest Path) sont comparées aux 4 agents RL. PPO fait 4 208,75 t/h vs Fixed Assignment 4 074,0 t/h (+3,3 %), confirmé par test statistique.
**Pour aller plus loin :** 4 baselines vs 4 agents RL, 3 scénarios, 10 réplications (seeds 42-51). Welch PPO vs Fixed nominal : $t=6{,}356$, $p=3{,}0\times10^{-6}$.

---

## Politique (en RL)
**Définition :** C'est la « façon de réagir » de l'agent — ce qu'il fait dans chaque situation, comme une habitude apprise.
**Explication :** Une politique associe à chaque situation observée une action (ou une distribution sur les actions). Au début de l'entraînement, ces habitudes sont presque aléatoires. À force d'essais et de corrections, elles deviennent de plus en plus pertinentes.
**Exemple :** La politique de PPO ne dit pas « fais exactement ça », mais donne des probabilités : 5 % pour l'action 1, 40 % pour l'action 3, etc. DQN en exploitation a une politique déterministe : il choisit toujours $\arg\max_a Q(s,a)$.
**Pour aller plus loin :** Politique stochastique : $\pi(a|s) = P(a_t=a|s_t=s)$ (PPO). Déterministe : $\pi(s) = \arg\max_a Q(s,a)$ (DQN en exploitation). Politique optimale : $\pi^*(s) = \arg\max_a Q^*(s,a)$.

---

## Algorithme glouton
> *📌 Contexte de base — seule « l'action gloutonne » (argmax) est nommée dans le mémoire ; le concept général n'y est pas défini.*

**Définition :** Un algorithme qui, à chaque étape, choisit toujours l'option localement optimale sans considérer les conséquences futures — rapide et simple, mais souvent sous-optimal sur le long terme.
**Explication :** C'est comme choisir toujours la file la plus courte à la caisse sans anticiper que tout le monde autour fait pareil — créant finalement un embouteillage sur cette caisse. L'algorithme glouton ne regarde jamais plus loin que le prochain pas. Il est efficace quand les décisions locales optimales mènent à une solution globale optimale (ex : algorithme de Dijkstra), mais échoue pour les problèmes où optimiser localement nuit à l'optimal global.
**Exemple :** L'heuristique Nearest Shovel est un algorithme glouton : elle envoie toujours le camion vers la pelle la plus proche, sans considérer les files d'attente futures. En high_load, ce choix localement optimal provoque 200 min d'attente — un résultat globalement catastrophique. L'agent RL n'est pas glouton : grâce à γ = 0,99, il considère les conséquences sur les 100 prochains pas.
**Pour aller plus loin :** En RL, la politique gloutonne dérive de la Q-table : $\pi_g(s) = \arg\max_a Q(s,a)$. C'est la politique d'exploitation pure (ε = 0). La stratégie ε-greedy combine greediness (1-ε) et exploration aléatoire (ε) pour éviter les optima locaux pendant l'entraînement.

---

## REINFORCE
> *📌 Contexte de base — non nommé dans le mémoire, mais permet de comprendre pourquoi PPO est une amélioration du gradient de politique de base.*

**Définition :** Le premier algorithme de gradient de politique — calcule le gradient en utilisant le retour Monte Carlo complet de chaque épisode pour mettre à jour directement la politique.
**Explication :** REINFORCE attend la fin d'un épisode entier, calcule le retour G_t pour chaque décision prise, puis ajuste les probabilités des actions : si G_t est élevé, augmenter la probabilité des actions prises ; si G_t est faible, la diminuer. Simple et correct en théorie, mais très bruité en pratique car G_t dépend de tout ce qui s'est passé après chaque décision, y compris du hasard.
**Exemple :** Si PPO est comme un entraîneur sportif qui analyse les 10 dernières minutes de jeu pour corriger les erreurs (GAE avec λ=0,95), REINFORCE est comme un entraîneur qui attend la fin du match entier avant de donner le moindre feedback — ce qui est juste mais peu précis. PPO est une version stabilisée de REINFORCE avec bootstrap (GAE) et clipping.
**Pour aller plus loin :** Mise à jour REINFORCE : $\theta \leftarrow \theta + \alpha \sum_t G_t \nabla_\theta \log \pi_\theta(a_t|s_t)$. Problème : variance très élevée de $G_t$. Solutions : soustraire une baseline $b(s_t)$ (→ réduit la variance sans biais), utiliser GAE au lieu de $G_t$ (→ PPO). REINFORCE avec baseline = précurseur direct de l'acteur-critique.

---

## Fonction de récompense
**Définition :** Le système de points qui dit à l'agent, après chaque action, si c'était une bonne ou une mauvaise idée — le seul signal d'apprentissage du RL.
**Explication :** C'est exactement un score de jeu vidéo : l'agent gagne ou perd des points selon plusieurs critères. Le critère principal (livrer du minerai) compte beaucoup plus que les autres (équité des files, distance du trajet), qui servent surtout à guider l'apprentissage intermédiaire.
**Exemple :** Livraison 140t → +1,0. File longue à la pelle choisie → -0,1. Trajet long → -0,025. Variance des files élevée → -0,004. Score net ≈ 0,871 pour une bonne décision.
**Pour aller plus loin :** $R_t = w_1 \frac{\Delta\text{tonnage}}{140} + w_2 \frac{-\text{Var}(\{q_p\})}{100} + w_3 \frac{-d_{\text{trajet}}}{5} + w_4(-\min(\frac{q_{p_i}}{30},1))$, $w_1=1{,}0$, $w_2=0{,}1$, $w_3=0{,}05$, $w_4=0{,}3$.

---

## Reward Shaping
**Définition :** Le fait d'ajouter des récompenses intermédiaires pour guider l'apprentissage, au lieu de récompenser seulement le résultat final.
**Explication :** Si on ne récompense que la livraison finale, l'agent reçoit très peu de signal — la plupart de ses décisions semblent « neutres ». Le reward shaping ajoute de petits indices en cours de route (pénaliser un mauvais choix de pelle) pour guider l'agent vers le bon comportement plus rapidement.
**Exemple :** Dans le projet, sans le terme $w_4$ (pénalité de file), la différence de récompense entre deux actions n'était que $6\times10^{-4}$ — l'apprentissage tabulaire était impossible. Avec $w_4=0{,}3$, la différence passe à $3{,}2\times10^{-2}$ (facteur ~50).
**Pour aller plus loin :** Sans $w_4$ : signal non-discriminant, Q-Learning et SARSA ne progressent pas. Avec $w_4$ : signal différencié, convergence observable. Théorème de Ng et al. (1999) : le reward shaping préserve la politique optimale si la fonction de shaping est une différence de potentiel.

---

## Exploration vs Exploitation
**Définition :** Le dilemme entre essayer quelque chose de nouveau (exploration) et refaire ce qu'on sait déjà bien marcher (exploitation).
**Explication :** Si on n'explore jamais, on risque de rester bloqué sur une solution moyenne sans découvrir la meilleure. Si on explore tout le temps, on n'utilise jamais ce qu'on a appris. Les agents explorent beaucoup au début (peu de connaissances) et de moins en moins au fil de l'entraînement.
**Exemple :** Q-Learning choisit presque au hasard au début, puis presque toujours la meilleure action à la fin. PPO garde plus de diversité (entropie résiduelle 0,50 nat) tandis que DQN se concentre sur 2/7 actions (80,3 % des décisions) en fin d'entraînement.
**Pour aller plus loin :** $\epsilon$-greedy : $\epsilon$ décroît de 1,0 à 0,01 (tabulaire) ou 0,02 (DQN). PPO : coefficient d'entropie $c_\text{ent}=0{,}02$ dans la fonction de perte pour maintenir l'exploration.

---

## Entropie (de la politique)
**Définition :** Un chiffre qui mesure à quel point la politique de l'agent est indécise — entropie élevée = beaucoup de diversité dans les choix ; entropie faible = une action largement dominante.
**Explication :** Entropie maximale = l'agent choisit au hasard parmi toutes les actions (début d'entraînement). Entropie nulle = l'agent choisit toujours la même action (politique déterministe). PPO maintient une entropie non nulle pour continuer à explorer légèrement même après convergence.
**Exemple :** Entropie de la politique PPO : démarre à 1,94 nat (max pour 7 actions = ln 7), finit à 0,50 (nominal) / 0,68 (high_load) / 0,55 (high_breakdown). La valeur plus élevée en high_load traduit l'instabilité de la politique dans ce scénario difficile.
**Pour aller plus loin :** $H(\pi(\cdot|s)) = -\sum_a \pi(a|s)\log\pi(a|s)$. Max $H = \ln 7 \approx 1{,}946$ nats. Coefficient $c_\text{ent}=0{,}02$ dans la perte PPO : $L = L^\text{CLIP} - c_1 L^V + c_2 H(\pi)$.

---

## Replay buffer
**Définition :** Une mémoire tampon où DQN stocke ses expériences passées pour les relire aléatoirement lors de l'apprentissage — brise les corrélations temporelles et améliore la stabilité.
**Explication :** Si DQN apprenait uniquement de l'expérience la plus récente, les exemples consécutifs seraient très similaires (le même camion faisant des trajets similaires) — ce qui biaise l'apprentissage. En stockant 200 000 expériences et en en piochant 64 au hasard, DQN apprend sur des situations variées, y compris des pannes survenues il y a longtemps.
**Exemple :** Dans le projet, DQN garde les 200 000 dernières transitions $(s,a,r,s')$. À chaque step d'apprentissage, il pioche 64 au hasard. PPO n'a pas de replay buffer — il apprend uniquement de ses interactions actuelles (on-policy).
**Pour aller plus loin :** Buffer FIFO de capacité 200 000. Mini-batch de 64 tiré uniformément. Variante : Prioritized Experience Replay (PER) — poids les expériences par leur erreur TD $|\delta_t|$. Non utilisé dans ce projet.

---

## On-policy / Off-policy
**Définition :** Off-policy = l'agent peut apprendre à partir de données collectées avec une ancienne politique ; on-policy = l'agent ne peut apprendre que de ses interactions actuelles.
**Explication :** Off-policy : comme réviser avec tes anciens devoirs, même ceux faits il y a longtemps avec une méthode différente — ils restent valides pour estimer la valeur optimale. On-policy : tu dois réviser uniquement tes exercices récents, faits avec ta méthode actuelle — sinon tes statistiques seraient faussées.
**Exemple :** DQN (off-policy) : réutilise des expériences vieilles de 200 000 pas. PPO (on-policy) : jette les données après chaque mise à jour et en collecte de nouvelles. Q-Learning = off-policy. SARSA = on-policy.
**Pour aller plus loin :** Off-policy : cible $\max_{a'}Q(s',a')$ — estime la politique optimale indépendamment de ce qui est joué. On-policy : cible $Q(s',a_{t+1})$ ou ratio $r_t(\theta)$ — dépend de la politique courante. Implication pratique : DQN peut réutiliser un replay buffer ; PPO ne peut pas.

---

## Temporal difference
**Définition :** Une façon de corriger ses estimations pas à pas, sans attendre la fin de l'épisode — le principe commun à Q-Learning, SARSA, DQN et PPO.
**Explication :** Au lieu d'attendre la fin de la journée pour corriger ses estimations (Monte Carlo), l'agent corrige immédiatement à chaque pas : « je pensais que cette situation valait 40, j'ai gagné 8 et la situation suivante semble valoir 45, donc j'avais sous-estimé — je corrige légèrement vers le haut ». Ce principe est fondamental à tout le RL moderne.
**Exemple :** $V = 40$, $r = 8$, $\gamma = 0{,}99$, $V' = 45$, $\alpha = 0{,}1$ → $\delta_t = 8 + 0{,}99 \times 45 - 40 = 12{,}55$ → $V \leftarrow 41{,}26$.
**Pour aller plus loin :** $V(s_t) \leftarrow V(s_t) + \alpha[r_t + \gamma V(s_{t+1}) - V(s_t)]$. Erreur TD : $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$. GAE est une somme pondérée d'erreurs TD sur plusieurs pas : $\hat{A}_t = \sum_{l\geq 0}(\gamma\lambda)^l \delta_{t+l}$.

---

## Discount factor (γ)
**Définition :** Un coefficient entre 0 et 1 qui contrôle à quel point l'agent valorise les récompenses futures par rapport aux récompenses immédiates.
**Explication :** Chaque récompense future perd un peu de sa valeur à chaque pas qui passe, comme de la glace qui fond. γ proche de 1 : fonte lente, l'agent planifie loin. γ proche de 0 : fonte rapide, l'agent devient myope et ne pense qu'à l'instant présent.
**Exemple :** Dans le projet, γ = 0,99 : l'agent conserve 37 % de la valeur d'une récompense 100 pas plus tard ($0{,}99^{100} \approx 0{,}366$). Une décision maintenant qui évite un embouteillage dans 40 pas reste significativement valorisée.
**Pour aller plus loin :** $0{,}99^{100} \approx 0{,}366$ (37 % conservé après 100 pas). $0{,}9^{100} \approx 0{,}00003$ (quasi nul). γ = 1 : somme non actualisée (problèmes avec horizon infini). γ = 0,99 : horizon effectif ≈ $1/(1-\gamma) = 100$ pas.

---

## Learning rate (α)
**Définition :** La taille du pas de correction à chaque mise à jour — trop grand : instabilité ; trop petit : convergence trop lente.
**Explication :** Imagine corriger le tir d'une fléchette : corriger trop fort → oscillations. Corriger trop peu → très long. Le learning rate contrôle l'amplitude de chaque correction. Les méthodes tabulaires (Q-Learning, SARSA) utilisent des α plus grands car chaque case du carnet est indépendante. Les réseaux de neurones utilisent des α très petits car modifier un poids impacte toutes les prédictions.
**Exemple :** Q-Learning/SARSA : α = 0,1. DQN/PPO : α = 3×10⁻⁴. Un α trop grand pour un réseau peut tout déstabiliser en une seule mise à jour.
**Pour aller plus loin :** $\theta \leftarrow \theta - \alpha \nabla_\theta L(\theta)$. Q-Learning/SARSA : $\alpha=0{,}1$ (mise à jour de la Q-table). Réseaux : $\alpha=3\times10^{-4}$ (Adam avec moments adaptatifs). Décroissance du learning rate (scheduling) parfois utilisée pour améliorer la convergence finale.

---

## Descente de gradient stochastique (SGD)
**Définition :** L'algorithme fondamental d'optimisation des réseaux de neurones : ajuster les poids dans la direction qui réduit la fonction de perte, en utilisant une estimation basée sur un sous-ensemble des données (mini-batch).
**Explication :** La descente de gradient calcule la direction de descente de l'erreur ($-\nabla_\theta L$) et fait un petit pas dans cette direction. « Stochastique » signifie qu'on n'utilise pas toutes les données à chaque pas, mais un mini-batch aléatoire — ce qui est plus rapide et introduit du bruit utile pour échapper aux optima locaux. Adam est une version améliorée de SGD qui adapte le pas individuellement pour chaque poids.
**Exemple :** DQN : à chaque step, on tire 64 expériences du replay buffer, on calcule l'erreur quadratique entre Q prédit et Q cible, puis on fait un pas SGD via Adam ($\alpha = 3 \times 10^{-4}$) pour réduire cette erreur. PPO fait pareil mais sur les données collectées depuis la dernière mise à jour de politique.
**Pour aller plus loin :** Mise à jour SGD : $\theta \leftarrow \theta - \alpha \nabla_\theta L(\theta; \mathcal{B})$ avec $\mathcal{B}$ = mini-batch. Gradient complet (batch GD) : $O(N \times p)$ calculs par pas. SGD : $O(B \times p)$ avec $B \ll N$. Adam = SGD + moments adaptatifs. Convergence SGD garantie sous conditions de convexité et learning rate décroissant.

---

## Fonction d'approximation
**Définition :** Un système qui estime la valeur de situations jamais vues en se basant sur des situations similaires — ce que font les réseaux de neurones dans DQN et PPO.
**Explication :** Une table Q ne peut stocker que les situations déjà vues. Une fonction d'approximation (réseau de neurones) généralise : deux situations similaires donnent des estimations similaires, même si l'une n'a jamais été vue. C'est la clé du passage du RL tabulaire au Deep RL.
**Exemple :** DQN reçoit 147 chiffres et calcule 7 valeurs Q — même pour une situation jamais vue exactement. La table Q équivalente aurait besoin d'une ligne pour chaque combinaison des 147 variables : un nombre infini.
**Pour aller plus loin :** Table Q : 17 034 états effectifs (cas réel). MLP 128×128 : entrée = 147 valeurs dans $[0,1]$, sortie = 7 valeurs Q. Approximation linéaire, arbres de décision, ou réseaux de neurones — tous sont des fonctions d'approximation.

---

## Surapprentissage (Overfitting) et généralisation
> *📌 Contexte de base — terme absent du mémoire, mais le jury peut demander « votre modèle n'a-t-il pas fait d'overfitting ? »*

**Définition :** Le surapprentissage survient quand un modèle mémorise les données d'entraînement au lieu d'en extraire des règles générales — il performe bien sur ce qu'il a vu, mal sur le reste. La généralisation est la capacité opposée : bien fonctionner sur des situations nouvelles.
**Explication :** Imagine un étudiant qui mémorise toutes les réponses d'un examen passé sans comprendre les concepts — il réussit cet examen mais échoue sur tout autre sujet similaire. Un modèle qui overfitte a la même pathologie. En RL, l'overfitting signifie que l'agent est trop spécialisé sur les états vus pendant l'entraînement et ne sait pas gérer des états légèrement différents.
**Exemple :** Dans ce projet, le risque d'overfitting est limité car l'espace d'état est continu et immense : le MLP ne peut pas mémoriser 2M transitions distinctes avec seulement 35 700 poids. De plus, la stochasticité de l'environnement (temps de trajet variables, pannes aléatoires) oblige l'agent à apprendre des règles générales plutôt que des réponses mémorisées. La validation sur 10 réplications avec seeds différentes confirme la généralisation.
**Pour aller plus loin :** Indicateurs d'overfitting en RL : divergence entre performance d'entraînement et performance d'évaluation, explained variance qui continue à croître sans amélioration des KPIs. Mécanismes de régularisation non utilisés ici (pas nécessaires) : dropout, L2 weight decay, early stopping. La stochasticité de l'environnement joue naturellement le rôle de régularisateur.

---

## MLP (Perceptron multicouche)
**Définition :** Le type de réseau de neurones utilisé par DQN et PPO — plusieurs couches de neurones empilées, chacune transformant les données de la précédente pour en extraire des patterns de plus en plus abstraits.
**Explication :** Imagine une chaîne de traducteurs : le premier lit les 147 chiffres et en fait 128 « impressions », le second affine ces 128 impressions en 128 abstractions, le dernier convertit ces abstractions en 7 scores d'action. Chaque couche capture des relations non-linéaires grâce à ReLU.
**Exemple :** Architecture du projet : $147 \to 128 \to 128 \to 7$. ≈ 35 700 poids. Modèle compact (0,6–0,9 Mo), entraînable sans GPU en ≈ 2h14 sur CPU.
**Pour aller plus loin :** Paramètres : couche 1 → $147 \times 128 + 128 = 18\,944$ ; couche 2 → $128 \times 128 + 128 = 16\,512$ ; sortie → $128 \times 7 + 7 = 903$ ; total ≈ 36 359. Implémenté via `MlpPolicy` de Stable-Baselines3 (PyTorch).

---

## ReLU (Rectified Linear Unit)
**Définition :** La fonction d'activation des couches cachées du MLP : $f(x) = \max(0, x)$ — garde les valeurs positives, met à zéro les négatives.
**Explication :** Sans fonction d'activation non-linéaire, empiler des couches de neurones ne servirait à rien (une composition de fonctions linéaires reste linéaire). ReLU introduit la non-linéarité nécessaire pour capturer des relations complexes entre les variables d'état et les valeurs d'action.
**Exemple :** Pour les valeurs -3, 0, 5 en entrée, ReLU renvoie 0, 0, 5. Dans le MLP 128×128, chaque neurone applique ReLU après avoir combiné les informations — ce qui permet de détecter des patterns complexes inaccessibles à une combinaison linéaire.
**Pour aller plus loin :** $f(x) = \max(0,x)$. Avantages sur sigmoïde/tanh : évite le vanishing gradient pour les grandes valeurs positives, calcul très rapide. Utilisée dans les couches cachées. Softmax en couche de sortie (PPO) ou sans activation (DQN, sortie Q-valeurs brutes).

---

## Softmax
**Définition :** Une fonction qui transforme des scores bruts en une distribution de probabilités valide — utilisée dans la couche de sortie de PPO.
**Explication :** Le réseau calcule 7 scores bruts (pouvant être négatifs ou très grands). Softmax les convertit en 7 probabilités positives qui s'additionnent à 1 : le score le plus élevé obtient la probabilité la plus haute. L'agent « tire au sort » selon ces probabilités, maintenant une légère exploration.
**Exemple :** Scores [0,5 ; 2,0 ; 1,0 ; 1,5 ; 0,3 ; 1,8 ; 0,2] → Softmax ≈ [5% ; 30% ; 11% ; 18% ; 4% ; 26% ; 4%]. En fin d'entraînement PPO, une action atteint 40–60% (entropie ≈ 0,50 nat).
**Pour aller plus loin :** $\text{Softmax}(z_i) = e^{z_i} / \sum_j e^{z_j}$. PPO uniquement (politique stochastique). DQN utilise $\arg\max_a Q_\theta(s,a)$ — pas de Softmax. ReLU dans les couches cachées, Softmax uniquement en sortie de l'acteur PPO.

---

## Rétropropagation (backpropagation)
**Définition :** L'algorithme qui calcule comment chaque poids du réseau de neurones a contribué à l'erreur, puis l'ajuste dans la bonne direction — couche par couche, de la sortie vers l'entrée.
**Explication :** La règle de dérivation en chaîne (chain rule) permet de calculer $\partial L / \partial w_i$ pour chaque poids $w_i$, même s'il se trouve à plusieurs couches de la sortie. C'est ce qui rend l'entraînement des réseaux profonds possible : chaque poids reçoit un signal précis indiquant comment il doit changer pour réduire l'erreur.
**Exemple :** Dans DQN, quand l'écart entre Q-valeur prédite et cible vaut 0,5, la rétropropagation calcule comment ajuster les ≈ 35 700 poids du MLP pour réduire cet écart. Géré automatiquement par PyTorch via `loss.backward()`.
**Pour aller plus loin :** $\partial L / \partial w_i = (\partial L / \partial a) \cdot (\partial a / \partial z) \cdot (\partial z / \partial w_i)$. Coût : $O(p)$ avec $p$ = nb de paramètres. Pour MLP 147→128→128→7 : $O(35\,700)$ par pas. Suivi par `optimizer.step()` (Adam).

---

## Optimiseur Adam
**Définition :** L'algorithme de mise à jour des poids utilisé par DQN et PPO — une version améliorée de la descente de gradient qui adapte automatiquement le pas pour chaque poids.
**Explication :** La descente de gradient simple applique le même α à tous les poids. Adam (Adaptive Moment Estimation) surveille l'historique des gradients de chaque poids individuellement et adapte le pas — grand pour les poids peu mis à jour, petit pour les fréquemment ajustés. Converge plus vite et est moins sensible au choix du learning rate.
**Exemple :** Dans le projet, DQN et PPO utilisent Adam avec α = 3×10⁻⁴. Après 2M steps, ≈ 31 250 mises à jour (2M ÷ 64 mini-batches). C'est cette adaptation individuelle qui permet à PPO d'atteindre explained\_variance = 0,999.
**Pour aller plus loin :** Moments : $m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$, $v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$. Mise à jour : $\theta \leftarrow \theta - \alpha \hat{m}_t / (\sqrt{\hat{v}_t} + \varepsilon)$. Valeurs SB3 : $\beta_1=0{,}9$, $\beta_2=0{,}999$, $\varepsilon=10^{-8}$.

---

## Époque (Epoch)
**Définition :** Un passage complet sur un ensemble de données d'entraînement pour mettre à jour les poids du réseau — PPO effectue plusieurs époques sur chaque batch de données collectées avant d'en collecter de nouvelles.
**Explication :** En deep learning classique, une époque = un passage sur tout le dataset. En PPO, une époque = un passage sur le batch de transitions collectées depuis la dernière mise à jour de politique. PPO fait plusieurs époques (n_epochs) sur le même batch car les données on-policy sont coûteuses à collecter — mais pas trop d'époques, sinon le ratio $r_t(\theta)$ s'éloignerait trop de 1 et le clipping perdrait son sens.
**Exemple :** Dans le projet, PPO avec n_steps=1024 collecte 1024 transitions, puis fait 10 époques sur ce batch (chacune découpée en mini-batches de 64). DQN n'a pas d'époques : il fait une mise à jour SGD par step, directement depuis le replay buffer. Les méthodes tabulaires non plus : une mise à jour Q par transition, sans batch.
**Pour aller plus loin :** PPO SB3 par défaut : `n_epochs=10`. Chaque époque parcourt le batch en mini-batches de 64. Total mises à jour par cycle : $(1024/64) \times 10 = 160$ passes SGD. Contrainte on-policy : après 10 époques, la politique a suffisamment changé que les données collectées sous $\pi_{\theta_\text{old}}$ ne sont plus utilisables — on repart en collecte.

---

## Acteur-critique
**Définition :** Une architecture à deux composantes : un acteur qui choisit les actions et un critique qui évalue la qualité des situations — utilisée par PPO.
**Explication :** L'acteur (joueur) choisit les actions. Le critique (coach) évalue si la situation est bonne ou mauvaise — en estimant combien de récompenses futures on peut espérer. Quand l'acteur fait une action meilleure que ce que le critique attendait, le critique encourage l'acteur à refaire ce choix. Cette comparaison « mieux que prévu » est plus précise qu'un simple score brut.
**Exemple :** Dans PPO, le même réseau MLP a deux sorties : une pour les probabilités d'action (acteur), une pour V(s) (critique). Explained variance = 0,999 : le critique prédit les retours futurs avec 99,9 % de précision.
**Pour aller plus loin :** $\hat{A}_t = G_t - V_\phi(s_t)$. Deux réseaux séparés ou un réseau avec deux têtes. L'avantage $\hat{A}$ est utilisé par l'acteur pour mettre à jour $\pi_\theta$ via $L^\text{CLIP}$. Le critique minimise $L^V = (V_\phi(s_t) - G_t)^2$.

---

## Fonction de valeur V(s)
**Définition :** Un chiffre qui estime « combien de récompenses totales peut-on espérer accumuler à partir de maintenant » — le score global d'une situation, indépendamment de l'action choisie.
**Explication :** V(s) ne dit pas QUELLE action faire, juste si la situation est bonne ou mauvaise en termes de gains futurs cumulés. C'est le rôle du critique dans PPO : il apprend à estimer V(s) pour aider l'acteur à savoir si ses actions sont vraiment bonnes ou si la situation était déjà favorable indépendamment de ses choix.
**Exemple :** V(situation) = 3,5 → on espère 3,5 unités de récompense. Action prise : 0,8 + 0,99×2,9 = 3,671 > 3,5 → avantage positif → PPO augmente la probabilité de cette action. Explained variance = 0,999 : V(s) prédit les retours réels avec 99,9 % de précision.
**Pour aller plus loin :** $V^\pi(s) = \mathbb{E}_\pi[\sum_{k=0}^\infty \gamma^k r_{t+k} \mid s_t=s]$. Relation : $V^\pi(s) = \sum_a \pi(a|s)Q^\pi(s,a)$. Dans PPO, le critique apprend $V_\phi(s)$ en minimisant $(V_\phi(s_t) - G_t)^2$. Explained variance $= 1 - \text{Var}(G_t - V_\phi(s_t))/\text{Var}(G_t)$.

---

## Avantage A(s, a)
**Définition :** La mesure de combien une action est meilleure ou moins bonne que la moyenne dans la même situation — « est-ce que j'ai fait mieux que ce qu'on attendait ? »
**Explication :** Au lieu de dire « cette action vaut 3,5 points » (dépend de la situation, pas de l'action), l'avantage dit « cette action vaut 0,17 de plus que ce qu'une action moyenne aurait rapporté ». Signal plus précis pour améliorer la politique : A > 0 → augmenter la probabilité ; A < 0 → la diminuer.
**Exemple :** V(s) = 3,5 ; action choisie → 0,8 + 0,99×2,9 = 3,671 ; A = +0,171 → PPO augmente légèrement cette action. Si A = -0,3 → PPO la diminue. C'est cette correction fine, répétée des milliers de fois, qui produit la politique finale.
**Pour aller plus loin :** $A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$. Estimé par GAE : $\hat{A}_t = \sum_{l\geq0}(\gamma\lambda)^l \delta_{t+l}$ où $\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$. $(\gamma\lambda)^{10} \approx 0{,}54$ → demi-vie ≈ 10 pas de décision.

---

## Scénario (expérimental)
**Définition :** Une configuration spécifique d'entraînement/évaluation qui fait varier les conditions de l'environnement — comme un niveau de difficulté différent pour tester la robustesse des agents.
**Explication :** Tester un agent sur un seul contexte ne suffit pas pour valider sa généralisation. En changeant les paramètres (nombre de camions, taux de pannes), on vérifie si l'agent reste performant dans des conditions différentes de celles d'entraînement. C'est une pratique standard en RL pour mesurer la robustesse et la généralisation.
**Exemple :** 3 scénarios dans le projet : nominal (conditions de référence), high_load (surcharge : 18 camions), high_breakdown (pannes fréquentes : $p_b = 10\%$). Chaque agent est entraîné sur son scénario cible et évalué sur les 3.
**Pour aller plus loin :** Nominal (12 camions, $p_b=2\%$, MF=4,0) ; high\_load (18 camions, MF=6,0) ; high\_breakdown ($p_b=10\%$). 8 politiques × 3 scénarios × 10 réplications = 240 évaluations.

---

## Robustesse (en RL)
**Définition :** La capacité d'un agent appris à maintenir de bonnes performances quand les conditions changent par rapport à l'entraînement — une propriété clé pour l'applicabilité industrielle.
**Explication :** Un agent « fragile » voit ses performances s'effondrer si les conditions changent (plus de pannes, plus de camions). Un agent « robuste » s'adapte grâce à ce qu'il a appris — sa politique généralisée gère mieux les perturbations qu'une règle fixe programmée.
**Exemple :** En high_breakdown ($p_b$ multiplié par 5), DQN et PPO surpassent Fixed Assignment car ils réaffectent dynamiquement les camions restants. Une règle fixe maintient les affectations même quand la flotte est réduite. Test de Welch : $t=5{,}863$, $p=1{,}3\times10^{-5}$.
**Pour aller plus loin :** Robustesse = gap de performance entre conditions nominales et perturbées. DQN : 4 168,5 → 3 991,8 t/h (-4,2 %) en high\_breakdown. PPO : 4 208,75 → 3 946,25 t/h (-6,2 %). Fixed : 4 074,0 → 3 860,5 t/h (-5,2 %).

---

## Explained Variance
**Définition :** Une métrique qui indique à quel point le critique de PPO prédit correctement les retours futurs — proche de 1,0 signifie une prédiction quasi-parfaite.
**Explication :** Si le critique dit systématiquement à l'avance le retour qui va vraiment arriver, il « explique » toute la variance (EV = 1,0). S'il ne fait pas mieux qu'une constante (prédire toujours la moyenne), EV = 0. Si ses prédictions sont pires qu'une constante, EV < 0 — signe d'un problème d'apprentissage.
**Exemple :** Dans le projet, le critique PPO atteint EV = 0,999 (nominal), 0,9999 (high_load), 0,9998 (high_breakdown). Cette valeur proche de 1 confirme que l'apprentissage du critique est complet et fiable pour guider l'acteur.
**Pour aller plus loin :** $EV = 1 - \dfrac{\text{Var}(G_t - V_\phi(s_t))}{\text{Var}(G_t)}$. EV = 1 : prédiction parfaite. EV = 0 : prédiction constante. EV < 0 : pire qu'une constante. Indicateur clé de la convergence du critique dans les méthodes acteur-critique.

---

## Policy Loss / Value Loss
**Définition :** Les deux fonctions de perte que PPO minimise pendant l'entraînement — Policy Loss pour l'acteur, Value Loss pour le critique.
**Explication :** Policy Loss mesure comment l'acteur doit ajuster ses probabilités d'action pour augmenter le retour espéré (via l'objectif clippé $L^\text{CLIP}$). Value Loss mesure l'écart entre ce que le critique prédit et ce qui s'est vraiment passé. La Value Loss devrait diminuer au fil du temps (le critique s'améliore) ; la Policy Loss reste petite et stable (les mises à jour sont modérées par le clipping).
**Exemple :** PPO nominal : Value Loss $32{,}0 \to 0{,}17$ (forte amélioration du critique). Policy gradient loss stable entre -0,025 et -0,014. DQN : train/loss $0{,}040 \to 0{,}025$, avec pic à 0,30 (instabilité due au replay buffer).
**Pour aller plus loin :** $L^V = \mathbb{E}[(V_\phi(s_t) - G_t)^2]$. $L^\text{CLIP} = \mathbb{E}[\min(r_t \hat{A}_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon)\hat{A}_t)]$. Perte totale PPO : $L = -L^\text{CLIP} + c_1 L^V - c_2 H$. DQN : $L = \frac{1}{B}\sum_i (Q_\theta(s_i,a_i) - y_i)^2$ avec $y_i = r_i + \gamma\max_{a'}Q_{\theta^-}(s'_i,a')$.

---

## Efficacité d'échantillonnage (Sample Efficiency)
**Définition :** La quantité de données (interactions avec l'environnement) nécessaire pour qu'un agent apprenne une politique performante — plus c'est petit, plus l'agent est efficace.
**Explication :** Certains algorithmes apprennent vite avec peu d'expériences (haute sample efficiency) ; d'autres ont besoin de beaucoup plus d'interactions pour atteindre le même niveau. DQN est plus sample-efficient que PPO car il réutilise ses expériences passées (replay buffer) ; PPO jette chaque batch après utilisation.
**Exemple :** Dans le projet, DQN et PPO atteignent l'essentiel de leur niveau final après 28 à 80 % de leurs 2M steps. Q-Learning/SARSA ont besoin de quasi 100 % de leurs 30 000 épisodes. SARSA en high\_breakdown : 0 % de progression nette (efficacité nulle).
**Pour aller plus loin :** Seuil : performance lissée à moins de 5 % de la valeur finale. DQN/PPO : 28–80 % de 2M steps. Tabulaires : 83–99 % de 30 000 épisodes. Off-policy (DQN) généralement plus sample-efficient qu'on-policy (PPO) grâce au replay buffer.

---

## Convergence (d'un algorithme RL)
**Définition :** Le moment où l'agent a fini d'apprendre l'essentiel — quand la politique cesse de s'améliorer significativement et se stabilise.
**Explication :** Comme une courbe de progression sportive : au début on progresse vite, puis les gains diminuent jusqu'à se stabiliser. Un algorithme RL a « convergé » quand la récompense lissée ne monte plus de façon visible. C'est le signal qu'on peut arrêter l'entraînement et évaluer.
**Exemple :** Q-Learning passe de 155 à 175 sur 30 000 épisodes en nominal. DQN/PPO atteignent l'essentiel après 28–80 % de 2M steps. Cas pathologique : SARSA en high\_breakdown reste stable à ≈ 142 dès le début — aucune convergence.
**Pour aller plus loin :** Critère du projet : performance lissée à < 5 % de sa valeur finale. PPO : `explained_variance = 0,999` confirme la convergence. Q-Learning : erreur TD $|\delta_t|$ décroissante confirme la stabilisation. Convergence théorique de Q-Learning garantie sous conditions de visite (Watkins, 1989).

---

## Significativité statistique (Welch, ANOVA, Tukey HSD)
**Définition :** Des tests qui permettent de répondre à la question « cette différence entre deux méthodes est-elle réelle, ou juste due au hasard des réplications ? »
**Explication :** Si on teste chaque méthode 10 fois et qu'on observe des résultats légèrement différents, comment savoir si la différence est réelle ou juste due à la variabilité aléatoire ? Les tests statistiques quantifient cette probabilité via la p-value : si p < 0,05, la différence est jugée réelle (significative).
**Exemple :** ANOVA sur 8 méthodes en nominal : $F(7,72) = 481{,}25$, $p < 10^{-56}$ — différence globale significative. Welch PPO vs Fixed : $t=6{,}356$, $p=3{,}0\times10^{-6}$. Tukey PPO vs DQN : $p=0{,}799$ — statistiquement équivalents malgré 40 t/h d'écart.
**Pour aller plus loin :** Test de Welch : $t$ pour deux groupes avec variances potentiellement inégales. ANOVA : $F$ pour $k$ groupes (ici 8 méthodes). Tukey HSD : comparaisons par paires avec correction pour comparaisons multiples. 10 réplications × 8 méthodes × 3 scénarios = 240 observations.

---

## Graine aléatoire (seed) et réplications
**Définition :** Une graine fixe le hasard d'une simulation pour la rejouer identiquement ; répliquer avec différentes graines permet de mesurer la variabilité et la reproductibilité des résultats.
**Explication :** La simulation est stochastique (temps de trajet variables, pannes aléatoires). Un seul résultat pourrait être dû à la chance. On répète 10 fois avec des graines différentes mais connues (42 à 51) pour obtenir une moyenne et un écart-type fiables, et vérifier que les différences entre méthodes sont stables.
**Exemple :** Dans le projet, 8 méthodes × 3 scénarios × 10 réplications (seeds 42-51) = 240 évaluations. Entraînement avec seed=42 ; évaluation sur les 10 seeds. L'écart-type PPO en nominal : ±49 t/h. En high_load : ±350 t/h (instabilité).
**Pour aller plus loin :** `numpy.random.seed(seed)`, `env.reset(seed=seed)`. Reproductibilité garantie si toutes les sources de hasard sont seedées. Les tests statistiques (Welch, ANOVA, Tukey) exploitent les 10 réplications pour calculer la significativité des différences.