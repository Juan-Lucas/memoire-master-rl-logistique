# Glossaire logistique minière, optimisation et RL

Ce glossaire est aligné sur les termes utilisés dans le projet et les chapitres 4 à 6 du mémoire. Chaque entrée comprend :
- Définition très simple
- Explication avec une analogie du quotidien
- Exemple concret
- (Optionnel) Pour aller plus loin : formules et chiffres précis, pour les questions techniques du jury

---

## Apprentissage par renforcement (RL)
**Définition :** C'est une façon d'apprendre en essayant des choses, en se trompant, et en gardant ce qui marche le mieux — un peu comme on apprend à faire du vélo.
**Explication :** Un agent (le « cerveau » du programme) se retrouve dans une situation, doit choisir une action, et reçoit ensuite une récompense (bonne ou mauvaise) selon le résultat. Personne ne lui dit à l'avance quoi faire : il essaie, observe ce qui se passe, et ajuste petit à petit sa façon de choisir pour obtenir le plus de récompenses possible sur la durée — pas seulement tout de suite, mais sur l'ensemble de la « partie ».
**Exemple :** Dans le projet, l'agent regarde l'état du site minier (files d'attente, position des camions, heure de la journée) et choisit où envoyer un camion. Au début, ses choix sont presque au hasard et la production est faible. Après des milliers d'essais simulés, il a appris une bonne stratégie : avec PPO, la production moyenne atteint 4 208,75 t/h.
**Pour aller plus loin :** Formellement, à chaque pas $t$, l'agent observe $s_t$, choisit $a_t\sim\pi(\cdot|s_t)$, reçoit $r_t$ et passe à $s_{t+1}$ ; l'objectif est de maximiser le retour actualisé $G_t=\sum_{k=0}^{\infty} \gamma^k r_{t+k}$.

---

## MDP (Processus de Décision Markovien)
**Définition :** C'est la façon « officielle » de décrire un problème de décision : on dit clairement ce que l'agent voit, ce qu'il peut faire, comment le monde change, et ce qui lui rapporte des points.
**Explication :** Pense à un jeu de société : à chaque tour, tout ce qui compte pour décider du prochain coup, c'est la position actuelle du plateau — pas l'historique complet de la partie. Un MDP, c'est exactement cette idée : ce qui se passe ensuite ne dépend que de la situation présente et de l'action choisie maintenant, jamais de ce qui s'est passé avant. Grâce à cette règle, l'agent n'a pas besoin de « se souvenir » de tout son passé : la situation actuelle lui suffit pour décider.
**Exemple :** Dans le projet, la « situation actuelle » regroupe tout ce qu'il faut savoir : la longueur des files d'attente aux pelles, où sont les 12 camions et ce qu'ils font, et l'heure de la journée. L'agent peut choisir entre 7 actions (envoyer le camion courant vers une pelle/dump ou attendre). La journée simulée dure 8h (480 minutes), sans « fin de partie » particulière — on regarde juste le tonnage transporté au total.
**Pour aller plus loin :** Formalisation par le 4-tuple $(\mathcal{S},\mathcal{A},\mathcal{P},\mathcal{R})$ ; $s_t=(\{q_p\},\{x_c\},\{z_r\},t_{\text{courant}})$, 147 valeurs normalisées ; $\mathcal{A}=\text{Discrete}(7)=|\mathcal{P}|\times|\mathcal{D}|+1$ (3 pelles × 2 dumps + ATTENDRE) ; propriété markovienne $\mathcal{P}(s_{t+1}|s_t,a_t)$ ; horizon fini $T=480$ min sans état terminal.

---

## Environnement Gymnasium
**Définition :** C'est un cadre standard, comme une prise électrique universelle, qui permet de brancher n'importe quel « cerveau » d'agent sur n'importe quelle simulation.
**Explication :** Imagine que tous les jeux vidéo du monde avaient exactement le même type de manette et les mêmes boutons « recommencer » et « jouer un coup ». Un programmeur qui sait utiliser une manette saurait alors jouer à n'importe quel jeu sans rien réapprendre. Gymnasium fait ça pour les simulations : chaque environnement a un bouton « recommencer » (reset) qui remet tout à zéro, et un bouton « joue ce coup » (step) qui fait avancer la simulation d'un pas et dit ce qui s'est passé.
**Exemple :** La mine simulée du projet est un environnement Gymnasium. Quand l'agent appelle « step » avec l'action 3, ça veut dire « envoie le camion courant vers la pelle 2 et le dump 1 » : la simulation avance un peu, et renvoie la nouvelle situation et la récompense obtenue.
**Pour aller plus loin :** Contrat standard `reset()` → $s_0$, `step(action)` → `(observation, reward, terminated, truncated, info)` ; `observation_space = Box(0,1,shape=(147,))`, `action_space = Discrete(7)`.

---

## Stable-Baselines3
**Définition :** Une boîte à outils en Python qui contient déjà tout prêt les algorithmes d'apprentissage par renforcement les plus connus.
**Explication :** C'est un peu comme un livre de recettes de cuisine : au lieu d'inventer chaque recette (algorithme) soi-même, on prend une recette déjà testée et on l'applique à ses propres ingrédients (l'environnement). On choisit la recette (PPO, DQN...), on lui donne l'environnement, et on lance la cuisson (l'entraînement) — la bibliothèque s'occupe de toute la mécanique compliquée derrière.
**Exemple :** Dans le projet, pour entraîner PPO, il suffit d'écrire quelques lignes de code qui disent « utilise PPO, avec ces réglages, sur cet environnement, et entraîne-toi pendant 2 millions de pas ». Cela prend environ 2h sur un ordinateur normal.
**Pour aller plus loin :** `PPO("MlpPolicy", env, learning_rate=3e-4, gamma=0.99, gae_lambda=0.95, clip_range=0.2, batch_size=64).learn(total_timesteps=2_000_000)`.

---

## Q-Learning
**Définition :** Une méthode qui tient un grand carnet où, pour chaque situation et chaque action possible, on note « combien de points ça vaut à peu près ».
**Explication :** Imagine un carnet géant où chaque page correspond à une situation (« il y a 5 camions qui attendent à la pelle 1 ») et chaque ligne à une action possible (« envoyer le camion vers la pelle 2 »). Au début, toutes les notes valent zéro. Chaque fois que l'agent essaie une action, il regarde ce qui s'est passé et corrige un peu la note dans son carnet — en se disant « si je joue ensuite le mieux possible, combien ça vaudra ». Petit à petit, le carnet devient fiable et l'agent n'a plus qu'à choisir, dans chaque situation, l'action qui a la meilleure note.
**Exemple :** Si une action avait une note de 50, qu'elle a rapporté 10 points tout de suite, et que la meilleure action suivante vaut 55, l'agent corrige légèrement la note vers le haut (elle passe à 51,45). Dans le projet, ce carnet contient 17 034 situations différentes, et chaque action correspond à « envoyer un camion vers une pelle/dump » ou « attendre ».
**Pour aller plus loin :** $Q(s_t,a_t) \leftarrow Q(s_t,a_t) + \alpha[r_t + \gamma \max_{a'} Q(s_{t+1},a') - Q(s_t,a_t)]$ ; exemple $Q=50,\ r=10,\ \gamma=0{,}99,\ \max Q'=55,\ \alpha=0{,}1 \Rightarrow 51{,}45$. Off-policy ($\max_{a'}$ = meilleure action possible, pas forcément celle jouée) ; $\epsilon$ décroît de 1,0 à 0,01.

---

## SARSA
**Définition :** Une cousine de Q-Learning, mais plus prudente : elle note la valeur de l'action qu'on va *vraiment* faire ensuite, pas la meilleure action imaginable.
**Explication :** Q-Learning corrige son carnet en imaginant « si je jouais ensuite le coup parfait ». SARSA, elle, regarde ce que l'agent fait *réellement* au coup suivant — même si ce n'est pas le meilleur coup possible (par exemple parce que l'agent était encore un peu en mode exploration/essai). Du coup, SARSA est plus « réaliste » et un peu plus prudente : elle n'apprend pas à se baser sur un coup parfait hypothétique qui ne sera peut-être jamais joué.
**Exemple :** Avec les mêmes chiffres que pour Q-Learning (note actuelle 50, récompense 10), si le coup vraiment joué ensuite vaut 52 (au lieu du meilleur possible, 55), la nouvelle note ne monte qu'à 51,15 au lieu de 51,45. Concrètement, SARSA évite davantage d'envoyer un camion vers une pelle déjà très chargée, car elle tient compte de ce qui se passe vraiment ensuite, pas d'un idéal théorique.
**Pour aller plus loin :** $Q(s_t,a_t) \leftarrow Q(s_t,a_t) + \alpha[r_t + \gamma Q(s_{t+1},a_{t+1}) - Q(s_t,a_t)]$ ; exemple $Q=50, r=10,\gamma=0{,}99,\alpha=0{,}1, Q(s',a_{t+1})=52 \Rightarrow 51{,}15$. On-policy.

---

## DQN (Deep Q-Network)
**Définition :** Comme Q-Learning, mais au lieu d'un carnet avec une page par situation, on utilise un réseau de neurones capable de « deviner » la note même pour des situations jamais vues.
**Explication :** Le problème du carnet de Q-Learning, c'est que dans notre projet il y a bien trop de situations possibles pour toutes les écrire (147 chiffres différents décrivent chaque situation). DQN remplace donc le carnet par un réseau de neurones : on lui montre la situation, et il calcule directement une note pour chaque action possible — même pour des situations qu'il n'a jamais vues exactement, en se basant sur des situations similaires déjà rencontrées. Deux astuces l'aident à apprendre sans s'emmêler : il garde un « journal » de ses expériences passées qu'il relit au hasard (au lieu d'apprendre uniquement du dernier instant), et il utilise une copie « figée » de lui-même comme cible, pour ne pas courir après un objectif qui bouge tout le temps.
**Exemple :** Dans le projet, DQN atteint 4 168,5 t/h en moyenne en conditions normales, et reste très stable même quand les pannes sont fréquentes (3 991,8 t/h en high_breakdown), avec très peu de variations entre les essais.
**Pour aller plus loin :** Réseau MLP 128×128, ReLU ; replay buffer de 200 000 transitions, batch=64 ; réseau cible $Q_{\theta^-}$ pour calculer la cible $r+\gamma\max_{a'}Q_{\theta^-}(s',a')$ ; $\epsilon$ décroît de 1,0 à 0,02 sur 2 000 000 steps ; écart-type DQN $\sigma\approx65$ vs PPO $\sigma=350$ en high_load.

---

## PPO (Proximal Policy Optimization)
**Définition :** Une méthode d'apprentissage qui améliore la stratégie de l'agent petit pas par petit pas, avec une « ceinture de sécurité » pour ne jamais changer trop brusquement.
**Explication :** Imagine que tu ajustes la recette d'un plat : si une modification a l'air bonne, tu ne changes pas toute la recette d'un coup au cas où tu te serais trompé sur l'ampleur du changement — tu ajustes un peu, tu goûtes, tu réajustes. PPO fait pareil avec la stratégie de l'agent : si une action s'est révélée bonne, PPO augmente un peu sa probabilité d'être choisie à nouveau, mais une « ceinture de sécurité » empêche cette probabilité de changer trop violemment en une seule fois — ce qui évite de tout casser si l'estimation était un peu fausse.
**Exemple :** Avec un réseau de 128×128 neurones entraîné sur 2 millions de pas (environ 9 677 parties simulées complètes), PPO atteint 4 208,75 t/h en moyenne en conditions normales — la meilleure productivité de toutes les méthodes testées dans le projet.
**Pour aller plus loin :** Ratio $r_t(\theta)=\pi_\theta(a|s)/\pi_{\theta_{old}}(a|s)$ ; objectif clippé $L^{CLIP}=\mathbb{E}[\min(r_t\hat A_t,\text{clip}(r_t,1-\epsilon,1+\epsilon)\hat A_t)]$, $\epsilon=0{,}2$ donc ratio limité à $[0{,}8;1{,}2]$ ; exemple $\hat A_t=+10$, $r_t=3{,}0 \Rightarrow$ terme non clippé $=30$, clippé $=12$. On-policy, pas de replay buffer.

---

## GAE (Generalized Advantage Estimation)
**Définition :** Une façon équilibrée de juger « est-ce que cette décision était bonne ? » — ni trop à court terme, ni trop à long terme.
**Explication :** Pour juger si une décision était bonne, on pourrait regarder seulement ce qui se passe juste après (rapide à calculer mais peu fiable, car ça dépend beaucoup du hasard du moment), ou attendre la fin de toute la journée pour voir le résultat final (fiable mais très lent, et ça mélange l'effet de plein d'autres décisions prises entre-temps). GAE fait un compromis : il regarde les conséquences sur les prochaines minutes qui suivent la décision, en donnant un peu moins d'importance à ce qui est plus loin dans le temps — comme regarder l'effet d'un coup sur les 10 prochains coups, sans pour autant attendre la fin de la partie.
**Exemple :** Dans le projet, GAE permet de juger qu'envoyer un camion vers une pelle plutôt qu'une autre était une bonne ou une mauvaise idée en regardant l'effet sur les dix prochaines minutes environ — assez pour voir si ça a désengorgé une file d'attente, sans attendre la fin des 8h de simulation.
**Pour aller plus loin :** $\hat{A}_t = \sum_{l=0}^{\infty}(\gamma\lambda)^l \delta_{t+l}$ ; avec $\gamma=0{,}99$, $\lambda=0{,}95$ : $\gamma\lambda=0{,}9405$, $(0{,}9405)^{10}\approx0{,}54$.

---

## Baseline
**Définition :** Une méthode « témoin », simple et fixe, à laquelle on compare l'agent pour voir s'il fait vraiment mieux.
**Explication :** Avant de dire « mon agent est intelligent et performant », il faut une référence pour comparer — sinon, comment savoir si 4 208 t/h, c'est bien ou pas ? Une baseline applique une règle toute simple et fixe (par exemple « envoie toujours le camion vers la pelle la plus proche »), sans aucun apprentissage. Si l'agent ne fait pas mieux que la meilleure de ces règles simples, alors tout l'apprentissage n'a servi à rien.
**Exemple :** Dans le projet, la baseline « Fixed Assignment » (chaque camion est toujours affecté à la même pelle) atteint 4 074,0 t/h en conditions normales. PPO fait mieux avec 4 208,75 t/h (+3,3 %), et ce n'est pas juste de la chance : un test statistique confirme que cet écart est réel.
**Pour aller plus loin :** 4 baselines (FIFO, Fixed Assignment, Nearest Shovel, Shortest Path) vs 4 agents RL, 3 scénarios, 10 réplications (seeds 42-51). Welch PPO vs Fixed nominal : $t=6{,}356$, $p=3{,}0\times10^{-6}$.

---

## Heuristique
**Définition :** Une règle simple et rapide pour décider, sans réfléchir aux conséquences à long terme.
**Explication :** C'est comme suivre la règle « prends toujours la file la plus courte au supermarché », sans te demander si tout le monde autour de toi suit la même règle au même moment — au risque que tout le monde se rue vers la même caisse et la rende finalement la plus longue ! Une heuristique décide vite, en regardant juste ce qui est sous ses yeux, sans vision d'ensemble ni anticipation.
**Exemple :** L'heuristique « Nearest Shovel » envoie toujours chaque camion vers la pelle la plus proche, sans regarder si elle est déjà encombrée. Quand il y a beaucoup de camions (scénario high_load), ça crée un énorme bouchon : les camions attendent en moyenne 200 minutes, contre environ 70-75 minutes dans un scénario normal — un agent qui voit l'ensemble du site évite largement ce problème.
**Pour aller plus loin :** En high_load (18 camions, MF=6), Nearest Shovel atteint 200,2 min d'attente moyenne.

---

## Méta-heuristique
**Définition :** Une famille de méthodes qui cherchent une bonne solution en essayant, en gardant les meilleures idées, et en les mélangeant — un peu comme l'évolution naturelle.
**Explication :** Imagine que tu essaies plein de façons différentes d'organiser le travail des camions, que tu gardes les meilleures, que tu les « croises » entre elles pour créer de nouvelles combinaisons, et que tu répètes ce processus encore et encore jusqu'à trouver quelque chose de pas mal. C'est l'idée des méta-heuristiques (inspirées de l'évolution, ou de la façon dont le métal refroidit lentement pour devenir plus solide). La différence avec le RL : une méta-heuristique ne « retient » rien d'une fois sur l'autre — si la situation change un peu, il faut tout recommencer depuis le début.
**Exemple :** Pour le dispatching minier, une méta-heuristique essaierait plein d'affectations possibles « camion → pelle → dump », testerait chacune sur une journée simulée, et garderait/combinerait les meilleures sur plusieurs tours. Mais contrairement à l'agent RL entraîné, qui réutilise instantanément ce qu'il a appris pour n'importe quelle nouvelle situation, la méta-heuristique devrait recommencer toute sa recherche si la situation change.
**Pour aller plus loin :** Exemples : algorithme génétique (mutation, croisement, fitness = productivité simulée), recuit simulé (température décroissante).

---

## Politique (en RL)
**Définition :** C'est la « façon de réagir » de l'agent — ce qu'il fait dans chaque situation, comme une habitude.
**Explication :** Une politique, c'est un peu comme les habitudes d'une personne : face à une situation donnée, elle réagit toujours plus ou moins de la même façon. Au début de l'entraînement, ces « habitudes » sont presque aléatoires. À force d'essais et de corrections, elles deviennent de plus en plus pertinentes : l'agent privilégie de plus en plus les actions qui ont bien marché par le passé dans des situations similaires.
**Exemple :** Face à une situation donnée, la politique de PPO ne dit pas « fais exactement ça », mais donne des chances à chaque action : par exemple 5 % de chance pour l'action 1, 40 % pour l'action 3, etc. L'agent tire au sort selon ces chances — l'action la plus probable sera choisie le plus souvent, mais pas toujours, ce qui lui laisse une marge pour essayer d'autres choses.
**Pour aller plus loin :** Politique portée par les poids $\theta$ d'un réseau (PPO/DQN) ou une table $Q$ (Q-Learning/SARSA, politique gloutonne $\pi(s)=\arg\max_a Q(s,a)$). PPO = politique stochastique (softmax sur 7 actions) ; DQN en exploitation = politique déterministe.

---

## Fonction de récompense
**Définition :** Le système de points qui dit à l'agent, après chaque action, si c'était une bonne ou une mauvaise idée.
**Explication :** C'est exactement comme un score dans un jeu vidéo : chaque fois que l'agent fait quelque chose, il gagne ou perd des points selon plusieurs critères à la fois — est-ce qu'un camion a livré son chargement ? est-ce que les files d'attente sont équilibrées ? est-ce que le trajet était court ? est-ce que la pelle choisie n'était pas déjà surchargée ? Le critère le plus important (livrer du minerai) compte beaucoup plus que les autres, qui servent surtout à « départager » entre deux bonnes décisions.
**Exemple :** Quand un camion de 140 tonnes arrive à destination, l'agent gagne un gros point. S'il choisit une pelle dont la file est déjà longue, il perd un petit peu de points. Au final, pour une décision donnée, l'agent peut par exemple obtenir un score net de 0,871 sur cette action — surtout grâce à la livraison réussie, légèrement réduit par la file d'attente et la distance du trajet.
**Pour aller plus loin :** $R_t = w_1 \cdot \frac{\Delta\text{tonnage}}{140} + w_2 \cdot \frac{-\text{Var}(\{q_p\})}{100} + w_3 \cdot \frac{-d_{\text{trajet}}}{5} + w_4 \cdot (-\min(\frac{q_{p_i}}{30},1))$, $w_1=1{,}0,\ w_2=0{,}1,\ w_3=0{,}05,\ w_4=0{,}3$. Exemple : livraison 140t (+1,0), variance files=4 (-0,004), trajet 2,5km (-0,025), file pelle=10 (-0,1) ⇒ $R_t=0{,}871$.

---

## Reward Shaping
**Définition :** Le fait d'ajouter de petits indices/points en cours de route, plutôt que de tout récompenser seulement à la fin.
**Explication :** Si on ne donnait des points que quand un camion termine complètement son trajet et livre son chargement, l'agent recevrait très peu de signaux — la plupart de ses décisions ne seraient « ni bonnes ni mauvaises » à ses yeux, et il aurait énormément de mal à apprendre quoi faire. Le reward shaping, c'est comme donner des petits encouragements en cours de route (par exemple, pénaliser légèrement le fait d'envoyer un camion vers une file déjà longue) pour guider l'agent vers le bon comportement, sans attendre la fin.
**Exemple :** Dans le projet, ajouter une petite pénalité quand un camion est envoyé vers une pelle déjà très chargée a rendu l'apprentissage beaucoup plus efficace pour Q-Learning et SARSA — sans cet indice supplémentaire, presque toutes les décisions se ressemblaient du point de vue des points obtenus, et l'agent n'apprenait presque rien.
**Pour aller plus loin :** Sans le terme $w_4$ (pénalité de file), la différence moyenne de récompense entre deux actions n'était que de $6\times10^{-4}$ ; avec $w_4=0{,}3$, elle passe à $3{,}2\times10^{-2}$ (facteur ~50).

---

## Exploration vs Exploitation
**Définition :** Le dilemme entre essayer quelque chose de nouveau (au cas où ce serait encore mieux) et refaire ce qu'on sait déjà bien marcher.
**Explication :** C'est exactement le choix entre retourner dans ton restaurant préféré (tu sais que c'est bon, « exploitation ») ou tester un nouveau restaurant (ça pourrait être encore meilleur, mais aussi moins bon, « exploration »). Si on n'explore jamais, on risque de rester bloqué sur une solution moyenne sans jamais découvrir la meilleure. Si on explore tout le temps, on n'utilise jamais ce qu'on a appris. Les agents du projet font les deux : ils explorent beaucoup au début (quand ils ne savent encore rien), et de moins en moins au fil de l'entraînement, à mesure qu'ils gagnent en confiance dans leurs choix.
**Exemple :** Au tout début de l'entraînement, l'agent Q-Learning choisit presque au hasard parmi ses 7 actions possibles. À la fin de l'entraînement, il choisit presque toujours l'action qu'il juge la meilleure. PPO garde quant à lui un peu de « diversité » dans ses choix même à la fin — alors que DQN, lui, a tendance à se concentrer sur seulement 2 des 7 actions possibles en fin d'entraînement.
**Pour aller plus loin :** $\epsilon$-greedy : $\epsilon$ décroît de 1,0 à 0,01 (tabulaire) ou 0,02 (DQN). PPO : coefficient d'entropie $c_{ent}=0{,}02$. §6.3.3 : DQN concentre 80,3% des décisions sur 2/7 actions vs PPO qui diversifie (ATTENDRE 17,7%).

---

## Entropie (de la politique)
**Définition :** Un chiffre qui dit à quel point l'agent est encore « indécis » dans ses choix.
**Explication :** Imagine que tu demandes à quelqu'un « que veux-tu manger ce soir ? ». S'il répond « n'importe quoi, ça m'est égal », c'est une entropie élevée — il est totalement indécis entre toutes les options. S'il répond « des pâtes, sans hésiter », c'est une entropie basse — il a une préférence claire. PPO surveille ce niveau d'indécision dans ses choix d'action, et est volontairement encouragé à ne pas devenir trop vite « sûr de lui » : ça lui permet de continuer à essayer différentes options même après avoir trouvé quelque chose qui marche, au cas où il y aurait encore mieux.
**Exemple :** Au début de l'entraînement, PPO est très indécis entre ses 7 actions possibles (entropie maximale). À la fin, en conditions normales, il est devenu beaucoup plus tranché dans ses choix. Mais dans le scénario de forte charge (high_load), il reste un peu plus « hésitant » qu'en conditions normales — ce qui correspond aussi au fait que ses résultats varient un peu plus dans ce scénario.
**Pour aller plus loin :** $H(\pi_\theta(\cdot|s)) = -\sum_{a}\pi_\theta(a|s)\log\pi_\theta(a|s)$ ; max $H=\ln 7\approx1{,}946$ nats. Entropie PPO : démarre à 1,94 nat, finit à 0,50 (nominal) / 0,68 (high_load) / 0,55 (high_breakdown). Coefficient $c_{ent}=0{,}02$ dans la perte.

---

## Replay buffer
**Définition :** Une sorte de carnet de souvenirs où l'agent note ce qui lui est arrivé, pour le relire plus tard et réapprendre.
**Explication :** Plutôt que d'apprendre uniquement de ce qui vient juste de se passer (ce qui ressemble beaucoup à ce qui s'est passé juste avant — pas très varié), l'agent garde un grand carnet de toutes ses expériences récentes. Pour apprendre, il pioche au hasard des pages de ce carnet, parfois très récentes, parfois plus anciennes. Ça lui permet de continuer à apprendre de situations rares (comme une panne) bien après qu'elles soient arrivées, et d'éviter d'apprendre toujours la même chose en boucle.
**Exemple :** Dans le projet, DQN garde les 200 000 dernières expériences dans son carnet. À chaque étape d'apprentissage, il pioche 64 expériences au hasard parmi celles-ci — certaines très récentes, d'autres remontant à des milliers de pas — y compris d'éventuelles pannes survenues bien plus tôt dans l'entraînement.
**Pour aller plus loin :** Buffer FIFO de 200 000 transitions $(s,a,r,s')$, mini-batch de 64 tiré aléatoirement à chaque étape d'apprentissage.

---

## On-policy / Off-policy
**Définition :** Off-policy = l'agent peut apprendre même à partir de vieux essais faits avec une stratégie différente d'avant ; on-policy = l'agent doit apprendre uniquement à partir de ce qu'il fait avec sa stratégie actuelle.
**Explication :** Imagine deux façons de réviser pour un examen : la première (off-policy), tu peux réutiliser tes anciens devoirs et erreurs, même ceux faits il y a longtemps avec une méthode différente — ils restent utiles pour juger ce qui est « objectivement » la meilleure réponse. La seconde (on-policy), tu dois réviser uniquement à partir de tes exercices les plus récents, faits avec ta méthode actuelle — sinon tes statistiques seraient faussées par une ancienne version de toi-même.
**Exemple :** DQN (off-policy) garde ses 200 000 expériences même très anciennes et continue de s'en servir sans problème. PPO (on-policy), au contraire, ne garde jamais de grandes quantités d'anciennes expériences : dès qu'il met à jour sa stratégie, il jette les données collectées avec la version précédente et en récolte de nouvelles.
**Pour aller plus loin :** Q-Learning/DQN off-policy car la cible $\max_{a'}Q(s',a')$ estime la politique optimale, indépendamment de l'action réellement jouée. SARSA/PPO on-policy : cible dépend de la politique courante ($Q(s',a_{t+1})$ ou ratio $r_t(\theta)$).

---

## Temporal difference
**Définition :** Une façon de corriger ses estimations petit à petit, au fur et à mesure, sans attendre la fin de toute l'histoire pour savoir si on avait raison.
**Explication :** Imagine que tu prévoies la météo de demain. Tu n'as pas besoin d'attendre demain pour ajuster ta prévision : si dans l'heure qui vient le temps se dégrade plus vite que prévu, tu corriges déjà ta prévision pour demain. La « différence temporelle », c'est exactement ça : comparer ce qu'on avait estimé avec ce qui vient de se passer, et ajuster un petit peu son estimation en conséquence — sans attendre la fin de toute la journée (ou de l'épisode) pour savoir si on avait raison.
**Exemple :** Si l'agent pensait qu'une situation valait 40 points, mais qu'il vient de gagner 8 points et d'arriver dans une situation qui semble valoir 45 points, alors la situation de départ valait en fait un peu plus que prévu (45+8 > 40) — l'agent ajuste donc son estimation à la hausse, vers 41,26 environ.
**Pour aller plus loin :** $V(s_t) \leftarrow V(s_t)+\alpha[r_t+\gamma V(s_{t+1})-V(s_t)]$, $\delta_t$=erreur TD. Exemple : $V=40,r=8,\gamma=0{,}99,V'=45,\alpha=0{,}1 \Rightarrow \delta_t=12{,}55$, $V\leftarrow41{,}26$.

---

## Discount factor (γ)
**Définition :** Un réglage qui dit à l'agent à quel point il doit se soucier du futur par rapport au présent.
**Explication :** Imagine que chaque récompense future perd un peu de sa valeur à chaque seconde qui passe, comme de la glace qui fond. γ contrôle la vitesse de fonte : proche de 1, la fonte est très lente et l'agent garde en tête des récompenses lointaines comme si elles comptaient presque autant que celles d'aujourd'hui (il « planifie loin »). Proche de 0, la fonte est rapide : seul l'instant présent compte vraiment, et l'agent devient « myope », un peu comme une heuristique simple qui ne pense pas au lendemain.
**Exemple :** Dans le projet, γ=0,99 : l'agent garde quasiment toute sa motivation même pour des récompenses obtenues une centaine de pas plus tard. C'est important car une bonne décision maintenant (envoyer un camion vers une pelle peu chargée) peut éviter un embouteillage 30 ou 40 pas plus tard — l'agent doit « s'en souvenir » au moment de décider.
**Pour aller plus loin :** $0{,}99^{100}\approx0{,}366$ (37% de la valeur initiale conservée après 100 pas) ; avec $\gamma=0{,}9$, $0{,}9^{100}\approx0{,}00003$ (quasi nul).

---

## Learning rate (α)
**Définition :** La taille du « pas » que l'agent fait chaque fois qu'il corrige une erreur.
**Explication :** Imagine que tu corriges le tir d'une fléchette qui a un peu manqué la cible. Si tu corriges trop fort à chaque essai, tu vas osciller sans jamais te stabiliser (trop à gauche, puis trop à droite, etc.). Si tu corriges trop peu, tu vas mettre une éternité à t'améliorer. Le learning rate, c'est exactement ce réglage : il dit de combien on ajuste l'estimation à chaque correction.
**Exemple :** Dans le projet, les « carnets » de Q-Learning et SARSA se corrigent avec des pas assez larges (α=0,1), car chaque case du carnet est indépendante des autres — corriger une case ne dérange pas le reste. Les réseaux de neurones de DQN et PPO, eux, utilisent des pas beaucoup plus petits (α=0,0003), car une seule grosse correction pourrait perturber tout le réseau d'un coup, puisque ses réglages sont partagés entre toutes les situations.
**Pour aller plus loin :** $\theta \leftarrow \theta - \alpha \nabla_\theta L(\theta)$. Tabulaire (Q-Learning/SARSA/TD) : $\alpha=0{,}1$ ; réseaux (DQN/PPO) : $\alpha=3\times10^{-4}$.

---

## Fonction d'approximation
**Définition :** Un système qui devine une bonne estimation pour des situations qu'il n'a jamais vues exactement, en se basant sur des situations ressemblantes déjà rencontrées.
**Explication :** Le carnet de Q-Learning a besoin d'une ligne par situation déjà rencontrée — mais avec 147 chiffres différents pour décrire chaque situation, il y a quasiment un nombre infini de situations possibles, impossible à toutes noter une par une. Une fonction d'approximation (comme un réseau de neurones) remplace ce carnet géant par un système qui « généralise » : deux situations qui se ressemblent beaucoup donneront des estimations qui se ressemblent aussi, même si l'une d'elles n'a jamais été vue pendant l'entraînement.
**Exemple :** Le réseau de DQN reçoit les 147 chiffres qui décrivent la situation actuelle, et calcule directement une estimation pour chacune des 7 actions possibles — sans avoir besoin d'avoir déjà vu exactement cette situation auparavant. Un carnet équivalent aurait dû stocker une ligne pour chaque combinaison possible de ces 147 chiffres, un nombre absolument énorme.
**Pour aller plus loin :** Table $Q$ : 17 034 états effectifs (cas réel du projet). Réseau MLP 128×128 : entrée = 147 valeurs continues dans $[0,1]$, sortie = 7 valeurs $Q$, poids stockés en quelques centaines de Ko.

---

## ReLU (Rectified Linear Unit)
**Définition :** Une règle toute simple appliquée dans le réseau de neurones : si le nombre est positif, on le garde tel quel ; s'il est négatif, on le remplace par zéro.
**Explication :** Imagine une vanne qui laisse passer l'eau seulement dans un sens : si la pression « pousse » dans le bon sens (nombre positif), elle passe sans changement ; si elle « pousse » dans le mauvais sens (nombre négatif), la vanne bloque tout (résultat = zéro). Cette règle très simple, répétée des milliers de fois dans le réseau, lui permet de combiner les informations de façon beaucoup plus riche qu'une simple addition — c'est ce qui permet au réseau d'apprendre des choses compliquées plutôt que de simples relations toutes droites.
**Exemple :** Pour les nombres -3, 0 et 5 en entrée, ReLU renvoie 0, 0 et 5. Dans les réseaux de PPO et DQN (128×128 neurones), chaque neurone applique cette règle après avoir combiné les informations sur l'état du site minier — c'est ce qui permet au réseau de repérer des liens compliqués entre la situation observée et l'intérêt de chaque action, que de simples additions ne pourraient pas capturer.
**Pour aller plus loin :** $f(x)=\max(0,x)$. Utilisée dans le MLP 128×128 de PPO et DQN.

---

## Acteur-critique
**Définition :** Une organisation à deux rôles : un « joueur » qui choisit les actions, et un « coach » qui évalue si la situation se présente bien ou mal.
**Explication :** Si le joueur (acteur) devait juger seul si son coup était bon, il aurait du mal : il sait juste « j'ai obtenu tel score à la fin », sans savoir si c'est un bon ou un mauvais score dans l'absolu pour cette situation. Le coach (critique) apprend justement à estimer « à quoi peut-on s'attendre dans cette situation » — et compare ensuite ce qui s'est vraiment passé à cette attente. Si le résultat est meilleur que prévu, le joueur est encouragé à refaire ce choix ; si c'est moins bien que prévu, il est découragé. Cette comparaison « mieux ou moins bien que prévu » est beaucoup plus utile au joueur qu'un simple score brut.
**Exemple :** Dans PPO, le même réseau de neurones a deux « sorties » : une qui propose une action (l'acteur), une qui estime « à quoi s'attendre dans cette situation » (le critique). Si le critique pensait qu'une situation valait 4 100 t/h et que le résultat réel est 4 200 t/h, l'action prise est encouragée car elle a fait mieux que prévu. Dans le projet, ce « coach » est très fiable : il prédit correctement plus de 99,9 % de ce qui va se passer.
**Pour aller plus loin :** $\hat{A}_t = G_t - V_\phi(s_t)$. Explained variance PPO = 0,999 (nominal).

---

## Taux d'utilisation
**Définition :** Le pourcentage du temps pendant lequel une pelle (ou un camion) travaille vraiment, plutôt que de rester sans rien faire.
**Explication :** C'est comme mesurer combien de temps un employé travaille réellement pendant ses heures de présence, par rapport au temps où il attend qu'on lui amène du travail. Un taux élevé veut dire que la machine est presque toujours occupée — mais attention, « occupée » ne veut pas forcément dire « productive de la meilleure façon possible » : on peut être occupé à faire des choses peu utiles.
**Exemple :** Si une pelle charge des camions pendant 45 minutes sur une heure (et attend 15 minutes sans camion), son taux d'utilisation est de 75 %. Dans le projet, la baseline « Fixed Assignment » obtient le taux d'utilisation le plus élevé (96,5 %) en conditions normales, mais PPO (94,6 %) transporte malgré tout plus de minerai au total (4 208,75 vs 4 074,0 t/h) — un taux légèrement plus faible peut être compensé par de meilleures décisions.
**Pour aller plus loin :** $\text{Taux} = \frac{\text{temps actif}}{\text{temps disponible}} \times 100$.

---

## Productivité
**Définition :** La quantité de minerai transportée par heure — le « score final » du jeu.
**Explication :** C'est le chiffre le plus important : tout ce que fait l'agent ne compte vraiment que si, à la fin, ça se traduit par plus de minerai amené aux points de traitement. Chaque fois qu'un camion termine sa livraison, son chargement (140 tonnes) s'ajoute au total de la journée. À la fin de la journée simulée (8h), on divise le total par 8 pour avoir un chiffre « par heure », facile à comparer entre méthodes.
**Exemple :** PPO atteint 4 208,75 t/h en moyenne en conditions normales. Sur une journée de 8h, ça représente environ 33 670 tonnes transportées en tout, soit l'équivalent d'environ 240 trajets complets de camion.
**Pour aller plus loin :** $R_{\text{rendement}}=\sum_{c\in D_t}\text{capacité}(c)$, $D_t$ = camions déchargés dans $[t,t+\Delta t]$.

---

## Temps d'attente
**Définition :** Le temps perdu par un camion à patienter dans une file, sans rien faire.
**Explication :** C'est exactement comme une file d'attente à la caisse d'un magasin : le temps que tu passes à attendre ton tour ne te sert à rien, c'est du temps « gaspillé ». Pour un camion, ce chrono démarre dès qu'il arrive devant une pelle ou un point de déchargement déjà occupé, et s'arrête quand c'est enfin son tour. Un temps d'attente faible veut dire que les camions sont bien répartis entre les ressources disponibles, sans que tout le monde se précipite au même endroit.
**Exemple :** En conditions normales, la baseline « Fixed Assignment » a le temps d'attente le plus faible (environ 28 minutes sur une journée de 8h), car chaque pelle a toujours exactement le bon nombre de camions affectés. PPO attend un peu plus (environ 32 minutes) car il privilégie la production totale, mais reste le meilleur parmi les agents qui apprennent. Dans le scénario où il y a trop de camions par rapport aux pelles, une règle simple comme « pelle la plus proche » fait grimper l'attente à 200 minutes — presque la moitié de la journée passée à l'arrêt !
**Pour aller plus loin :** Nominal : Fixed=27,9 min, PPO=31,8 min. High_load : Nearest Shovel=200,2 min (≈42% du temps de l'épisode).

---

## Consommation spécifique
**Définition :** La quantité de carburant utilisée pour transporter une tonne de minerai — un peu comme la consommation aux 100 km d'une voiture, mais « par tonne transportée ».
**Explication :** Deux méthodes peuvent transporter la même quantité de minerai, mais l'une peut le faire en utilisant moins de carburant — par exemple en évitant d'envoyer un camion vers une pelle déjà pleine, puis de devoir le réorienter vers une autre. C'est donc un indicateur « complémentaire » à la production : il dit si le transport se fait de façon économe, ou avec beaucoup de trajets inutiles.
**Exemple :** En conditions normales, PPO et DQN consomment environ 4 à 6 % de carburant en moins par tonne transportée que la méthode « Fixed Assignment », et cet avantage se maintient même quand il y a beaucoup de pannes — preuve que les agents ne se contentent pas de transporter plus, ils évitent aussi les trajets superflus.
**Pour aller plus loin :** $E_e = \frac{\text{carburant total}}{\text{tonnage total}}$. Nominal : PPO=0,0412 L/t (-4,4%), DQN=0,0404 L/t (-6,3%), Fixed=0,0431 L/t (Welch $t=-10{,}01$, $p<10^{-8}$).

---

## Scénario
**Définition :** Une « configuration » différente de la simulation — comme un niveau de jeu plus ou moins difficile.
**Explication :** Imagine un jeu vidéo avec plusieurs niveaux de difficulté : « normal », « beaucoup de monde » et « plein de pannes ». Le terrain de jeu reste le même (même carte, mêmes règles physiques), mais le nombre de camions ou la fréquence des pannes change. Chaque méthode (agent ou règle simple) est testée sur chacun de ces niveaux, plusieurs fois, pour voir comment elle se comporte selon les conditions.
**Exemple :** Le mémoire utilise 3 scénarios : « nominal » (conditions stables, la référence), « high_load » (beaucoup plus de camions que d'habitude, pour tester l'équilibrage des files) et « high_breakdown » (5 fois plus de pannes que la normale, pour tester la résistance aux imprévus). Par exemple, DQN ne perd que très peu de productivité entre le scénario normal et celui avec beaucoup de pannes, malgré ce changement important.
**Pour aller plus loin :** Nominal (12 camions, 3 pelles, 2 dumps, $p_b=2\%$, MF=4,0) ; high_load (18 camions, MF=6,0) ; high_breakdown ($p_b=10\%$). DQN : 4 168,5 t/h (nominal) → 3 991,8 t/h (high_breakdown), soit -4,2%.

---

## Robustesse
**Définition :** La capacité à rester performant même quand les conditions deviennent difficiles.
**Explication :** C'est la différence entre une personne qui s'effondre complètement au moindre imprévu, et une autre qui s'adapte et continue à bien faire son travail même quand les choses se compliquent. Une méthode « fragile » voit ses résultats chuter fortement dès que les conditions changent (plus de camions, plus de pannes) ; une méthode « robuste » garde des résultats proches de ses résultats habituels, car elle s'adapte à la nouvelle situation au lieu de suivre une règle figée qui ne marche bien que dans un seul cas précis.
**Exemple :** Quand on multiplie par 5 le taux de pannes, DQN et PPO continuent à transporter plus de minerai que la règle fixe « Fixed Assignment » — et l'écart est suffisamment net pour ne pas être dû au hasard. À l'inverse, quand il y a beaucoup plus de camions que d'habitude, des règles simples comme « pelle la plus proche » voient leur temps d'attente tripler, alors que DQN ne recule que très légèrement.
**Pour aller plus loin :** High_breakdown : DQN=3 991,8 t/h, PPO=3 946,25 t/h > Fixed=3 860,5 t/h (Welch DQN vs Fixed $t=5{,}863$, $p=1{,}3\times10^{-5}$, validant H1).

---

## Congestion
**Définition :** Un embouteillage : trop de camions qui veulent utiliser la même pelle au même moment.
**Explication :** C'est exactement comme un bouchon sur l'autoroute : si tout le monde prend la même sortie parce qu'elle est « la plus proche », la sortie devient vite débordée alors qu'une autre sortie un peu plus loin reste libre. Une règle simple ne voit pas ce problème arriver : elle continue d'envoyer des camions vers la pelle « la plus proche » même si celle-ci est déjà débordée — un cercle vicieux. Un agent qui observe l'état des files d'attente, lui, peut voir le problème venir et répartir les camions vers des pelles moins chargées, même si elles sont un peu plus loin.
**Exemple :** Quand il y a beaucoup de camions sur le site (scénario high_load), les règles simples comme « pelle la plus proche » provoquent un énorme embouteillage : les camions attendent plus de 200 minutes, soit plus de 40 % de la journée. DQN, qui regarde l'état des files avant de décider, limite beaucoup mieux les dégâts.
**Pour aller plus loin :** High_load (18 camions, 3 pelles, MF=6,0) : Nearest Shovel et Shortest Path ≥ 200 min d'attente ; DQN limite la perte à -3,6% vs Fixed (5 664,8 t/h).

---

## Zone tampon
**Définition :** Une zone d'attente, comme un parking, où les camions patientent avant que ce soit leur tour à une pelle ou un point de déchargement.
**Explication :** C'est la « file d'attente physique » devant chaque pelle ou point de déchargement : quand un camion arrive et que la ressource est occupée, il ne bloque pas la circulation, il va patienter dans cette zone. L'agent peut voir combien de camions attendent déjà dans chacune de ces zones, et utilise cette information pour décider où envoyer le prochain camion — un peu comme regarder combien de monde fait la queue avant de choisir sa caisse au supermarché.
**Exemple :** Si la zone tampon d'une pelle contient déjà 10 camions, l'agent reçoit un petit « moins de points » s'il envoie encore un camion vers cette pelle — ce qui l'incite à choisir une autre pelle moins encombrée plutôt que d'aggraver l'embouteillage.
**Pour aller plus loin :** $q_p$ = longueur de la file de la pelle $p$, une des 147 variables de l'observation. Pénalité $w_4 \cdot (-\min(\frac{q_{p_i}}{30},1))$ : pour $q_{p_i}=10$, terme $=0{,}3\times(-10/30)=-0{,}1$.

---

## Stérile
**Définition :** La roche « sans valeur » qu'il faut creuser et enlever pour pouvoir accéder au minerai utile en dessous.
**Explication :** Imagine que tu dois enlever beaucoup de terre et de cailloux avant d'arriver au trésor enfoui. Cette terre et ces cailloux (le « stérile ») ne valent rien en eux-mêmes, mais il faut quand même les transporter ailleurs avec des camions, ce qui prend du temps et coûte de l'argent — sans rapporter de revenu direct. C'est un élément important du contexte minier réel, même si le projet ne s'en occupe pas directement pour l'instant.
**Exemple :** Ce mémoire se concentre sur le transport du minerai vers 2 points de traitement. Le transport du stérile vers une zone de stockage n'est pas inclus dans la simulation actuelle, mais les mêmes mécanismes (camions, pelles, files d'attente) pourraient s'appliquer à ce troisième type de destination — une piste d'amélioration mentionnée pour la suite du projet.
**Pour aller plus loin :** Ratio de décapage = volume de stérile / volume de minerai (non modélisé dans le simulateur actuel).

---

## Match Factor (MF)
**Définition :** Un chiffre qui dit s'il y a « juste assez », « trop » ou « pas assez » de camions par rapport aux pelles disponibles.
**Explication :** Imagine un restaurant avec un certain nombre de serveurs et un certain nombre de clients : s'il y a trop de clients par serveur, il y a de l'attente partout ; s'il y a trop peu de clients, les serveurs restent les bras croisés. Le Match Factor, c'est ce rapport entre le nombre de camions et le nombre de pelles. Comme chaque camion passe une bonne partie de son temps « en trajet » plutôt qu'en train de charger, il faut en réalité plusieurs camions par pelle pour qu'elle ne s'arrête jamais — donc un MF « idéal » est nettement supérieur à 1.
**Exemple :** Le scénario normal du projet a 12 camions pour 3 pelles, soit un MF de 4 — un équilibre jugé correct pour ce site. Le scénario « high_load » a 18 camions pour 3 pelles (MF=6) : il y a « trop » de camions par pelle, ce qui crée de l'attente pour les règles simples qui ne s'adaptent pas.
**Pour aller plus loin :** $MF = \frac{N_{\text{camions}}}{N_{\text{pelles}}}$. Nominal MF=4,0 ; high_load MF=6,0 ; high_breakdown MF=4,0 (seul $p_b$ change).

---

## Explained Variance
**Définition :** Un chiffre qui dit si le « coach » (le critique de PPO) prédit bien ou mal ce qui va se passer.
**Explication :** Si le coach annonce systématiquement à l'avance le résultat qui va vraiment arriver, on dit qu'il « explique » 100 % de ce qui se passe (1,0). S'il ne fait pas mieux qu'une moyenne générale, peu importe la situation, il explique 0 % (0,0). Et s'il se trompe carrément plus qu'une moyenne aléatoire, le chiffre devient négatif — signe que quelque chose ne va pas dans l'apprentissage. C'est un indicateur de confiance important : si le coach se trompe beaucoup, ses conseils au joueur (l'acteur) ne valent pas grand-chose.
**Exemple :** Dans le projet, le coach de PPO est extrêmement fiable : il prédit correctement plus de 99,9 % de ce qui va se passer, dans les trois scénarios testés (normal, surcharge, beaucoup de pannes). On peut donc faire confiance à ses conseils pour ajuster la stratégie du joueur.
**Pour aller plus loin :** $EV = 1 - \dfrac{\mathrm{Var}(G_t - V_\phi(s_t))}{\mathrm{Var}(G_t)}$. PPO : $EV=0{,}999$ (nominal), $0{,}9999$ (high_load), $0{,}9998$ (high_breakdown).

---

## Policy Loss / Value Loss
**Définition :** Des chiffres qui montrent à quel point le « joueur » et le « coach » sont encore en train d'ajuster leurs estimations pendant l'entraînement.
**Explication :** Pendant l'entraînement, le coach (critique) compare sans cesse ses prédictions avec ce qui se passe vraiment, et corrige son estimation — l'écart entre prédiction et réalité (la « value loss ») diminue au fil du temps, signe que le coach devient de plus en plus fiable. Le joueur (acteur), lui, ajuste sa façon de choisir les actions — sa « policy loss » reste petite et stable plutôt que de tomber à zéro, ce qui est normal : ça veut juste dire que les ajustements restent raisonnables d'une étape à l'autre, pas que le joueur a arrêté de progresser.
**Exemple :** Dans le projet, l'erreur du coach de PPO diminue énormément au cours de l'entraînement (il devient beaucoup plus précis), tandis que l'ajustement du joueur reste petit et stable tout au long. Pour DQN, l'erreur de ses estimations diminue aussi globalement, mais avec quelques variations dues au fait qu'il pioche ses exemples d'apprentissage au hasard dans son carnet de souvenirs (replay buffer), qui contient des expériences de qualité très variable.
**Pour aller plus loin :** PPO nominal : value loss $32{,}0\to0{,}17$ ; policy gradient loss stable entre -0,025 et -0,014. DQN : train/loss $0{,}040\to0{,}025$, pic à 0,30.

---

## Efficacité d'échantillonnage (Sample Efficiency)
**Définition :** Le pourcentage du temps d'entraînement qu'il a vraiment fallu pour que l'agent devienne « bon » — le reste n'apportant que des petits réglages.
**Explication :** C'est comme apprendre à conduire : après un certain nombre d'heures, on conduit déjà correctement, et les heures suivantes n'apportent que de petites améliorations. Si un agent atteint l'essentiel de son niveau final après seulement un tiers de son entraînement, c'est qu'il « apprend vite » par rapport au temps qu'on lui a donné — et qu'on aurait peut-être pu s'arrêter plus tôt sans trop perdre en qualité.
**Exemple :** Dans le projet, DQN et PPO atteignent l'essentiel de leur niveau final après environ un quart à quatre cinquièmes de leur entraînement (selon le scénario), alors que Q-Learning et SARSA ont besoin de presque tout leur temps d'entraînement pour se stabiliser. Un cas particulier : dans le scénario avec beaucoup de pannes, SARSA ne progresse quasiment pas du tout — il reste stable autour de son niveau de départ dès le début.
**Pour aller plus loin :** Seuil = performance lissée à moins de 5% de sa valeur finale, jusqu'à la fin de l'entraînement. DQN/PPO : 28% à 80% de 2M steps ; Q-Learning/SARSA : 83% à 99% de 30 000 épisodes. SARSA high_breakdown = 0,0% (pas de progression nette, stable ~142).

---

## Significativité statistique (Welch, ANOVA, Tukey HSD)
**Définition :** Des tests qui permettent de répondre à la question « cette différence entre deux méthodes est-elle réelle, ou est-ce juste un coup de chance ? »
**Explication :** Si tu lances une pièce 10 fois et obtiens 7 fois « face », est-ce que la pièce est vraiment truquée, ou est-ce juste le hasard ? Ces tests statistiques répondent à ce genre de question, mais appliqués aux performances des méthodes du projet (testées chacune 10 fois avec des conditions de départ différentes). Un test compare deux méthodes entre elles ; un autre vérifie d'abord si, parmi toutes les méthodes, il y en a au moins une qui se démarque vraiment ; un troisième compare ensuite toutes les paires de méthodes deux à deux pour voir lesquelles sont vraiment différentes.
**Exemple :** Dans le projet, ces tests confirment que PPO transporte vraiment plus de minerai que la règle fixe « Fixed Assignment » (pas un coup de chance). Mais ils montrent aussi que PPO et DQN, malgré une petite différence de chiffres entre eux, sont en réalité statistiquement équivalents en production — leur différence pourrait très bien être due au hasard. Ce type de test permet de répondre précisément à un membre du jury qui demanderait « êtes-vous sûr que cet écart est réel ? »
**Pour aller plus loin :** ANOVA nominal $F(7,72)=481{,}25$, $p<10^{-56}$. Welch PPO vs Fixed nominal $t=6{,}356$, $p=3{,}0\times10^{-6}$. Tukey HSD PPO vs DQN nominal $p=0{,}799$ (équivalents malgré 4 208,75 vs 4 168,5 t/h). 10 réplications, seeds 42-51.

---

## Graine aléatoire (seed) et réplications
**Définition :** Une « graine » fixe le hasard d'une simulation pour pouvoir la rejouer exactement pareil ; une réplication, c'est rejouer toute la simulation avec une graine différente, pour voir si le résultat tient toujours.
**Explication :** La simulation contient beaucoup de hasard (temps de trajet qui varient, pannes qui arrivent au hasard...). Si on lance la simulation une seule fois et qu'on obtient un bon résultat, comment savoir si c'est parce que la méthode est vraiment bonne, ou juste parce qu'on a eu de la « chance » sur ce tirage en particulier ? La solution : rejouer plusieurs fois avec des « graines » différentes (mais connues et fixes, pour que l'expérience reste reproductible), et regarder la moyenne et la dispersion des résultats obtenus.
**Exemple :** Dans le projet, chaque méthode est évaluée 10 fois, avec 10 graines différentes mais toujours les mêmes (de 42 à 51) — ce qui permet de calculer une moyenne fiable et de savoir si les différences entre méthodes sont solides ou pourraient changer d'une fois à l'autre. Au total, ça représente 8 méthodes × 3 scénarios × 10 graines = 240 résultats pour l'ensemble du benchmark.
**Pour aller plus loin :** Entraînement avec seed=42 ; évaluation sur 10 réplications (seeds 42-51). $8 \times 3 \times 10 = 240$ lignes de résultats.
