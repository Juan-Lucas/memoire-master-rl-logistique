# Explication simple des formulations mathématiques

Ce document explique de manière simple et concrète les formulations mathématiques des chapitres 1 à 3 du mémoire.

## Chapitre 1 : concepts et cadre général

### 1. Processus de décision markovien (MDP)
Le MDP est le cadre mathématique principal utilisé pour formuler le problème de dispatching minier.

- On représente un problème séquentiel par un quadruplet :
  - $\mathcal{S}$ : ensemble des états possibles.
  - $\mathcal{A}$ : ensemble des actions possibles.
  - $\mathcal{P}(s' \mid s, a)$ : probabilité de passer à l'état $s'$ quand on prend l'action $a$ depuis l'état $s$.
  - $\mathcal{R}(s, a, s')$ : récompense associée à cette transition.

Exemple simple :
- État $s$ = « camion A est vide près de la pelle 1, camion B est en route vers le concasseur ».
- Action $a$ = « envoyer camion A vers la pelle 2 ».
- Transition $s' =$ nouvel état après quelques minutes.
- Récompense $R =$ score qui mesure si cette décision améliore la productivité.

> Remarque : le chapitre 1 reste surtout conceptuel. Il définit le vocabulaire et le cadre MDP sans détailler encore les formules spécifiques du problème minier.

## Chapitre 2 : état de l'art

Le chapitre 2 compare des familles de méthodes (heuristiques, méta-heuristiques, systèmes industriels, RL). Il n’introduit pas de nouvelles formules mathématiques spécifiques du modèle proposé.

- On y trouve des notions de graphe pour modéliser le réseau routier minier.
- On y explique pourquoi certaines approches sont "+ myopes +" ou "+ déterministes +".

> Point important : le chapitre 2 justifie le besoin d’un modèle mathématique plus explicite et d’un agent qui apprend en tenant compte de l’incertitude.

## Chapitre 3 : modélisation du problème

C’est ici que passent les formules mathématiques principales du mémoire.

### 1. Coût de déplacement sur le réseau minier

La distance, le temps et l’usure sont combinés pour calculer un coût de trajet entre deux nœuds :

$$
C_{ij} = \alpha \cdot T_{ij} + \beta \cdot D_{ij} + \gamma \cdot E_{ij}
$$

- $D_{ij}$ = distance entre le point $i$ et le point $j$.
- $T_{ij}$ = temps de trajet moyen entre ces deux points.
- $E_{ij}$ = estimation du coût énergétique ou de l’usure.
- $\alpha, \beta, \gamma$ = poids choisis pour équilibrer ces facteurs.

Exemple : si $D_{ij}=10$, $T_{ij}=20$, $E_{ij}=5$ et $\alpha=1$, $\beta=1$, $\gamma=1$, alors $C_{ij}=35$.

### 2. Variabilité stochastique des temps de trajet

Le réseau réel est variable, donc on ajoute un terme aléatoire :

$$
T_{ij}(t) = \bar{T}_{ij} + \epsilon_{ij}(t)
$$

- $\bar{T}_{ij}$ = temps moyen.
- $\epsilon_{ij}(t)$ = bruit aléatoire.
- On suppose $\epsilon_{ij}(t) \sim \mathcal{N}(0, \sigma_{ij}^2)$, c’est-à-dire un bruit gaussien centré.

Exemple : si le temps moyen est 20 minutes et le bruit est +2 minutes, le temps réel peut devenir 22 minutes.

### 3. Formule du MDP appliquée au dispatching minier

Le chapitre 3 formalise l’état du système par :

$$
s_t = \Big( \{q_p\}_{p \in \text{Pelles}},\ \{x_c\}_{c \in \text{Camions}},\ \{z_r\}_{r \in \text{Ressources}},\ t_{\text{courant}} \Big)
$$

- $q_p$ = nombre de camions en file d’attente à la pelle $p$.
- $x_c$ = position/statut du camion $c$.
- $z_r$ = état de disponibilité de la ressource $r$.
- $t_{\text{courant}}$ = instant de la simulation.

Exemple concret :
- Si la pelle A a 3 camions en attente et la pelle B en a 1, alors $\{q_p\} = \{3, 1\}$.
- Si le camion 1 est libre près de la pelle A et le camion 2 est en route vers le concasseur, cela fait partie de $\{x_c\}$.

### 4. Action de dispatching

L’action prise par l’agent est une affectation de camions aux pelles, ou bien la décision d’attendre :

$$
a_t = \text{Assignment}(c_1 \rightarrow p_j, c_2 \rightarrow p_k, \ldots) \quad \text{ou} \quad a_t = \text{ATTENDRE}
$$

- Une action est une liste de paires "camion → pelle".
- ATTENDRE signifie ne rien changer pour ce pas de décision.

Exemple :
- $a_t = \{ c_1 \rightarrow p_2,\ c_3 \rightarrow p_1 \}$ signifie envoyer le camion 1 à la pelle 2 et le camion 3 à la pelle 1.

### 5. Transition d’état

Après avoir pris l’action, le système change d’état selon la simulation :

$$
s'_t = \text{Simuler}(s_t, a_t, \Delta t)
$$

On peut représenter cette transition comme :

$$
s'_t = f(s_t, a_t) + \eta_t, \quad \eta_t \sim \mathcal{N}(0, \Sigma)
$$

- $f(s_t,a_t)$ = évolution déterministe liée à l’action.
- $\eta_t$ = perturbation aléatoire.

Exemple :
- Si l’action envoie un camion à une pelle et que le temps de trajet varie, le nouvel état $s'_t$ reflète la position actualisée des camions et l’évolution des files.

### 6. Fonction de récompense

La récompense combine plusieurs objectifs : rendement, équité et coût.

$$
R_t = w_1 \cdot R_{\text{rendement}}(s_t, s'_t) + w_2 \cdot R_{\text{équité}}(s_t) + w_3 \cdot R_{\text{coût}}(a_t)
$$

Chaque composante se comprend ainsi :

#### 6.1 Rendement

On calcule le tonnage livré pendant le pas de temps :

$$
D_t = \{c \in \text{Camions} \mid c \text{ est déchargé dans } [t, t+\Delta t]\}
$$

$$
R_{\text{rendement}}(s_t, s'_t) = \sum_{c \in D_t} \mathrm{capacité}(c)
$$

Exemple : si deux camions livrent 100 t et 80 t, alors $R_{\text{rendement}} = 180$.

#### 6.2 Équité

L’équité cherche à éviter que certaines pelles restent surchargées tandis que d’autres sont sous-utilisées :

$$
R_{\text{équité}}(s_t) = -\lambda \cdot \phi\left(\{q_p\}_{p \in \text{Pelles}}\right)
$$

avec

$$
\phi\left(\{q_p\}_{p \in \text{Pelles}}\right) =
\begin{cases}
    \mathrm{Var}\left(\{q_p\}_{p \in \text{Pelles}}\right), & \text{si } |\text{Pelles}| \ge 2, \\
    0, & \text{si } |\text{Pelles}| = 1.
\end{cases}
$$

- $\mathrm{Var}$ = variance, mesurant l’écart entre les files.
- $\lambda$ = coefficient positif de pondération.

Exemple :
- Si les files sont $\{3, 3\}$, la variance est 0, donc l’équité est maximale.
- Si les files sont $\{1, 5\}$, la variance est élevée, la pénalité est plus forte.

#### 6.3 Coût

On pénalise les actions qui génèrent des déplacements coûteux :

$$
R_{\text{coût}}(a_t) = -\sum_{(c \rightarrow p) \in a_t} C(\text{pos}(c), \text{pos}(p))
$$

- $C(\text{pos}(c), \text{pos}(p))$ = coût du trajet entre la position du camion et la pelle.
- Le signe négatif indique qu’un trajet coûteux réduit la récompense.

Exemple : si deux affectations valent 10 et 8, alors $R_{\text{coût}} = -18$.

## Synthèse pratique

### Ce qu’il faut retenir pour expliquer demain

- Le chapitre 1 présente le cadre général : un problème de décision séquentielle se modélise avec un MDP.
- Le chapitre 2 justifie pourquoi les méthodes existantes ne suffisent pas et pourquoi le RL est prometteur.
- Le chapitre 3 donne les formules clés : coût de trajet, temps de trajet stochastique, état $s_t$, action $a_t$, transition $s'$, et récompense $R_t$.

### Comment l’expliquer simplement

1. Expliquer d’abord le principe du MDP : état, action, transition, récompense.
2. Montrer que l’état contient les files aux pelles, la position des camions et la disponibilité des ressources.
3. Montrer que l’action est un choix d’affectation ou une attente.
4. Expliquer que la transition est une simulation de ce qui se passe ensuite, avec un peu de hasard dans les temps de trajet.
5. Expliquer que la récompense mesure simultanément la productivité, l’équité et le coût.

> Astuce : utilisez un exemple de mini-mine avec deux pelles et trois camions pour illustrer chaque notion (état, action, transition, récompense).

## Glossaire rapide des symboles

- $\mathcal{S}$ : états possibles.
- $\mathcal{A}$ : actions possibles.
- $\mathcal{P}(s' \mid s, a)$ : transition probabiliste.
- $\mathcal{R}(s, a, s')$ : récompense.
- $s_t$ : état à l’instant $t$.
- $a_t$ : action à l’instant $t$.
- $s'_t$ : état suivant.
- $\Delta t$ : pas de décision.
- $D_t$ : ensemble des camions déchargés entre $t$ et $t+\Delta t$.
- $\phi$ : fonction d’équité.
- $C_{ij}$ : coût de déplacement entre deux nœuds.
- $\epsilon_{ij}(t)$ : bruit fluctuant du temps de trajet.
