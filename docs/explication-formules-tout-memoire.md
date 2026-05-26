# Explication simple des formulations mathématiques

Ce document explique de manière simple et concrète les formulations mathématiques de tout le mémoire.

## Chapitre 1 : Concepts et cadre général

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

## Chapitre 2 : État de l'art

Le chapitre 2 compare des familles de méthodes (heuristiques, méta-heuristiques, systèmes industriels, RL). Il n'introduit pas de nouvelles formules mathématiques spécifiques du modèle proposé.

- On y trouve des notions de graphe pour modéliser le réseau routier minier.
- On y explique pourquoi certaines approches sont "myopes" ou "déterministes".

> Point important : le chapitre 2 justifie le besoin d'un modèle mathématique plus explicite et d'un agent qui apprend en tenant compte de l'incertitude.

## Chapitre 3 : Modélisation du problème

C'est ici que passent les formules mathématiques principales du mémoire.

### 1. Coût de déplacement sur le réseau minier

La distance, le temps et l'usure sont combinés pour calculer un coût de trajet entre deux nœuds :

$$
C_{ij} = \alpha \cdot T_{ij} + \beta \cdot D_{ij} + \gamma \cdot E_{ij}
$$

- $D_{ij}$ = distance entre le point $i$ et le point $j$.
- $T_{ij}$ = temps de trajet moyen entre ces deux points.
- $E_{ij}$ = estimation du coût énergétique ou de l'usure.
- $\alpha, \beta, \gamma$ = poids choisis pour équilibrer ces facteurs.

Exemple (mémoire) : Si $\alpha = 0{,}5$, $\beta = 0{,}3$, $\gamma = 0{,}2$, $T_{ij} = 15$ minutes, $D_{ij} = 2{,}5$ km, $E_{ij} = 0{,}8$, alors $C_{ij} = 0{,}5 \times 15 + 0{,}3 \times 2{,}5 + 0{,}2 \times 0{,}8 = 8{,}41$.

### 2. Variabilité stochastique des temps de trajet

Le réseau réel est variable, donc on ajoute un terme aléatoire :

$$
T_{ij}(t) = \bar{T}_{ij} + \epsilon_{ij}(t)
$$

- $\bar{T}_{ij}$ = temps moyen.
- $\epsilon_{ij}(t)$ = bruit aléatoire.
- On suppose $\epsilon_{ij}(t) \sim \mathcal{N}(0, \sigma_{ij}^2)$, c'est-à-dire un bruit gaussien centré.

Exemple (mémoire) : Si $\bar{T}_{ij} = 12$ minutes et $\eta_{ij}(t) = 2{,}3$ minutes (échantillon d'une distribution normale avec $\sigma = 2{,}5$), alors $T_{ij}(t) = 12 + 2{,}3 = 14{,}3$ minutes.

### 3. Formule du MDP appliquée au dispatching minier

Le chapitre 3 formalise l'état du système par :

$$
s_t = \Big( \{q_p\}_{p \in \text{Pelles}},\ \{x_c\}_{c \in \text{Camions}},\ \{z_r\}_{r \in \text{Ressources}},\ t_{\text{courant}} \Big)
$$

- $q_p$ = nombre de camions en file d'attente à la pelle $p$.
- $x_c$ = position/statut du camion $c$.
- $z_r$ = état de disponibilité de la ressource $r$.
- $t_{\text{courant}}$ = instant de la simulation.

Exemple (mémoire) : Pour une mine avec 3 pelles, 12 camions et 2 dumps à $t_{\text{courant}} = 240$ minutes (4 heures) : $s_t = (\{2{,}3,1\},\ \{\text{yard},\text{shovel}_1,\text{dump}_1,\ldots\},\ \{0{,}5\},\ 240)$ où les files d'attente aux pelles sont $\{2,3,1\}$ camions respectivement, les camions sont répartis sur le réseau, et les dumps seront disponibles dans $\{0,5\}$ minutes.

### 4. Action de dispatching

L'action prise par l'agent est une affectation de camions aux pelles, ou bien la décision d'attendre :

$$
a_t = \text{Assignment}(c_1 \rightarrow p_j, c_2 \rightarrow p_k, \ldots) \quad \text{ou} \quad a_t = \text{ATTENDRE}
$$

- Une action est une liste de paires "camion → pelle".
- ATTENDRE signifie ne rien changer pour ce pas de décision.

Exemple :
- $a_t = \{ c_1 \rightarrow p_2,\ c_3 \rightarrow p_1 \}$ signifie envoyer le camion 1 à la pelle 2 et le camion 3 à la pelle 1.

### 5. Transition d'état

Après avoir pris l'action, le système change d'état selon la simulation :

$$
s'_t = \text{Simuler}(s_t, a_t, \Delta t)
$$

On peut représenter cette transition comme :

$$
s'_t = f(s_t, a_t) + \eta_t, \quad \eta_t \sim \mathcal{N}(0, \Sigma)
$$

- $f(s_t,a_t)$ = évolution déterministe liée à l'action.
- $\eta_t$ = perturbation aléatoire.

Exemple :
- Si l'action envoie un camion à une pelle et que le temps de trajet varie, le nouvel état $s'_t$ reflète la position actualisée des camions et l'évolution des files.

### 6. Fonction de récompense

La récompense combine plusieurs objectifs : rendement, équité et coût.

$$
R_t = w_1 \cdot R_{\text{rendement}}(s_t, s'_t) + w_2 \cdot R_{\text{équité}}(s_t) + w_3 \cdot R_{\text{coût}}(a_t)
$$

Exemple (mémoire) : Si $w_1 = 1{,}0$, $w_2 = 0{,}1$, $w_3 = 0{,}05$, $R_{\text{rendement}} = 280$ tonnes, $R_{\text{équité}} = -5{,}2$ (pénalité due à files inégales), $R_{\text{coût}} = -120$ (pénalité de distance), alors $R_t = 1{,}0 \times 280 + 0{,}1 \times (-5{,}2) + 0{,}05 \times (-120) = 273{,}48$.

Chaque composante se comprend ainsi :

#### 6.1 Rendement

On calcule le tonnage livré pendant le pas de temps :

$$
D_t = \{c \in \text{Camions} \mid c \text{ est déchargé dans } [t, t+\Delta t]\}
$$

$$
R_{\text{rendement}}(s_t, s'_t) = \sum_{c \in D_t} \mathrm{capacité}(c)
$$

Exemple (mémoire) : Si 2 camions sont déchargés dans l'intervalle $[t, t+\Delta t]$ avec des capacités de $140$ tonnes chacun, alors $R_{\text{rendement}} = 140 + 140 = 280$ tonnes.

#### 6.2 Équité

L'équité cherche à éviter que certaines pelles restent surchargées tandis que d'autres sont sous-utilisées :

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

- $\mathrm{Var}$ = variance, mesurant l'écart entre les files.
- $\lambda$ = coefficient positif de pondération.

Exemple (mémoire) : Si $\lambda = 0{,}5$ et les files d'attente sont $\{q_p\} = \{2, 5, 3\}$ camions, la variance est $\phi(\{2,5,3\}) = \text{Var}(\{2,5,3\}) = 2{,}33$, alors $R_{\text{équité}} = -0{,}5 \times 2{,}33 = -1{,}17$.

#### 6.3 Coût

On pénalise les actions qui génèrent des déplacements coûteux :

$$
R_{\text{coût}}(a_t) = -\sum_{(c \rightarrow p) \in a_t} C(\text{pos}(c), \text{pos}(p))
$$

- $C(\text{pos}(c), \text{pos}(p))$ = coût du trajet entre la position du camion et la pelle.
- Le signe négatif indique qu'un trajet coûteux réduit la récompense.

Exemple (mémoire) : Si l'action $a_t$ assigne 2 camions avec des coûts de trajet $C = 8{,}41$ et $C = 6{,}25$, alors $R_{\text{coût}} = -(8{,}41 + 6{,}25) = -14{,}66$.

## Chapitre 4 : Méthodologie et algorithmes RL

### Espace d'action (Eq. 4.1)

L'espace d'action est discret et encode les paires (pelle, dump) :

$$
\mathcal{A} = \text{Discrete}\big(|\mathcal{P}| \times |\mathcal{D}| + 1\big)
$$

Exemple (mémoire) : Pour 3 pelles et 2 dumps, $|\mathcal{P}| \times |\mathcal{D}| + 1 = 7$ actions (0-6). Si $a_t = 4$, alors $\text{shovel\_idx} = 4 \div 2 = 2$ (pelle 3) et $\text{dump\_idx} = 4 \bmod 2 = 0$ (dump 1).

### Q-Learning (Eq. 4.3)

Méthode tabulaire off-policy qui apprend la fonction Q(s,a) :

$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \Big[ r_t + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t) \Big]
$$

Exemple (mémoire) : Si $Q(s_t, a_t) = 50$, $r_t = 10$, $\gamma = 0{,}99$, $\max_{a'} Q(s_{t+1}, a') = 55$, $\alpha = 0{,}1$, alors $Q(s_t, a_t) \leftarrow 50 + 0{,}1 \times (10 + 0{,}99 \times 55 - 50) = 51{,}45$.

### TD Learning (Eq. 4.4)

Principe général de mise à jour par différence temporelle :

$$
V(s_t) \leftarrow V(s_t) + \alpha \Big[ r_t + \gamma V(s_{t+1}) - V(s_t) \Big]
$$

Exemple (mémoire) : Si $V(s_t) = 40$, $r_t = 8$, $\gamma = 0{,}99$, $V(s_{t+1}) = 45$, $\alpha = 0{,}1$, alors $V(s_t) \leftarrow 40 + 0{,}1 \times (8 + 0{,}99 \times 45 - 40) = 41{,}26$.

### SARSA (Eq. 4.5)

Variante on-policy qui utilise l'action effectivement prise :

$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \Big[ r_t + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t) \Big]
$$

Exemple (mémoire) : Si $Q(s_t, a_t) = 50$, $r_t = 10$, $\gamma = 0{,}99$, $Q(s_{t+1}, a_{t+1}) = 52$ (action effectivement prise), $\alpha = 0{,}1$, alors $Q(s_t, a_t) \leftarrow 50 + 0{,}1 \times (10 + 0{,}99 \times 52 - 50) = 51{,}15$.

### DQN (Deep Q-Network)

Extension du Q-Learning utilisant un réseau de neurones profond pour approximer Q(s,a) :

$$
Q_\theta(s, a) \approx Q^*(s, a)
$$

- $Q_\theta$ est paramétré par les poids $\theta$ du réseau neuronal.
- Architecture MLP 128×128 avec ReLU (Section 4.5.2).
- Utilise l'expérience replay et un réseau cible pour la stabilité.

Exemple : Le réseau prend l'état normalisé $s_t$ en entrée (vecteur de dimension $d$) et produit les valeurs Q pour chaque action possible. Pour 7 actions, la sortie est un vecteur $[Q(s, a_0), Q(s, a_1), ..., Q(s, a_6)]$.

### PPO (Proximal Policy Optimization)

Algorithme moderne qui optimise directement une politique avec une contrainte de proximité :

$$
L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]
$$

- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ est le ratio de probabilités.
- $\epsilon = 0{,}2$ est le paramètre de clipping.
- $\hat{A}_t$ est l'avantage estimé avec GAE ($\lambda = 0{,}95$).
- Architecture MLP 128×128 avec ReLU (Section 4.5.4).

Exemple : Si l'action choisie a une probabilité de 0.3 avant mise à jour et 0.35 après, alors $r_t = 0{,}35 / 0{,}3 = 1{,}17$. Avec $\epsilon = 0{,}2$, le ratio est borné entre 0.8 et 1.2, donc $r_t$ est limité à 1.2.

## Chapitre 5 : Implémentation et expérimentation

Ce chapitre ne contient pas de nouvelles équations mathématiques mais décrit l'architecture logicielle et le protocole expérimental.

### Architecture globale

Le système est basé sur l'interface Gymnasium avec les composants suivants :

- **Environment** : Orchestrateur implémentant `reset()` et `step()`
- **ObservationBuilder** : Construit l'observation normalisée $s_t$
- **ActionMask** : Masque les actions invalides dynamiquement
- **RewardCalculator** : Calcule $R_t$ selon Eq. 3.7

### Normalisation des observations

Les observations sont normalisées dans $[0, 1]$ :

$$
s_{t,\text{norm}} = \frac{s_t - s_{\min}}{s_{\max} - s_{\min}}
$$

Exemple : Si la position d'un camion varie de 0 à 100 mètres et est à 50 mètres, alors $s_{\text{norm}} = (50 - 0) / (100 - 0) = 0{,}5$.

### Protocole d'entraînement

- Nombre d'épisodes : 100 pour PPO, 1000 pour Q-Learning/SARSA
- Taille du batch : 64
- Taux d'apprentissage : $3 \times 10^{-4}$ pour PPO, $0{,}1$ pour Q-Learning/SARSA

### Scénarios de simulation

Les scénarios testent la robustesse :
- Nominal : configuration standard
- High-load : plus de camions que de pelles
- Low-load : moins de camions que de pelles
- High-breakdown : taux de panne élevé
- Single-shovel : une seule pelle disponible
- Short-shift : durée de simulation réduite

## Chapitre 6 : Résultats et analyse

Ce chapitre présente l'analyse statistique des résultats. Bien qu'il ne contienne pas de nouvelles équations théoriques, il utilise des métriques statistiques.

### Indicateurs de performance (KPIs)

Conformes au Tableau 4.8 du mémoire (Section 4.7.4) :

- **Productivité** : $\frac{\sum T_i}{H}$ (tonnes par heure)
- **Utilisation** : $\frac{\text{Temps actif}}{\text{Temps total}} \times 100$ (%)
- **Temps d'attente moyen** : $\frac{1}{N} \sum w_i$ (minutes)
- **Consommation spécifique** : $\frac{\text{Carburant}}{\text{Tonnage}}$ (L/tonne)
- **Coût par cycle** : $\frac{\text{Carburant}}{\text{Cycles}}$ (L/cycle)

### Analyse statistique

Les résultats sont présentés sous forme de moyenne ± écart-type :

$$
\text{KPI} = \mu \pm \sigma
$$

Exemple : Si la productivité est $8000 \pm 250$ t/h sur 10 réplications, cela signifie que la moyenne est 8000 t/h avec un écart-type de 250 t/h.

### Comparaison de méthodes

Comparaison des méthodes (baselines vs RL) sur différents scénarios :

- FIFO : baseline classique
- Shortest Path : baseline gloutonne
- Fixed Assignment : baseline zéro-niveau
- Nearest Shovel : baseline géographique
- Q-Learning : RL tabulaire
- SARSA : RL on-policy
- DQN : Deep RL valeur
- PPO : Deep RL politique (méthode principale)

## Glossaire rapide des symboles

### Chapitre 3
- $\mathcal{S}$ : états possibles.
- $\mathcal{A}$ : actions possibles.
- $\mathcal{P}(s' \mid s, a)$ : transition probabiliste.
- $\mathcal{R}(s, a, s')$ : récompense.
- $s_t$ : état à l'instant $t$.
- $a_t$ : action à l'instant $t$.
- $s'_t$ : état suivant.
- $\Delta t$ : pas de décision.
- $D_t$ : ensemble des camions déchargés entre $t$ et $t+\Delta t$.
- $\phi$ : fonction d'équité.
- $C_{ij}$ : coût de déplacement entre deux nœuds.
- $\epsilon_{ij}(t)$ : bruit fluctuant du temps de trajet.

### Chapitre 4
- $\alpha$ : Taux d'apprentissage.
- $\gamma$ : Facteur d'actualisation.
- $\epsilon$ : Taux d'exploration (Q-Learning, SARSA) ou paramètre de clipping (PPO).
- $Q(s, a)$ : Fonction de valeur d'action (Q-Learning, SARSA).
- $V(s)$ : Fonction de valeur d'état (TD Learning).
- $Q_\theta(s, a)$ : Approximation par réseau neuronal (DQN).
- $\max_{a'}$ : Maximum sur toutes les actions (Q-Learning off-policy).
- $\delta_t$ : Erreur de différence temporelle (TD error).
- $\lambda$ : Coefficient de GAE pour PPO.
- $r_t(\theta)$ : Ratio de probabilités PPO.
- $\hat{A}_t$ : Avantage estimé par PPO.
- $|P|$ : Nombre de pelles (espace d'action).
- $|D|$ : Nombre de dumps (espace d'action).

### Chapitre 5
- $s_{t,\text{norm}}$ : Observation normalisée.
- $\mu$ : Moyenne statistique.
- $\sigma$ : Écart-type statistique.

## Synthèse pratique

### Ce qu'il faut retenir pour expliquer le mémoire

- **Chapitre 1** : Cadre MDP pour la décision séquentielle
- **Chapitre 2** : Justification du RL par rapport aux méthodes existantes
- **Chapitre 3** : Formules clés du modèle (coût, temps, état, action, récompense)
- **Chapitre 4** : Algorithmes RL (Q-Learning, SARSA, DQN, PPO)
- **Chapitre 5** : Implémentation Gymnasium, normalisation, protocole expérimental
- **Chapitre 6** : Analyse statistique des résultats, comparaison des méthodes

### Comment l'expliquer simplement

1. Commencer par le MDP : état, action, transition, récompense
2. Montrer l'état du dispatching minier (files, positions, disponibilité)
3. Expliquer l'action comme choix d'affectation
4. Décrire la transition comme simulation avec stochastique
5. Présenter la récompense multi-objectif (rendement, équité, coût)
6. Introduire les algorithmes RL (tabulaires puis profonds)
7. Expliquer l'implémentation avec Gymnasium
8. Présenter les résultats et la comparaison avec les baselines

> Astuce : utilisez un exemple de mini-mine avec 2 pelles et 3 camions pour illustrer chaque notion.
