# Explication simple des algorithmes — basée sur les pseudocodes du mémoire

Ce document reprend **mot pour mot** les pseudocodes des Algorithmes 4.1 à 4.4
du mémoire et les explique en langage courant, ligne par ligne.
Objectif : pouvoir expliquer n'importe quel algo en 2 minutes au tableau.

---

## Algorithme 4.1 — Q-Learning (Section 4.4.1)

### Le pseudocode du mémoire ligne par ligne

```
ENTRÉES : Environnement minier E, α, γ
          ε = 1.0 (exploration initiale), ε_min = 0.01

1. Initialiser Q(s, a) ← 0 pour tout (s, a)
   → On part de zéro : on ne sait rien au départ.

2. Pour chaque épisode (= poste de 8h) :
3.   s₀ ← reset(E)         → La mine repart à zéro : tous les camions au dépôt.

4.   Tant que l'épisode n'est pas fini :
5.     Choisir a_t par ε-greedy :
         Avec prob. ε  → action aléatoire         (on explore)
         Sinon         → argmax Q(s_t, a)          (on exploite ce qu'on sait)
6.     Décoder a_t en (pelle p_i, dump d_j)
7.     Exécuter a_t : observer r_t et s_{t+1}
8.     Mettre à jour :
       Q(s,a) ← Q(s,a) + α × [ r_t + γ × max Q(s',a') − Q(s,a) ]
       ↑ on corrige notre estimation vers ce qu'on a réellement obtenu
9.     s_t ← s_{t+1}

10.  ε ← max(ε_min, ε − Δε)    → On explore de moins en moins au fil du temps.

RETOUR : π*(s) = argmax Q(s, a)   → La meilleure action connue pour chaque état.
```

### Ce que fait la formule de mise à jour
```
Q(s,a) ← Q(s,a) + α × [ r_t + γ × max Q(s',a') − Q(s,a) ]
                          ↑                ↑             ↑
                   ce qu'on      meilleur futur    ce qu'on
                   a gagné        estimé           pensait gagner
```
La différence entre ce qu'on espérait et ce qu'on a obtenu s'appelle
**l'erreur TD**. L'agent corrige son carnet de notes proportionnellement à α.

### Propriété clé
**Off-policy** : à la ligne 8, on utilise `max Q(s', a')` — la meilleure
action *théorique* — même si on a choisi une action aléatoire à l'étape
précédente. L'agent apprend toujours vers l'optimal, peu importe ce qu'il fait.

---

## Algorithme 4.2 — TD Learning (Section 4.4.2)

### Le pseudocode du mémoire ligne par ligne

```
ENTRÉES : Environnement E, α, γ, fonction de valeur V(s)

1. Initialiser V(s) ← 0 pour tout s
   → On estime la valeur de chaque situation, pas des actions.

2. Pour chaque épisode :
3.   s₀ ← reset(E)

4.   Tant que l'épisode n'est pas fini :
5.     Choisir a_t selon la politique π
6.     Décoder a_t en (pelle p_i, dump d_j)
7.     Exécuter a_t : observer r_t, s_{t+1}
8.     Calculer l'erreur TD : δ_t = r_t + γ V(s_{t+1}) − V(s_t)
       ↑ "cette situation était-elle meilleure ou moins bonne que prévu ?"
9.     Mettre à jour : V(s_t) ← V(s_t) + α × δ_t
10.    s_t ← s_{t+1}

RETOUR : Fonction de valeur V(s)
```

### Rôle dans le mémoire
TD Learning est la **brique de base** des autres algorithmes :
- Q-Learning applique δ_t pour mettre à jour Q(s, a)
- SARSA utilise la même formule mais on-policy
- PPO calcule le GAE comme une somme pondérée de δ_t successifs

---

## Algorithme 4.3 — SARSA (Section 4.4.3)

### Le pseudocode du mémoire ligne par ligne

```
ENTRÉES : Environnement E, α, γ
          ε = 1.0, ε_min = 0.01

1. Initialiser Q(s, a) ← 0 pour tout (s, a)

2. Pour chaque épisode :
3.   s₀ ← reset(E)
4.   Choisir a₀ par ε-greedy    ← DIFFÉRENCE : SARSA choisit l'action AVANT la boucle

5.   Tant que l'épisode n'est pas fini :
6.     Décoder a_t en (pelle p_i, dump d_j)
7.     Exécuter a_t : observer r_t, s_{t+1}
8.     Choisir a_{t+1} par ε-greedy depuis Q(s_{t+1}, ·)
       ← DIFFÉRENCE : on choisit la prochaine action MAINTENANT

9.     Mettre à jour (on-policy) :
       Q(s,a) ← Q(s,a) + α × [ r_t + γ × Q(s_{t+1}, a_{t+1}) − Q(s,a) ]
                                              ↑
                                    action RÉELLEMENT choisie, pas le max

10.    s_t ← s_{t+1},  a_t ← a_{t+1}

11.  ε ← max(ε_min, ε − Δε)

RETOUR : π*(s) = argmax Q(s, a)
```

### La différence structurelle avec Q-Learning (telle qu'expliquée dans le mémoire)

| Ligne de mise à jour | Q-Learning | SARSA |
|---|---|---|
| Formule | `max Q(s', a')` | `Q(s', a_{t+1})` |
| Type | Off-policy | **On-policy** |
| Signification | Meilleure action possible | Action **réellement** choisie |

> *"Q-Learning utilise max_{a'} Q(s_{t+1}, a') (off-policy) tandis que SARSA
> utilise Q(s_{t+1}, a_{t+1}) (on-policy), où a_{t+1} est l'action
> effectivement choisie par la politique ε-greedy."* — mémoire Section 4.4.3

### Conséquence pratique
SARSA intègre le coût de ses propres erreurs d'exploration → politique
**plus conservative** que Q-Learning. Résultat nominal : 3 222 t/h vs 3 273 t/h.

---

## Algorithme 4.4 — PPO (Section 4.8)

### Le pseudocode du mémoire ligne par ligne

```
ENTRÉES : Environnement E (graphe G, camions C, pelles P, dumps D)
          Acteur π_θ et Critique V_φ (MLP 128×128, ReLU)
          α=0.0003, γ=0.99, λ_GAE=0.95, ε_clip=0.2, T_max steps

1. Initialiser θ et φ aléatoirement

2. Pour k = 1, 2, ..., T_max/N :     ← itérations d'entraînement

   // Phase 1 : COLLECTER des expériences
3. Pour t = 1, ..., N :
4.   Observer s_t = ({q_p}, {x_c}, {z_r}, t_courant)
5.   Normaliser s_t dans [0, 1]
6.   Échantillonner a_t ~ π_θ(· | s_t)    ← l'acteur choisit
7.   Décoder a_t en (pelle p_i, dump d_j) :
       i ← a_t ÷ |D|,  j ← a_t mod |D|
       Si a_t = |P|×|D| : ATTENDRE
8.   Simuler le cycle du camion courant :
       Trajet c → p_i  (temps log-normal, Eq. 3.2)
       Attente + chargement à p_i
       Trajet chargé p_i → d_j
       Attente + déchargement à d_j
       Retour à vide d_j → p_i
9.   Calculer la récompense R_t = w1·R_rendement + w2·R_équité + w3·R_coût
10.  Observer s_{t+1}
11.  Stocker (s_t, a_t, R_t, s_{t+1}, log π_θ(a_t|s_t))

   // Phase 2 : CALCULER les avantages (GAE)
12. Â_t = Σ (γ·λ_GAE)^l × δ_{t+l}
         où δ_t = R_t + γ·V_φ(s_{t+1}) − V_φ(s_t)
    ↑ "Cette action était-elle meilleure que ce que le critique attendait ?"

   // Phase 3 : OPTIMISER sur plusieurs mini-batches
13. Pour chaque époque (mini-batches de 64) :
14.   Calculer le ratio : r_t(θ) = π_θ(a_t|s_t) / π_θk(a_t|s_t)
      ↑ "combien la politique a changé depuis la collecte ?"
15.   Objectif acteur (CLIPPING) :
      L^CLIP = E[ min( r_t·Â_t,  clip(r_t, 0.8, 1.2)·Â_t ) ]
      ↑ si r_t > 1.2 (trop de changement), on plafonne → stabilité
16.   Objectif critique :
      L^V = E[ (V_φ(s_t) − R_t^cible)² ]
17.   θ ← θ + α·∇L^CLIP     (acteur)
      φ ← φ − α·∇L^V        (critique)

RETOUR : Politique entraînée π_θ*
```

### Les 3 phases à retenir

**Phase 1 — Collecter** : l'acteur joue dans la mine et enregistre ce qui se passe.

**Phase 2 — Évaluer** : le critique calcule si chaque décision était bonne
ou mauvaise (*"avantage"* = différence entre récompense réelle et attendue).

**Phase 3 — Améliorer** : on met à jour la politique, mais avec un frein
(le clipping) pour ne pas changer trop radicalement d'un coup.

### Le mécanisme clé : le clipping (ligne 15)

```
clip(r_t, 0.8, 1.2) signifie :
  Si r_t = 1.5  → plafonné à 1.2   (n'exploite pas trop une bonne action)
  Si r_t = 0.5  → planché à 0.8    (ne pénalise pas trop une mauvaise)
  Si r_t = 1.1  → gardé tel quel
```

**Pourquoi c'est important** : sans ce frein, une seule bonne expérience
pourrait faire basculer toute la politique → instabilité. Le clipping garantit
des mises à jour progressives et stables.

### Différence fondamentale avec DQN

| | DQN | PPO |
|---|---|---|
| Apprend | Une fonction Q(s,a) | Une politique π(a\|s) directement |
| Type | Off-policy + replay | **On-policy** (données fraîches) |
| Stabilité | Moyenne (peut diverger) | **Élevée** (grâce au clipping) |
| Réseau(x) | 1 réseau Q | **2 réseaux** : acteur + critique |

---

## Tableau comparatif — ce qu'il faut retenir

| | Q-Learning | SARSA | DQN | PPO |
|---|---|---|---|---|
| Algo | 4.1 | 4.3 | — | 4.4 |
| Type | Off-policy | On-policy | Off-policy | On-policy |
| Représentation | Table Q | Table Q | Réseau Q_θ | Acteur π_θ + Critique V_φ |
| Mise à jour avec | max Q(s',a') | Q(s', a_{t+1}) | max Q_θ-(s',a') | Clipped surrogate |
| Résultat nominal | 3 273 t/h | 3 222 t/h | 3 311 t/h | 3 335 t/h |

---

## 3 questions du jury et leurs réponses

**"Quelle est la différence entre Q-Learning et SARSA ?"**
> Q-Learning utilise `max Q(s', a')` (off-policy) : il apprend toujours
> vers la meilleure action théorique. SARSA utilise `Q(s', a_{t+1})`
> (on-policy) : il apprend avec l'action réellement choisie, y compris les
> erreurs d'exploration. SARSA est plus conservateur, Q-Learning converge
> vers une meilleure politique finale.

**"Qu'est-ce que le GAE dans PPO ?"**
> Generalized Advantage Estimation (ligne 12 de l'algo). C'est la façon
> dont PPO mesure si une action était bonne : on calcule l'erreur TD
> δ_t = R_t + γ·V(s_{t+1}) − V(s_t) et on la somme sur plusieurs steps
> futurs, pondérée par (γ·λ). λ=0.95 équilibre biais (estimé sur peu de
> steps) et variance (estimé sur beaucoup de steps).

**"Pourquoi PPO et pas DQN comme algorithme final ?"**
> DQN est off-policy et peut diverger dans des environnements stochastiques.
> PPO est on-policy avec clipping → les mises à jour sont toujours bornées
> (entre 0.8 et 1.2 fois l'ancienne politique). Sur le scénario high_breakdown,
> PPO obtient 47.7 min d'attente contre 57.3 min pour DQN, confirmant
> sa meilleure robustesse aux perturbations.
