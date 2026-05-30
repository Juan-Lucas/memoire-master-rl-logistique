# Diagrammes et visualisations à créer

Ce document décrit les diagrammes et visualisations essentiels à créer pour votre soutenance et votre mémoire.

---

## 1. Architecture globale du système

### Description
Diagramme montrant l'architecture complète de votre solution RL pour le dispatching minier.

### Composants à inclure
- **Environnement Gymnasium** (central)
- **ObservationBuilder** → construit l'état s_t
- **ActionMask** → masque les actions invalides
- **RewardCalculator** → calcule R_t selon Eq. 3.7
- **Agent PPO** → prend les décisions
- **Simulateur** → modélise le système minier
- **Baseline policies** (FIFO, Shortest Path, etc.)

### Flèches et relations
- Agent PPO → Environnement (action a_t)
- Environnement → Agent PPO (observation s_t, récompense r_t)
- Simulateur → Environnement (transition s'_t)
- ObservationBuilder → Environnement (construction s_t)
- RewardCalculator → Environnement (calcul R_t)

### Outils recommandés
- **Draw.io** (gratuit, en ligne)
- **Lucidchart** (payant)
- **PowerPoint** (intégré)

### Exemple de structure
```
┌─────────────┐
│  Agent PPO  │
└──────┬──────┘
       │ a_t
       ↓
┌─────────────────────────────────────┐
│      Environnement Gymnasium        │
│  ┌───────────────────────────────┐  │
│  │ ObservationBuilder            │  │
│  │ (construit s_t)               │  │
│  └───────────────────────────────┘  │
│  ┌───────────────────────────────┐  │
│  │ ActionMask                    │  │
│  │ (masque actions invalides)     │  │
│  └───────────────────────────────┘  │
│  ┌───────────────────────────────┐  │
│  │ RewardCalculator              │  │
│  │ (calcule R_t)                 │  │
│  └───────────────────────────────┘  │
└──────┬──────────────────────────────┘
       │ s_t, r_t
       ↑
┌─────────────┐
│  Simulateur │
└─────────────┘
```

---

## 2. Réseau minier (graphe)

### Description
Schéma du réseau routier minier avec pelles, dumps, yard et routes.

### Composants à inclure
- **3 pelles** (P1, P2, P3) avec files d'attente
- **2 dumps** (D1, D2)
- **Yard** (zone de départ)
- **Routes** avec distances et temps de trajet
- **Camions** en mouvement (flèches)

### Exemple de structure
```
        P1 (file: 2 camions)
         ↑
         │ 3 km, 8 min
         │
    ┌────┴────┐
    │  Yard   │
    └────┬────┘
         │ 2 km, 5 min
         ↓
        P2 (file: 3 camions)
         ↑
         │ 4 km, 12 min
         │
        D1
```

### Outils recommandés
- **Draw.io** (formes prédéfinies)
- **PowerPoint** (formes personnalisées)
- **Visio** (professionnel)

---

## 3. Cycle MDP (état → action → transition → récompense)

### Description
Diagramme de flux montrant le cycle complet du MDP.

### Composants à inclure
- **État s_t** (files, positions, disponibilités)
- **Action a_t** (affectation camion→pelle ou attendre)
- **Transition** (simulation stochastique)
- **État s'_{t+1}** (nouvel état)
- **Récompense R_t** (rendement + équité + coût)

### Exemple de structure
```
┌──────────┐
│  s_t     │ → Files: {2,3,1}
│ (État)   │   Positions: {...}
└────┬─────┘   Disponibilités: {0,5}
     │
     │ a_t = {c1→P2, c3→P1}
     ↓
┌──────────┐
│ Transition│ → Simulation stochastique
└────┬─────┘
     │
     ↓
┌──────────┐
│ s'_{t+1} │ → Files: {1,2,2}
│ (Nouvel  │   Positions: {...}
│  état)   │   Disponibilités: {1,4}
└────┬─────┘
     │
     │ R_t = 273.48
     ↓
┌──────────┐
│Récompense│ → Rendement: 280 t
└──────────┘   Équité: -5.2
              Coût: -120
```

### Outils recommandés
- **Draw.io** (flux)
- **Lucidchart** (processus)

---

## 4. Architecture réseau PPO

### Description
Schéma de l'architecture du réseau de neurones PPO.

### Composants à inclure
- **Input layer** (dimension ~50)
- **Hidden layer 1** (128 neurones, ReLU)
- **Hidden layer 2** (128 neurones, ReLU)
- **Output layer** (7 actions, softmax)
- **Value head** (estimation V(s))

### Exemple de structure
```
Input (50 features)
    ↓
┌─────────────────────┐
│ Dense (128) + ReLU  │
└─────────────────────┘
    ↓
┌─────────────────────┐
│ Dense (128) + ReLU  │
└─────────────────────┘
    ↓
    ├─────────────┬─────────────┐
    ↓             ↓             ↓
Policy head   Value head    (optionnel)
(7 actions)   V(s)          Entropy bonus
(softmax)
```

### Outils recommandés
- **Draw.io** (réseaux de neurones)
- **NN-SVG** (générateur SVG)
- **PowerPoint** (formes)

---

## 5. Comparaison des méthodes (bar chart)

### Description
Graphique à barres comparant les 8 méthodes sur différents KPIs.

### Données à inclure
- **Productivité** (t/h) : PPO, DQN, Q-Learning, SARSA, FIFO, Shortest Path, Fixed Assignment, Nearest Shovel
- **Temps d'attente** (min) : même ordre
- **Consommation** (L/t) : même ordre
- **Utilisation** (%) : même ordre

### Exemple de structure
```
Productivité (t/h) — Scénario nominal
┌─────────────────────────────────────────────────────┐
│ Fixed  ████████████████████████████████████████████ │ 4074
│ NearS  █████████████████████████████████████        │ 3830
│ ShortP █████████████████████████████████████        │ 3830
│ FIFO   ████████████████████████████████             │ 3318
│ PPO    ████████████████████████████████             │ 3335
│ DQN    ████████████████████████████████             │ 3311
│ Q-Learn ███████████████████████████████             │ 3273
│ SARSA  ██████████████████████████████               │ 3222
└─────────────────────────────────────────────────────┘
```

### Outils recommandés
- **Python/Matplotlib** (génération automatique)
- **Excel** (graphiques simples)
- **PowerPoint** (graphiques intégrés)

---

## 6. Learning curves (PPO)

### Description
Courbe d'apprentissage montrant l'évolution de la récompense moyenne au fil des épisodes.

### Axes
- **X** : Épisodes (0-100)
- **Y** : Récompense moyenne (ou productivité)

### Courbes à inclure
- **Récompense moyenne** (ligne principale)
- **Récompense médiane** (ligne secondaire)
- **Intervalle de confiance** (zone ombrée)

### Exemple de structure
```
Récompense
    ↑
    │      ╱╲
    │     ╱  ╲
    │    ╱    ╲
    │   ╱      ╲─────
    │  ╱              ╲
    │ ╱                ╲
    │╱                  ╲
    └───────────────────────→ Épisodes
     0                   100
```

### Outils recommandés
- **Python/Matplotlib** (génération depuis logs)
- **Python/Seaborn** (graphiques statistiques)
- **TensorBoard** (si logs TensorBoard disponibles)

---

## 7. Robustesse sur scénarios perturbés

### Description
Graphique montrant la performance relative de PPO vs baselines sur différents scénarios.

### Scénarios
- Nominal
- High-load (15 camions)
- Low-load (8 camions)
- High-breakdown (5% pannes)
- Single-shovel (1 pelle)
- Short-shift (4h)

### Exemple de structure
```
Performance relative (% de nominal)
┌─────────────────────────────────────────────────────┐
│ Nominal      ████████████████████████████████████  │ 100%
│ High-load    ██████████████████████████████████    │  92%
│ Low-load     ████████████████████████████████████  │  95%
│ High-break   █████████████████████████████████     │  88%
│ Single-shovel █████████████████████████████████   │  85%
│ Short-shift  ████████████████████████████████████  │  90%
└─────────────────────────────────────────────────────┘
```

### Outils recommandés
- **Python/Matplotlib** (bar chart horizontal)
- **Excel** (graphiques simples)

---

## 8. Fonction de récompense multi-objectif

### Description
Schéma montrant les trois composantes de la récompense et leur combinaison.

### Composants
- **Rendement** (tonnage livré)
- **Équité** (variance des files)
- **Coût** (distance/consommation)
- **Combinaison pondérée** (w1, w2, w3)

### Exemple de structure
```
┌─────────────┐
│ Rendement   │ → 280 tonnes
│ (w1=1.0)    │   × 1.0 = 280
└──────┬──────┘
       │
       ↓
┌─────────────┐
│  Équité     │ → Variance = 2.33
│ (w2=0.1)    │   × 0.1 × (-1) = -0.23
└──────┬──────┘
       │
       ↓
┌─────────────┐
│   Coût      │ → Somme coûts = 14.66
│ (w3=0.05)   │   × 0.05 × (-1) = -0.73
└──────┬──────┘
       │
       ↓
┌─────────────┐
│   R_total   │ = 280 - 0.23 - 0.73 = 279.04
└─────────────┘
```

### Outils recommandés
- **Draw.io** (flux)
- **PowerPoint** (formes)

---

## 9. Timeline de la méthodologie

### Description
Chronologie montrant les étapes de votre méthodologie.

### Étapes
1. Formulation MDP
2. Implémentation environnement Gymnasium
3. Implémentation baselines
4. Implémentation agents RL
5. Entraînement et tuning
6. Évaluation sur scénarios
7. Analyse statistique

### Exemple de structure
```
┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│  MDP     │→│Gymnasium │→│Baselines │→│  Agents  │
│formulation│ │           ││           ││   RL     │
└──────────┘ └──────────┘ └──────────┘ └────┬─────┘
                                           ↓
                                    ┌──────────┐
                                    │Entraînement│
                                    └────┬─────┘
                                         ↓
                                    ┌──────────┐
                                    │ Évaluation│
                                    └────┬─────┘
                                         ↓
                                    ┌──────────┐
                                    │  Analyse  │
                                    └──────────┘
```

### Outils recommandés
- **Draw.io** (timeline)
- **PowerPoint** (timeline smartart)
- **Lucidchart** (timeline)

---

## 10. Tableau de comparaison complète

### Description
Tableau récapitulatif comparant toutes les méthodes sur tous les KPIs.

### Colonnes
- Méthode
- Productivité (t/h)
- Temps attente (min)
- Consommation (L/t)
- Utilisation camions (%)
- Utilisation pelles (%)
- Score global

### Exemple de structure
```
| Méthode       | Prod. (t/h) | Attente (min) | Conso (L/t) | Util. (%) |
|---------------|-------------|---------------|-------------|-----------|
| Fixed Assign. | 4074        | 27.9          | 0.0431      | 96.8      |
| Nearest Shovel| 3830        | 74.6          | 0.0443      | 87.8      |
| Shortest Path | 3830        | 74.6          | 0.0443      | 87.8      |
| FIFO          | 3318        | 32.8          | 0.0528      | 95.8      |
| PPO           | 3335        | 30.6          | 0.0525      | 97.1      |
| DQN           | 3311        | 33.4          | 0.0524      | 96.4      |
| Q-Learning    | 3273        | 36.9          | 0.0534      | 95.4      |
| SARSA         | 3222        | 52.5          | 0.0537      | 93.2      |
```

### Outils recommandés
- **LaTeX** (tableau dans mémoire)
- **Excel** (tableau pour présentation)
- **Python/Pandas** (génération depuis données)

---

## Outils pour créer les diagrammes

### Outils gratuits
- **Draw.io** (diagrams.net) : recommandé pour la plupart des diagrammes
- **Lucidchart** (version gratuite limitée)
- **PowerPoint** (intégré Office)
- **Google Slides** (gratuit en ligne)

### Outils Python (pour graphiques)
- **Matplotlib** : graphiques scientifiques
- **Seaborn** : graphiques statistiques
- **Plotly** : graphiques interactifs

### Outils professionnels
- **Visio** (Microsoft)
- **Adobe Illustrator** (vectoriel)

---

## Conseils pour les diagrammes

### Général
- **Simplicité** : éviter la surcharge d'information
- **Cohérence** : utiliser les mêmes couleurs/styles
- **Lisibilité** : police taille minimale 12pt
- **Légende** : toujours inclure une légende

### Couleurs
- **PPO** : bleu (méthode principale)
- **Baselines** : gris
- **Autres RL** : vert, orange, violet
- **Fond** : blanc ou très clair

### Tailles
- **Slides présentation** : 16:9 (1920×1080)
- **Mémoire** : adapter à la mise en page
- **Tableau blanc** : grand format lisible

---

## Checklist de création

### Priorité haute (obligatoire)
- [ ] Architecture globale du système
- [ ] Réseau minier
- [ ] Cycle MDP
- [ ] Comparaison des méthodes (bar chart)
- [ ] Learning curves PPO

### Priorité moyenne (recommandé)
- [ ] Architecture réseau PPO
- [ ] Robustesse scénarios perturbés
- [ ] Fonction de récompense multi-objectif
- [ ] Timeline méthodologie

### Priorité basse (optionnel)
- [ ] Tableau comparatif complet
- [ ] Diagramme séquentiel détaillé

---

**Nombre total de diagrammes** : 10
**Objectif minimal** : 5 (priorité haute)
**Objectif idéal** : 8 (priorité haute + moyenne)
