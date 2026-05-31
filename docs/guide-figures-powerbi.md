# Guide Power BI — Générer les 3 figures du Chapitre 6

Ce document explique étape par étape comment créer les 3 figures
manquantes du mémoire avec Power BI Desktop (gratuit).

---

## Prérequis

- **Power BI Desktop** installé (téléchargement gratuit sur microsoft.com)
- Les fichiers CSV du projet :
  - `data/results/benchmark_results.csv`
  - `data/results/q_learning_nominal/training_rewards.csv`
  - `data/results/sarsa_nominal/training_rewards.csv`
  - `data/results/q_learning_high_load/training_rewards.csv`
  - `data/results/sarsa_high_load/training_rewards.csv`

---

## FIGURE 1 — Barplot productivité comparative (3 scénarios)

**Objectif :** barres groupées montrant la productivité (t/h) de chaque
algorithme sur les scénarios nominal, high_load et high_breakdown.

### Étape 1 : Charger les données

1. Ouvrir Power BI Desktop
2. `Accueil` → `Obtenir des données` → `Texte/CSV`
3. Sélectionner `data/results/benchmark_results.csv`
4. Cliquer `Transformer les données` (Power Query s'ouvre)

### Étape 2 : Filtrer les scénarios

1. Dans Power Query, colonne `scenario` → clic droit → `Filtrer`
2. Garder uniquement : `nominal`, `high_load`, `high_breakdown`
3. Supprimer la colonne `nominal_asymétrique` si présente

### Étape 3 : Calculer la moyenne par algorithme et scénario

1. Onglet `Accueil` → `Regrouper par`
2. Regrouper par : `scenario` + `policy`
3. Nouvelle colonne : `prod_mean` = Moyenne de `productivity_tph`
4. Cliquer `OK`
5. `Fermer et appliquer`

### Étape 4 : Créer le visuel

1. Dans le volet `Visualisations` → choisir **Histogramme groupé**
2. Faire glisser :
   - `policy` → **Axe X** (légende des barres)
   - `prod_mean` → **Valeurs Y**
   - `scenario` → **Légende** (couleur des groupes)
3. Dans `Format visuel` :
   - Titre : `Productivité comparative (t/h) — 3 scénarios`
   - Couleurs : choisir une palette contrastée (bleu, orange, vert)

### Étape 5 : Trier et mettre en forme

1. Clic droit sur l'axe X → `Trier par` → `prod_mean` décroissant
2. Dans `Format` → `Axe Y` → activer le titre : `Productivité (t/h)`
3. Dans `Format` → `Axe X` → activer le titre : `Algorithme`
4. Activer les étiquettes de données : `Format` → `Étiquettes de données` → ON

### Étape 6 : Exporter

1. Clic droit sur le visuel → `Copier` → `Copier en tant qu'image`
2. Coller dans Paint ou Word, enregistrer en PNG
3. Nommer : `barplot_productivite_scenarios.png`
4. Copier dans `reports/figures/`

---

## FIGURE 2 — Courbes d'apprentissage Q-Learning et SARSA

**Objectif :** courbes de récompense cumulée par épisode montrant la
progression de Q-Learning et SARSA sur les scénarios nominal et high_load.

### Étape 1 : Charger les 4 fichiers CSV

1. `Accueil` → `Obtenir des données` → `Texte/CSV`
2. Charger `data/results/q_learning_nominal/training_rewards.csv`
3. Dans Power Query, renommer la table : `QL_nominal`
4. Ajouter une colonne calculée : `Algorithme` = `"Q-Learning"`, `Scénario` = `"Nominal"`
5. Répéter pour les 3 autres fichiers :
   - `sarsa_nominal/training_rewards.csv` → `SARSA_nominal`
     (Algorithme = `"SARSA"`, Scénario = `"Nominal"`)
   - `q_learning_high_load/training_rewards.csv` → `QL_highload`
     (Algorithme = `"Q-Learning"`, Scénario = `"High Load"`)
   - `sarsa_high_load/training_rewards.csv` → `SARSA_highload`
     (Algorithme = `"SARSA"`, Scénario = `"High Load"`)

### Étape 2 : Combiner les tables

1. Dans Power Query : `Accueil` → `Ajouter des requêtes` → `Ajouter des requêtes en tant que nouvelles`
2. Sélectionner les 4 tables
3. Nommer la table combinée : `LearningCurves`
4. `Fermer et appliquer`

### Étape 3 : Lisser les courbes (moyenne mobile)

1. Créer une mesure DAX :
   ```
   Reward_100ep = 
   AVERAGEX(
     FILTER(LearningCurves, 
       LearningCurves[episode] >= MAX(LearningCurves[episode]) - 100),
     LearningCurves[total_reward]
   )
   ```
   *(alternative simple : utiliser directement `total_reward`)*

### Étape 4 : Créer le visuel

1. Choisir **Graphique en courbes**
2. Faire glisser :
   - `episode` → **Axe X**
   - `total_reward` → **Valeurs Y**
   - `Algorithme` → **Légende**
3. Ajouter un **filtre de page** sur `Scénario` pour créer 2 graphiques séparés
4. Titres :
   - Graphique 1 : `Convergence — Scénario nominal`
   - Graphique 2 : `Convergence — Scénario high_load`

### Étape 5 : Mise en forme

1. Couleurs : Q-Learning en bleu, SARSA en orange
2. `Format` → `Axe Y` → titre : `Récompense cumulée`
3. `Format` → `Axe X` → titre : `Épisode`
4. Activer la légende

### Étape 6 : Exporter

1. Copier en tant qu'image
2. Nommer : `learning_curves_nominal.png` et `learning_curves_highload.png`
3. Copier dans `reports/figures/`

---

## FIGURE 3 — Comparaison temps d'attente (Fixed / PPO / DQN)

**Objectif :** graphique à barres groupées montrant les temps d'attente
moyens de Fixed Assignment, PPO et DQN sur les 3 scénarios côte à côte.

### Étape 1 : Charger et préparer

1. Utiliser la même table `benchmark_results.csv` que la Figure 1
2. Dans Power Query → `Filtrer` la colonne `policy` :
   garder uniquement `Fixed`, `PPO`, `DQN`
3. Recalculer la moyenne : `wait_mean` = Moyenne de `avg_wait_min_per_truck`

### Étape 2 : Créer le visuel

1. Choisir **Histogramme groupé**
2. Faire glisser :
   - `scenario` → **Axe X**
   - `wait_mean` → **Valeurs Y**
   - `policy` → **Légende**
3. Titre : `Temps d'attente moyen (min) — Fixed / PPO / DQN`

### Étape 3 : Ajouter les valeurs de référence

1. Ajouter une **ligne de constante** pour marquer 30 min (seuil cible)
2. `Format` → `Ligne de référence` → `Ajouter` → valeur = 30
3. Étiquette : `Seuil cible`

### Étape 4 : Mise en forme

1. Couleurs :
   - Fixed : gris
   - PPO : bleu
   - DQN : orange
2. `Format` → `Axe Y` → titre : `Temps d'attente moyen (min)`
3. `Format` → `Axe X` → renommer les libellés :
   - `nominal` → `Nominal`
   - `high_load` → `High Load`
   - `high_breakdown` → `High Breakdown`
4. Activer les étiquettes de données

### Étape 5 : Exporter

1. Copier en tant qu'image
2. Nommer : `wait_time_comparison.png`
3. Copier dans `reports/figures/`

---

## Intégration dans le mémoire LaTeX

Une fois les PNG dans `reports/figures/`, remplacer dans `rl_thesis.tex` :

**Pour les barplots (ligne ~1800) :**
```latex
\begin{figure}[H]
    \centering
    \includegraphics[width=0.90\textwidth]{figures/barplot_productivite_scenarios.png}
    \caption{Productivité comparative des algorithmes sur les trois scénarios.}
    \label{fig:barplot-productivite}
\end{figure}
```

**Pour les courbes d'apprentissage :**
```latex
\begin{figure}[H]
    \centering
    \includegraphics[width=0.85\textwidth]{figures/learning_curves_nominal.png}
    \caption{Courbes de convergence de Q-Learning et SARSA — scénario nominal.}
    \label{fig:learning-curves}
\end{figure}
```

**Pour la comparaison temps d'attente (ligne ~1854) :**
```latex
\begin{figure}[H]
    \centering
    \includegraphics[width=0.85\textwidth]{figures/wait_time_comparison.png}
    \caption{Temps d'attente moyen (min) — Fixed Assignment, PPO et DQN
             sur les trois scénarios.}
    \label{fig:wait-comparison}
\end{figure}
```

---

## Conseils pour l'export final

- **Résolution** : exporter en 300 DPI minimum pour l'impression
- **Format** : PNG (pas JPEG pour éviter les artefacts)
- **Fond** : fond blanc, pas transparent
- **Police** : taille minimum 10 pt pour être lisible une fois réduit dans le PDF
- **Cohérence** : utiliser les mêmes couleurs pour les mêmes algorithmes dans les 3 figures
