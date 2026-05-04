# Hypothèses de modélisation (récapitulatif et implications)

Ce document reprend les hypothèses formelles retenues pour la modélisation (chapitre 3) et précise leurs implications pratiques et tests à mener.

1. Horizon temporel (temps discret)
   - Implication : choix critique de `\Delta t` (trade-off fidélité / coût de calcul).
   - Test conseillé : sensibilité de la politique aux valeurs de `\Delta t`.

2. Stationnarité à court terme
   - Implication : entraînement stable sur fenêtres de quelques heures.
   - Test conseillé : simulations avec ruptures (pluie, incidents) pour mesurer robustesse.

3. Observation incomplète
   - Implication : possible non-markovianité; nécessité d'architectures robustes (RNN, croyances).
   - Test conseillé : comparer RL avec mémoire (LSTM) vs sans mémoire.

4. Regroupement des camions par classes (homogénéité par classe)
   - Implication : réduction de la dimension d'état mais perte de granularité.
   - Test conseillé : scénarios avec hétérogénéité poussée pour mesurer biais.

5. Points de chargement/déchargement fixes pendant un épisode
   - Implication : simplifie l'expérimentation, nécessité d'étendre pour études de généralisation.

6. Pas de défaillance majeure durant l'épisode
   - Implication : valider ensuite la robustesse en injectant pannes et exceptions.

---

Pour chaque hypothèse, il est recommandé de documenter le protocole expérimental de validation (scénarios, métriques, seuils d'alerte) afin d'évaluer la portée des conclusions obtenues.