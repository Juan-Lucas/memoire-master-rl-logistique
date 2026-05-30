# KPIs et mesures utilisées dans le mémoire

Ce document formalise les indicateurs de performance effectivement utilisés dans les chapitres 4, 5 et 6 du mémoire.

## 1. KPI principaux

### Productivité horaire
$$\text{Productivité} = \frac{\text{Tonnage total transporté}}{\text{Durée de l’épisode en heures}}$$

- Dans le code : $P_h = \frac{\text{total\_tonnage\_t}}{\text{episode\_minutes} / 60}$.
- Correspondance avec le mémoire : performance opérationnelle brute.

### Temps d’attente moyen par camion
$$E_t = \frac{\text{Temps d’attente total des camions}}{\text{Nombre de camions}}$$

- Dans le code : $E_t = \frac{\text{total\_wait\_min}}{\text{truck\_count}}$.
- KPI central pour la fluidité et la robustesse.

### Consommation spécifique
$$E_e = \frac{\text{Carburant total consommé}}{\text{Tonnage total transporté}}$$

- Dans le code : $E_e = \frac{\text{total\_fuel\_l}}{\text{total\_tonnage\_t}}$.
- Sert à mesurer l’efficacité énergétique.

### Taux d’utilisation des camions
$$U = \frac{\text{Temps actif total des camions}}{\text{Nombre de camions} \times \text{Durée d’épisode}} \times 100$$

- Dans le code : $U = \frac{\text{total\_active\_min}}{\text{truck\_count} \times \text{episode\_minutes}} \times 100$.
- Indicateur de disponibilité de la flotte.

### Coût moyen par cycle
$$C_{\text{cycle}} = \frac{\text{Carburant total consommé}}{\text{Nombre total de cycles}}$$

- Dans le code : $C_{\text{cycle}} = \frac{\text{total\_fuel\_l}}{\text{total\_cycles}}$.
- Estime le coût énergétique par tour de camion.

## 2. KPI complémentaires

### Reward cumulée
- Mesure la qualité globale de la politique RL pendant un épisode.
- Sert à suivre la convergence des agents Q-Learning, SARSA, DQN et PPO.

### Robustesse aux perturbations
- Mesurée par la stabilité des KPI sur les scénarios perturbés (`high_load`, `high_breakdown`).
- Une agent robuste conserve une performance proche de la baseline même en présence de pannes ou surcharge.

### Écart-type et intervalle de confiance
- Les résultats sont présentés en moyenne ± écart-type sur 10 réplications (seeds 42..51).
- Cette statistique permet d’évaluer la stabilité et la reproductibilité.

## 3. Liens avec les chapitres

- **Chapitre 4** : choix des KPI et justification de la fonction de récompense multi-objectif.
- **Chapitre 5** : protocole expérimental, scénarios et métriques d’évaluation.
- **Chapitre 6** : comparaison des méthodes sur les mêmes KPI et analyse de robustesse.

## 4. Pourquoi ces KPI ?

- La productivité capture la performance opérationnelle brute.
- Le temps d’attente mesure la fluidité du système.
- La consommation spécifique traduit l’efficacité énergétique.
- L’utilisation confirme que la flotte est mobilisée.
- Le coût par cycle relie l’efficacité énergétique à la productivité.
- La robustesse aux perturbations est critique pour les applications minières.

## 5. Notes d’implémentation

- Les KPI sont calculés dans `memoire_master_rl_logistique/simulation/kpi.py`.
- Les valeurs de sécurité `max(..., 1e-9)` garantissent l’absence de division par zéro.
- Les KPI sont exportés pour chaque épisode et utilisés dans les tableaux du chapitre 6.

---

*Ce document doit être utilisé comme référence pour la rédaction et l’interprétation des résultats.*
