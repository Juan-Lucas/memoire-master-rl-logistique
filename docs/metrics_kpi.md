# Formalisation mathématique des KPIs

Conforme au Tableau 4.8 du mémoire (Section 4.7.4).

## 1. Productivité et Performance

- **Tonnage transporté par heure** :
	$$\text{Productivité} = \frac{\sum_{i=1}^{N} T_i}{H}$$
	où $T_i$ est le tonnage transporté par le camion $i$ sur la période $H$ (heures).
- **Nombre de cycles camion-pelle** :
	$$\text{Cycles} = \sum_{i=1}^{N} C_i$$
	où $C_i$ est le nombre de cycles réalisés par le camion $i$.
- **Production supplémentaire** :
	$$\Delta P = P_{opt} - P_{ref}$$
	où $P_{opt}$ est la production obtenue avec l’optimisation, $P_{ref}$ la baseline.

## 2. Utilisation des équipements
- **Taux d’utilisation des camions** :
	$$\text{Utilisation} = \frac{\text{Temps actif}}{\text{Temps total}} \times 100$$
- **Match Factor (MF)** :
	$$\text{MF} = \frac{\text{Nombre de camions}}{\text{Nombre de pelles}}$$

## 3. Temps d’attente et fluidité
- **Temps moyen d’attente en zone tampon** :
	$$\text{Temps d’attente} = \frac{1}{N} \sum_{i=1}^{N} w_i$$
	où $w_i$ est le temps d’attente du camion $i$.
- **Nombre d’événements de congestion** :
	$$\text{Congestions} = \sum_{t=1}^{T} \mathbb{1}_{\text{congestion}}(t)$$
	où $\mathbb{1}_{\text{congestion}}(t)$ vaut 1 si congestion à l’instant $t$.

## 4. Consommation de carburant et coûts
- **Litres consommés par tonne** :
	$$\text{Consommation spécifique} = \frac{\text{Litres consommés}}{\text{Tonnage transporté}}$$
- **Coût opérationnel par tonne** :
	$$\text{Coût} = \frac{\text{Coût total}}{\text{Tonnage transporté}}$$
- **Coût moyen par cycle** :
	$$\text{Coût par cycle} = \frac{\text{Total carburant}}{\text{Nombre de cycles}}$$

## 5. Qualité de service et contraintes
- **Respect des contraintes** :
	$$\text{Respect} = \frac{\text{Nombre de cycles conformes}}{\text{Nombre total de cycles}} \times 100$$
- **Variabilité des temps de cycle** :
	$$\text{Variance} = \frac{1}{N} \sum_{i=1}^{N} (c_i - \bar{c})^2$$
	où $c_i$ est le temps de cycle du camion $i$, $\bar{c}$ la moyenne.

## 6. Adaptation et robustesse
- **Réactivité** :
	$$\text{Réactivité} = \frac{\text{Nombre de décisions adaptatives}}{\text{Nombre total de décisions}}$$

## 7. Sécurité
- **Incidents** :
	$$\text{Incidents} = \sum_{i=1}^{N} \mathbb{1}_{\text{incident}}(i)$$

---
## Justification du choix du "temps d’attente" comme KPI différenciant

Le temps d’attente est un KPI central et différenciant pour la logistique minière car il reflète directement l’efficacité du dispatching, la fluidité des opérations et l’adaptation aux perturbations. Un temps d’attente élevé indique une mauvaise synchronisation entre camions et pelles, des congestions ou une sous-utilisation des équipements. Les approches classiques (heuristiques) peinent à minimiser ce KPI, tandis que les méthodes avancées (RL, optimisation dynamique) visent explicitement à le réduire, ce qui se traduit par une amélioration globale de la productivité, une réduction des coûts et une meilleure robustesse du système. C’est donc un indicateur clé pour comparer les performances des différentes stratégies d’optimisation.
# Métriques de Succès (KPIs) – Synthèse de l’état de l’art

## 1. Productivité et Performance
- Tonnage transporté par heure ou par shift ([Afrapoli & Askari-Nasab 2017], [Ozdemir & Kumral 2019])
- Nombre de cycles camion-pelle
- Production supplémentaire obtenue (t/shift)
- Respect des objectifs de production

## 2. Utilisation des équipements
- Taux d’utilisation des camions (%) ([Munirathinam & Yingling 1994], [Ozdemir & Kumral 2019])
- Taux d’utilisation des pelles (%)
- Match Factor (MF) : équilibre entre flotte et pelles ([Ozdemir & Kumral 2019])

## 3. Temps d’attente et fluidité
- Temps moyen d’attente en zone tampon
- Temps moyen d’attente à la pelle ([Ozdemir & Kumral 2019])
- Nombre d’événements de congestion
- Temps perdu dû aux congestions

## 4. Consommation de carburant et coûts
- Litres consommés par tonne transportée ([Afrapoli & Askari-Nasab 2017])
- Coût opérationnel par tonne
- Coût total du transport minier

## 5. Qualité de service et contraintes
- Respect des contraintes de production et de qualité ([Munirathinam & Yingling 1994])
- Variabilité des temps de cycle
- Respect des ratios de décapage et de teneur

## 6. Adaptation et robustesse
- Réactivité face aux perturbations (routes, météo, pannes)
- Nombre de décisions adaptatives prises ([Nazari et al. 2018])
- Performance de l’agent RL dans des scénarios perturbés

## 7. Sécurité
- Nombre d’incidents ou quasi-incidents liés au dispatching

---
**Origine des KPIs** :
- Afrapoli & Askari-Nasab (2017) : revue des modèles et algorithmes FMS, importance de la productivité, des coûts et de l’adaptation.
- Munirathinam & Yingling (1994) : classification des heuristiques, métriques de productivité, utilisation, respect des contraintes.
- Nazari et al. (2018) : KPIs pour RL appliqué au VRP, adaptation dynamique, performance en généralisation.
- Ozdemir & Kumral (2019) : simulation et optimisation, match factor, temps d’attente, productivité.

Ces KPIs sont recommandés pour évaluer quantitativement les solutions d’optimisation (heuristiques, RL) dans la logistique minière, en s’appuyant sur les standards de la littérature.