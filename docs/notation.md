# Notation utilisée (chapitre 3)

Ce fichier recense la notation mathématique et symboles employés dans le chapitre 3.

- `s_t` : État du système à l'instant t (tuple d'informations, voir `rl_thesis.tex`).
- `q_p` : Longueur de la file d'attente à la pelle `p` (nombre de camions en attente).
- `x_c` : Localisation (nœud) du camion `c` et son statut (libre, en route, en charge, en décharge).
- `z_r` : État de disponibilité de la ressource `r` (disponible / maintenance / indisponible).
- `t_{courant}` : Temps courant ou étape de simulation.
- `a_t` : Action prise à l'instant t (affectations camion→pelle ou `ATTENDRE`).
- `\Delta t` : Pas de temps de décision / fenêtre de simulation entre deux décisions.
- `D_t` : Ensemble des camions déchargés dans l'intervalle `[t, t+\Delta t]`.
- `R_t` : Récompense globale au pas t (combinaison pondérée des composantes).
- `R_{rendement}` : Composante de récompense liée au tonnage livré pendant `\Delta t`.
- `R_{\text{équité}}` : Composante favorisant une distribution équilibrée des files (fonction `\phi`).
- `\phi(\cdot)` : Fonction d'équité (helper) — ex. variance conditionnelle sur le nombre de pelles.
- `\mathrm{Var}(\cdot)` : Variance d'un ensemble de valeurs.
- `f(s_t,a_t)` : Transition déterministe (partie déterministe du simulateur).
- `\eta_t` : Bruit/perturbation stochastique (ex. `\eta_t \sim \mathcal{N}(0,\Sigma)`).
- `C(\mathrm{pos}(c),\mathrm{pos}(p))` : Coût (distance/énergie/usure) entre la position du camion et une pelle.
- `w_1,w_2,w_3` : Poids scalaires de combinaison des composantes de récompense.

_Pour toute autre notation rencontrée dans le chapitre 3, ajoutez une entrée ici et je l'explique._