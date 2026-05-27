# Fiches de compréhension du code

Ce document contient des fiches détaillées pour comprendre et maîtriser chaque fichier de code du projet.

---

## Fiche 1 : simulation/graph_model.py

**Rôle du fichier** : Modélise le réseau routier minier comme un graphe orienté pondéré (NetworkX).

**Dépendances** : NetworkX, dataclasses, random, math

**Fonctions/classes principales** :
- `RoadEdge` : Dataclass représentant une arête routière (src, dst, distance, pente, état)
- `RoadGraph` : Classe encapsulant nx.DiGraph du réseau routier
- `build_mine_graph()` : Fonction constructrice du graphe minier

---

### Classe : RoadGraph

**Variables critiques** :
- `_g` : nx.DiGraph interne (NetworkX)
- `_edge_cache` : dictionnaire {(src, dst): RoadEdge} pour accès rapide

**Méthodes clés** :

**add_node(node_id, node_type)**
- **Entrées** : identifiant nœud (str), type (str)
- **Sorties** : None
- **Rôle** : Ajoute un nœud au graphe (lève ValueError si doublon)
- **Hypothèse** : node_id unique
- **Cas limite** : Doublon → ValueError

**add_edge(src, dst, distance_km, slope_pct, road_state=1.0)**
- **Entrées** : source, destination, distance, pente, état route
- **Sorties** : None
- **Rôle** : Ajoute arête orientée + met en cache
- **Hypothèse** : src et dst existent déjà
- **Cas limite** : Nœuds inexistants → ValueError

**shortest_path(src, dst, weight="distance_km")**
- **Entrées** : source, destination, poids à minimiser
- **Sorties** : liste des nœuds du chemin
- **Rôle** : Algorithme Dijkstra (Section 3.2 mémoire)
- **Hypothèse** : graphe connexe
- **Cas limite** : Aucun chemin → liste vide

**sample_travel_time_minutes(src, dst, loaded, rng)**
- **Entrées** : source, destination, chargé?, générateur aléatoire
- **Sorties** : temps de trajet (minutes)
- **Rôle** : Échantillonne temps log-normal (Eq. 3.2)
- **Constantes** : base_speed_kmh = 26.0 (chargé) / 32.0 (vide), sigma = 0.12
- **Hypothèse** : Distribution log-normale réaliste
- **Cas limite** : Temps < 0.2 min → clamp à 0.2

**Ligne sensible** : Ligne 169 `slope_factor = max(0.55, 1.0 - max(0.0, edge.slope_pct) * 0.03)`
- **Pourquoi critique** : Ajuste vitesse selon pente, impact direct sur temps de trajet
- **Effet si modifiée** : Change la sensibilité à la pente, affecte les résultats

---

### Fonction : build_mine_graph(shovel_count=3, dump_count=2)

**Rôle** : Construit topologie réaliste yard → junction → pelles/dumps

**Topologie** :
- Yard central connecté à junction_1
- Junction connecté à toutes les pelles et dumps
- Arêtes bidirectionnelles
- Trajets directs pelle→dump et dump→pelle

**Paramètres réalistes** (basés sur Afrapoli 2019, Mohtasham 2023) :
- Distances : 0.8 km (yard-junction), 1.5-2.5 km (junction-pelles), 2.5-3.2 km (junction-dumps)
- Pentes : 1-8% (réalistes pour mines à ciel ouvert)

**Hypothèses** :
- Topologie étoile autour de junction
- Paramètres cycliques si plus de 5 pelles/dumps

**Validation personnelle** :
- [ ] Je peux expliquer la topologie sans regarder
- [ ] Je sais pourquoi les distances/pentes sont ces valeurs
- [ ] Je comprends le rôle de l'edge_cache

---

## Fiche 2 : simulation/entities.py

**Rôle du fichier** : Définit les entités physiques du système (camions, pelles, dumps).

**Dépendances** : dataclasses

**Correspondance MDP (Tableau 4.2)** :
- Truck → x_c (localisation, statut) — Eq. 3.3
- Shovel → q_p (file d'attente), z_r — Eq. 3.3
- DumpSite → z_r (disponibilité) — Eq. 3.3

---

### Classe : Truck

**Variables critiques** :
- `truck_id` : identifiant unique
- `capacity_tonnes` : capacité (défaut 140 t, Caterpillar 785C)
- `available_at_min` : moment où camion sera disponible
- `total_wait_min` : temps d'attente accumulé
- `total_active_min` : temps actif accumulé
- `cycles_completed` : nombre de cycles effectués
- `total_tonnage_t` : tonnage total transporté
- `total_fuel_l` : carburant total consommé
- `history` : liste d'historique (optionnel)

**Hypothèses** :
- Capacité fixe 140 t (standard industriel)
- KPIs accumulés sur tout l'épisode

**Cas limite** : capacité = 0 → division par zéro possible dans KPIs

---

### Classe : Shovel

**Variables critiques** :
- `shovel_id` : identifiant
- `node_id` : nœud du graphe où se trouve la pelle
- `load_time_mean_min` : temps moyen chargement (défaut 2.0 min)
- `load_time_std_min` : écart-type chargement (défaut 0.3 min)
- `available_at_min` : moment où pelle sera libre

**Hypothèses** :
- Distribution normale pour temps de chargement
- Paramètres basés sur Hitachi 2500 (spécifications industrielles)

**Ligne sensible** : Ligne 43-44 (paramètres stochastiques)
- **Pourquoi critique** : Définit la stochasticité du chargement
- **Effet si modifiée** : Change la variabilité, impacte les files d'attente

---

### Classe : DumpSite

**Variables critiques** :
- `dump_id` : identifiant
- `node_id` : nœud du graphe
- `unload_time_mean_min` : temps moyen déchargement (défaut 1.0 min)
- `unload_time_std_min` : écart-type déchargement (défaut 0.2 min)
- `available_at_min` : moment où dump sera libre

**Hypothèses** :
- Déchargement plus rapide que chargement (1.0 vs 2.0 min)
- Même stochasticité que pelles

**Validation personnelle** :
- [ ] Je peux expliquer chaque variable de Truck
- [ ] Je comprends la différence entre mean et std
- [ ] Je sais pourquoi les temps sont ces valeurs

---

## Fiche 3 : simulation/kpi.py

**Rôle du fichier** : Calcule les KPIs de performance (Tableau 4.8 mémoire).

**Dépendances** : Aucune (fonction pure)

---

### Fonction : compute_kpis(...)

**Entrées** :
- episode_minutes : durée épisode
- truck_count : nombre de camions
- total_tonnage_t : tonnage total
- total_wait_min : temps d'attente total
- total_active_min : temps actif total
- total_fuel_l : carburant total
- total_cycles : nombre de cycles (optionnel, défaut 0)
- total_distance_km : distance totale (optionnel, défaut 0.0)

**Sorties** : dictionnaire avec 11 KPIs

**KPIs calculés** :
1. **productivity_tph** = total_tonnage_t / episode_hours
2. **avg_wait_min_per_truck** = total_wait_min / truck_count
3. **utilization_pct** = (total_active_min / max_available_minutes) * 100
4. **specific_fuel_l_per_ton** = total_fuel_l / total_tonnage_t
5. **cost_per_cycle** = total_fuel_l / total_cycles

**Constantes de sécurité** :
- `max(episode_minutes / 60.0, 1e-9)` : évite division par zéro
- `max(total_tonnage_t, 1e-9)` : évite division par zéro

**Ligne sensible** : Ligne 24 `cost_per_cycle = total_fuel_l / max(float(total_cycles), 1.0)`
- **Pourquoi critique** : Division par zéro si cycles = 0
- **Effet si modifiée** : DivisionError si cycles = 0 sans max()

**Validation personnelle** :
- [ ] Je peux expliquer chaque KPI sans regarder
- [ ] Je comprends pourquoi les sécurités max(..., 1e-9)
- [ ] Je peux calculer un KPI à la main

---

## Fiche 4 : env/mine_env.py

**Rôle du fichier** : Environnement Gymnasium implémentant l'interface reset/step/render.

**Dépendances** : gymnasium, numpy, simulation (entities, graph_model, fuel_model), env (observation_builder)

**Correspondance mémoire (Section 4.3)** :
- Implémente interface Gymnasium standard
- Orchestre boucle Agent–Environnement
- Tableau 4.2 mapping classes → variables MDP

---

### Classe : MineEnv

**Variables critiques** :
- `truck_count`, `shovel_count`, `dump_count` : configuration
- `episode_minutes` : durée épisode (défaut 480 min = 8h)
- `breakdown_probability` : probabilité panne (défaut 0.02 = 2%)
- `reward_weights` : (w1, w2, w3) pour récompense multi-objectif
- `graph` : RoadGraph du réseau
- `trucks`, `shovels`, `dumps` : listes d'entités
- `current_time_min` : temps courant simulation
- `current_truck_idx` : index du camion actif
- `truck_locations`, `truck_statuses` : états des camions

**Constantes de statut** :
- STATUS_IDLE = 0
- STATUS_TO_SHOVEL = 1
- STATUS_LOADING = 2
- STATUS_TO_DUMP = 3
- STATUS_UNLOADING = 4
- STATUS_RETURNING = 5

---

### Méthode : reset(seed=None, options=None)

**Rôle** : Initialise la mine pour un nouvel épisode

**Étapes** :
1. Initialise générateur aléatoire avec seed
2. Construit graphe via build_mine_graph()
3. Crée pelles (Shovel) avec paramètres stochastiques
4. Crée dumps (DumpSite) avec paramètres stochastiques
5. Crée camions (Truck) avec capacité
6. Initialise temps courant à 0
7. Initialise tous camions au yard, statut IDLE
8. Retourne observation + info

**Hypothèses** :
- Seed fixé pour reproductibilité
- Paramètres par défaut réalistes

**Validation personnelle** :
- [ ] Je peux expliquer chaque étape du reset
- [ ] Je comprends pourquoi la seed est importante

---

### Méthode : step(action)

**Rôle** : Exécute un cycle complet pour le camion courant

**Décodage action** :
- Actions 0..(S*D-1) : shovel_idx = action // D, dump_idx = action % D
- Action S*D : ATTENDRE (pelle et dump les plus tôt disponibles)

**Cycle camion complet** :
1. **Trajet vers pelle** : _travel_route() → temps + carburant
2. **Attente pelle** : si pelle occupée → wait + carburant ralenti
3. **Chargement** : _sample_duration() → temps stochastique
4. **Trajet chargé vers dump** : _travel_route() → temps + carburant
5. **Attente dump** : si dump occupé → wait + carburant ralenti
6. **Déchargement** : _sample_duration() → temps stochastique
7. **Retour à vide** : _travel_route() → temps + carburant
8. **Panne éventuelle** : avec probabilité breakdown_probability

**Récompense** : _compute_reward(action) → Eq. 3.7

**Avancement temps** :
- current_time_min = min(available_at_min de tous camions)
- current_truck_idx = argmin(available_at_min)

**Terminaison** :
- terminated = False (jamais terminé prématurément)
- truncated = current_time_min >= episode_minutes

**Ligne sensible** : Ligne 254-259 (avancement temps)
- **Pourquoi critique** : Détermine quel camion est actif next
- **Effet si modifiée** : Change l'ordre de traitement, affecte résultats

**Validation personnelle** :
- [ ] Je peux expliquer le cycle complet
- [ ] Je comprends le décodage de l'action
- [ ] Je sais comment l'avancement temps fonctionne

---

### Méthode : _compute_reward(action)

**Rôle** : Calcule récompense multi-objectif (Eq. 3.7)

**Composantes** :
- **R_rendement** = tonnage livré pendant ce step
- **R_équité** = -variance des temps d'attente pelles
- **R_coût** = -distance parcourue vers pelle

**Normalisation** :
- R_rendement_norm = R_rendement / capacity_tonnes
- R_équité_norm = R_équité / 100.0
- R_coût_norm = R_coût / 5.0

**Formule finale** :
R_t = w1 * R_rendement_norm + w2 * R_équité_norm + w3 * R_coût_norm

**Ligne sensible** : Ligne 301-303 (normalisation)
- **Pourquoi critique** : Met à l'échelle les composantes
- **Effet si modifiée** : Change l'équilibre des composantes

**Validation personnelle** :
- [ ] Je peux expliquer chaque composante
- [ ] Je comprends pourquoi la normalisation est nécessaire
- [ ] Je peux calculer une récompense à la main

---

### Méthode : _decode_action(action)

**Rôle** : Décode action en paire (pelle, dump)

**Logique** :
- Si action >= S*D : ATTENDRE → pelle et dump les plus tôt disponibles
- Sinon : shovel_idx = action // D, dump_idx = action % D

**Hypothèse** : Action dans [0, S*D]

**Validation personnelle** :
- [ ] Je peux décoder une action sans regarder
- [ ] Je comprends pourquoi ATTENDRE est S*D

---

### Méthode : _travel_route(src, dst, loaded)

**Rôle** : Trajet multi-sauts via graphe → (temps, carburant)

**Étapes** :
1. Trouve chemin le plus court via Dijkstra
2. Pour chaque arête du chemin : appelle _travel()
3. Somme temps et carburant

**Hypothèse** : Chemin existe

**Validation personnelle** :
- [ ] Je comprends la différence _travel_route vs _travel
- [ ] Je sais pourquoi Dijkstra est utilisé

---

### Méthode : _travel(src, dst, loaded)

**Rôle** : Trajet direct avec stochasticité

**Étapes** :
1. Échantillonne temps via graph.sample_travel_time_minutes()
2. Calcule carburant via estimate_travel_fuel_l()
3. Avec probabilité 5% : dégrade road_state

**Stochasticité** : Temps log-normal (Eq. 3.2)

**Ligne sensible** : Ligne 371-372 (dégradation route)
- **Pourquoi critique** : Modélise usure route
- **Effet si modifiée** : Change la robustesse aux perturbations

**Validation personnelle** :
- [ ] Je comprends la stochasticité du temps
- [ ] Je sais pourquoi la route se dégrade

---

## Fiche 5 : env/observation_builder.py

**Rôle du fichier** : Construit vecteur d'observation normalisé pour l'agent RL.

**Dépendances** : numpy, simulation (entities, graph_model)

**Correspondance mémoire (Section 4.3.3)** :
- Encode l'état s_t du MDP (Eq. 3.3)
- Normalisation dans [0,1] pour stabiliser apprentissage

---

### Fonction : build_observation(...)

**Entrées** :
- trucks : liste des camions
- shovels : liste des pelles
- dumps : liste des dumps
- graph : graphe routier
- current_time_min : temps courant
- episode_minutes : durée épisode
- truck_locations : positions camions
- truck_statuses : statuts camions

**Sorties** : np.ndarray normalisé dans [0,1]

**Composantes de l'observation** :
1. **Files pelles** : wait_until / _MAX_TIME_MIN (1 feature par pelle)
2. **Disponibilité dumps** : wait_until / _MAX_TIME_MIN (1 feature par dump)
3. **Statut camions** : status / 5.0 (1 feature par camion)
4. **Disponibilité camions** : avail / _MAX_TIME_MIN (1 feature par camion)
5. **Tonnage camions** : tonnage / 2000.0 (1 feature par camion)
6. **Carburant camions** : fuel / 500.0 (1 feature par camion)
7. **Temps courant** : current_time / episode_minutes (1 feature)

**Constantes de normalisation** :
- _MAX_QUEUE = 10.0
- _MAX_DISTANCE_KM = 10.0
- _MAX_TIME_MIN = 480.0 (8 heures)

**Taille vecteur** : num_shovels + num_dumps + num_trucks * 4 + 1

**Ligne sensible** : Ligne 52 `obs_parts.append(min(wait_until / _MAX_TIME_MIN, 1.0))`
- **Pourquoi critique** : Clamp dans [0,1], évite valeurs >1
- **Effet si modifiée** : Valeurs >1 peuvent destabiliser apprentissage

**Validation personnelle** :
- [ ] Je peux expliquer chaque composante
- [ ] Je comprends pourquoi la normalisation est nécessaire
- [ ] Je peux calculer la taille du vecteur

---

### Fonction : observation_size(num_trucks, num_shovels, num_dumps)

**Rôle** : Retourne taille du vecteur d'observation

**Formule** : num_shovels + num_dumps + num_trucks * 4 + 1

**Validation personnelle** :
- [ ] Je peux calculer la taille sans regarder
- [ ] Je comprends pourquoi 4 features par camion

---

## Exercices de validation

### Exercice 1 : Compréhension graph_model.py
1. Dessiner la topologie du graphe sur papier
2. Calculer le temps de trajet yard→shovel_1 (chargé) à la main
3. Expliquer pourquoi slope_factor = max(0.55, ...)

### Exercice 2 : Compréhension mine_env.py
1. Expliquer un step complet de mine_env.step(action=4)
2. Décoder l'action 4 pour 3 pelles, 2 dumps
3. Calculer une récompense avec des valeurs arbitraires

### Exercice 3 : Compréhension observation_builder.py
1. Calculer la taille de l'observation pour 12 camions, 3 pelles, 2 dumps
2. Expliquer pourquoi le tonnage est divisé par 2000.0
3. Dessiner la structure du vecteur d'observation

---

## Checklist de maîtrise finale

- [ ] Je peux expliquer graph_model.py sans regarder
- [ ] Je peux expliquer entities.py sans regarder
- [ ] Je peux expliquer kpi.py sans regarder
- [ ] Je peux expliquer mine_env.py sans regarder
- [ ] Je peux expliquer observation_builder.py sans regarder
- [ ] Je peux tracer un cycle complet de step()
- [ ] Je peux calculer une récompense à la main
- [ ] Je peux décoder une action sans regarder
- [ ] Je comprends toutes les constantes de normalisation
- [ ] Je peux justifier chaque choix technique

---

**Score de maîtrise** : ____ / 10 cases cochées

**Objectif minimal** : 8 / 10
**Objectif idéal** : 10 / 10

---

## Fiche 6 : simulation/fuel_model.py

**Rôle du fichier** : Modélise la consommation de carburant pour les trajets et les temps d'attente.

**Dépendances** : Aucune (fonctions pures)

---

### Fonction : estimate_travel_fuel_l(distance_km, slope_pct, road_state, loaded)

**Entrées** :
- distance_km : distance du segment (km)
- slope_pct : pente (%)
- road_state : état de la route (1.0 = bon, <1.0 = dégradé)
- loaded : camion chargé? (bool)

**Sorties** : carburant consommé (litres)

**Formule** :
- base_l_per_km = 0.65 (chargé) / 0.45 (vide)
- slope_factor = 1.0 + slope_pct * 0.03 (plus de pente = plus de carburant)
- road_factor = 1.0 + (1.0 - road_state) * 0.5 (route dégradée = plus de carburant)
- fuel = distance_km * base_l_per_km * slope_factor * road_factor

**Ligne sensible** : Ligne 11 `road_factor = 1.0 + max(0.0, 1.0 - road_state) * 0.5`
- **Pourquoi critique** : Définit l'impact de l'état de la route
- **Effet si modifiée** : Change la sensibilité à la dégradation

**Validation personnelle** :
- [ ] Je peux expliquer chaque facteur
- [ ] Je comprends pourquoi chargé consomme plus que vide

---

### Fonction : estimate_idle_fuel_l(idle_minutes, idle_l_per_hour=10.0)

**Entrées** :
- idle_minutes : temps d'attente (minutes)
- idle_l_per_hour : consommation au ralenti (L/h, défaut 10.0)

**Sorties** : carburant consommé (litres)

**Formule** : idle_l_per_hour * (idle_minutes / 60.0)

**Hypothèse** : Consommation au ralenti constante 10 L/h (standard diesel)

**Validation personnelle** :
- [ ] Je peux expliquer la formule
- [ ] Je comprends pourquoi idle_l_per_hour = 10.0

---

## Fiche 7 : env/action_mask.py

**Rôle du fichier** : Masquage dynamique des actions invalides (Section 3.4.2 mémoire).

**Dépendances** : numpy, simulation (entities)

---

### Fonction : compute_action_mask(shovels, dumps, current_time_min, include_wait=True)

**Entrées** :
- shovels : liste des pelles
- dumps : liste des dumps
- current_time_min : temps courant
- include_wait : inclure action ATTENDRE?

**Sorties** : np.ndarray de masque (1 = valide, 0 = invalide)

**Conformité mémoire (Section 3.4.2)** :
- Espace d'action : A = Discrete(|P| × |D| + 1)
- Actions 0..(S×D-1) : paires (pelle, dump)
- Action S×D : ATTENDRE
- Toutes les paires sont considérées valides (agent peut attendre disponibilité)

**Implémentation** :
- Retourne un masque de uns (toutes actions valides)
- ATTENDRE toujours valide si include_wait=True

**Hypothèse** : L'agent apprend à attendre si nécessaire, pas besoin de masquer

**Validation personnelle** :
- [ ] Je comprends pourquoi toutes les actions sont valides
- [ ] Je peux expliquer la différence avec un masque strict

---

## Fiche 8 : baselines/fifo_policy.py

**Rôle du fichier** : Implémente la baseline FIFO (First In, First Out).

**Dépendances** : numpy, simulation (entities)

**Correspondance mémoire (Section 4.7.1)** : Baseline classique, camions affectés par ordre d'arrivée

---

### Classe : FIFOPolicy

**Variables critiques** :
- num_shovels : nombre de pelles
- num_dumps : nombre de dumps
- _counter : compteur pour round-robin

**Méthode predict(observation, info=None)**
- **Entrées** : observation (non utilisé), info (non utilisé)
- **Sorties** : action encodée
- **Logique** : round-robin cyclique sur pelles et dumps
- **Action** : shovel_idx = counter % num_shovels, dump_idx = counter % num_dumps
- **Hypothèse** : Ordre FIFO simple, sans considération d'état

**Ligne sensible** : Ligne 33-34 (round-robin)
- **Pourquoi critique** : Définit la logique FIFO
- **Effet si modifiée** : Change la distribution cyclique

**Validation personnelle** :
- [ ] Je peux expliquer la logique round-robin
- [ ] Je comprends pourquoi FIFO est simple mais myope

---

## Fiche 9 : baselines/shortest_path_policy.py

**Rôle du fichier** : Implémente la baseline Shortest Path (Chemin le plus court).

**Dépendances** : numpy, simulation (graph_model)

**Correspondance mémoire (Section 4.7.1)** : Baseline classique, minimise coût de trajet C_ij (Eq. 3.1)

---

### Classe : ShortestPathPolicy

**Variables critiques** :
- graph : RoadGraph du réseau
- shovel_node_ids : liste des nœuds de pelles
- dump_node_ids : liste des nœuds de dumps

**Méthode _estimate_cost(src, dst)**
- **Entrées** : source, destination
- **Sorties** : coût estimé
- **Logique** : utilise shortest_distance via Dijkstra comme proxy de coût Eq. 3.1
- **Hypothèse** : distance = proxy de coût total

**Méthode predict(observation, info=None, truck_location="yard")**
- **Entrées** : observation, info, position camion
- **Sorties** : action avec coût minimal
- **Logique** :
  - Pour chaque pelle : coût camion→pelle
  - Pour chaque dump : coût pelle→dump
  - Choisit paire avec coût total minimal
- **Hypothèse** : Optimisation locale sans vision globale

**Ligne sensible** : Ligne 66-68 (sélection meilleur coût)
- **Pourquoi critique** : Définit la décision greedy
- **Effet si modifiée** : Change la sélection de la paire optimale

**Validation personnelle** :
- [ ] Je peux expliquer pourquoi c'est une optimisation locale
- [ ] Je comprends la différence avec vision globale

---

## Fiche 10 : baselines/fixed_policy.py

**Rôle du fichier** : Implémente la baseline Fixed Assignment (Affectation fixe).

**Dépendances** : numpy

**Correspondance mémoire (Section 4.7.1)** : Baseline zéro-niveau, chaque camion assigné à une paire fixe cyclique

---

### Classe : FixedAssignmentPolicy

**Variables critiques** :
- num_shovels : nombre de pelles
- num_dumps : nombre de dumps

**Méthode predict(observation, info=None)**
- **Entrées** : observation, info
- **Sorties** : action encodée
- **Logique** :
  - truck_idx = info["current_truck_idx"] (0 par défaut)
  - shovel_idx = truck_idx % num_shovels
  - dump_idx = truck_idx % num_dumps
- **Hypothèse** : Assignation fixe cyclique, aucune adaptation
- **Cas limite** : info=None → truck_idx = 0

**Ligne sensible** : Ligne 31 (truck_idx depuis info)
- **Pourquoi critique** : Détermine quelle assignation fixe
- **Effet si modifiée** : Change le mapping camion→pelle

**Validation personnelle** :
- [ ] Je comprends pourquoi c'est une baseline zéro-niveau
- [ ] Je peux expliquer l'assignation cyclique

---

## Fiche 11 : baselines/nearest_policy.py

**Rôle du fichier** : Implémente la baseline Nearest Shovel (Pelle la plus proche).

**Dépendances** : numpy, simulation (graph_model)

**Correspondance mémoire (Section 4.7.1)** : Baseline classique, approche gloutonne géographique

---

### Classe : NearestShovelPolicy

**Variables critiques** :
- graph : RoadGraph du réseau
- shovel_node_ids : liste des nœuds de pelles
- dump_node_ids : liste des nœuds de dumps

**Méthode _estimate_distance(src, dst)**
- **Entrées** : source, destination
- **Sorties** : distance
- **Logique** : shortest_distance via Dijkstra

**Méthode predict(observation, info=None, truck_location="yard")**
- **Entrées** : observation, info, position camion
- **Sorties** : action
- **Logique** :
  1. Trouve pelle la plus proche du camion
  2. Trouve dump le plus proche de cette pelle
  3. Retourne action encodée
- **Hypothèse** : Distance géographique = proxy de bon choix
- **Limite** : Ignore files d'attente (myope)

**Ligne sensible** : Ligne 46-50 (sélection pelle la plus proche)
- **Pourquoi critique** : Définit la logique greedy
- **Effet si modifiée** : Change la sélection de la pelle

**Validation personnelle** :
- [ ] Je comprends pourquoi c'est glouton
- [ ] Je peux expliquer la limitation (ignore files)

---

## Fiche 12 : rl/train_ppo.py

**Rôle du fichier** : Entraînement de l'agent PPO (Section 4.5.4 mémoire).

**Dépendances** : Stable-Baselines3, env (mine_env), rl (callbacks)

**Correspondance mémoire (Section 4.5.4, Tableau 4.6)** : Algorithme PPO avec hyperparamètres

---

### Fonction : create_env(...)

**Rôle** : Crée instance MineEnv avec paramètres

**Paramètres** : truck_count, shovel_count, dump_count, episode_minutes, seed, reward_weights

**Validation personnelle** :
- [ ] Je peux créer un environnement avec différents paramètres

---

### Fonction : train_ppo(total_timesteps=50000, ...)

**Rôle** : Entraîne agent PPO sur environnement minier

**Hyperparamètres (Tableau 4.6)** :
- learning_rate = 3e-4
- gamma = 0.99
- batch_size = 64
- gae_lambda = 0.95
- clip_range = 0.2
- n_steps = 2048
- n_epochs = 10
- ent_coef = 0.01
- policy_kwargs = {"net_arch": [128, 128]} (MLP 128×128 ReLU)

**Étapes** :
1. Crée environnement vectorisé (make_vec_env)
2. Initialise modèle PPO avec hyperparamètres
3. Crée callback KPILoggerCallback
4. Entraîne avec model.learn()
5. Sauvegarde modèle

**Ligne sensible** : Ligne 84-92 (hyperparamètres PPO)
- **Pourquoi critique** : Définit l'architecture et l'entraînement
- **Effet si modifiée** : Change la convergence et la performance

**Validation personnelle** :
- [ ] Je peux expliquer chaque hyperparamètre
- [ ] Je comprends pourquoi MLP 128×128

---

## Fiche 13 : rl/train_dqn.py

**Rôle du fichier** : Entraînement DQN (Section 4.5.2 mémoire).

**Dépendances** : Stable-Baselines3, env (mine_env), rl (callbacks)

**Correspondance mémoire (Section 4.5.2)** : Extension Deep RL de Q-Learning

---

### Fonction : train_dqn(total_timesteps=50000, ...)

**Rôle** : Entraîne agent DQN sur environnement minier

**Hyperparamètres** :
- learning_rate = 3e-4
- gamma = 0.99
- batch_size = 64
- buffer_size = 50_000
- learning_starts = 1000
- target_update_interval = 500
- exploration_fraction = 0.3
- exploration_initial_eps = 1.0
- exploration_final_eps = 0.05
- policy_kwargs = {"net_arch": [128, 128]}

**Différences avec PPO** :
- Utilise replay buffer (50_000 transitions)
- Utilise réseau cible (target_update_interval=500)
- Exploration ε-greedy (1.0→0.05)

**Ligne sensible** : Ligne 81-97 (hyperparamètres DQN)
- **Pourquoi critique** : Définit stabilité DQN
- **Effet si modifiée** : Peut causer instabilité sans replay buffer

**Validation personnelle** :
- [ ] Je peux expliquer replay buffer
- [ ] Je comprends la différence avec PPO

---

## Fiche 14 : rl/train_q_learning.py

**Rôle du fichier** : Entraînement Q-Learning tabulaire (Section 4.4.1 mémoire).

**Dépendances** : numpy, env (mine_env)

**Correspondance mémoire (Section 4.4.1)** : RL tabulaire off-policy

---

### Fonction : _discretize_obs(obs, n_bins=5)

**Rôle** : Convertit observation continue [0,1] en tuple d'indices discrets

**Logique** :
- Clip obs dans [0, 1]
- Multiplie par n_bins
- Convertit en entier
- Retourne tuple

**Hypothèse** : n_bins=5 divisions égales par dimension

**Validation personnelle** :
- [ ] Je peux discrétiser une observation à la main

---

### Fonction : train_q_learning(n_episodes=10000, alpha=0.1, gamma=0.99, ...)

**Rôle** : Entraîne agent Q-Learning tabulaire

**Hyperparamètres (Tableau 4.4)** :
- alpha = 0.1 (taux d'apprentissage)
- gamma = 0.99 (facteur d'actualisation)
- epsilon_start = 1.0, epsilon_min = 0.01 (exploration)
- n_bins = 5 (discrétisation)

**Boucle d'entraînement** :
1. Reset environnement
2. Pour chaque step :
   - Sélection action ε-greedy
   - Exécute step
   - Met à jour Q-table (off-policy) : Q[s,a] += α[r + γ max Q[s',:] - Q[s,a]]
3. Décroissance ε linéairement

**Ligne sensible** : Ligne 96-99 (mise à jour Q-Learning)
- **Pourquoi critique** : C'est l'équation fondamentale (Eq. 4.3)
- **Effet si modifiée** : Change la convergence

**Validation personnelle** :
- [ ] Je peux écrire l'équation Q-Learning de mémoire
- [ ] Je comprends off-policy vs on-policy

---

### Classe : QLearningPolicy

**Rôle** : Politique gloutonne basée sur Q-table entraînée

**Méthode predict(observation)**
- **Logique** : discrétise observation, retourne argmax Q[state]
- **Cas limite** : state pas dans Q-table → retourne 0

**Validation personnelle** :
- [ ] Je peux charger une Q-table et l'utiliser

---

## Fiche 15 : rl/train_sarsa.py

**Rôle du fichier** : Entraînement SARSA tabulaire (Section 4.4.3 mémoire).

**Dépendances** : numpy, env (mine_env), rl (train_q_learning)

**Correspondance mémoire (Section 4.4.3)** : RL tabulaire on-policy

---

### Fonction : train_sarsa(n_episodes=10000, alpha=0.1, gamma=0.99, ...)

**Rôle** : Entraîne agent SARSA tabulaire

**Hyperparamètres (Tableau 4.5)** :
- alpha = 0.1, gamma = 0.99
- epsilon_start = 1.0, epsilon_min = 0.01
- n_bins = 5

**Différence clé avec Q-Learning** :
- Choix next_action AVANT mise à jour (on-policy)
- Mise à jour : Q[s,a] += α[r + γ Q[s',a'] - Q[s,a]]
- Utilise action effectivement prise, pas max

**Ligne sensible** : Ligne 87-96 (choix next_action avant update)
- **Pourquoi critique** : C'est ce qui distingue SARSA (on-policy)
- **Effet si modifiée** : Deviendrait Q-Learning (off-policy)

**Validation personnelle** :
- [ ] Je peux expliquer la différence on-policy vs off-policy
- [ ] Je peux écrire l'équation SARSA de mémoire

---

### Classe : SarsaPolicy

**Rôle** : Politique gloutonne basée sur table SARSA entraînée

**Méthode predict(observation)**
- **Logique** : discrétise observation, retourne argmax Q[state]

**Validation personnelle** :
- [ ] Je peux charger une table SARSA et l'utiliser

---

## Fiche 16 : rl/evaluate_agent.py

**Rôle du fichier** : Évaluation des agents entraînés et baselines (Section 4.7 mémoire).

**Dépendances** : Stable-Baselines3, baselines (4 politiques), env (mine_env), simulation (kpi)

---

### Fonction : evaluate_policy(env, policy_fn, n_episodes=10, policy_name="policy")

**Rôle** : Évalue une politique sur n épisodes

**Étapes** :
1. Pour chaque épisode (seed différent) :
   - Reset environnement
   - Boucle step jusqu'à terminated/truncated
   - Calcule KPIs via compute_kpis()
2. Calcule moyenne et écart-type des KPIs

**Sorties** : dictionnaire avec KPIs moyens + std

**Validation personnelle** :
- [ ] Je peux expliquer pourquoi n_episodes=10
- [ ] Je comprends le rôle des seeds

---

### Fonction : evaluate_ppo(model_path, n_episodes=10, ...)

**Rôle** : Évalue agent PPO entraîné

**Étapes** :
1. Charge modèle PPO depuis model_path
2. Définit policy_fn qui utilise model.predict(obs, deterministic=True)
3. Appelle evaluate_policy()

**Validation personnelle** :
- [ ] Je peux charger un modèle PPO et l'évaluer

---

### Fonction : evaluate_all_baselines(n_episodes=10, ...)

**Rôle** : Évalue toutes les baselines (Section 4.7.1)

**Baselines évaluées** :
- FIFO
- Fixed Assignment
- Nearest Shovel
- Shortest Path

**Pour chaque baseline** :
1. Crée environnement
2. Initialise politique
3. Appelle evaluate_policy()

**Validation personnelle** :
- [ ] Je peux expliquer la différence entre chaque baseline
- [ ] Je comprends pourquoi 4 baselines

---

## Fiche 17 : rl/callbacks.py

**Rôle du fichier** : Callbacks personnalisés pour enregistrement KPIs (Section 4.6 mémoire).

**Dépendances** : Stable-Baselines3 (BaseCallback)

---

### Classe : KPILoggerCallback(BaseCallback)

**Rôle** : Enregistre KPIs dans CSV pendant entraînement

**Variables** :
- log_path : chemin fichier CSV
- log_freq : fréquence d'enregistrement (défaut 1000 steps)
- _episode_tonnages : liste tonnages épisodes
- _episode_waits : liste temps d'attente
- _episode_fuels : liste carburant

**Méthode _on_step()**
- **Logique** :
  - À chaque step terminé (done=True) : enregistre KPIs depuis info
  - Tous les log_freq steps : calcule moyennes, enregistre dans logger
- **KPIs enregistrés** :
  - mine/avg_tonnage_t
  - mine/avg_wait_min
  - mine/avg_fuel_l
  - mine/episodes_logged

**Méthode _on_training_end()**
- **Rôle** : Sauvegarde CSV final avec toutes les données

**Ligne sensible** : Ligne 42 (log_freq)
- **Pourquoi critique** : Détermine fréquence d'enregistrement
- **Effet si modifiée** : Trop fréquent = slowdown, trop rare = manque de données

**Validation personnelle** :
- [ ] Je comprends pourquoi utiliser un callback
- [ ] Je peux expliquer le rôle de BaseCallback

---

## Fiche 18 : experiments/scenarios.py

**Rôle du fichier** : Définit scénarios expérimentaux (Section 5.4 mémoire).

**Dépendances** : dataclasses

---

### Classe : Scenario

**Rôle** : Configuration d'un scénario expérimental

**Variables** :
- name : identifiant scénario
- description : description textuelle
- truck_count : nombre de camions (défaut 12)
- shovel_count : nombre de pelles (défaut 3)
- dump_count : nombre de dumps (défaut 2)
- episode_minutes : durée épisode (défaut 480 = 8h)
- breakdown_probability : probabilité panne (défaut 0.02 = 2%)
- reward_weights : (w1, w2, w3) = (1.0, 0.1, 0.05)
- seeds : liste seeds (défaut 42..51 = 10 seeds)
- total_timesteps : steps entraînement (défaut 50_000)

---

### Dictionnaire SCENARIOS

**Scénarios définis** :
1. **nominal** : 12 camions, 3 pelles, 2 dumps, breakdown=2%
2. **high_load** : 18 camions (surcharge)
3. **low_load** : 6 camions
4. **high_breakdown** : breakdown=10% (robustesse)
5. **single_shovel** : 1 pelle, 1 dump (goulot)
6. **short_shift** : 4 heures au lieu de 8

**Justification valeurs** :
- Basé sur littérature (Kangwa 2021, Mohtasham 2023)
- 10 réplications (seeds 42..51) pour statistiques

**Validation personnelle** :
- [ ] Je peux expliquer chaque scénario
- [ ] Je comprends pourquoi 10 seeds

---

## Fiche 19 : experiments/run_benchmark.py

**Rôle du fichier** : Benchmark comparatif complet (Section 5.4 mémoire).

**Dépendances** : csv, Stable-Baselines3, baselines, env, experiments (scenarios), rl (tous train), simulation (kpi)

---

### Fonction : run_single_episode(env, policy_fn, seed)

**Rôle** : Lance un épisode et retourne KPIs

**Étapes** :
1. Reset environnement avec seed
2. Boucle step jusqu'à terminated/truncated
3. Calcule KPIs via compute_kpis()
4. Retourne dictionnaire KPIs

**Validation personnelle** :
- [ ] Je peux lancer un épisode manuellement

---

### Fonction : run_benchmark(scenario, output_dir, train_ppo_flag=True)

**Rôle** : Exécute benchmark pour un scénario

**Méthodes comparées** :
- Heuristiques : FIFO, Fixed, Nearest, ShortestPath
- RL classiques : Q-Learning, SARSA
- Deep RL : DQN, PPO (si train_ppo_flag=True)

**Pour chaque méthode** :
1. Charge ou entraîne modèle
2. Définit policy_fn
3. Pour chaque seed du scénario :
   - Lance run_single_episode()
   - Enregistre résultats

**Ligne sensible** : Ligne 223-240 (boucle policies × seeds)
- **Pourquoi critique** : C'est le cœur du benchmark
- **Effet si modifiée** : Change la comparaison

**Validation personnelle** :
- [ ] Je peux expliquer pourquoi 8 méthodes
- [ ] Je comprends le protocole (mêmes scénarios, mêmes seeds)

---

### Fonction : run_all_benchmarks(output_dir, train_ppo_flag=True, scenario_names=None)

**Rôle** : Lance benchmarks sur tous les scénarios

**Étapes** :
1. Crée dossier output_dir
2. Filtre scénarios si scenario_names fourni
3. Pour chaque scénario : appelle run_benchmark()
4. Sauvegarde CSV avec tous les résultats

**Validation personnelle** :
- [ ] Je peux lancer un benchmark complet

---

## Fiche 20 : main.py

**Rôle du fichier** : Point d'entrée principal du système (CLI).

**Dépendances** : argparse, experiments (run_benchmark), rl (tous train)

---

### Fonction : run_check_env(truck_count=12, shovel_count=3, dump_count=2)

**Rôle** : Valide environnement Gymnasium avec check_env

**Étapes** :
1. Crée MineEnv
2. Appelle check_env (Gymnasium)
3. Affiche observation_space et action_space

**Validation personnelle** :
- [ ] Je peux valider mon environnement

---

### Fonction : main()

**Rôle** : Point d'entrée CLI avec argparse

**Arguments** :
- --train-only : entraînement PPO uniquement
- --train-q-learning : entraînement Q-Learning
- --train-sarsa : entraînement SARSA
- --train-dqn : entraînement DQN
- --episodes : nombre épisodes Q-Learning/SARSA (défaut 10000)
- --benchmark-only : benchmark uniquement
- --check-env : validation environnement
- --truck-count, --shovel-count, --dump-count : configuration
- --timesteps : steps PPO (défaut 50000)
- --visualize-graph : générer figure graphe

**Pipeline complet** (si aucun argument) :
1. Validation environnement (check_env)
2. Entraînement PPO
3. Benchmark comparatif

**Ligne sensible** : Ligne 212-242 (pipeline complet)
- **Pourquoi critique** : Définit le workflow standard
- **Effet si modifiée** : Change l'ordre des étapes

**Validation personnelle** :
- [ ] Je peux utiliser la CLI
- [ ] Je comprends le pipeline complet

---

## Checklist finale de maîtrise complète

### Module simulation
- [ ] graph_model.py maîtrisé
- [ ] entities.py maîtrisé
- [ ] kpi.py maîtrisé
- [ ] fuel_model.py maîtrisé

### Module env
- [ ] mine_env.py maîtrisé
- [ ] observation_builder.py maîtrisé
- [ ] action_mask.py maîtrisé

### Module baselines
- [ ] fifo_policy.py maîtrisé
- [ ] shortest_path_policy.py maîtrisé
- [ ] fixed_policy.py maîtrisé
- [ ] nearest_policy.py maîtrisé

### Module rl
- [ ] train_ppo.py maîtrisé
- [ ] train_dqn.py maîtrisé
- [ ] train_q_learning.py maîtrisé
- [ ] train_sarsa.py maîtrisé
- [ ] evaluate_agent.py maîtrisé
- [ ] callbacks.py maîtrisé

### Module experiments
- [ ] scenarios.py maîtrisé
- [ ] run_benchmark.py maîtrisé

### Point d'entrée
- [ ] main.py maîtrisé

---

**Score de maîtrise complète** : ____ / 20 cases cochées

**Objectif minimal** : 18 / 20
**Objectif idéal** : 20 / 20
