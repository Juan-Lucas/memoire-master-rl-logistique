# Classification des Articles - Revue de Littérature

Cette structure organise les articles selon les trois piliers de la thèse et l'ordre de lecture recommandé.

## 📚 Structure des Dossiers

### 1️⃣ État de l'Art et FMS (`1-etat-de-art-fms/`)
**Objectif** : Comprendre les systèmes de gestion de flotte existants (FMS), leurs logiques et leurs limites.

**Articles à placer ici** :
- ✅ **Afrapoli & Askari-Nasab (2017)** - *Mining fleet management systems: a review of models and algorithms* → `Mining fleet management systems a review of models and algorithms.pdf`
- ✅ **Alarie & Gamache (2002)** - *Overview of Solution Strategies Used in Truck Dispatching Systems for Open Pit Mines* → `Overview of Solution Strategies Used in Truck Dispatching Systems for Open Pit Mines.pdf`
- ✅ **Newman et al. (2010)** - *A Review of Operations Research in Mine Planning* → `A Review of Operations Research in Mine Planning.pdf`
- ✅ **Mohtasham et al. (2023)** - *Evaluating the performance of the DISPATCH algorithm* → `Evaluating the performance of the DISPATCH algorithm, a commercial software, in the Sungun copper mine.pdf`

**Ordre de lecture** : 
1. Afrapoli & Askari-Nasab (2017) - **Commencer par celui-ci**
2. Alarie & Gamache (2002)
3. Newman et al. (2010) - En diagonale pour le contexte global
4. Mohtasham et al. (2023) - Étude de cas récente

---

### 2️⃣ Méthodes Classiques - Baselines (`2-methodes-classiques-baselines/`)
**Objectif** : Maîtriser les approches d'optimisation classiques contre lesquelles l'agent RL sera comparé.

**Articles à placer ici** :
- ✅ **Munirathinam & Yingling (1994)** - *A review of computer-based truck dispatching strategies* → `A review of computer-based truck dispatching strategies for surface mining operations.pdf`
- ✅ **Subtil et al. (2011)** - *A Practical Approach to Truck Dispatch* → `A Practical Approach to Truck Dispatch for Open Pit Mines.pdf`
- ✅ **Souza et al. (2010)** - *A hybrid heuristic algorithm* → `A hybrid heuristic algorithm for the open-pit-mining operational planning problem.pdf`
- ✅ **Ahangaran et al. (2012)** - *Real-time dispatching modelling* → `Real Time Dispatching Modelling For Trucks With Different Capacities In Open Pit Mines.pdf`
- ✅ **Ta et al. (2013)** - *A linear model for surface mining* → `A linear model for surface mining haul truck allocation incorporating shovel idle probabilities.pdf`
- ✅ **Wang (2022)** - *Truck Dispatching Optimization Model* → `Truck Dispatching Optimization Model and Algorithm Based on 0-1 Decision Variables.pdf`

**Ordre de lecture** :
1. Munirathinam & Yingling (1994) - **Heuristiques de base**
2. Subtil et al. (2011) - Approche hybride LP + Simulation
3. Souza et al. (2010) - Métaheuristiques (GRASP, VNS)
4. Les autres en lecture rapide pour voir la diversité des approches

---

### 3️⃣ RL et Nouvelles Approches (`3-rl-nouvelles-approches/`)
**Objectif** : S'inspirer des approches par Reinforcement Learning et optimisation moderne.

**Articles à placer ici** :
- ✅ **Nazari et al. (2018)** - *Reinforcement Learning for Solving the Vehicle Routing Problem* → `Reinforcement Learning for Solving the Vehicle Routing Problem.pdf`
- ✅ **Zhang et al. (2020)** - *Learning to Dispatch for Job Shop Scheduling* → `Learning to Dispatch for Job Shop Scheduling via Deep Reinforcement Learning.pdf`
- ✅ **Liu & Chai (2019)** - *Optimizing Open-Pit Truck Route* → `Optimizing Open-Pit Truck Route Based on Minimization of Time-Varying Transport Energy Consumption.pdf`
- ✅ **Optimization-Based Dispatching Policies** → `Optimization-Based Dispatching Policies for Open-Pit Mining.pdf`

**Ordre de lecture** :
1. **Nazari et al. (2018) - À LIRE EN PROFONDEUR** (définition MDP, architecture réseau)
2. Liu & Chai (2019) - Pour la fonction de récompense (énergie/carburant)
3. Zhang et al. (2020) - Approche GNN (Graph Neural Network) - Avancé

---

### 4️⃣ Études de Cas et Compléments (`4-etudes-cas-complementaires/`)
**Objectif** : Collecter exemples concrets, résultats chiffrés et techniques complémentaires.

**Articles à placer ici** :
- ✅ **Ozdemir & Kumral (2019)** - *Simulation based optimization* → `Simulation based optimization of truck shovel material handling systems in multi pit surface mines.pdf`
- ✅ **Abolghasemian et al. (2020)** - *A Two-Phase Simulation-Based Optimization* → `A Two-Phase Simulation-Based Optimization of Hauling System in Open-Pit Mine.pdf`
- ✅ **Concurrent Simulation And Optimization** → `Concurrent Simulation And Optimization Models For Mining Planning.pdf`
- ✅ **Simulation and optimization approach** → `Simulation and optimization approach for uncertainty based short-term planning in open pit mines.pdf`
- ✅ **Open Pit Truck Shovel Haulage System Simulation** → `Open Pit Truck Shovel Haulage System Simulation.pdf`
- ✅ **A simulation model to study truck allocation options** → `A simulation model to study truckallocation options.pdf`
- ✅ **Truck dispatching in an open pit mine** → `Truck dispatching in an open pit mine.pdf`
- ✅ **Use of Machine Learning Algorithm Models** → `Use of Machine Learning Algorithm Models to Optimize the Fleet Management System in Opencast Mines.pdf`

**Ordre de lecture** : Lecture rapide/ciblée pour collecter :
- Modèles utilisés (LP, MILP, Simulation...)
- Résultats obtenus (ex: amélioration de X%)
- Limites identifiées par les auteurs

---

## 🎯 Ordre de Lecture Stratégique Global

### **Étape 1 : Big Picture (2-3 jours)**
Dossier `1-etat-de-art-fms/`
1. Afrapoli & Askari-Nasab (2017) ⭐ **PRIORITÉ 1**
2. Alarie & Gamache (2002)
3. Newman et al. (2010) - Survol

### **Étape 2 : Plonger dans les Méthodes (1 semaine)**
Dossiers `2-methodes-classiques-baselines/` + `3-rl-nouvelles-approches/`
1. Munirathinam & Yingling (1994) - Heuristiques baselines
2. **Nazari et al. (2018)** ⭐ **PRIORITÉ 2 - Cœur de ta contribution**
3. Liu & Chai (2019) - Fonction de récompense
4. Zhang et al. (2020) - Si temps disponible (GNN avancé)

### **Étape 3 : Compléter (lecture rapide)**
Dossiers `2-methodes-classiques-baselines/` + `4-etudes-cas-complementaires/`
- Lire les autres articles en se concentrant sur :
  - Résultats chiffrés
  - Limites identifiées
  - Métriques utilisées

---

## 📋 Commandes pour Organiser

```powershell
# Déplacer les articles dans leurs dossiers respectifs

# 1. État de l'Art et FMS
Move-Item "Mining fleet management systems a review of models and algorithms.pdf" "1-etat-de-art-fms/"
Move-Item "Overview of Solution Strategies Used in Truck Dispatching Systems for Open Pit Mines.pdf" "1-etat-de-art-fms/"
Move-Item "A Review of Operations Research in Mine Planning.pdf" "1-etat-de-art-fms/"
Move-Item "Evaluating the performance of the DISPATCH algorithm, a commercial software, in the Sungun copper mine.pdf" "1-etat-de-art-fms/"

# 2. Méthodes Classiques - Baselines
Move-Item "A review of computer-based truck dispatching strategies for surface mining operations.pdf" "2-methodes-classiques-baselines/"
Move-Item "A Practical Approach to Truck Dispatch for Open Pit Mines.pdf" "2-methodes-classiques-baselines/"
Move-Item "A hybrid heuristic algorithm for the open-pit-mining operational planning problem.pdf" "2-methodes-classiques-baselines/"
Move-Item "Real Time Dispatching Modelling For Trucks With Different Capacities In Open Pit Mines.pdf" "2-methodes-classiques-baselines/"
Move-Item "A linear model for surface mining haul truck allocation incorporating shovel idle probabilities.pdf" "2-methodes-classiques-baselines/"
Move-Item "Truck Dispatching Optimization Model and Algorithm Based on 0-1 Decision Variables.pdf" "2-methodes-classiques-baselines/"

# 3. RL et Nouvelles Approches
Move-Item "Reinforcement Learning for Solving the Vehicle Routing Problem.pdf" "3-rl-nouvelles-approches/"
Move-Item "Learning to Dispatch for Job Shop Scheduling via Deep Reinforcement Learning.pdf" "3-rl-nouvelles-approches/"
Move-Item "Optimizing Open-Pit Truck Route Based on Minimization of Time-Varying Transport Energy Consumption.pdf" "3-rl-nouvelles-approches/"
Move-Item "Optimization-Based Dispatching Policies for Open-Pit Mining.pdf" "3-rl-nouvelles-approches/"

# 4. Études de Cas et Compléments
Move-Item "Simulation based optimization of truck shovel material handling systems in multi pit surface mines.pdf" "4-etudes-cas-complementaires/"
Move-Item "A Two-Phase Simulation-Based Optimization of Hauling System in Open-Pit Mine.pdf" "4-etudes-cas-complementaires/"
Move-Item "Concurrent Simulation And Optimization Models For Mining Planning.pdf" "4-etudes-cas-complementaires/"
Move-Item "Simulation and optimization approach for uncertainty based short-term planning in open pit mines.pdf" "4-etudes-cas-complementaires/"
Move-Item "Open Pit Truck Shovel Haulage System Simulation.pdf" "4-etudes-cas-complementaires/"
Move-Item "A simulation model to study truckallocation options.pdf" "4-etudes-cas-complementaires/"
Move-Item "Truck dispatching in an open pit mine.pdf" "4-etudes-cas-complementaires/"
Move-Item "Use of Machine Learning Algorithm Models to Optimize the Fleet Management System in Opencast Mines.pdf" "4-etudes-cas-complementaires/"
```

---

## 📊 Tableau Récapitulatif par Priorité

| Priorité | Article | Catégorie | Pourquoi Crucial |
|----------|---------|-----------|------------------|
| ⭐⭐⭐ | Afrapoli & Askari-Nasab (2017) | État de l'art FMS | Review complet, détaille DISPATCH, pointe les limites |
| ⭐⭐⭐ | Nazari et al. (2018) | RL nouvelles approches | **Cœur méthodologique** - MDP pour VRP |
| ⭐⭐ | Alarie & Gamache (2002) | État de l'art FMS | Classique cité, structure single-stage vs multi-stage |
| ⭐⭐ | Munirathinam & Yingling (1994) | Baselines classiques | Heuristiques de base pour ta baseline |
| ⭐⭐ | Liu & Chai (2019) | RL nouvelles approches | Fonction de récompense (énergie/carburant) |
| ⭐ | Zhang et al. (2020) | RL nouvelles approches | GNN avancé (si temps) |
| ⭐ | Mohtasham et al. (2023) | État de l'art FMS | Étude de cas récente DISPATCH |

---

## ✅ Checklist de Lecture

### Phase 1 (Priorité Absolue - 1 semaine)
- [x] Afrapoli & Askari-Nasab (2017) - Annoté, fiché
- [x] Alarie & Gamache (2002) - Annoté, fiché
- [x] Nazari et al. (2018) - **Décortiqué en profondeur** (MDP, architecture, reward)

### Phase 2 (Important - 1 semaine)
- [x] Munirathinam & Yingling (1994) - Heuristiques notées
- [x] Liu & Chai (2019) - Formules de consommation extraites
- [x] Newman et al. (2010) - Contexte global noté

### Phase 3 (Compléments - Au fur et à mesure)
- [x] Autres articles lus rapidement (focus résultats + limites)
- [x] Tableau comparatif créé (approches, métriques, résultats)

---

**Note** : Commite régulièrement tes fiches de lecture dans le notebook `1.0-business-and-data-understanding.ipynb` avec des messages comme :
```bash
git commit -m "docs(understanding): Synthèse Afrapoli & Askari-Nasab 2017"
```
