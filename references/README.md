# Références de l'article et du mémoire

Ce dossier contient les PDF des travaux cités et leurs fiches de lecture (un carnet `fiches-lecture.ipynb` par dossier, une cellule par article).

- **Les PDF ne sont pas versionnés** (`references/**/*.pdf` dans `.gitignore`) : ils restent sur la machine et ne sont pas publiés sur GitHub. Les fiches, elles, sont versionnées.
- **Mise à jour du 22/10/2026** :
  - inventaire des 50 références citées dans l'article v3 (`reports/article/RL_article_FR_v3.tex`) ;
  - téléchargement des PDF librement accessibles ;
  - nouvelles fiches et fiches réécrites.

  Chaque fiche ajoutée ce jour-là dit si elle repose sur le **texte intégral** ou sur le **résumé seulement**. Ce sont des lectures assistées, à relire.

## Organisation

| Dossier | Contenu | Fiches |
|---|---|---|
| `1-etat-de-art-fms/` | Systèmes de gestion de flotte, revues, DISPATCH | 6 |
| `2-methodes-classiques-baselines/` | Heuristiques et optimisation (références de comparaison) | 6 |
| `3-rl-nouvelles-approches/` | RL appliqué au dispatching et aux problèmes voisins | 9 |
| `4-etudes-cas-complementaires/` | Simulation et optimisation en mine, études de cas | 8 |
| `5-fondements-rl/` (nouveau) | Algorithmes, processus semi-markoviens, bibliothèques | 15 |
| `6-evaluation-statistique/` (nouveau) | Tests, bootstrap, reproductibilité | 7 |

## Inventaire des références citées dans l'article v3

Légende :
- **PDF** : présent dans ce dossier ;
- **Fiche** : *existante* = fiche déjà présente avant le 22/10/2026, non modifiée ; *texte* = fiche fondée sur le texte intégral ; *résumé* = fondée sur le résumé seulement ; *non lu* = ouvrage ou article payant, la fiche dit seulement à quoi il sert.

### Domaine : gestion de flotte et dispatching minier

| Clé `.bib` | Dossier | PDF | Fiche |
|---|---|---|---|
| Afrapoli2017 | 1 | ✅ | existante |
| Alarie2002 | 1 | ✅ | existante |
| AlexandraNewman2010 | 1 | ✅ | existante |
| MehrnazMohtasham2023 | 1 | ✅ | existante |
| Hazrathosseini2024 | 1 | ❌ (SAGE bloque le téléchargement automatique ; essayer dans un navigateur) | résumé |
| BurtCaccetta2014 | 1 | ❌ (accès fermé) | résumé |
| MohanMunirathinam1994 | 2 | ✅ | existante |
| Subtil2011 | 2 | ✅ | existante |
| Souza2010 | 2 | ✅ | existante |
| Ahangaran2012 | 2 | ✅ | existante |
| ChungTa2013 | 2 | ✅ | existante |
| ZhongxinWang2022 | 2 | ✅ | existante |
| MohammadrezaNazari2018 | 3 | ✅ | existante |
| CongZhang2020 | 3 | ✅ | existante |
| GuangweiLiu2019 | 3 | ✅ | existante |
| AmandaSmith2020 | 3 | ✅ | existante |
| deCarvalho2023 | 3 | ✅ (nouveau) | texte (réécrite) |
| Zhao2025 | 3 | ❌ (accès fermé) | résumé |
| Pei2025 | 3 | ❌ (accès fermé) | résumé |
| FranciscoRosales2025 | 3 | ✅ (nouveau) | texte (réécrite) |
| Banerjee2025 | 3 | ✅ (nouveau) | texte (réécrite) |
| BurakOzdemir2019 | 4 | ✅ | existante |
| MiladAbolghasemian2020 | 4 | ✅ | existante |
| MarceloFioroni2008 | 4 | ✅ | existante |
| ShivPrakashUpadhyay2017 | 4 | ✅ | existante |
| NecmettinCetin2004 | 4 | ✅ | existante |
| Zeng2022 | 4 | ✅ | existante |
| Soumis1989 | 4 | ✅ | existante |

### Fondements de l'apprentissage par renforcement et outils (dossier 5)

| Clé `.bib` | PDF | Fiche |
|---|---|---|
| RichardSutton2018 | ✅ (version en ligne des auteurs) | texte |
| WatkinsDayan1992 | ✅ | texte |
| RummeryNiranjan1994 | ✅ | texte |
| Mnih2015 | ✅ | texte |
| JohnSchulman2017 | ✅ | texte |
| Schulman2016GAE | ✅ | texte |
| Haarnoja2018 | ✅ | texte |
| BradtkeDuff1995 | ✅ | texte |
| SuttonPrecupSingh1999 | ✅ | texte |
| AntoninRaffin2021 | ✅ | texte |
| MarkTowers2025 | ✅ | texte |
| Puterman1994 | ❌ (ouvrage payant) | non lu |
| MoezKrichen2022 | ❌ (HAL bloque le téléchargement automatique ; libre dans un navigateur) | résumé |
| KingaSzatmari2025 | ❌ (accès libre, mais ScienceDirect bloque le téléchargement automatique) | résumé |
| SampsonNwachukwu2025 | ❌ (accès libre, mais IEEE bloque le téléchargement automatique) | résumé |

### Évaluation statistique (dossier 6)

| Clé `.bib` | PDF | Fiche |
|---|---|---|
| Henderson2018 | ✅ | texte |
| Agarwal2021 | ✅ | texte |
| Demsar2006 | ✅ | texte |
| Wilcoxon1945 | ❌ (JSTOR, payant) | non lu |
| Friedman1937 | ❌ (payant) | non lu |
| Holm1979 | ❌ (JSTOR, payant) | non lu |
| EfronTibshirani1993 | ❌ (ouvrage payant) | non lu |

**Bilan** : 50 références citées, 38 PDF présents, 50 fiches (21 existantes ; 17 nouvelles ou réécrites sur le texte intégral ; 7 sur le résumé ; 5 « non lu »).

## PDF à récupérer à la main

**Accès libre** (dans un navigateur, le site bloque seulement les téléchargements automatiques) :
- Szatmári et al. (2025) : https://doi.org/10.1016/j.ceja.2025.100753
- Nwachukwu et al. (2025) : https://doi.org/10.1109/ACCESS.2025.3561807
- Krichen (2022) : https://hal.science/hal-04139754
- Hazrathosseini & Moradi Afrapoli (2024), accès hybride, probablement libre : https://doi.org/10.1177/25726668231222998

**Accès payant**, par la bibliothèque ou en écrivant aux auteurs :
- Zhao et al. (2025) : https://doi.org/10.1080/23249935.2025.2522360
- Pei et al. (2025) : https://doi.org/10.1080/17480930.2025.2486316
- Burt & Caccetta (2014) : https://doi.org/10.1287/inte.2013.0732
- Burt & Caccetta (2007), source probable de la formule du facteur d'appariement : https://doi.org/10.1080/17480930701388606
- Wilcoxon (1945), Friedman (1937), Holm (1979), Efron & Tibshirani (1993), Puterman (1994)

## Erreurs trouvées dans le `.bib` (à corriger au lot 2)

| Clé | Problème | Correction |
|---|---|---|
| FranciscoRosales2025 | Mauvaise publication, et fiche avec un mauvais co-auteur (« D'Angelo ») | `@inproceedings` : Rosales Marticorena, L. F. ; Diaz, A. *Proc. 12th Int. Conf. of Management and Industrial Engineering (ICMIE)*, vol. 12, p. 207-217, 2025, DOI 10.56177/12icmie2025.202 |
| deCarvalho2023 | Titre incomplet | « Integrating short-term stochastic production planning updating with mining fleet management… », *Applied Intelligence* 53, 23179-23202, DOI 10.1007/s10489-023-04774-3 |
| Banerjee2025 | Titre de la v1 | « … for Open-Pit Truck Dispatch Optimization » (v2), arXiv:2503.19195 |
| MarkTowers2025 | Titre | « A **Standardized** Interface… », NeurIPS 2025 |
| Hazrathosseini2024 | Incomplet | 133(1), 50-73, DOI 10.1177/25726668231222998 |
| Zhao2025 | Incomplet | 7e auteur Xiaolei Ma, DOI 10.1080/23249935.2025.2522360 |
| Pei2025 | Incomplet | 40(2), 89-106, DOI 10.1080/17480930.2025.2486316 (en ligne en 2025) |

## Points à revoir dans l'article (lot 2)

- **Szatmári et al. (2025)** est une revue : « démontrent des architectures multi-niveaux » (l. 187) devient « passent en revue ».
- **Krichen (2022)** est un document HAL non évalué par les pairs : le retirer de l'introduction (l. 181).
- **Nwachukwu et al. (2025)** est hors domaine : Haarnoja et al. (2018) suffit pour SAC (l. 1603).
- **Pei et al. (2025)** évaluent déjà leur politique sans réentraînement, sur des flottes de 20 à 25 camions : ne pas présenter l'évaluation sans réentraînement comme entièrement nouvelle.
- **Rosales & Diaz (2025)** ne donnent aucun résultat chiffré : ne pas leur attribuer de gain.
- **Facteur d'appariement** (l. 935) : vérifier la formule dans Burt & Caccetta (2007) ou (2014).
- **Watkins & Dayan (1992)** : leurs conditions de convergence (table, exploration infinie, pas décroissants) ne sont pas réunies pour nos agents tabulaires. C'est un argument pour expliquer leur mauvaise convergence.
