# 🥤 Projet détection ecocup

Ce projet vise à entrainer un **classifieur binaire** permettant la détection des *ecocup* (gobelets réutilisables en plastique). Pour ce faire, nous utiliserons les outils et techniques vues en **SY32** (*Vision et apprentissage*) lors du semestre P25

---

## 📁 Structure du projet

SY32_detection_ecopcup/

│

├── src/                      # Tout le code source

│   ├── const/                # Constient les constantes du projet

│

├── data/                     # Données du projet

│

├── docs/                     # Documentation et rapports/soutenances. Contient les attendus du projet

├── README.md                 # You are here 📌

├── contributing.md                 # Normes pour la contribution (PR, commits messages...)

└── requirements.txt          # Fichier contenant les bibliothèques nécessaires

---

## 🎯 Objectifs

- Entrainer un classifieur capable de détecter une écocup
- Utiliser les méthodes de cours pour détecter toutes les ecocups

---

## 🔧 Technologies

- **Python** : traitement de données, ETL, modélisation
- **Numpy** : manipulation des données
- **Pandas** : manipulation des données
- **Scikit Image**: manipulation des images
- **Scikit learn** : machine learning pour l’analyse statistique
- **Git / GitHub** : gestion de version et collaboration
- **Matplotlib** : Affichage

---

## 🚀 Lancement du projet

1. **Cloner le dépôt**
```bash
git clone --recurse-submodules https://github.com/EngelCalon/SY32_detection_ecocup.git
cd SY32_detection_ecocup
```
ou s'il est déjà cloné, il faut récupérer les données d'entrainement:

```bash
git submodule update --init --recursive
```

Commande utilisée pour cloner le jeu d'entrainement dans le repo (ne pas refaire):

```bash
git submodule add https://github.com/user/data-repo.git data
```

2. **Installer les bibliothèques**
```bash
pip install -r requirements.txt
```

3. Pull les changements du jeu d'entrainement

```bash
git submodule update
```

---

## Codes à analyser/lancer

- Le notebook de visualisation des résultats sur les images de test est : results_on_test_set.ipynb
- Le notebook principal pour l'entraînement et la détection sur les images de test est : detections_on_test_set.ipynb
- Les données filtrées étapes par étapes et les codes permettant de préparer les données sont dans local_data/
- Les fonctions utilisées par l'algorithme de détection sont dans utils/

Le reste est à l'état de brouillon, mais nous pouvons tout de même décrire l'utilité de chacun : 

- Les notebooks classifieur.ipynb et classifieur_v2.ipynb ont servi pour l'évaluation des différents classifieurs et extracteurs de descriptions d'image pour notre cas
- Le notebook optimisation_engel.ipynb est celui ayant servi à calculer les paramètres optimaux pour le classifieur retenu
- fenetre_glissante.ipynb et fenetre_glissante_prepa.ipynb ont servi pour le test et la calibration de l'algorithme de détection sur les images d'entraînement 
- test_data/ contient les résultats enregistrés de la détection brute sur les images de tests (en retenant toutes les détections avec une confiance de plus de 0.5, ces résultats sont ensuite filtrés à l'étape de visualisation)
---

## 👥 Collaboration

- Travail en binôme

- Suivi des tâches via issues GitHub

- Gestion des rushs avec les fonctionnalités Projet Github

- Normes pour les actions git précisées dans **contributing.md**

---

## 🧠 Auteurs

- Engel CALON (*@EngelCalon*)

- Adrien Duqué (*@riuza-c-137*)

---

## 🙏 Remerciements

Un grand merci à **Julien MOREAU** pour son enseignement et son suivi tout au long du semestre. Ses conseils et cours auront été cruciaux lors du développement du projet.

