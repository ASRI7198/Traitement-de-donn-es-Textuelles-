## Atelier Pratique : Étude Comparative d'Algorithmes Multi-Label

## Description

Dans cet atelier pratique, l'objectif est de réaliser une étude comparative entre plusieurs algorithmes d’apprentissage supervisé multi-label sur un jeu de données textuelles en Python. Vous explorerez diverses techniques de prétraitement et de modélisation.

##Prérequis

Python 3.8+

Jupyter Notebook

Les bibliothèques Python suivantes :

numpy

pandas

matplotlib

scikit-learn

nltk

gensim

Installez les bibliothèques avec la commande suivante si elles ne sont pas disponibles :

pip install numpy pandas matplotlib scikit-learn nltk gensim

Lancement de Jupyter Notebook

Pour ouvrir un notebook, tapez la commande suivante dans votre terminal :

jupyter notebook

Cela ouvrira une interface dans votre navigateur. Créez un nouveau notebook Python et commencez par exécuter le code suivant dans une cellule :

import numpy as np
np.set_printoptions(threshold=10000, suppress=True)
import pandas as pd
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

#Jeu de Données

Le jeu de données utilisé est intitulé PubMed Multi-label Dataset et est disponible dans le fichier PubMed-multi-label-dataset.csv. Chaque texte est associé à un ensemble de labels parmi les 14 suivants :

Anatomy [A]

Organisms [B]

Diseases [C]

Chemicals and Drugs [D]

Analytical, Diagnostic and Therapeutic Techniques, and Equipment [E]

Psychiatry and Psychology [F]

Phenomena and Processes [G]

Disciplines and Occupations [H]

Anthropology, Education, Sociology, and Social Phenomena [I]

Technology, Industry, and Agriculture [J]

Information Science [L]

Named Groups [M]

Health Care [N]

Geographicals [Z]

#Tâches à Réaliser

#1. Prétraitement des Données

Importer le fichier avec pandas.read_csv.

Analyser la distribution des labels (target).

Nettoyer les textes en supprimant les stop words, en transformant le texte en minuscules et en retirant les ponctuations (.,!$()@%*).

Diviser les données en deux ensembles :

Jeu d’apprentissage : 50%

Jeu de test : 50%

# 2. Modélisation Multi-Label

Implémentez une fonction run_models qui :

Compare les approches EnsembleClassifierChain et MultiOutputClassifier de sklearn.multioutput.

Utilise deux modèles de base : un réseau de neurones à deux couches et un k-plus proches voisins.

Évalue les modèles avec les métriques suivantes :

Micro-F1

Macro-F1

Zero-One Loss

# 3. Vectorisation des Textes

Effectuer une vectorisation initiale en utilisant TF-IDF (en ajustant la taille du vocabulaire).

Appliquer la réduction de dimension avec TruncatedSVD pour générer des concepts (thèmes). Utilisez la fonction suivante pour afficher les mots les plus pertinents :

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = f"Concept #{topic_idx}: "
        message += " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()

Interprétez les résultats obtenus.

# 4. Word2Vec

Créez un nouveau notebook pour :

Apprendre un modèle Word2Vec sur les données textuelles (cf. Word2Vec_creation.ipynb sur Moodle).

Évaluer visuellement et numériquement le modèle sur quelques mots-clés.

Utiliser le modèle Word2Vec pour vectoriser les textes en moyenne pondérée par les scores TF-IDF.

Comparer les résultats avec ceux obtenus à l’étape précédente.

# 5. Automatisation

Construisez un pipeline automatisé pour réaliser l’ensemble des traitements et modélisations.

Liens Utiles

Documentation Scikit-learn

Gensim pour Word2Vec

Modèle Word2Vec pré-entraîné de Google
