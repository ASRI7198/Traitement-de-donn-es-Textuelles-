import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, zero_one_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, zero_one_loss
from sklearn.multioutput import MultiOutputClassifier, ClassifierChain
import pandas as pd
from sklearn.decomposition import TruncatedSVD
import warnings
import gensim
from gensim.models import KeyedVectors

nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')

np.set_printoptions(threshold=10000, suppress = True)
warnings.filterwarnings('ignore')

def import_data(file_name):
    df = pd.read_csv(file_name)
    return df


def analyze_data(df):
    # change types
    for column in df.select_dtypes(include=['object']).columns:
        df[column] = df[column].astype(str)
    print("Types de colonnes :")
    print(df.dtypes)

    nombre_lignes, nombre_colonnes = df.shape
    print(f"Nombre de lignes : {nombre_lignes}")
    print(f"Nombre de colonnes : {nombre_colonnes}")

    label_columns = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'L', 'M', 'N', 'Z']
    label_counts = df[label_columns].sum()
    print("Fréquence des labels :")
    print(label_counts)

    plt.figure(figsize=(10, 6))
    label_counts.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title("Fréquence des labels dans le dataset")
    plt.xlabel("Labels")
    plt.ylabel("Nombre d'occurrences")
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


def traiter_data(df):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    def clean_text(text):
        if not isinstance(text, str):
            text = str(text)
        # Conversion en minuscules
        text = text.lower()
        # Suppression des ponctuations et des caractères spéciaux
        text = re.sub(r'[^\w\s]', '', text)
        # Fractionner le text en mots
        words = text.split()
        # Supprimer les stop words et appliquer la lemmatization
        words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
        # Reconstruire le texte en une seule chaîne
        return " ".join(words)
    
    columns = ["abstractText", 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'L', 'M', 'N', 'Z']

    df["abstractText"] = df["abstractText"].apply(clean_text)

    return df[columns]


def separer_data(prepared_df, y):
    if not y.empty:
        X_train, X_test, y_train, y_test = train_test_split(
            prepared_df, y, test_size=0.5, random_state=42
        )
        return X_train, X_test, y_train, y_test

    labels_columns = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'L', 'M', 'N', 'Z']
    X = prepared_df["abstractText"]
    y = prepared_df[labels_columns]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=42
    )
    return X_train, X_test, y_train, y_test


def vectorisation_tfidf(X_train, X_test):
    tfidf_vectorizer = TfidfVectorizer(max_features=2000)
    X_train_vectorised = tfidf_vectorizer.fit_transform(X_train)
    X_test_vectorised = tfidf_vectorizer.transform(X_test)
    return X_train_vectorised, X_test_vectorised, tfidf_vectorizer


def vocabulaire(vectorizer):
    if hasattr(vectorizer, 'get_feature_names_out'):
        return vectorizer.get_feature_names_out()
    else:
        raise ValueError("The vectorizer must be fitted before calling vocabulaire().")


def optimize_k(X_train_vectorised, y_train):
    param_grid = {'n_neighbors': range(3, 20)}
    grid_search = GridSearchCV(KNeighborsClassifier(metric='euclidean'), param_grid, cv=5, scoring='f1_micro')
    grid_search.fit(X_train_vectorised, y_train)
    return grid_search.best_params_['n_neighbors']


def run_models(X_train_vectorised, y_train, X_test_vectorised, y_test, num_chains=3):
    results = {}

    # Optimisation du nombre de voisins pour KNN
    best_k = optimize_k(X_train_vectorised, y_train)
    print(f"Nombre optimal de voisins pour KNN : {best_k}")

    knn = KNeighborsClassifier(n_neighbors=best_k, metric='euclidean')

    # KNN avec ClassifierChain
    print("Entraînement de KNN avec ClassifierChain...")
    chains_knn = [ClassifierChain(knn, order="random", random_state=i) for i in range(num_chains)]
    for chain in chains_knn:
        chain.fit(X_train_vectorised, y_train)

    y_pred_chains_knn = []
    for chain in chains_knn:
        y_pred_chain = chain.predict_proba(X_test_vectorised) >= 0.5
        y_pred_chains_knn.append(y_pred_chain)

    y_pred_chains_knn = np.array(y_pred_chains_knn)
    y_pred_final_knn_chain = (y_pred_chains_knn.sum(axis=0) > (len(chains_knn) / 2)).astype(int)

    results['KNN_ClassifierChain'] = {
        'micro-F1': f1_score(y_test, y_pred_final_knn_chain, average='micro'),
        'macro-F1': f1_score(y_test, y_pred_final_knn_chain, average='macro'),
        'zero-one-loss': zero_one_loss(y_test, y_pred_final_knn_chain)
    }

    # KNN avec MultiOutputClassifier
    print("Entraînement de KNN avec MultiOutputClassifier...")
    model_knn = MultiOutputClassifier(knn)
    model_knn.fit(X_train_vectorised, y_train)
    y_pred_knn_multi = model_knn.predict(X_test_vectorised)

    results['KNN_MultiOutput'] = {
        'micro-F1': f1_score(y_test, y_pred_knn_multi, average='micro'),
        'macro-F1': f1_score(y_test, y_pred_knn_multi, average='macro'),
        'zero-one-loss': zero_one_loss(y_test, y_pred_knn_multi)
    }

    # Réseau de neurones
    nn = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=42
    )

    # NN avec ClassifierChain
    print("Entraînement de NN avec ClassifierChain...")
    chains_nn = [ClassifierChain(nn, order="random", random_state=i) for i in range(num_chains)]
    for chain in chains_nn:
        chain.fit(X_train_vectorised, y_train)

    y_pred_chains_nn = []
    for chain in chains_nn:
        y_pred_chain = chain.predict_proba(X_test_vectorised) >= 0.5
        y_pred_chains_nn.append(y_pred_chain)

    y_pred_chains_nn = np.array(y_pred_chains_nn)
    y_pred_final_nn_chain = (y_pred_chains_nn.sum(axis=0) > (len(chains_nn) / 2)).astype(int)

    results['NN_ClassifierChain'] = {
        'micro-F1': f1_score(y_test, y_pred_final_nn_chain, average='micro'),
        'macro-F1': f1_score(y_test, y_pred_final_nn_chain, average='macro'),
        'zero-one-loss': zero_one_loss(y_test, y_pred_final_nn_chain)
    }

    # NN avec MultiOutputClassifier
    print("Entraînement de NN avec MultiOutputClassifier...")
    model_nn = MultiOutputClassifier(nn)
    model_nn.fit(X_train_vectorised, y_train)
    y_pred_nn_multi = model_nn.predict(X_test_vectorised)

    results['NN_MultiOutput'] = {
        'micro-F1': f1_score(y_test, y_pred_nn_multi, average='micro'),
        'macro-F1': f1_score(y_test, y_pred_nn_multi, average='macro'),
        'zero-one-loss': zero_one_loss(y_test, y_pred_nn_multi)
    }

    # Affichage des résultats
    results_df = pd.DataFrame(results).T
    print("\nComparaison des performances des modèles :")
    print(results_df.to_string())

    return results


def plot_svd_cumulative_variance(X_train_vectorised, threshold=0.95):
    svd = TruncatedSVD(n_components=X_train_vectorised.shape[1], random_state=42)
    svd.fit(X_train_vectorised)
    cumulative_variance = svd.explained_variance_ratio_.cumsum()

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'{threshold*100}% variance')
    plt.xlabel('Nombre de composantes')
    plt.ylabel('Variance expliquée cumulée')
    plt.title('Variance expliquée cumulée pour SVD')
    plt.legend()
    plt.grid()
    plt.show()


def apply_svd(X_train_vectorised, X_test_vectorised, tfidf_vectorizer, n_components=100):
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_train_svd = svd.fit_transform(X_train_vectorised)
    X_test_svd = svd.transform(X_test_vectorised)

    print("Top words for each concept:")
    print_top_words(svd, feature_names=vocabulaire(tfidf_vectorizer), n_top_words=10)

    return X_train_svd, X_test_svd


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Concept #%d: " % topic_idx
        message += " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()


def word2vec_generator(texts,model,vector_size):
    dict_word2vec = {}
    for index, word_list in enumerate(texts):
        arr = np.array([0.0 for i in range(0, vector_size)])
        nb_word=0
        for word in word_list:
            try:
                arr += model[word]
                nb_word=nb_word+1
            except KeyError:
                continue
        if(len(word_list) == 0):
            dict_word2vec[index] = arr
        else:
            dict_word2vec[index] = arr / nb_word
    df_word2vec = pd.DataFrame(dict_word2vec).T
    return df_word2vec


def word2vec_generator_tfidf(texts, model, vector_size, tfidf_vectorizer):
    tfidf_scores = dict(zip(tfidf_vectorizer.get_feature_names_out(), tfidf_vectorizer.idf_))
    dict_word2vec = {}
    for index, word_list in enumerate(texts):
        arr = np.zeros(vector_size)
        total_weight = 0
        for word in word_list:
            try:
                word_vector = model[word]
                tfidf_score = tfidf_scores.get(word, 0)
                arr += word_vector * tfidf_score
                total_weight += tfidf_score
            except KeyError:
                continue
        if total_weight > 0:
            dict_word2vec[index] = arr / total_weight
        else:
            dict_word2vec[index] = arr
    df_word2vec = pd.DataFrame(dict_word2vec).T
    return df_word2vec


def pipeline(data):
    data = traiter_data(data)
    print("Preparation des données ☑️")

    y = data[['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'L', 'M', 'N', 'Z']]
    corpus = data['abstractText']
    corpus = corpus.apply(lambda line : gensim.utils.simple_preprocess((line)))
    X_train, X_test, y_train, y_test = separer_data(corpus, y)
    print("Séparation des données ☑️")

    model = gensim.models.Word2Vec.load('Word2vec_entraine.h5')
    X_train_word2vec = word2vec_generator(X_train, model.wv, model.wv.vector_size)
    X_test_word2vec = word2vec_generator(X_test, model.wv, model.wv.vector_size)
    print("Création des embedding ☑️")

    print("Entraînement des Modèles...")
    results_word2vec = run_models(X_train_word2vec, y_train, X_test_word2vec, y_test)
    
    return results_word2vec