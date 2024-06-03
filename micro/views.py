from django.shortcuts import render
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# Create your views here.
def home(request):
    if request.method == 'POST':
        montant = int(request.POST.get("montant"))
        revenu = int(request.POST.get("revenu"))
        antecedent = int(request.POST.get("antecedent"))
        result = regressionLogistic(montant, revenu, antecedent)
        return render(request, "home.html", {'result': result})
    return render(request, "home.html")

def regressionLogistic(montant, revenu, antecedent):
    # Charger les données CSV dans un DataFrame Pandas
    data = pd.read_csv('micro/static/microfinance_data.csv')

    # Extraire les caractéristiques et la cible
    X = data.iloc[:, 1:-1].values  # toutes les lignes de toutes les colonnes sauf la première et derniere
    y = data.iloc[:, -1].values  # toutes les lignes de la derniere colonne
    y = y.reshape((y.shape[0], 1))

    # Vérifier les types de données
    print(f"les types de données: X = {X.dtype} et y = {y.dtype}")

    # Normaliser les caractéristiques numériques
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)
    plt.hist(X_norm[:, 0])  # Visualiser la distribution de la première caractéristique normalisée
    plt.hist(X_norm[:, 1])  # Visualiser la distribution de la deuxième caractéristique normalisée
    plt.show()

    # Créer le nuage de points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='summer')
    # Définir le point de référence central pour accéder à diverses ressources du projet
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    # Définir le chemin d'enregistrement
    save_path = os.path.join(BASE_DIR, 'static', 'plot1.png')
    plt.title("Diagramme de points")
    # Ajouter des étiquettes aux axes
    plt.xlabel("Montant")
    plt.ylabel("Revenu annuel")
    # Enregistrer le graphique au chemin d'accès spécifié
    plt.savefig(save_path)

    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Créer un modèle de régression logistique
    model = LogisticRegression()

    # Entraîner le modèle sur les données d'entraînement
    model.fit(X_train, y_train)

    # Faire des prédictions sur les données de test
    y_pred = model.predict(X_test)

    # Évaluer les performances du modèle
    accuracy = accuracy_score(y_test, y_pred) #la précision du modèle (pourcentage de prédictions correctes)
    print(f"Précision du modèle: {accuracy:.2f}")
    # Visualiser les erreurs de classification
    confusion = confusion_matrix(y_test, y_pred)
    print("Matrice de confusion :\n", confusion)
    classification = classification_report(y_test, y_pred)
    print("Rapport de classification :\n", classification)

    # Afficher les coefficients du modèle ou Interpréter les coefficients du modèle (facultatif)
    print("Coefficients du modèle:")
    print(model.coef_)

    # Prédire la probabilité de remboursement pour un nouveau client
    new_customer = np.array([[montant, revenu, antecedent]])  # Montant du prêt, Revenu annuel, Antécédents de crédit

    # Créer le nuage de points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='summer', edgecolors='k')
    plt.scatter(new_customer[:, 0], new_customer[:, 1], c='r')
    # Définir le point de référence central pour accéder à diverses ressources du projet
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    # Définir le chemin d'enregistrement
    save_path = os.path.join(BASE_DIR, 'static', 'plot2.png')
    plt.title("Diagramme de points pour le nouveau client")
    # Ajouter des étiquettes aux axes
    plt.xlabel("Montant")
    plt.ylabel("Revenu annuel")
    # Enregistrer le graphique au chemin d'accès spécifié
    plt.savefig(save_path)

    probability = model.predict_proba(new_customer)[0][1]
    print(model.predict(new_customer))
    return "{:.2f}%".format(probability * 100)
