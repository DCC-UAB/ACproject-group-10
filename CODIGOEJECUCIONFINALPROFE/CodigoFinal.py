import os
import re
import kagglehub
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from time import sleep
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    confusion_matrix
)
print("Descomprimir CSVs en la carpeta CODIGOEJECUCIONFINALPROFE y ejecutar este script")
print("El codigo de este script es el usado durante toda la practica, solo reorganizado para ejecutarlo de manera secuencial con ligertos cambios o excepciones")
###############################################################################
#                          DESCARGA DE RECURSOS NLTK
###############################################################################
nltk.download('stopwords')
nltk.download('punkt')


print("descarga de dataset grande, dataset base esta en el directorio actual")
# Descargar el dataset al directorio actual
current_directory = os.getcwd()

# Descargar el dataset al directorio actual
print("Si aparece un error de descarga, borrar la linea de descarga y simplemente no se usara el dataset grande, cuyos resultados igual estan en el repositorio de antemano")
sleep(1)
print("Tambien se puede descargar manualmente de https://www.kaggle.com/snap/amazon-fine-food-reviews")
sleep(5)
path = kagglehub.dataset_download("snap/amazon-fine-food-reviews", current_directory)
print("Path to dataset files:", path)
###############################################################################
#                           FUNCIÓN DE LIMPIEZA DE TEXTO
###############################################################################
def clean_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    words = text.split()
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

###############################################################################
#                        FUNCIÓN PARA ENTRENAR Y EVALUAR MODELOS
###############################################################################
def train_and_evaluate(models, X_train, X_test, y_train, y_test, folder_name, output_filename):
    """
    Entrena y evalúa cada modelo en 'models'.
    Genera matriz de confusión y curva ROC. Guarda métricas en un diccionario.
    """
    # Crear directorio de salida si no existe
    os.makedirs(folder_name, exist_ok=True)

    # Archivo para guardar los resultados
    output_file = open(output_filename, 'w')

    results = {}
    for model_name, model in models.items():
        print(f"Entrenando {model_name}...")
        # Manejo especial de GaussianNB con X sparse
        if model_name == "Naive Bayes (Gaussian)":
            model.fit(X_train.toarray(), y_train)
            y_pred = model.predict(X_test.toarray())
            y_pred_prob = model.predict_proba(X_test.toarray())[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_prob = model.predict_proba(X_test)[:, 1]

        # Cálculo de métricas
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary')
        recall = recall_score(y_test, y_pred, average='binary')
        f1 = f1_score(y_test, y_pred, average='binary')
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)

        # Almacenar resultados
        results[model_name] = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "ROC AUC": roc_auc,
            "FPR": fpr,
            "TPR": tpr
        }

        # Matriz de confusión
        cm = confusion_matrix(y_test, y_pred, labels=[1, 0])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Positivo', 'Negativo'],
                    yticklabels=['Positivo', 'Negativo'])
        plt.title(f'Matriz de Confusión para {model_name}')
        plt.xlabel('Predicción')
        plt.ylabel('Real')
        plt.savefig(f'{folder_name}/confusion_matrix_{model_name.replace(" ", "_").lower()}.png')
        plt.close()

    # Imprimir y guardar resultados en archivo
    for model_name, metrics in results.items():
        output_file.write(f"Resultados para {model_name}:\n")
        for metric, value in metrics.items():
            if isinstance(value, np.ndarray):
                continue
            output_file.write(f"{metric}: {value:.4f}\n")
        output_file.write("\n")

    # Crear gráfico comparativo de la curva ROC
    plt.figure(figsize=(10, 5))
    for model_name, metrics in results.items():
        plt.plot(metrics["FPR"], metrics["TPR"], label=f'{model_name} (AUC = {metrics["ROC AUC"]:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Curva ROC Comparativa')
    plt.legend(loc="lower right")
    plt.savefig(f'{folder_name}/roc_comparativa.png')
    plt.close()

    output_file.close()
    return results

###############################################################################
#                   FUNCIÓN PARA BÚSQUEDA DE HIPERPARÁMETROS
###############################################################################
def hyperparameter_search(model, param_grid, X_train, y_train):
    """
    Realiza búsqueda de hiperparámetros con GridSearchCV.
    Retorna grid_search ya ajustado.
    """
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                               cv=3, scoring='accuracy', n_jobs=-1,
                               return_train_score=True)
    grid_search.fit(X_train, y_train)
    return grid_search

###############################################################################
#                        PRIMER DATASET (amazon_reviews.csv)
###############################################################################
print("Entrenamiento de datos en primer dataset")
sleep(5)

# Cargar y limpiar el dataset
df = pd.read_csv('amazon_reviews.csv')
dfSimple = df.drop(
    columns=[
        'reviewerName', 'reviewTime', 'day_diff', 'helpful_yes', 'helpful_no',
        'total_vote', 'score_pos_neg_diff', 'score_average_rating',
        'wilson_lower_bound'
    ]
)

# Binarizar la variable objetivo
dfSimpleBinary = dfSimple.copy()
dfSimpleBinary['overall'] = dfSimpleBinary['overall'].apply(lambda x: 1 if x > 2.5 else 0)
dfSimpleBinary.dropna(subset=["reviewText"], inplace=True)
dfSimpleBinary['reviewText'] = dfSimpleBinary['reviewText'].apply(clean_text)

# Vectorización TF-IDF
tfidf_1 = TfidfVectorizer(max_features=5000)
X_1 = tfidf_1.fit_transform(dfSimpleBinary['reviewText'])
y_1 = dfSimpleBinary['overall']

# División de datos en entrenamiento y prueba
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_1, y_1, test_size=0.2, random_state=42)

# Modelos iniciales
models_1 = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "Naive Bayes (Multinomial)": MultinomialNB(),
    "Naive Bayes (Bernoulli)": BernoulliNB(),
    "Naive Bayes (Gaussian)": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(),
    "SVM": SVC(probability=True),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100)
}

# Entrenamiento y evaluación
results_1 = train_and_evaluate(models_1, X_train_1, X_test_1, y_train_1, y_test_1,
                               'plots_primer_dataset', 'results_primer_dataset.txt')

###############################################################################
#          BÚSQUEDA DE HIPERPARÁMETROS PARA ALGUNOS MODELOS (1er dataset)
###############################################################################
# Ejemplo: Logistic Regression y Random Forest
param_grid_1 = {
    "Logistic Regression": {
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear', 'lbfgs']
    },
    "Random Forest": {
        'n_estimators': [50, 100],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [None, 10, 20]
    }
}

best_params_1 = {}
for model_name in ["Logistic Regression", "Random Forest"]:
    print(f"\nBúsqueda de hiperparámetros para {model_name} (dataset 1)")
    base_model = models_1[model_name]
    grid_search_1 = hyperparameter_search(base_model, param_grid_1[model_name], X_train_1, y_train_1)
    best_params_1[model_name] = grid_search_1.best_params_
    print("Mejores hiperparámetros:", grid_search_1.best_params_)

###############################################################################
#                        SEGUNDO DATASET (balanceado)
###############################################################################
print("\nEntrenamiento de datos en segundo dataset (primer dataset balanceado)")
sleep(5)

# Crear conjunto de datos balanceado para el 2do experimento
df_positive_2 = dfSimpleBinary[dfSimpleBinary['overall'] == 1]
df_negative_2 = dfSimpleBinary[dfSimpleBinary['overall'] == 0]
df_balanced_2 = pd.concat([df_positive_2.sample(len(df_negative_2), random_state=42), df_negative_2])

X_2 = tfidf_1.transform(df_balanced_2['reviewText'])
y_2 = df_balanced_2['overall']

X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_2, y_2, test_size=0.2, random_state=42)

# Reutilizamos los mismos modelos básicos
models_2 = models_1

# Entrenamiento y evaluación
results_2 = train_and_evaluate(models_2, X_train_2, X_test_2, y_train_2, y_test_2,
                               'plots_segundo_dataset', 'results_segundo_dataset.txt')

###############################################################################
#   BÚSQUEDA DE HIPERPARÁMETROS PARA ALGUNOS MODELOS (2do dataset balanceado)
###############################################################################
best_params_2 = {}
for model_name in ["Logistic Regression", "Random Forest"]:
    print(f"\nBúsqueda de hiperparámetros para {model_name} (dataset 2)")
    base_model = models_2[model_name]
    grid_search_2 = hyperparameter_search(base_model, param_grid_1[model_name], X_train_2, y_train_2)
    best_params_2[model_name] = grid_search_2.best_params_
    print("Mejores hiperparámetros:", grid_search_2.best_params_)

###############################################################################
#            TERCER DATASET (reviews.csv), CON FORMATO UNIFICADO
###############################################################################
print("\nEntrenamiento y resultados dataset Grande (3er dataset)")
sleep(5)

# Crear directorio para guardar los gráficos
os.makedirs('hyperParams2Balanced_plots', exist_ok=True)

# Archivo para guardar las salidas de texto
output_file_3 = open('Hyperparams2Balanced.txt', 'w')

# Cargar y limpiar datos
dfSimpleBinary_3 = pd.read_csv('reviews.csv')
dfSimpleBinary_3['Score'] = dfSimpleBinary_3['Score'].apply(lambda x: 1 if x > 2.5 else 0)
dfSimpleBinary_3.dropna(subset=["Text"], inplace=True)
dfSimpleBinary_3['Text'] = dfSimpleBinary_3['Text'].apply(clean_text)

# Vectorizar (nueva TF-IDF para este dataset si se desea independiente)
tfidf_3 = TfidfVectorizer(max_features=5000)
X_3 = tfidf_3.fit_transform(dfSimpleBinary_3['Text'])
y_3 = dfSimpleBinary_3['Score']

# Crear un conjunto de datos balanceado (como en los otros ejemplos)
df_positive_3 = dfSimpleBinary_3[dfSimpleBinary_3['Score'] == 1]
df_negative_3 = dfSimpleBinary_3[dfSimpleBinary_3['Score'] == 0]
df_balanced_3 = pd.concat([df_positive_3.sample(len(df_negative_3), random_state=42), df_negative_3])

X_bal_3 = tfidf_3.transform(df_balanced_3['Text'])
y_bal_3 = df_balanced_3['Score']

# Dividir datos
X_train_3, X_test_3, y_train_3, y_test_3 = train_test_split(X_bal_3, y_bal_3, test_size=0.2, random_state=42)

# Modelos básicos (antes de la búsqueda de hiperparámetros)
models_3_before = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes (Multinomial)": MultinomialNB(),
    "Naive Bayes (Bernoulli)": BernoulliNB(),
}

# Entrenamiento y evaluación (versión "antes")
results_3_before = train_and_evaluate(
    models_3_before, X_train_3, X_test_3, y_train_3, y_test_3,
    'hyperParams2Balanced_plots', 'results_3_before.txt'
)

# Búsqueda de hiperparámetros (adaptada al 3er dataset)
param_grid_3 = {
    "Logistic Regression": {
        'C': [0.1, 1, 10],
        'solver': ['newton-cg', 'lbfgs', 'liblinear']
    },
    "Naive Bayes (Multinomial)": {
        # Ejemplo: En MultinomialNB no hay muchos hiperparámetros relevantes,
        # pero se podría modificar 'alpha'
        'alpha': [0.1, 1.0, 10.0]
    },
    "Naive Bayes (Bernoulli)": {
        'alpha': [0.1, 1.0, 10.0]
    },
}

best_params_3 = {}
for model_name, model in models_3_before.items():
    print(f"\nBúsqueda de hiperparámetros para {model_name} (3er dataset)")
    grid_search_3 = hyperparameter_search(model, param_grid_3[model_name], X_train_3, y_train_3)
    best_params_3[model_name] = grid_search_3.best_params_
    print(f"Mejores hiperparámetros para {model_name}:", grid_search_3.best_params_)
    output_file_3.write(f"Mejores hiperparámetros para {model_name}: {grid_search_3.best_params_}\n")

# Modelos con los mejores hiperparámetros
models_3_after = {
    "Logistic Regression": LogisticRegression(max_iter=1000, **best_params_3["Logistic Regression"]),
    "Naive Bayes (Multinomial)": MultinomialNB(**best_params_3["Naive Bayes (Multinomial)"]),
    "Naive Bayes (Bernoulli)": BernoulliNB(**best_params_3["Naive Bayes (Bernoulli)"])
}

# Entrenamiento y evaluación (versión "después")
results_3_after = train_and_evaluate(
    models_3_after, X_train_3, X_test_3, y_train_3, y_test_3,
    'hyperParams2Balanced_plots', 'results_3_after.txt'
)

output_file_3.close()

print("Proceso finalizado. Se generaron los archivos y gráficos correspondientes.")
