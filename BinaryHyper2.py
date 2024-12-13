import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Crear directorio para guardar los gráficos
os.makedirs('hyperParams2_plots', exist_ok=True)

# Archivo para guardar las salidas de texto
output_file = open('Hyperparams2.txt', 'w')

# Cargar y preprocesar los datos
dfSimpleBinary = pd.read_csv('amazon_reviews2_simpleBinary.csv')
dfSimpleBinary['Text'] = dfSimpleBinary['Text'].fillna('')

tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(dfSimpleBinary['Text'])
y = dfSimpleBinary['Score']

# División de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir modelos sin ajuste de hiperparámetros
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes (Multinomial)": MultinomialNB(),
    "Naive Bayes (Bernoulli)": BernoulliNB(),
    #"Naive Bayes (Gaussian)": GaussianNB(),

    "Random Forest": RandomForestClassifier(n_estimators=100),
    "Decision Tree": DecisionTreeClassifier(),
    #"SVM": SVC(probability=True),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100)
}

# Diccionario para almacenar los resultados
results_before = {}

for model_name, model in models.items():
    output_file.write(f"Entrenando modelo: {model_name}\n")
    # Entrenar el modelo
    model.fit(X_train, y_train)
    # Predecir en el conjunto de prueba
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    # Evaluar el rendimiento
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    # Guardar los resultados
    results_before[model_name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC AUC": roc_auc,
        "FPR": fpr,
        "TPR": tpr
    }

# Definir los hiperparámetros a buscar para cada modelo
param_grid = {
    "Logistic Regression": {
        'C': [0.1, 1, 10],
        'solver': ['newton-cg', 'lbfgs']
    },
    "Random Forest": {
        'n_estimators': [50, 100],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [None, 10, 20],
        'criterion': ['gini']
    },
    "SVM": {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf']
    },
    "K-Nearest Neighbors": {
        'n_neighbors': [3, 5],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean']
    },
    "Gradient Boosting": {
        'n_estimators': [50, 100],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5]
    }
}

# Crear los modelos base
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(),
    #"Naive Bayes (Multinomial)": MultinomialNB(),
    #"Naive Bayes (Bernoulli)": BernoulliNB(),
    #"Naive Bayes (Gaussian)": GaussianNB(),
    # "Decision Tree": DecisionTreeClassifier(),
    # "SVM": SVC(probability=True),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Gradient Boosting": GradientBoostingClassifier()
}

# Buscar los mejores hiperparámetros
best_params = {}
for model_name, model in models.items():
    output_file.write(f"Buscando mejores hiperparámetros para: {model_name}\n")
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid[model_name], cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_params[model_name] = grid_search.best_params_
output_file.write(f"{best_params}\n")

# Definir modelos con los mejores hiperparámetros
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, **best_params["Logistic Regression"]),
    "Random Forest": RandomForestClassifier(**best_params["Random Forest"]),
    "Naive Bayes (Multinomial)": MultinomialNB(),
    "Naive Bayes (Bernoulli)": BernoulliNB(),
    #"Naive Bayes (Gaussian)": GaussianNB(),
    #"Decision Tree": DecisionTreeClassifier(),
    #"SVM": SVC(probability=True, **best_params["SVM"]),
    "K-Nearest Neighbors": KNeighborsClassifier(**best_params["K-Nearest Neighbors"]),
    "Gradient Boosting": GradientBoostingClassifier(**best_params["Gradient Boosting"])
}

# Diccionario para almacenar los resultados después del ajuste de hiperparámetros
results_after = {}

for model_name, model in models.items():
    output_file.write(f"Entrenando modelo con mejores hiperparámetros: {model_name}\n")
    # Entrenar el modelo
    model.fit(X_train, y_train)
    # Predecir en el conjunto de prueba
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    # Evaluar el rendimiento
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    # Guardar los resultados
    results_after[model_name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC AUC": roc_auc,
        "FPR": fpr,
        "TPR": tpr
    }

# Aplicar los modelos entrenados al dataset completo
X_full = tfidf.transform(dfSimpleBinary['Text'])
y_full = dfSimpleBinary['Score']

for model_name, model in models.items():
    output_file.write(f"Aplicando modelo al dataset completo: {model_name}\n")
    y_pred_full = model.predict(X_full)
    y_pred_prob_full = model.predict_proba(X_full)[:, 1]
    # Evaluar el rendimiento en el dataset completo
    accuracy_full = accuracy_score(y_full, y_pred_full)
    precision_full = precision_score(y_full, y_pred_full, average='binary')
    recall_full = recall_score(y_full, y_pred_full, average='binary')
    f1_full = f1_score(y_full, y_pred_full, average='binary')
    fpr_full, tpr_full, _ = roc_curve(y_full, y_pred_prob_full)
    roc_auc_full = auc(fpr_full, tpr_full)
    # Guardar los resultados en el archivo de texto
    output_file.write(f"Resultados para {model_name} en el dataset completo:\n")
    output_file.write(f"Accuracy: {accuracy_full:.4f}\n")
    output_file.write(f"Precision: {precision_full:.4f}\n")
    output_file.write(f"Recall: {recall_full:.4f}\n")
    output_file.write(f"F1 Score: {f1_full:.4f}\n")
    output_file.write(f"ROC AUC: {roc_auc_full:.4f}\n\n")

# Guardar gráficos
# Grafico antes de ajuste
plt.figure(figsize=(10, 5))
for model_name, metrics in results_before.items():
    plt.plot(metrics["FPR"], metrics["TPR"], label=f'{model_name} (AUC = {metrics["ROC AUC"]:.4f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curva ROC Comparativa antes de parámetros')
plt.legend(loc="lower right")
plt.savefig('hyperParams2_plots/roc_before.png')
plt.close()

# Grafico después
plt.figure(figsize=(10, 5))
for model_name, metrics in results_after.items():
    plt.plot(metrics["FPR"], metrics["TPR"], label=f'{model_name} (AUC = {metrics["ROC AUC"]:.4f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curva ROC Comparativa después de parámetros')
plt.legend(loc="lower right")
plt.savefig('hyperParams2_plots/roc_after.png')
plt.close()

# Crear el gráfico comparativo de la curva ROC antes y después del ajuste de hiperparámetros
plt.figure(figsize=(10, 5))
for model_name, metrics in results_before.items():
    plt.plot(metrics["FPR"], metrics["TPR"], label=f'{model_name} Before (AUC = {metrics["ROC AUC"]:.4f})')

for model_name, metrics in results_after.items():
    plt.plot(metrics["FPR"], metrics["TPR"], label=f'{model_name} After (AUC = {metrics["ROC AUC"]:.4f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curva ROC Comparativa antes y después del ajuste de hiperparámetros')
plt.legend(loc="lower right")
plt.savefig('hyperParams2_plots/roc_comparative.png')
plt.close()

feature_names = tfidf.get_feature_names_out()

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred, labels=[1, 0])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['Positivo', 'Negativo'], yticklabels=['Positivo', 'Negativo'])
    plt.title(f'Matriz de Confusión para {model_name}')
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.savefig(f'hyperParams2_plots/confusion_matrix_{model_name}.png')
    plt.close()

    # Cálculo de métricas
    VP = cm[0, 0]  # Verdaderos Positivos
    FN = cm[0, 1]  # Falsos Negativos
    FP = cm[1, 0]  # Falsos Positivos
    VN = cm[1, 1]  # Verdaderos Negativos

    accuracy = (VP + VN) / (VP + FP + FN + VN)
    precision = VP / (VP + FP) if (VP + FP) != 0 else 0
    recall = VP / (VP + FN) if (VP + FN) != 0 else 0
    specificity = VN / (VN + FP) if (VN + FP) != 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    output_file.write(f'Matriz de Confusión para {model_name}:\n')
    output_file.write(f'Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | Specificity: {specificity:.4f} | F1 Score: {f1:.4f}\n\n')

    # Palabras más importantes
    if hasattr(model, 'coef_'):
        coefs = model.coef_[0].toarray() if hasattr(model.coef_[0], 'toarray') else model.coef_[0]
    elif hasattr(model, 'feature_importances_'):
        coefs = model.feature_importances_
    else:
        continue

    # Asegurarse de que coefs y feature_names tengan la misma longitud
    if len(coefs) == len(feature_names):
        top_features = np.argsort(coefs)[-10:]
        top_weights = coefs[top_features]
        top_words = [feature_names[i] for i in top_features]

        plt.figure(figsize=(10, 6))
        sns.barplot(x=top_weights, y=top_words, orient='h')
        plt.title(f'Palabras más importantes para {model_name}')
        plt.xlabel('Peso')
        plt.ylabel('Palabras')
        plt.savefig(f'hyperParams2_plots/top_words_{model_name}.png')
        plt.close()
    else:
        output_file.write(f"Error: La longitud de coefs ({len(coefs)}) no coincide con la longitud de feature_names ({len(feature_names)}) para {model_name}\n")

# Cerrar el archivo de salida
output_file.close()