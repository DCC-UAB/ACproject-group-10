[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/USx538Ll)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-2e0aaae1b6195c2367325f4f02e2d04e9abb55f0b24a779b69b11b9e10269abc.svg)](https://classroom.github.com/online_ide?assignment_repo_id=17281635&assignment_repo_type=AssignmentRepo)



## Preguntas Semana 1

1. Cargar, limpiar y preparar los datos. Que
2. Convertir texto a caracteristicas
3. Entrenar un modelo de clasificación
4. Comparar diferentes modelos en los mismos datos

### Respuestas
1. Simplificamos datos hasta tener solo 2 columnas: `overall` y `reviewtext`. Luego, eliminamos filas con valores nulos.
2. Utilizamos una funcion llamada clean_text hecha por nosotros para limpiar el texto. Luego, utilizamos TfidfVectorizer para convertir texto a caracteristicas.
3. Utilizamos un modelo de clasificación llamado `LogisticRegression` para entrenar el modelo. Tambien entrenamos de inicio con `RandomForestClassifier`, que consideramos seria el que tendria mejor funcionamiento antes de hacer pruebas
4. Ejecutamos muchos modelos diferentes y comparamos las matrices de confusion y la roc curve para determinar cual modelo es mejor. En este caso, el modelo `LogisticRegression` fue el mejor en un sistema de puntaje binario. Tenemos la idea de hacer pruebas prediciendo el valor exacto de cada review para la semana que viene
## Preguntas Semana 2
1. Entrenar Hiperparametros de un modelo
2. Solucionar problemas de desbalanceo de clases
3. Validar un modelo utilizando diferentes metricas
4. Seleccionar el mejor modelo
5. Realizar predicciones sobre nuevos datos

### Respuestas
1. Utilizamos GridSearchCV para entrenar hiperparametros de un modelo. En este caso, utilizamos `LogisticRegression` y `RandomForestClassifier` 
2. Directamente utilizamos pd.concat para concatenar los datos de train y test usando el total de datos negativos para escoger datos positovos random
3. Utilizamos diferentes metricas como `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `roc_auc_score` y `confusion_matrix` para validar el modelo
4. Utilizamos `LogisticRegression` como mejor modelo, ya que tuvo mejor desempeño en todas las metricas
5. Realizamos predicciones sobre nuevos datos utilizando el modelo `LogisticRegression`, `random forest` y `naive bayes`.  

## Instrucciones ejecucion final
En la carpeta `CODIGOEJECUCIONFINAL` se encuentran los archivos necesarios para ejecutar el codigo final. Para ejecutar el codigo, se debe correr el archivo `codigofinal.py` que deberia usar `amazon_reviews` y descargar `reviews.csv` para poder ejecutarse. 
Despues este mismo codigo ira almacenando los diferentes plots de ejemplo en 3 directorios diferentes para mostrar parte de lo que se ha hecho durante la practica. Realmente en el archivo de `Codigousadodurantepractica` esta el codigo que fuimos utilizando durante
la practica y la mayoria de  los plots que se fueron generando durante la misma. 

