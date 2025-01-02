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
