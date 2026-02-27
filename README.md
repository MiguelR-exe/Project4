# 🎥 Reconocimiento de Actividades Humanas (UCI-HAR) con Machine Learning

## 👥 Integrantes
- Meza Leon, Ricardo Manuel  
- Ramos Bonilla, Miguel Angel 
- Cabezas Ramirez, Dylan Andres

---

## 📌 Descripción del Proyecto

Este proyecto implementa y visualiza distintos modelos de **Machine Learning** aplicados al dataset **UCI HAR (Human Activity Recognition)**.

El objetivo es clasificar actividades humanas (caminar, subir escaleras, sentarse, pararse, acostarse, etc.) utilizando datos reales de sensores de smartphones.

Además del entrenamiento de modelos, el proyecto genera una **visualización animada con Manim** donde se explica paso a paso cómo funciona cada algoritmo y cómo se comparan sus resultados.

---

## 🧠 Dataset Utilizado

Se utiliza el **UCI HAR Dataset**, el cual contiene:

- Datos de 30 voluntarios
- Ventanas de 2.56 segundos (128 lecturas a 50Hz)
- 561 features por muestra (dominio del tiempo y frecuencia)
- 6 actividades:
  - Caminar
  - Subir escaleras
  - Bajar escaleras
  - Sentarse
  - Pararse
  - Acostarse

Cada muestra es un vector de 561 características extraídas de señales de acelerómetro y giroscopio.

---

## ⚙️ Modelos Implementados

Se entrenaron y compararon los siguientes modelos:

- Regresión Logística
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM con kernel RBF)
- Random Forest
- Multi-Layer Perceptron (MLP)

Todos los modelos fueron evaluados con:

- Accuracy
- Precision
- Recall
- F1-Score
- Matriz de Confusión

---

## 🔄 Pipeline del Proyecto

1. Carga del dataset
2. Normalización de datos con `StandardScaler`
3. Reducción de dimensionalidad con `PCA` (visualización en 2D)
4. Entrenamiento de modelos
5. Evaluación con métricas reales
6. Generación de animación explicativa con Manim

---

## 📊 Tecnologías Utilizadas

- Python
- NumPy
- Scikit-learn
- Manim
- Joblib

---

## ▶️ Cómo Ejecutar el Proyecto

### 1️⃣ Descargar el Dataset

Descargar el **UCI HAR Dataset** y colocarlo en la raíz del proyecto con la siguiente estructura:
UCI HAR Dataset/
train/
test/

---

### 2️⃣ Instalar Dependencias

```bash
pip install numpy scikit-learn manim joblib
manim -pqh archivo.py HARVideo
```
## 🎯 Objetivo Académico

Este proyecto fue desarrollado con fines educativos para:

- Comprender el funcionamiento interno de distintos algoritmos de clasificación.  
- Comparar modelos lineales y no lineales.  
- Visualizar conceptos como:  
  - PCA  
  - Margen máximo en SVM  
  - Voto en Random Forest  
  - Vecinos en KNN  
  - Forward pass y Backpropagation en MLP  
- Analizar métricas reales sobre un dataset del mundo real.  

---

## 📌 Conclusión

El proyecto demuestra cómo diferentes enfoques de Machine Learning pueden resolver el mismo problema con distintos niveles de complejidad y rendimiento, permitiendo analizar sus ventajas y limitaciones utilizando datos reales y visualización interactiva.
