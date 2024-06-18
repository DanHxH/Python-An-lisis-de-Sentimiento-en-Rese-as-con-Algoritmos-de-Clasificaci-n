import os
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import collections

# Cargar el archivo CSV
df = pd.read_csv('reviewText.csv')

# Crear un objeto PorterStemmer
stemmer = PorterStemmer()

# Importar palabras vacías en inglés
stop_words = set(stopwords.words('english'))

# Definir una expresión regular para eliminar signos de puntuación y números
re_punc = re.compile('[%s]' % re.escape(string.punctuation))
re_num = re.compile('\d')

# Preprocesar las reseñas
preprocessed_reviews = []
for i, row in df.iterrows():
    # Comprobar si la reseña está vacía
    if pd.isnull(row['reviewText']):
        continue

    # Convertir el texto a minúsculas
    text = row['reviewText'].lower()

    # Tokenizar el texto en palabras
    words = word_tokenize(text)

    # Aplicar la expresión regular para eliminar signos de puntuación y números
    stripped = [re_punc.sub('', w) for w in words]
    stripped = [re_num.sub('', w) for w in stripped]

    # Eliminar palabras vacías
    filtered_words = [word for word in stripped if word not in stop_words]

    # Aplicar el Porter Stemming a las palabras limpias
    stemmed_words = [stemmer.stem(word) for word in filtered_words]

    # Unir las palabras en una cadena de texto y eliminar los espacios dobles
    preprocessed_review = ' '.join(stemmed_words)
    preprocessed_review = re.sub(' +', ' ', preprocessed_review)

    # Guardar la reseña preprocesada en un archivo
    with open('reseñas_preprocesadas.txt', 'a', encoding='utf-8') as file:
        file.write(f"{i} | {preprocessed_review}\n")

    # Agregar la reseña preprocesada a la lista
    preprocessed_reviews.append(preprocessed_review)

# Leer las reseñas preprocesadas
with open('reseñas_preprocesadas.txt', 'r', encoding='utf-8') as file:
    reviews = file.read()

# Dividir las reseñas en palabras
words = reviews.split()

# Eliminar el símbolo '|'
words = [word for word in words if word != '|']

# Eliminar números
words = [word for word in words if not word.isdigit()]

# Ordenar las palabras alfabéticamente
words.sort()

# Contar las palabras
word_counts = collections.Counter(words)

# Imprimir la longitud del vocabulario y los 50 términos más comunes
print(f"Longitud del vocabulario: {len(word_counts)}")
print("50 términos más comunes:")
for word, count in word_counts.most_common(50):
    print(f"{word}: {count}")
print()

# Guardar el vocabulario en un archivo
with open('vocabulario.txt', 'w', encoding='utf-8') as file:
    for word, count in word_counts.items():
        file.write(f"{word}: {count}\n")

# Eliminar las reseñas vacías
df = df.dropna(subset=['reviewText'])

# Definir las reseñas positivas, neutrales y negativas
df['review_type'] = 'neutral'
df.loc[df['overall'] >= 4, 'review_type'] = 'positive'
df.loc[df['overall'] <= 2, 'review_type'] = 'negative'

# Guardar las opiniones etiquetadas en un archivo CSV
df_etiquetas = pd.DataFrame({
    'review': preprocessed_reviews,
    'etiqueta': df['review_type']
})
df_etiquetas.to_csv('reseñas_etiquetas.csv', index=False)

# Contar el número de opiniones positivas, negativas y neutrales
review_counts = df['review_type'].value_counts()

# Generar la gráfica de pastel
plt.figure(figsize=(6,6))
plt.pie(review_counts, labels = review_counts.index, autopct='%1.1f%%')
plt.title('Distribución de opiniones')
# Guardar la gráfica como una imagen
plt.savefig('grafica.png')
plt.show()

# Definir las características (X) y la variable objetivo (y)
X = df['reviewText']
y = df['review_type']

# Dividir los datos en un conjunto de entrenamiento y un conjunto de prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear un objeto TfidfVectorizer
vectorizer = TfidfVectorizer()

# Ajustar y transformar las reseñas de entrenamiento
X_train_tfidf = vectorizer.fit_transform(X_train)

# Transformar las reseñas de prueba
X_test_tfidf = vectorizer.transform(X_test)

# Crear los modelos
svm_model = svm.SVC()
dt_model = DecisionTreeClassifier()
lr_model = LogisticRegression()

# Entrenar y evaluar el modelo SVM
svm_model.fit(X_train_tfidf, y_train)
svm_predictions = svm_model.predict(X_test_tfidf)

# Entrenar y evaluar el modelo C4.5
dt_model.fit(X_train_tfidf, y_train)
dt_predictions = dt_model.predict(X_test_tfidf)

# Entrenar y evaluar el modelo de Regresión Logística
lr_model.fit(X_train_tfidf, y_train)
lr_predictions = lr_model.predict(X_test_tfidf)

# Separar las reseñas positivas y negativas
positive_reviews = [review for review, label in zip(preprocessed_reviews, df['review_type']) if label == 'positive']
negative_reviews = [review for review, label in zip(preprocessed_reviews, df['review_type']) if label == 'negative']

# Crear una cadena de texto con todas las reseñas positivas y negativas
positive_text = ' '.join(review for review in positive_reviews)
negative_text = ' '.join(review for review in negative_reviews)

# Crear las nubes de palabras
positive_wordcloud = WordCloud(width=800, height=800, max_words=500 , max_font_size=100).generate(positive_text)
negative_wordcloud = WordCloud(width=800, height=800, max_words=500 , max_font_size=100).generate(negative_text)

# Guardar las nubes de palabras como imágenes
positive_wordcloud.to_file('nube_positiva.png')
negative_wordcloud.to_file('nube_negativa.png')

# Mostrar la nube de palabras de las reseñas positivas
plt.figure(figsize=(10,10))
plt.imshow(positive_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.title('Reseñas positivas')
plt.show()

# Mostrar la nube de palabras de las reseñas negativas
plt.figure(figsize=(10,10))
plt.imshow(negative_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.title('Reseñas negativas')
plt.show()

# Definir una función para evaluar un modelo y devolver la exactitud
def evaluate_model(predictions, model_name, y_positive, y_negative):
    # Filtrar las etiquetas de prueba y las predicciones solo para opiniones positivas y negativas
    y_test_filtered = y_test[(y_test == y_positive) | (y_test == y_negative)]
    predictions_filtered = predictions[(y_test == y_positive) | (y_test == y_negative)]

    # Calcular la matriz de confusión, precisión, recuerdo, medida F y exactitud
    cm = confusion_matrix(y_test_filtered, predictions_filtered, labels=[y_positive, y_negative])
    precision = precision_score(y_test_filtered, predictions_filtered, average='weighted', zero_division=0)
    recall = recall_score(y_test_filtered, predictions_filtered, average='weighted', zero_division=0)
    f1 = f1_score(y_test_filtered, predictions_filtered, average='weighted', zero_division=0)
    accuracy = accuracy_score(y_test_filtered, predictions_filtered)

    # Imprimir los resultados
    print(f"{model_name} Métricas:")
    print(f"Exactitud: {accuracy}")
    print(f"Matriz de confusión:\n{cm}")
    report = classification_report(y_test_filtered, predictions_filtered, zero_division=1, output_dict=True)
    print("Informe de clasificación:")
    print("              precision    recall  f1-score   support")
    for label, scores in report.items():
        if label in [y_negative, y_positive]:
            print(f"{label:11} {scores['precision']:>10.2f} {scores['recall']:>9.2f} {scores['f1-score']:>9.2f} {scores['support']:>8}")

    # Guardar los resultados en un archivo
    with open(f'{model_name}_metrics.txt', 'w') as f:
        f.write(f"{model_name} Métricas:\n")
        f.write(f"Exactitud: {accuracy}\n")
        f.write(f"Matriz de confusión:\n{cm}\n")
        f.write("Informe de clasificación:\n")
        f.write("              precision    recall  f1-score   support\n")
        for label, scores in report.items():
            if label in [y_negative, y_positive]:
                f.write(f"{label:11} {scores['precision']:>10.2f} {scores['recall']:>9.2f} {scores['f1-score']:>9.2f} {scores['support']:>8}\n")

    # Devolver la exactitud
    return accuracy


# Evaluar los modelos y guardar las exactitudes
svm_accuracy = evaluate_model(svm_predictions, 'SVM', 'positive', 'negative')
print()

dt_accuracy = evaluate_model(dt_predictions, 'C4.5', 'positive', 'negative')
print()

lr_accuracy = evaluate_model(lr_predictions, 'Regresión Logística', 'positive', 'negative')
print()


# Crear un DataFrame para almacenar las métricas de exactitud
accuracy_df = pd.DataFrame({
    'Clasificador': ['SVM', 'C4.5', 'Regresion Logistica'],
    'Exactitud': [svm_accuracy, dt_accuracy, lr_accuracy]
})

# Imprimir el DataFrame en formato de tabla
print("\nClasificador       Exactitud")
for i in range(len(accuracy_df)):
    print(f"{accuracy_df['Clasificador'][i]:<20} {accuracy_df['Exactitud'][i]}")

# Guardar el DataFrame en un archivo CSV
accuracy_df.to_csv('exactitud_modelos.csv', index=False)

# Determinar el mejor clasificador
best_model = accuracy_df.loc[accuracy_df['Exactitud'].idxmax(), 'Clasificador']
best_accuracy = accuracy_df['Exactitud'].max()

# Imprimir el mejor clasificador
print("\nEl mejor clasificador fue: {} con un valor de exactitud de: {}".format(best_model, best_accuracy))
