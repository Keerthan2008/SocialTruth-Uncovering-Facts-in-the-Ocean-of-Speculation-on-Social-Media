import json
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import joblib


# Define function for text preprocessing
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    # Join tokens back into string
    text = ' '.join(filtered_tokens)
    return text

# Read data and preprocess
data = []
labels = []
rumour_dirs = ["charliehebdo", "ferguson", "germanwings-crash", "ottawashooting", "sydneysiege"]

for rumour_dir in rumour_dirs:
    rumour_path = os.path.join("D:\\project\\pheme-rnr-dataset", rumour_dir, "rumours")
    non_rumour_path = os.path.join("D:\\project\\pheme-rnr-dataset", rumour_dir, "non-rumours")

    # Read rumour files
    rumour_files = os.listdir(rumour_path)
    for file in rumour_files:
        with open(os.path.join(rumour_path, file, "source-tweet", file + ".json"), 'r') as f:
            tweet = json.load(f)
            data.append(preprocess_text(tweet['text']))
            labels.append(1)  # Rumour label

    # Read non-rumour files
    non_rumour_files = os.listdir(non_rumour_path)
    for file in non_rumour_files:
        with open(os.path.join(non_rumour_path, file, "source-tweet", file + ".json"), 'r') as f:
            tweet = json.load(f)
            data.append(preprocess_text(tweet['text']))
            labels.append(0)  # Non-rumour label

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = SVC()
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

report = classification_report(y_test, y_pred, target_names=['Non-rumour', 'Rumour'])
print("Classification Report:")
print(report)

joblib.dump(model, 'D:\\project\\svm_model.joblib')

joblib.dump(vectorizer, 'vectorizer.joblib')

