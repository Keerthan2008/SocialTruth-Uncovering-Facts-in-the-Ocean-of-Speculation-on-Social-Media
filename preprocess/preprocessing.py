import json
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    text = ' '.join(filtered_tokens)
    return text

data = []
labels = []
rumour_dirs = ["charliehebdo", "ferguson", "germanwings-crash", "ottawashooting", "sydneysiege"]

for rumour_dir in rumour_dirs:
    rumour_path = os.path.join("D:\\project\\pheme-rnr-dataset", rumour_dir, "rumours")
    non_rumour_path = os.path.join("D:\\project\\pheme-rnr-dataset", rumour_dir, "non-rumours")

    rumour_files = os.listdir(rumour_path)
    for file in rumour_files:
        with open(os.path.join(rumour_path, file, "source-tweet", file + ".json"), 'r') as f:
            tweet = json.load(f)
            data.append(preprocess_text(tweet['text']))
            labels.append(1)  

    non_rumour_files = os.listdir(non_rumour_path)
    for file in non_rumour_files:
        with open(os.path.join(non_rumour_path, file, "source-tweet", file + ".json"), 'r') as f:
            tweet = json.load(f)
            data.append(preprocess_text(tweet['text']))
            labels.append(0) 

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

report = classification_report(y_test, y_pred, target_names=['Non-rumour', 'Rumour'])
print("Classification Report:")
print(report)