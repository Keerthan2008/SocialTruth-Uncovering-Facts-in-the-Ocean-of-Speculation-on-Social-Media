from flask import Flask, render_template, request, redirect, url_for
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

app = Flask(__name__)

model = joblib.load('svm_model.joblib')
vectorizer = joblib.load('vectorizer.joblib')

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    text = ' '.join(filtered_tokens)
    return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        preprocessed_text = preprocess_text(text)
        vectorized_text = vectorizer.transform([preprocessed_text])
        prediction = model.predict(vectorized_text)[0]
        if prediction == 1:
            return redirect(url_for('rumour_result', text=text))
        else:
            return redirect(url_for('non_rumour_result', text=text))

@app.route('/rumour_result')
def rumour_result():
    text = request.args.get('text', '')
    return render_template('rumour_result.html', text=text)

@app.route('/non_rumour_result')
def non_rumour_result():
    text = request.args.get('text', '')
    return render_template('non_rumour_result.html', text=text)

if __name__ == '__main__':
    app.run(debug=True)
