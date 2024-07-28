from flask import Flask, request, render_template
import joblib

app = Flask(__name__)
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form['message']
    input_tfidf = vectorizer.transform([user_input])
    sentiment = model.predict(input_tfidf)
    return render_template('index.html', prediction=sentiment[0])

if __name__ == '__main__':
    app.run(debug=True)
