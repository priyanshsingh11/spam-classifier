from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    try:
        input_text = request.form['text']
        tokens = tokenizer.transform([input_text])
        prediction = model.predict(tokens)
        result = "Spam" if prediction[0] == 1 else "Not Spam"
        return jsonify({'input': input_text, 'classification': result})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
