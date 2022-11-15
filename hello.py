from flask import Flask, render_template, request
import test

app = Flask(__name__)


@app.route('/')
def hello_world():
    return render_template('hello.html')


@app.route('/submit', methods=['POST', 'GET'])
def submit():
    input_sentence = request.form.get("sentence")
    predicted_sentence = test.translate(input_sentence.lower(), test.transformer)
    return predicted_sentence
