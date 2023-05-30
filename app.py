# app.py
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
app = Flask(__name__)
app.debug=True

# Main page
@app.route('/')
def start():
    return render_template('index.html')

@app.route('/door')
def door():
    return render_template('index2.html')

@app.route('/molding')
def molding():
    return render_template('index2.html')

@app.route('/drying')
def drying():
    lstm_ae = load_model('./data/lstm-ae1.h5')
    return render_template('index2.html')

@app.route('/ques_concern', methods=['GET', 'POST'])
def question_type():        
    return render_template('index3.html')


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)