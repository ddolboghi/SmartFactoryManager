# app.py
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
# 데이터 분석 툴
import pandas as pd
import numpy as np
# 운영체계와 관련된 툴
import os
import glob
# 시각화
import seaborn as sns
import matplotlib.pyplot as plt # matplotlib
import librosa

# 경고 방지
import warnings # 워닝 방지
warnings.filterwarnings('ignore')

plt.figure(figsize=(15,15))
plt.style.use('ggplot')

app = Flask(__name__)
app.debug=True

# Main page
@app.route('/')
def start():
    return render_template('index.html')

# 민경
@app.route('/door')
def door():
    return render_template('drying/index2.html')

# 승훈
@app.route('/molding')
def molding():
    return render_template('molding/index3.html')

# 오준, 재윤
@app.route('/drying')
def drying():
    lstm_ae = load_model('./data/lstm-ae1.h5')
    return render_template('sound/dataframe.html', table=df.to_html())

@app.route('/ques_concern', methods=['GET', 'POST'])
def question_type():        
    return render_template('index3.html')


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)