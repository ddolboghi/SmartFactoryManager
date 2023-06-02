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

import pickle
from sklearn import metrics
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
@app.route('/drying', methods=['GET', 'POST'])
def drying():
    if request.method == 'POST':
        audio_files = request.files.getlist('audio')
  

        df = pd.DataFrame(columns=['mfcc_min', 'mfcc_max', 'spectrum_min', 'spectrum_max'])
        df['NG'] = 0

        spectrum_min = []
        spectrum_max = []
        mfcc_min = []
        mfcc_max = []
        ng = []

        for audio_file in audio_files:
            err, ser = librosa.load(audio_file, sr=100)
            left_spectrum, left_f_ng = mk_Frequency(err, ser)
            mfcc = librosa.feature.mfcc(y=err, sr=ser)
            
            if 'error' in audio_file.filename:
                ng.append(1)
            else:
                ng.append(0)

            spectrum_min.append(min(left_spectrum))
            spectrum_max.append(max(left_spectrum))
            mfcc_min.append(mfcc.min())
            mfcc_max.append(mfcc.max())

        df['spectrum_min'] = spectrum_min
        df['spectrum_max'] = spectrum_max
        df['mfcc_min'] = mfcc_min
        df['mfcc_max'] = mfcc_max
        df['NG'] = ng

        data = df.iloc[:, :-2]
        # target = df.iloc[:, -1:]
        
        ments = []
        
        if mfcc.min() < -411 :
            ments.append("mfcc_min")
        elif ((mfcc.max() > 33) and mfcc.max() < 20 ):
            ments.append("mfcc_max")            
            
        with open('C:\\SmartFactoryManager\\model.dtc', 'rb') as file:
            loaded_model = pickle.load(file)

        pred_y = loaded_model.predict(data)
       
        
        return render_template('/sound/sound.html', pred_y = pred_y, ments = ments)


    return render_template('/sound/upload.html')


def mk_Frequency(y, sr):
    fft = np.fft.fft(y)
    magnitude = np.abs(fft)
    fre = np.linspace(0, sr, len(magnitude))
    haf_spectrum = magnitude[:int(len(magnitude)/2)]
    haf_fre = fre[:int(len(magnitude)/2)]
    return haf_spectrum, haf_fre







    
    
@app.route('/ques_concern', methods=['GET', 'POST'])
def question_type():        
    return render_template('index3.html')


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)