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
    err_Name='C:/final_prj/dataset_sound/data/FAN_sound_error'
    ok_Name='C:/final_prj/dataset_sound/data/FAN_sound_OK'

    df=pd.DataFrame(columns=['mfcc_min','mfcc_max','spectrum_min','spectrum_max'])
    df['NG']=0
     
    def mk_Frequency(y,sr):
        fft=np.fft.fft(y)
        magnitude=np.abs(fft)
        fre=np.linspace(0,sr,len(magnitude))
        haf_spectrum=magnitude[:int(len(magnitude)/2)]
        haf_fre=fre[:int(len(magnitude)/2)]
        return haf_spectrum, haf_fre
    
    spectrum_min=list() 
    spectrum_max=list() 
    mfcc_min=list() 
    mfcc_max=list()
    ng=list()
    a=0 
    b=0 
    for n in range(183):
        if (len(glob.glob(err_Name+'/*'))-1) >= n:
            path=glob.glob(ok_Name+'/*')[a]
            err,ser=librosa.load(path,sr=100)
            left_spectrum,left_f_ng=mk_Frequency(err,ser)
            mfcc=librosa.feature.mfcc(y=err,sr=ser)
            ng.append(1)
            a+=1
        else:
            path=glob.glob(ok_Name+'/*')[b]
            okr,skr=librosa.load(path,sr=100)
            left_spectrum,left_f_ok=mk_Frequency(okr,skr)
            mfcc=librosa.feature.mfcc(y=okr,sr=skr)
            ng.append(0)
            b+=1
            
        spectrum_min.append(min(left_spectrum))
        spectrum_max.append(max(left_spectrum))
        mfcc_min.append(mfcc.min())
        mfcc_max.append(mfcc.max())
    df['spectrum_min']=spectrum_min 
    df['spectrum_max']=spectrum_max
    df['mfcc_min']=mfcc_min 
    df['mfcc_max']=mfcc_max
    df['NG']=ng
    
    return render_template('sound/dataframe.html', table=df.to_html())

@app.route('/ques_concern', methods=['GET', 'POST'])
def question_type():        
    return render_template('index3.html')


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)