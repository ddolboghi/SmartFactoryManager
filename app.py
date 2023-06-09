# app.py
from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import *
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
#도어용
import io
import torch
from PIL import Image
# 몰딩용
from werkzeug.utils import secure_filename


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
    return render_template('main.html')


# 민경
RESULT_FOLDER = os.path.join('static')
app.config['RESULT_FOLDER'] = RESULT_FOLDER

def find_model():
    model_name = 'best.pt'
    model_path = os.path.abspath(model_name)
    if os.path.exists(model_path):
        return model_path
    else:
        print(f"Model file '{model_name}' not found in the current directory!")
        return None

def get_prediction(img_bytes):
    img = Image.open(io.BytesIO(img_bytes))
    imgs = [img]
    results = model(imgs, size=640)

    labels_and_probs = []
    for result in results.pred[0]:
        label = int(result[-1])
        prob = float(result[-2])
        labels_and_probs.append((label, prob))

    # If no defects detected, consider the image as normal
    if not labels_and_probs:
        labels_and_probs.append(('정상사진입니다.', 1.0))  # You can change the probability to any value you want

    return results, labels_and_probs


@app.route('/door', methods=['GET', 'POST'])
def door():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return

        img_bytes = file.read()
        results, labels_and_probs = get_prediction(img_bytes)
        results.save(save_dir='static')
        filename = 'image0.jpg'

        id_to_class = {0: '스크래치 불량입니다.', 1: '외관손상 불량입니다.'}

        class_names_and_probs = []
        for label, prob in labels_and_probs:
            class_name = id_to_class.get(label, '해당 이미지는 정상입니다.')
            class_names_and_probs.append((class_name, prob))

        return render_template('./door/result1.html', result_image=filename, model_name=model_name, class_names_and_probs=class_names_and_probs)

    return render_template('./door/index.html')

torch.hub._validate_not_a_forked_repo = lambda a, b, c: True

model_name = find_model()

if model_name is not None:
    model = torch.hub.load('WongKinYiu/yolov7', 'custom', path_or_model=model_name)
    model.eval()

# 승훈
model_path = os.path.join(app.root_path, 'templates', 'molding', 'model.h5')
model = load_model(model_path)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/molding', methods=['GET', 'POST'])
def windshield_molding():
    return render_template('molding/index3.html')

@app.route('/results', methods=['POST'])
def results():
    if 'file' not in request.files:
        return redirect(url_for('molding'))

    files = request.files.getlist('file')

    results = []

    for file in files:
        image = Image.open(file)
        # 이미지 처리
        processed_image = preprocess_image(image)
        # 예측
        result = model.predict(processed_image)
        result = result.flatten()
        results.append({'image_file': file.filename, 'prediction': result})

    return render_template('molding/results.html', results=results)


def preprocess_image(image):
    # 이미지 전처리 로직을 구현하세요
    # 예시: 이미지 크기 조정, 정규화 등
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def predict(image):
    # 모델 예측
    prediction = model.predict(image)
    return prediction

def file_exists(file_path):
    return os.path.exists(file_path)

# Register the custom filter in the Jinja2 environment
app.jinja_env.filters['file_exists'] = file_exists

# 재윤 
@app.route('/temp')
def temp():
    ## 온도 데이터
    ### data ~~~~ 전처리 
    TIME_STEP=36
    def create_sequences(X, y, time_steps=TIME_STEP):
        Xs, ys = [], []
        for i in range(len(X)-time_steps):
            Xs.append(X.iloc[i:(i+time_steps)].values)
            ys.append(y.iloc[i:(i+time_steps)].values)
            # ys.append(y.iloc[i+time_steps])
        return np.array(Xs), np.array(ys)
    
    def flatten(X):
        # 다차원 -> 1차원 변환 함수
        flattened_X = np.empty((X.shape[0], X.shape[2]))  
        for i in range(X.shape[0]):
            flattened_X[i] = X[i, (X.shape[1]-1), :]
        return (flattened_X)
    
    test_data = pd.read_csv("./test_data.csv")
    # learn_data =  실시간 학습 구현을 위한 학습데이터 불러오기 
    test_data = test_data[:2016]
    X_test, Y_test = create_sequences(test_data[['Temp']], test_data[['NG']])
    lstm_ae1 = load_model('./lstm-ae1.h5')
    prediction = lstm_ae1.predict(X_test) 
    
    label = flatten(Y_test).reshape(-1)
    
    mse = np.mean(np.power(X_test - prediction, 2), axis=1)
    error_df = pd.DataFrame({'reconstruction_error': mse.reshape(-1),
                            'true_class':label}) 
    best_thr = np.percentile(mse.reshape(-1),93)

    pred_y = ["불량" if e > best_thr else "정상" for e in error_df['reconstruction_error'].values]
    temp_values = test_data['Temp'][:len(pred_y)]
    process_values = test_data['Process'][:len(pred_y)]
    result = {temp_values[i]: {'process': process_values[i], 'prediction': pred_y[i]} for i in range(len(pred_y))}

    # Render the result in an HTML template
    return render_template('sound/temp.html', result=result)







# 오준
@app.route('/drying', methods=['GET', 'POST'])
def drying():
    ## 팬 사운드 데이터
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

        with open('./model.dtc', 'rb') as file:
            loaded_model = pickle.load(file)

        pred_y = loaded_model.predict(data)

        return render_template('/sound/sound.html', pred_y=pred_y, ments=ments, mfcc_min = mfcc_min, mfcc_max = mfcc_max, spectrum_min = spectrum_min)

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