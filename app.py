# app.py
from flask import Flask, request, render_template, redirect
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

#도어용
import io
import torch
from PIL import Image


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
@app.route('/molding')
def molding():
    return render_template('molding/index3.html')

# 오준, 재윤
@app.route('/drying')
def drying():
    lstm_ae = load_model('./data/lstm-ae1.h5')
    return render_template('sound/dataframe.html')

@app.route('/ques_concern', methods=['GET', 'POST'])
def question_type():        
    return render_template('index3.html')


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)