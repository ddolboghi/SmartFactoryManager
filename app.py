# app.py
from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from sklearn.metrics import *
import pandas as pd
import numpy as np
import os
# 시각화
import seaborn as sns
import matplotlib.pyplot as plt

#도어용
import io
import torch
from PIL import Image
# 몰딩용
from werkzeug.utils import secure_filename
import random   

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


RESULT_FOLDER = os.path.join('static')
app.config['RESULT_FOLDER'] = RESULT_FOLDER


# yolo
def find_model():
    model_name = 'best.pt'
    yolo_path = os.path.abspath(model_name)
    if os.path.exists(yolo_path):
        return yolo_path
    else:
        print(f"Model file '{model_name}' not found in the current directory!")
        return None

def get_prediction(img_bytes):
    img = Image.open(io.BytesIO(img_bytes))
    imgs = [img]
    print(imgs)
    results = yolo_model(imgs, size=640)

    labels_and_probs = []
    for result in results.pred[0]:
        label = int(result[-1])
        prob = round(float(result[-2])* 100, 2)
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
        filename = './static/images/image0.jpg'

        id_to_class = {0: '스크래치 불량', 1: '외관손상 불량'}

        class_names_and_probs = []
        for label, prob in labels_and_probs:
            class_name = id_to_class.get(label, '정상적인 도어입니다.')
            class_names_and_probs.append((class_name, prob))

        return render_template('./door/result1.html', result_image=filename, model_name=yolo_name, class_names_and_probs=class_names_and_probs)

    return render_template('./door/index.html')

torch.hub._validate_not_a_forked_repo = lambda a, b, c: True

yolo_name = find_model()

if yolo_name is not None:
    yolo_model = torch.hub.load('WongKinYiu/yolov7', 'custom', path_or_model=yolo_name)
    yolo_model.eval()


# 승훈
model_path = os.path.join(app.root_path, 'templates', 'molding', 'model.h5')
model = load_model(model_path)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/molding', methods=['GET', 'POST'])
def windshield_molding():
    result_data = pd.read_csv("./static/data/molding_result.csv")

    # Convert DataFrame to a list of dictionaries
    data = result_data.to_dict('records')
    random.shuffle(data)

    return render_template('molding/results.html', data=data)


# 재윤 
@app.route('/temp')
def temp():
    result_data = pd.read_csv("./static/data/result.csv")

    # Convert DataFrame to a list of dictionaries
    data = result_data.to_dict('records')
    random.shuffle(data)

    # Render the result in an HTML template
    return render_template('sound/temp.html', data=data)


# 오준
@app.route('/drying')
def drying():
    result_data = pd.read_csv("./static/data/sound_result.csv")

    # Convert DataFrame to a list of dictionaries
    data = result_data.to_dict('records')
    random.shuffle(data)

    # Render the result in an HTML template
    return render_template('sound/sound.html', data=data)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)