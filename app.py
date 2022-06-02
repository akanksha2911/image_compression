from ast import Global
from fileinput import filename
from flask import Flask, flash, request, redirect, send_file, url_for, render_template
import os
from werkzeug.utils import secure_filename
import cv2 as cv
import numpy as np
from PIL import Image as im

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def K_means(img):
    if len(img.shape) < 3:
        Z = img.reshape((-1, -1))
    elif len(img.shape) == 3:
        Z = img.reshape((-1, 3))
    K = 200
    Z = np.float32(Z)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv.kmeans(
        Z, K, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    Clustered_img = res.reshape(img.shape)
    return Clustered_img


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        file.filename = 'orgimg.jpg'
        filename = file.filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        img = cv.imread('static/uploads/orgimg.jpg', 1)
        compimg = K_means(img)
        path = 'static/uploads'
        cv.imwrite(os.path.join(path, 'compimg.jpg'), compimg)
        compfilename = os.path.basename('static/uploads/compimg.jpg')
        print('upload_image filename: ' + compfilename)
        flash('Image successfully uploaded and displayed below')
        return render_template('index.html', filename=compfilename)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)


@app.route('/downloads')
def download_image():
    p = 'static/uploads/compimg.jpg'
    return send_file(p, as_attachment=True)


@app.route('/display/<filename>')
def display_image(filename):
    # print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


if __name__ == "__main__":
    app.run()
