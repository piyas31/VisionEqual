import os
from flask import Flask, request, render_template, send_file
import cv2
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'processed'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

def enhance_contrast(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    equ = cv2.equalizeHist(gray)
    return cv2.cvtColor(equ, cv2.COLOR_GRAY2BGR)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return "No file part", 400
    
    file = request.files['image']
    if file.filename == '':
        return "No selected file", 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    img = cv2.imread(filepath)
    processed_img = enhance_contrast(img)

    processed_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)
    cv2.imwrite(processed_path, processed_img)

    return send_file(processed_path, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
