import os
from flask import Flask, request, render_template, jsonify, send_from_directory
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import base64

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'processed'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

def image_to_base64(path):
    with open(path, "rb") as img_file:
        return "data:image/jpeg;base64," + base64.b64encode(img_file.read()).decode()

def enhance_contrast(img, alpha=1.0):
    # alpha = contrast factor
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.convertScaleAbs(l, alpha=alpha, beta=0)
    lab = cv2.merge((l,a,b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return enhanced

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def daltonize(img):
    # Simple Red-Green colorblind simulation (deuteranopia)
    # Matrix from colorblind simulation literature
    transform = np.array([[0.625, 0.7, 0],
                          [0.7,   0.625, 0],
                          [0,     0,     1]])
    img_float = img.astype(float) / 255.0
    img_cb = np.clip(np.dot(img_float, transform.T), 0, 1)
    img_cb = (img_cb * 255).astype(np.uint8)
    return img_cb

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    img = cv2.imread(filepath)
    if img is None:
        return jsonify({'error': 'Invalid image'}), 400

    # Save original image path
    original_path = filepath

    # Grayscale image
    gray_img = grayscale(img)
    gray_path = os.path.join(app.config['PROCESSED_FOLDER'], f"gray_{filename}")
    cv2.imwrite(gray_path, gray_img)

    # Daltonized image
    daltonized_img = daltonize(img)
    daltonized_path = os.path.join(app.config['PROCESSED_FOLDER'], f"daltonized_{filename}")
    cv2.imwrite(daltonized_path, daltonized_img)

    # Enhanced contrast image with default alpha=1 (you can make this dynamic)
    enhanced_img = enhance_contrast(img, alpha=1.0)
    enhanced_path = os.path.join(app.config['PROCESSED_FOLDER'], f"enhanced_{filename}")
    cv2.imwrite(enhanced_path, enhanced_img)

    # Convert images to base64 for frontend display
    data = {
        'original': image_to_base64(original_path),
        'grayscale': image_to_base64(gray_path),
        'daltonized': image_to_base64(daltonized_path),
        'enhanced': image_to_base64(enhanced_path),
    }

    return jsonify(data)


if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)

