from flask import Flask, request, render_template, send_file, jsonify, url_for
import cv2
import numpy as np
import os
from io import BytesIO
from PIL import Image

app = Flask(__name__, static_url_path='/static')

# Directory to save uploaded images
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/uploads/<filename>')
def get_uploaded_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        print('No file part')
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        print('No selected file')
        return jsonify({'error': 'No selected file'}), 400
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        segmented_image_url, masked_image_url = process_image(filepath)
        return jsonify({
            "segmented_image": segmented_image_url,
            "masked_image": masked_image_url
        })

@app.route('/ourteam')
def our_team():
    return render_template('ourteam.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')    

def process_image(filepath):
    try:
        # read the image
        image = cv2.imread(filepath)
        if image is None:
            print(f"Failed to read image: {filepath}")
            return None, None

        # convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # reshape the image to a 2D array of pixels and 3 color values (RGB)
        pixel_values = image.reshape((-1, 3))
        # convert to float
        pixel_values = np.float32(pixel_values)

        # define stopping criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

        # number of clusters (K)
        k = 3
        compactness, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # convert back to 8 bit values
        centers = np.uint8(centers)

        # flatten the labels array
        labels = labels.flatten()

        # convert all pixels to the color of the centroids
        segmented_image = centers[labels]

        # reshape back to the original image dimension
        segmented_image = segmented_image.reshape(image.shape)

        # save segmented image
        segmented_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'segmented_image.png')
        Image.fromarray(segmented_image).save(segmented_image_path)

        # disable only the cluster number 2 (turn the pixel into black)
        masked_image = np.copy(image)
        # convert to the shape of a vector of pixel values
        masked_image = masked_image.reshape((-1, 3))
       
        cluster = 2
        masked_image[labels == cluster] = [0, 0, 0]

        masked_image = masked_image.reshape(image.shape)

        masked_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'masked_image.png')
        Image.fromarray(masked_image).save(masked_image_path)

        return url_for('get_uploaded_file', filename='segmented_image.png'), url_for('get_uploaded_file', filename='masked_image.png')
    except Exception as e:
        print(f"Error processing image: {e}")
        return None, None

if __name__ == '__main__':
    app.run(debug=True)
