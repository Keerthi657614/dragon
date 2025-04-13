import os
import gdown  # Install using: pip install gdown
from flask import Flask, request, render_template, flash, redirect, url_for, send_file
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import numpy as np
import tempfile

# Flask app setup
app = Flask(__name__, template_folder='templates')
app.secret_key = "supersecretkey"
UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure necessary directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Google Drive Model Download Setup
MODEL_PATH = "inception_model.h5"
GOOGLE_DRIVE_FILE_ID = "1xfT5dmRTnw6hzhft-xKYUZv7rH2ZcRSq"

def download_model():
    """Download the model from Google Drive if not found locally."""
    if not os.path.exists(MODEL_PATH):
        print("Model not found locally. Downloading from Google Drive...")
        url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
        print("Download complete.")
    else:
        print("Model already exists.")

# Load or create InceptionV3 model
def create_inceptionv3_model():
    """Create an InceptionV3 model if no pretrained model is found."""
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Freeze base layers
    for layer in base_model.layers:
        layer.trainable = False
    
    return model

# Ensure model is available before loading
download_model()

# Load the model
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    print("Model not found. Creating new InceptionV3 model...")
    model = create_inceptionv3_model()
    print("Please upload your trained model weights if available.")

# File extension checker
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Image preprocessing for InceptionV3
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(299, 299))  # Resize to 299x299 for InceptionV3
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)  # Apply InceptionV3 preprocessing
    return img_array

# Routes
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)

        # Save the file temporarily using tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            file.save(temp_file.name)
            file_path = temp_file.name  # The temporary file path

            # Preprocess and predict
            img_array = preprocess_image(file_path)
            prediction = model.predict(img_array)[0][0]
            label = "Fresh" if prediction > 0.5 else "Defect"
            confidence = prediction if prediction > 0.5 else 1 - prediction

            # Serve the image dynamically and display prediction
            return render_template('result.html', 
                                   filename=filename, 
                                   prediction=label, 
                                   confidence=f"{confidence:.4f}", 
                                   image_path=file_path)
    else:
        flash('Allowed file types are png, jpg, jpeg')
        return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)
