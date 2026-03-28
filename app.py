import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

# =====================================
# INITIALIZE FLASK
# =====================================
app = Flask(__name__)

# =====================================
# FIX FOR softmax_v2 ERROR
# =====================================
custom_objects = {
    "softmax_v2": tf.nn.softmax
}

# =====================================
# LOAD MODEL (JSON + WEIGHTS)
# =====================================
with open("model_vgg.json", "r") as json_file:
    loaded_model_json = json_file.read()

model = model_from_json(loaded_model_json, custom_objects=custom_objects)
model.load_weights("model_vgg.weights.h5")

print("✅ Model loaded successfully!")

# =====================================
# CLASS LABELS
# =====================================
CLASS_NAMES = [
    "Black Soil",
    "Cinder Soil",
    "Laterite Soil",
    "Peat Soil",
    "Yellow Soil"
]

# =====================================
# SOIL → CROPS MAPPING
# =====================================
SOIL_CROPS = {
    "Black Soil": {
        "vegetables": ["Tomato", "Onion", "Chili"],
        "fruits": ["Grapes", "Orange", "Pomegranate"]
    },
    "Cinder Soil": {
        "vegetables": ["Potato", "Carrot"],
        "fruits": ["Apple", "Pear"]
    },
    "Laterite Soil": {
        "vegetables": ["Brinjal", "Okra"],
        "fruits": ["Cashew", "Mango"]
    },
    "Peat Soil": {
        "vegetables": ["Lettuce", "Spinach"],
        "fruits": ["Blueberry", "Strawberry"]
    },
    "Yellow Soil": {
        "vegetables": ["Cabbage", "Beans"],
        "fruits": ["Banana", "Papaya"]
    }
}

# =====================================
# UPLOAD FOLDER CONFIGURATION
# =====================================
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# =====================================
# HOME PAGE
# =====================================
@app.route("/")
def home():
    return render_template("index.html")


# =====================================
# PREDICTION ROUTE
# =====================================
@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return "No file uploaded"

    file = request.files["file"]

    if file.filename == "":
        return "No selected file"

    # Secure filename
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    # =====================================
    # IMAGE PREPROCESSING
    # =====================================
    img = image.load_img(filepath, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # =====================================
    # MODEL PREDICTION
    # =====================================
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    predicted_class = CLASS_NAMES[class_index]
    confidence = float(np.max(prediction)) * 100

    # =====================================
    # GET CROPS BASED ON SOIL
    # =====================================
    crops = SOIL_CROPS.get(predicted_class, {})
    vegetables = crops.get("vegetables", [])
    fruits = crops.get("fruits", [])

    # =====================================
    # RETURN RESULT PAGE
    # =====================================
    return render_template(
        "result.html",
        soil=predicted_class,
        confidence=round(confidence, 2),
        vegetables=vegetables,
        fruits=fruits,
        image_path=filepath
    )


# =====================================
# RUN APP
# =====================================
if __name__ == "__main__":
    app.run(debug=True)
