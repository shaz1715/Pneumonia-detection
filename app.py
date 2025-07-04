from flask import Flask, request, render_template, jsonify, redirect, url_for, flash
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import json

app = Flask(__name__)
app.secret_key = "neha_secret"  # required for flash messages

model = tf.keras.models.load_model("model.h5")

def preprocess_image(image):
    image = image.convert('L').resize((150, 150))
    img_array = np.expand_dims(np.array(image), axis=(0, -1)) / 255.0
    return img_array

@app.route('/', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            image = Image.open(file.stream)
            input_data = preprocess_image(image)
            prediction = model.predict(input_data)[0][0]
            label = "Pneumonia" if prediction < 0.5 else "No Pneumonia"

            with open("latest_result.json", "w") as f:
                json.dump({"result": label}, f)

            flash(f"Prediction: {label}")
            return redirect(url_for('upload'))

    return render_template("index.html")

@app.route('/latest_prediction', methods=['GET'])
def get_latest():
    if os.path.exists("latest_result.json"):
        with open("latest_result.json") as f:
            return jsonify(json.load(f))
    return jsonify({"result": "No prediction yet"})

if __name__ == '_main_':
    app.run(host='0.0.0.0', port=5000,debug=True)
