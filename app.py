import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)

# ------------------------------
# Load TFLite model (super cepat)
# ------------------------------
interpreter = tf.lite.Interpreter(model_path="tb_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def preprocess(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
    return img_array

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("image")
    if file is None or file.filename == "":
        return "No file uploaded", 400

    upload_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(upload_path)

    # Preprocess
    img = image.load_img(upload_path, target_size=(224, 224))
    img_array = preprocess(img)

    # TFLite inferensi
    interpreter.set_tensor(input_details[0]["index"], img_array)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]["index"])[0][0]

    result = "TUBERCULOSIS" if prediction > 0.5 else "NORMAL"

    return render_template(
        "result.html",
        result=result,
        image_path="/" + upload_path
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
