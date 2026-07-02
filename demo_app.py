
import base64
import io
import os
import tempfile

from flask import Flask, request, jsonify, send_from_directory

from predict import score_image

app = Flask(__name__)


@app.route("/")
def index():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), "demo.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    img_b64 = data["image"].split(",")[1] 
    img_bytes = base64.b64decode(img_b64)

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        f.write(img_bytes)
        tmp_path = f.name

    try:
        score = score_image(tmp_path)
    finally:
        os.remove(tmp_path)

    return jsonify({"score": score})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
