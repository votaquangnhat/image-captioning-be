from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from PIL import Image
from inference import infer

app = Flask(__name__)
CORS(app)

@app.route("/caption", methods=["POST"])
def generate_caption():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "Empty filename"}), 400

        image = Image.open(file.stream).convert("RGB")
        
        with torch.no_grad():
            caption = infer(image)

        return jsonify({"caption": caption}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Ensure the server runs on port 5000
    app.run(host="0.0.0.0", port=5000, debug=True)