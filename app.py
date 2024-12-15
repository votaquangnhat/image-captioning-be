from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torchvision
from PIL import Image
from inference import generateCaption

app = Flask(__name__)
CORS(app)

model_list = ['dumb', 'trans', 'transpre']

@app.route('/')
def home():
    image = Image.open("test1.jpg").convert("RGB")
    captions = dict.fromkeys(model_list, [])
    with torch.no_grad():
        for model in captions.keys():
            try:
                captions[model] = generateCaption(image, model)
            except:
                captions[model] = 'Error!!!'

    return jsonify({"message": "system is working", 
                    "dumbModelCaption": captions['dumb'],
                    "transModelCaption": captions['trans'],
                    "transpreModelCaption": captions['transpre']}), 200

@app.route("/caption", methods=["POST"])
def generate_caption():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "Empty filename"}), 400
        
        model = request.form['model']

        image = Image.open(file.stream).convert("RGB")
        
        with torch.no_grad():
            try:
                caption = generateCaption(image, model)
            except:
                caption = 'Cannot use torchvision now'
            print(f'Caption: {caption}')

        return jsonify({'model': model, "caption": caption}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Ensure the server runs on port 5000
    app.run(host="0.0.0.0", port=5000, debug=True)