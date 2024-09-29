from flask import Flask, request, jsonify
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow CORS for frontend-backend communication

# Load BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

@app.route("/caption", methods=["POST"])
def generate_caption():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        image = Image.open(image_file)
        inputs = processor(images=image, return_tensors="pt")

        # Generate caption
        output = model.generate(**inputs)
        caption = processor.decode(output[0], skip_special_tokens=True)
        
        return jsonify({"caption": caption})

    except Exception as e:
        return jsonify({"error": f"Failed to generate caption: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
