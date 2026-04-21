from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import io
import base64
import numpy as np
import scipy.ndimage

app = Flask(__name__)
CORS(app)


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Linear(28 * 28, 256)
        self.second = nn.Linear(256, 128)
        self.third = nn.Linear(128, 64)
        self.fourth = nn.Linear(64, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.first(x))
        x = self.dropout(x)
        x = self.relu(self.second(x))
        x = self.dropout(x)
        x = self.relu(self.third(x))
        x = self.dropout(x)
        x = self.fourth(x)
        return x


# Load model
model = Classifier()
model.load_state_dict(torch.load("../digit_recognizer_model.pth", map_location=torch.device("cpu")))
model.eval()


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        image_data = data.get("image", "")

        # Decode base64 image
        if "," in image_data:
            image_data = image_data.split(",")[1]

        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert("L")

        img_array = np.array(image, dtype=np.float32)
        
        # The background in frontend is #0d0d15 (grey-ish dark).
        # We need to threshold to remove it so background is purely 0.
        img_array[img_array < 50] = 0
        
        # Find the bounding box of the drawn digit
        non_empty_columns = np.where(img_array.max(axis=0) > 0)[0]
        non_empty_rows = np.where(img_array.max(axis=1) > 0)[0]
        
        if len(non_empty_columns) > 0 and len(non_empty_rows) > 0:
            cropBox = (min(non_empty_rows), max(non_empty_rows), min(non_empty_columns), max(non_empty_columns))
            img_cropped = img_array[cropBox[0]:cropBox[1]+1, cropBox[2]:cropBox[3]+1]
            
            # Make the crop a square by padding the smaller dimension
            h, w = img_cropped.shape
            size = max(h, w)
            img_square = np.zeros((size, size), dtype=np.float32)
            y_offset = (size - h) // 2
            x_offset = (size - w) // 2
            img_square[y_offset:y_offset+h, x_offset:x_offset+w] = img_cropped
            
            # Resize the squared digit to 20x20 using Lanczos
            img_square_pil = Image.fromarray(img_square).resize((20, 20), Image.LANCZOS)
            img_20 = np.array(img_square_pil, dtype=np.float32)
            
            # Pad to 28x28 (adds exactly 4 pixels of black border on each side to match MNIST format)
            img_28 = np.pad(img_20, pad_width=4, mode='constant', constant_values=0)
            
            # Compute center of mass and shift image
            cy, cx = scipy.ndimage.center_of_mass(img_28)
            shift_y = np.round(14.0 - cy).astype(int)
            shift_x = np.round(14.0 - cx).astype(int)
            img_28 = scipy.ndimage.shift(img_28, (shift_y, shift_x), cval=0.0)
        else:
            # Handle empty canvas case
            img_28 = np.zeros((28, 28), dtype=np.float32)

        # Normalize pixel values to [0, 1]
        # MNIST images are mostly 0 or 1, and the brightest pixel is usually ~1.0
        # If the drawing gets faint due to resizing, this restores its strength
        max_val = img_28.max()
        if max_val > 0:
            img_28 = img_28 / max_val
        img_28 = np.clip(img_28, 0, 1)

        # Convert to tensor and add batch/channel dimensions
        tensor = torch.tensor(img_28).unsqueeze(0).unsqueeze(0)  # [1, 1, 28, 28]

        # Predict
        model.eval()
        with torch.no_grad():
            logits = model(tensor)
            probs = F.softmax(logits, dim=1)
            predicted = torch.argmax(probs, dim=1).item()
            confidence = probs[0][predicted].item()

        return jsonify({
            "prediction": predicted,
            "confidence": round(confidence * 100, 2),
            "probabilities": [round(p.item() * 100, 2) for p in probs[0]]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
