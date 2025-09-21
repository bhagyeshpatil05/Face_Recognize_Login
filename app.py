from flask import Flask, render_template, request
import cv2
import mediapipe as mp
import torch
from facenet_pytorch import InceptionResnetV1
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

# Load FaceNet model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Load known embeddings
KNOWN_FOLDER = "known_users"
known_embeddings = []
known_names = []

mp_face = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.6)

for file in os.listdir(KNOWN_FOLDER):
    img_path = os.path.join(KNOWN_FOLDER, file)
    img = np.array(Image.open(img_path).convert('RGB'))
    results = mp_face.process(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    if results.detections:
        bbox = results.detections[0].location_data.relative_bounding_box
        h, w, _ = img.shape
        x1 = int(bbox.xmin * w)
        y1 = int(bbox.ymin * h)
        x2 = x1 + int(bbox.width * w)
        y2 = y1 + int(bbox.height * h)
        face_crop = img[y1:y2, x1:x2]
        face_crop = Image.fromarray(face_crop).resize((160,160))
        face_tensor = torch.tensor(np.array(face_crop)).permute(2,0,1).float()/255.0
        face_tensor = face_tensor.unsqueeze(0).to(device)
        embedding = resnet(face_tensor).detach().cpu()
        known_embeddings.append(embedding)
        known_names.append(file.split(".")[0])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/verify', methods=['POST'])
def verify():
    file = request.files['file']
    img = np.array(Image.open(file).convert('RGB'))
    results = mp_face.process(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    if not results.detections:
        return "❌ No face detected"
    
    bbox = results.detections[0].location_data.relative_bounding_box
    h, w, _ = img.shape
    x1 = int(bbox.xmin * w)
    y1 = int(bbox.ymin * h)
    x2 = x1 + int(bbox.width * w)
    y2 = y1 + int(bbox.height * h)
    face_crop = img[y1:y2, x1:x2]
    face_crop = Image.fromarray(face_crop).resize((160,160))
    face_tensor = torch.tensor(np.array(face_crop)).permute(2,0,1).float()/255.0
    face_tensor = face_tensor.unsqueeze(0).to(device)
    embedding = resnet(face_tensor).detach().cpu()

    # Compare with known embeddings
    min_dist = float('inf')
    name = "Unknown"
    for i, known_emb in enumerate(known_embeddings):
        dist = (embedding - known_emb).norm().item()
        if dist < 0.8 and dist < min_dist:
            min_dist = dist
            name = known_names[i]

    if name != "Unknown":
        return f"✅ Login successful: {name}"
    else:
        return "❌ Face not recognized"

if __name__ == "__main__":
    app.run(debug=True)
