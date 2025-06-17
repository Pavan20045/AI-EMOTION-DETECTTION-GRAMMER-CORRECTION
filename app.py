from flask import Flask, request, render_template, jsonify
from pydub import AudioSegment
import librosa
import torch
import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# === Load model ===
device = "cuda" if torch.cuda.is_available() else "cpu"

# Use Hugging Face processor if local one is not available
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/wav2vec2-base", num_labels=7)
model.load_state_dict(torch.load("emotion_mod.pth", map_location=device))  # corrected name
model.to(device)
model.eval()

label_map = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happiness",
    4: "Neutral",
    5: "Confidence",
    6: "Sadness"
}

@app.route('/')
def index():
    return render_template('index4.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['audio_data']
    file_path = os.path.join(UPLOAD_FOLDER, "input.wav")
    file.save(file_path)
    print("file_path",file_path)
    # Convert to proper format
    audio = AudioSegment.from_file(file_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(file_path, format="wav")

    waveform, _ = librosa.load(file_path, sr=16000)
    inputs = processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True, truncation=True, max_length=32000)
    input_values = inputs.input_values.to(device)

    with torch.no_grad():
        outputs = model(input_values)
        pred = torch.argmax(outputs.logits, dim=-1).item()

    label = label_map[pred]
    return jsonify({'prediction': label})

if __name__ == "__main__":
    app.run(debug=True)







