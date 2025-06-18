import os
import warnings
from flask import Flask, request, render_template, jsonify
from pydub import AudioSegment
from pydub.utils import which
import librosa
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
from transformers.utils import logging as hf_logging

# Suppress warnings and logs
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
hf_logging.set_verbosity_error()

# Set ffmpeg path
AudioSegment.converter = which("ffmpeg")

# Initialize Flask app
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model and processor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/wav2vec2-base", num_labels=7)
model.load_state_dict(torch.load("emotion_model.pth", map_location=device))
model.to(device)  # type: ignore
model.eval()

# Emotion labels
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
    return render_template('index4(main).html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio_data' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    file = request.files['audio_data']
    original_path = os.path.join(UPLOAD_FOLDER, "input_original")
    file.save(original_path)

    try:
        audio = AudioSegment.from_file(original_path)
        audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        file_path = os.path.join(UPLOAD_FOLDER, "input.wav")
        audio.export(file_path, format="wav")
    except Exception as e:
        return jsonify({'error': f'Audio conversion error: {str(e)}'}), 500

    try:
        waveform, _ = librosa.load(file_path, sr=16000)
    except Exception as e:
        return jsonify({'error': f'Error loading audio: {str(e)}'}), 500

    try:
        inputs = processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True, truncation=True, max_length=32000)
        input_values = inputs['input_values'].to(device)

        with torch.no_grad():
            outputs = model(input_values)
            pred = torch.argmax(outputs.logits, dim=-1).item()
    except Exception as e:
        return jsonify({'error': f'Model inference error: {str(e)}'}), 500

    label = label_map.get(int(pred), "Unknown")
    return jsonify({'prediction': label})

if __name__ == "__main__":
    app.run(debug=False)
