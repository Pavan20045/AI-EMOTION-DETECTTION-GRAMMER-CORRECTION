Voice Analysis Suite: Emotion Detection & Grammar Correction

This project is an integrated web-based **speech analysis system** that includes:

- 🔊 **Emotion Detection** from audio using deep learning
- ✍️ **Grammar Correction** of transcribed text
- 🏠 A unified **home page** for navigation

---

## 📌 Project Structure
AI-POWERED-SPEECH-ANALYSIS-AND-COMMUNICATION-ENHANCEMENT/
│
├── templates/
│ ├── home(main).html # Home page interface
│ ├── index4(main).html # Emotion Detection frontend
│ └── public.html # Grammar Correction frontend
│
├── uploads/
│ ├── input_original/ # Folder to hold raw inputs
│ └── input.wav # Current audio file being processed
│
├── .env # Environment config (e.g., API keys)
├── .gitignore # Git ignore file
├── app.py # Emotion Detection backend (Flask)
├── emotion_model.pth # Trained PyTorch model for emotion
├── server.js # Grammar Correction backend (Node.js)
├── setupffmpeg.py # FFmpeg installation helper
├── package.json # Node.js dependencies
├── package-lock.json # Node.js dependency lock
├── requirements.txt # Python dependencies
└── README.md # Project documentation


---

## 🧠 Features

### 🔊 Emotion Detection
- **Frontend**: `index4(main).html`
- **Backend**: `app.py` (Flask + PyTorch)
- Uses Wav2Vec2 + custom classifier (`emotion_model.pth`)
- Supports `.wav` audio input
- Real-time emotion classification

### ✍️ Grammar Correction
- **Frontend**: `public.html`
- **Backend**: `server.js` (Node.js + LanguageTool or grammar API)
- Provides corrections to spoken/transcribed input

### 🏠 Home Page
- **File**: `home(main).html`
- Acts as a navigation hub between Emotion and Grammar modules

---

⚙️ Installation
git clone https://github.com/Pavan20045/AI-EMOTION-DETECTTION-GRAMMER-CORRECTION.git
cd AI-EMOTION-DETECTTION-GRAMMER-CORRECTION

pip install -r requirements.txt

npm install

choco install ffmpeg (or) python setupffmpeg.py

python app.py

node server.js

