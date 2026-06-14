# 🎙️ EmoWave — Speech Emotion Classifier

[![Hugging Face Space](https://img.shields.io/badge/🤗%20Hugging%20Face-EmoWave-orange)](https://huggingface.co/spaces/Avik128/EmoWave)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-red)
![License](https://img.shields.io/badge/License-MIT-green)

> Audio-based speech emotion recognition pipeline. MFCC + SVM baseline → EfficientNet transfer learning on Mel Spectrograms. 7 emotion classes, 4000+ audio clips, deployed to Hugging Face.

---

## 🚀 Live Demo

[![Hugging Face Space](https://img.shields.io/badge/🤗%20Hugging%20Face-EmoWave-orange)](https://huggingface.co/spaces/Avik128/EmoWave)

Try it live → [https://huggingface.co/spaces/Avik128/EmoWave](https://huggingface.co/spaces/Avik128/EmoWave)

---

## 🏆 Results

| Model | Dataset | Accuracy |
|---|---|---|
| SVM + 40 MFCC features | RAVDESS only | 68.75% |
| SVM + 40 MFCC features | RAVDESS + TESS | **88.15%** |
| Scratch CNN (3 conv blocks) | RAVDESS + TESS | 54.00% |
| EfficientNet (transfer learning) | RAVDESS + TESS | **88.51%** ✅ |

**Key insight:** Data quantity matters more than model complexity. Adding TESS to RAVDESS jumped SVM from 68.75% → 88.15% without changing the model at all.

---

## 🎭 Detectable Emotions

| Emotion | Description |
|---|---|
| 😡 Angry | Raised voice, harsh tone |
| 😨 Fearful | Trembling, high pitch |
| 😊 Happy | Bright, energetic tone |
| 😐 Neutral | Flat, monotone delivery |
| 😢 Sad | Low energy, slow speech |
| 🤢 Disgusted | Tense, low growling tone |
| 😲 Surprised | Sharp pitch changes |

---

## 📁 Project Structure

```
EmoWave/
│
├── data/
│   ├── Actor_01/ ... Actor_24/        ← RAVDESS audio files
│   ├── TESS Toronto emotional.../     ← TESS audio files
│   └── combined_emotion_data.csv      ← merged dataset metadata
│
├── Notebook/
│   ├── 01_exploration.ipynb           ← EDA, waveforms, spectrograms, MFCCs
│   ├── 02_classical_ml.ipynb          ← MFCC features + SVM/RF/XGBoost
│   ├── 03_cnn.ipynb                   ← Scratch CNN on Mel Spectrograms
│   └── 04_efficientnet.ipynb          ← EfficientNet transfer learning
│
├── models/
│   ├── svm_model.pkl                  ← trained SVM (probability=True)
│   ├── scaler.pkl                     ← StandardScaler
│   ├── label_encoder.pkl              ← LabelEncoder (number → emotion)
│   └── emowave_efficientnet1.pth      ← fine-tuned EfficientNet weights
│
├── app/
│   ├── app.py                         ← Gradio app
│   └── requirements.txt
│
└── README.md
```

---

## 🧠 How It Works

### Stage 1 — Data Collection & Exploration (notebook 01)

EmoWave uses two datasets combined:

**RAVDESS** — 1,440 clips, 24 professional actors (12M/12F), 8 emotions, two intensity levels. Labels encoded in filename (e.g. `03-01-05-02-01-01-12.wav` → position 3 = emotion code).

**TESS** — 2,800 clips, 2 female actresses, 7 emotions, word-level utterances. Labels encoded in folder name (e.g. `OAF_angry/` → angry).

After combining and dropping `calm` (TESS doesn't have it, only 196 RAVDESS samples):

```
Combined dataset: ~4,000 clips
Classes: 7 emotions
Balance: ~500+ clips per class
```

Audio representations explored:
- **Waveform** — raw pressure measurements at 44,100 samples/second
- **Mel Spectrogram** — 2D frequency-energy-time image (CNN input)
- **MFCCs** — 13 coefficient fingerprint of sound texture (SVM input)

---

### Stage 2 — Classical ML Pipeline (notebook 02)

40 features extracted per clip:

```
13 MFCC means        → average frequency texture
13 MFCC std devs     → temporal variation
12 Chroma means      → pitch class information
1  Spectral centroid → brightness of sound
1  Zero crossing rate → noisy vs tonal
= 40 features total
```

Models compared:

| Model | RAVDESS only | RAVDESS + TESS |
|---|---|---|
| SVM (RBF kernel) | 68.75% | 88.15% |
| Random Forest | 62.50% | 88.15% |
| XGBoost | 60.42% | 86.79% |

SVM with RBF kernel wins on both datasets — handles small, well-scaled continuous feature spaces better than tree-based models.

---

### Stage 3 — Scratch CNN (notebook 03)

```
Input: [batch, 1, 128, 130]   ← 3 second Mel Spectrogram
Conv2d(1→32) + BN + ReLU + MaxPool  →  [batch, 32, 64, 65]
Conv2d(32→64) + BN + ReLU + MaxPool →  [batch, 64, 32, 32]
Conv2d(64→128) + BN + ReLU + MaxPool → [batch, 128, 16, 16]
Flatten → Linear(32768→512) → Dropout(0.5) → Linear(512→7)
```

Result: **54%** — too little data for training from random weights.

Data augmentation applied (noise, time shift, pitch shift) — didn't help enough. Root cause: CNNs need more data than 4,000 clips to learn from scratch.

---

### Stage 4 — EfficientNet Transfer Learning (notebook 04)

EfficientNet-B0 pretrained on ImageNet adapted for audio:

```
Modification 1: first Conv2d → accept 1-channel grayscale spectrogram
Modification 2: classifier → Linear(1280 → 7)

Phase 1 (epochs 1–30):
  Freeze backbone → train classifier + first conv only → 22%*

Phase 2 (epochs 31–60):
  Unfreeze all layers → fine-tune with lr=0.0001 → 88.51% ✅
```

*Phase 1 was low due to SSL issue blocking pretrained weight download — resolved by disabling SSL verification.

The pretrained ImageNet features (edges, textures, patterns) transfer well to spectrogram images — same visual pattern recognition, different domain.

---

## 📊 Key Findings

**1. Data beats model complexity**
Adding TESS to RAVDESS improved SVM by +19.4% — bigger gain than switching from SVM to CNN.

**2. Transfer learning rescues small datasets**
EfficientNet with pretrained weights reached 88.51% on 4,000 clips. Scratch CNN only reached 54% on the same data.

**3. Domain shift is real**
Testing on CREMA-D (unseen dataset) showed lower accuracy — angry clips sometimes predicted as disgusted. Both emotions have similar acoustic profiles (raised voice, tension). Adding CREMA-D to training would improve cross-dataset generalization.

**4. Calm is hard**
RAVDESS `calm` (196 clips) was dropped — too few samples, TESS doesn't have it, acoustically similar to neutral.

---

## 🗃️ Datasets

**RAVDESS** — Ryerson Audio-Visual Database of Emotional Speech and Song
- 1,440 files, 24 actors, 8 emotions, 2 intensities
- Download: [Kaggle](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)

**TESS** — Toronto Emotional Speech Set
- 2,800 files, 2 female actresses, 7 emotions
- Download: [Kaggle](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess)

---

## ⚙️ Setup

```bash
conda create -n emotionnet python=3.10 -y
conda activate emotionnet
python -m pip install librosa soundfile numpy pandas scikit-learn matplotlib tqdm joblib
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python -m pip install gradio xgboost
```

---

## 🚀 Run The App Locally

```bash
cd app
python app.py
```

Upload any `.wav` speech clip and choose between EfficientNet or SVM model.

---

## 🔮 What's Next

- Add CREMA-D dataset to improve cross-dataset generalization
- Grad-CAM visualization to see which spectrogram regions activate per emotion
- Real-time microphone input in the Gradio app
- Multi-label emotion detection (mixed emotions)

---

## 🛠️ Tech Stack

Python · PyTorch · librosa · scikit-learn · EfficientNet-B0 · Gradio · XGBoost

---

## 👤 Author

**Avik Sarkar** — B.Tech CSE AI/ML, Brainware University (2024–2028)

GitHub: [aviksarkar0204-stack](https://github.com/aviksarkar0204-stack)
Hugging Face: [Avik128](https://huggingface.co/Avik128)

---

## 📌 Related Projects

- [EchoNet](https://github.com/aviksarkar0204-stack/EchoNet) — Environmental sound classification (ESC-50, 50 classes, 74.5% accuracy)
- [BirdCLEF](https://github.com/aviksarkar0204-stack/BirdCLEF) — Bird species detection from audio (in progress)
