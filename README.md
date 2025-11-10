🧠 Speech Emotion Recognition using Deep Learning (LSTM + CNN)

This project focuses on building a Speech Emotion Recognition (SER) system using Deep Learning techniques — specifically Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM) networks. The goal is to detect human emotions such as happy, sad, angry, neutral, etc. from audio speech signals.

🚀 Project Overview

The model takes raw audio signals as input, extracts meaningful Mel-Frequency Cepstral Coefficients (MFCCs) features, and classifies them into predefined emotion categories.
It combines:

CNN layers for spatial feature extraction from spectrograms.

LSTM layers for temporal feature learning, capturing the sequence of emotional patterns over time.

🧩 Key Features

Audio preprocessing using Librosa for MFCC and spectrogram extraction

CNN + LSTM hybrid model for accurate emotion detection

Support for common datasets like RAVDESS, TESS, and CREMA-D

Model evaluation with accuracy, confusion matrix, and emotion classification report

Real-time emotion prediction capability (optional extension)

🛠️ Tech Stack

Python, TensorFlow / Keras, NumPy, Librosa, Matplotlib, Scikit-learn

📊 Model Architecture

Feature Extraction: Convert audio to MFCCs or Mel Spectrograms

CNN Layers: Capture local frequency-time dependencies

LSTM Layers: Learn temporal emotional context

Dense Layers: Perform final emotion classification

📈 Results

Achieved promising accuracy and generalization across multiple emotion categories.
Further improvements can be done using data augmentation, attention mechanisms, or transfer learning with pre-trained audio embeddings.

💡 Future Scope

Integration with real-time speech systems or chatbots

Emotion-based human-computer interaction

Expansion to multilingual emotion recognition

📁 Repository Structure
├── data/                 # Audio dataset
├── features/             # Extracted MFCC features
├── models/               # Trained CNN-LSTM models
├── notebooks/            # Jupyter notebooks for training/testing
├── utils/                # Helper functions
└── main.py               # Main script for training and inference
