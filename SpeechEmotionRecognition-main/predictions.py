"""This module makes predictions on new voice recordings."""

import numpy as np
import joblib
from preprocessing import extract_features_single


def make_predictions(file_path):
    """
    This function takes in a file path and makes predictions on it.
    """
    # Extract features from the audio file
    features = extract_features_single(file_path)
    if features is None:
        raise Exception("Failed to extract features from the audio file")
    
    features = features.reshape(1, -1)  # Reshape for sklearn models

    try:
        # Load models
        lstm_model = joblib.load("models/lstm_model.joblib")
        cnn_model = joblib.load("models/cnn_model.joblib")
        
        # Make predictions
        lstm_pred = lstm_model.predict(features)[0]
        cnn_pred = cnn_model.predict(features)[0]

        # Probabilities / confidence
        lstm_proba = None
        cnn_proba = None
        if hasattr(lstm_model, "predict_proba"):
            try:
                lstm_proba = lstm_model.predict_proba(features)[0]
            except Exception:
                lstm_proba = None
        if hasattr(cnn_model, "predict_proba"):
            try:
                cnn_proba = cnn_model.predict_proba(features)[0]
            except Exception:
                cnn_proba = None

        # Define emotion labels
        emotions = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgusted", "surprised"]
        
        # Get predicted emotions
        lstm_emotion = emotions[lstm_pred]
        cnn_emotion = emotions[cnn_pred]

        # Confidence for chosen class (0-1 float)
        lstm_conf = float(lstm_proba[int(lstm_pred)]) if lstm_proba is not None else None
        cnn_conf = float(cnn_proba[int(cnn_pred)]) if cnn_proba is not None else None
        
        # Suggest a video URL matching the detected emotion (no interactive prompt)
        emotion_videos = {
            'neutral': 'https://www.youtube.com/watch?v=kRauhbZqJCY',
            'calm': 'https://www.youtube.com/watch?v=Zljg2ptExHc',
            'happy': 'https://www.youtube.com/watch?v=srYPJYgDaj8',
            'sad': 'https://www.youtube.com/watch?v=EvDQBIisG7c',
            'angry': 'https://www.youtube.com/watch?v=7D3zpOBRN9c',
            'fearful': 'https://www.youtube.com/watch?v=fcLl-DZGLZ8',
            'disgusted': 'https://www.youtube.com/watch?v=UM7ydNEK68w',
            'surprised': 'https://www.youtube.com/watch?v=JNQU-4YEnm4'
        }

        suggested_url = emotion_videos.get(lstm_emotion, emotion_videos['neutral'])

        return {
            "lstm_prediction": lstm_emotion,
            "lstm_confidence": lstm_conf,
            "cnn_prediction": cnn_emotion,
            "cnn_confidence": cnn_conf,
            "suggested_video": suggested_url
        }
        
    except Exception as e:
        print(f"Error making predictions: {str(e)}")
        return {
            "lstm_prediction": "error",
            "cnn_prediction": "error"
        }

if __name__ == "__main__":
    work_rec = "recordings/findahappyplace.wav"
    predictions = make_predictions(file_path=work_rec)
    print("\nPredictions:")
    print(f"LSTM Model predicts: {predictions['lstm_prediction']}")
    print(f"CNN Model predicts: {predictions['cnn_prediction']}")