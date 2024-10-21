from flask import Flask, request, render_template, jsonify
import os
import numpy as np
import soundfile as sf
import librosa
from pydub import AudioSegment, effects
import noisereduce as nr
import matplotlib.pyplot as plt
import io
import base64
import tensorflow as tf

# Define paths to the model and weights
MODEL_PATH = 'models/model.json'
WEIGHTS_PATH = 'models/model_weights.h5'

# Load the model architecture
with open(MODEL_PATH, 'r') as json_file:
    json_savedModel = json_file.read()

# Recreate the model from the architecture
model = tf.keras.models.model_from_json(json_savedModel)

# Load the model's saved weights
model.load_weights(WEIGHTS_PATH)

# Compile the model (as you did during training)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['categorical_accuracy'])

app = Flask(__name__)

# Preprocess function from your existing code
def preprocess(audio_data, sr, frame_length=2048, hop_length=512):
    audio_segment = AudioSegment(
        audio_data.tobytes(),
        frame_rate=sr,
        sample_width=audio_data.dtype.itemsize,
        channels=1
    )
    normalizedsound = effects.normalize(audio_segment, headroom=5.0)
    normal_x = np.array(normalizedsound.get_array_of_samples(), dtype='float32')

    if not np.isfinite(normal_x).all():
        raise ValueError("Audio data contains NaN or infinite values.")

    final_x = nr.reduce_noise(y=normal_x, sr=sr)

    if not np.isfinite(final_x).all():
        raise ValueError("Post-noise-reduction audio data contains NaN or infinite values.")

    f1 = librosa.feature.rms(y=final_x, frame_length=frame_length, hop_length=hop_length, center=True).T
    f2 = librosa.feature.zero_crossing_rate(y=final_x, frame_length=frame_length, hop_length=hop_length, center=True).T
    f3 = librosa.feature.mfcc(y=final_x, sr=sr, n_mfcc=13, hop_length=hop_length).T
    X = np.concatenate((f1, f2, f3), axis=1)
    X_3D = np.expand_dims(X, axis=0)

    return X_3D

# Emotion mapping
emotions = {
    0: 'neutral',
    1: 'calm',
    2: 'happy',
    3: 'sad',
    4: 'angry',
    5: 'fearful',
    6: 'disgust',
    7: 'surprised'
}

# Route for homepage
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle file upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400

    if file and file.filename.endswith('.wav'):
        # Read the uploaded audio file
        audio_data, samplerate = sf.read(file)

        # Preprocess and predict emotions
        x = preprocess(audio_data, samplerate)
        predictions = model.predict(x, verbose=1)

        pred_list = list(predictions)
        pred_np = np.squeeze(np.array(pred_list).tolist(), axis=0)

        # Create a bar chart for the emotion distribution
        fig = plt.figure(figsize=(10, 2))
        plt.bar(list(emotions.values()), pred_np, color='darkturquoise')
        plt.ylabel("Probability (%)")
        plt.title("Emotion Distribution")

        # Save the plot to a string buffer and encode it as a base64 string
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()

        # Get the emotion with the highest probability
        max_emo = np.argmax(predictions)
        result = emotions.get(max_emo, "Unknown")

        # Return the result as JSON
        return jsonify({
            "result": result,
            "distribution_image": img_str
        })

    return "Invalid file format. Please upload a .wav file.", 400

if __name__ == '__main__':
    app.run(debug=True)
