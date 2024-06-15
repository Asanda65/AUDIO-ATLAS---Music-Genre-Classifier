import json
import numpy as np
import librosa
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# Define the path to your new audio file
new_audio_file_path = 'path/to/your/new/audio.wav'

# Load the trained model
model = load_model('genre_classifier_model.h5')

# Preprocess the new audio file
# (Make sure this preprocessing is the same as what you did for your training data)
y, sr = librosa.load(new_audio_file_path, sr=None)
y = librosa.util.normalize(y)
y, _ = librosa.effects.trim(y)

# Extract features
# (Make sure you extract the same features in the same way as your training data)
mfcc = librosa.feature.mfcc(y=y, sr=sr).mean(axis=1)
chroma = librosa.feature.chroma_stft(y=y, sr=sr).mean(axis=1)
contrast = librosa.feature.spectral_contrast(y=y, sr=sr).mean(axis=1)
tonnetz = librosa.feature.tonnetz(y=y, sr=sr).mean(axis=1)

features = np.array([mfcc, chroma, contrast, tonnetz]).reshape(1, -1)

# Use the model to make a prediction
# Assuming your model was trained on scaled features, you might need to scale the features of the new data
prediction = model.predict(features)

# If you saved the LabelEncoder, you can use it to decode the predicted genre
# Load your saved label encoder
with open('label_encoder.json', 'r') as fp:
    label_encoder = LabelEncoder()
    label_encoder.classes_ = json.load(fp)

# Decode the prediction
predicted_genre_index = np.argmax(prediction)
predicted_genre = label_encoder.inverse_transform([predicted_genre_index])

print(f'The predicted genre is: {predicted_genre[0]}')
