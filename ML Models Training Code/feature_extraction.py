import os
import numpy as np
import tensorflow
from tqdm import tqdm
import librosa

# directory path
dirname = r"C:\pythonProject\Dataset\genres_original"

# Save audio paths and labels
audio_paths = []
audio_label = []

# Printing all the files in different directories
for root, dirs, files in os.walk(dirname, topdown=False):
    for filenames in files:
        if filenames.endswith('.wav'):
            audio_paths.append(os.path.join(root, filenames))
            label = filenames.split('.', 1)[0]
            audio_label.append(label)

audio_paths = np.array(audio_paths)
audio_label = np.array(audio_label)

# Preallocate arrays with dtype=np.float32 to save the features
AllSpec = np.empty((len(audio_paths), 1025, 1293), dtype=np.float32)  # Spectrogram
AllMel = np.empty((len(audio_paths), 128, 1293), dtype=np.float32)    # Mel-Spectrogram
AllMfcc = np.empty((len(audio_paths), 10, 1293), dtype=np.float32)    # MFCC
AllZcr = np.empty((len(audio_paths), 1293), dtype=np.float32)         # Zero-crossing rate
AllCen = np.empty((len(audio_paths), 1293), dtype=np.float32)         # Spectral centroid
AllChroma = np.empty((len(audio_paths), 12, 1293), dtype=np.float32)  # Chromagram

bad_index = []

for i in tqdm(range(len(audio_paths))):
    try:
        path = audio_paths[i]
        y, sr = librosa.load(path, dtype=np.float32)  # Ensure audio is loaded as float32
        # Compute features using librosa, ensuring all operations keep the data as float32
        X = librosa.stft(y)  # Spectrogram
        Xdb = librosa.amplitude_to_db(abs(X))
        AllSpec[i] = Xdb.astype(np.float32)

        M = librosa.feature.melspectrogram(y=y, sr=sr)
        M_db = librosa.power_to_db(M)
        AllMel[i] = M_db.astype(np.float32)

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10)
        AllMfcc[i] = mfcc.astype(np.float32)

        zcr = librosa.feature.zero_crossing_rate(y)[0]
        AllZcr[i] = zcr.astype(np.float32)

        sp_cen = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        AllCen[i] = sp_cen.astype(np.float32)

        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12, n_fft=4096)
        AllChroma[i] = chroma_stft.astype(np.float32)



    except Exception as e:
        bad_index.append(i)

# Delete data at corrupt indices
for array in [AllSpec, AllMel, AllMfcc, AllZcr, AllCen, AllChroma]:
    np.delete(array, bad_index, axis=0)

# Convert labels from string to numerical
genre_to_num = {'blues': 0, 'classical': 1, 'country': 2, 'disco': 3, 'hiphop': 4, 'jazz': 5, 'metal': 6, 'pop': 7, 'reggae': 8, 'rock': 9}
audio_label = np.array([genre_to_num[label] for label in audio_label if label not in bad_index])

# Convert labels from numerical to categorical data
y = tensorflow.keras.utils.to_categorical(audio_label, num_classes=10, dtype="int32")

# Save all the features and labels in a .npz file
np.savez_compressed(os.getcwd() + "/MusicFeatures.npz", spec=AllSpec, mel=AllMel, mfcc=AllMfcc, zcr=AllZcr, cen=AllCen, chroma=AllChroma, target=y)
