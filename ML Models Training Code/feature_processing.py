import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import os

# Load the .npz file of features
f = np.load(os.getcwd() + "/MusicFeatures.npz")
S = f['spec']
mfcc = f['mfcc']
mel = f['mel']
chroma = f['chroma']
y = f['target']

# Split train-test data
S_train, S_test, mfcc_train, mfcc_test, mel_train, mel_test, chroma_train, chroma_test, y_train, y_test = train_test_split(
    S, mfcc, mel, chroma, y, test_size=0.2)

# Process Spectrogram - feature scaling for uniformity
maximum1 = np.amax(S_train)
S_train /= maximum1
S_test /= maximum1

S_train = S_train.reshape((S_train.shape[0], S_train.shape[1], S_train.shape[2], 1))
S_test = S_test.reshape((S_test.shape[0], S_test.shape[1], S_test.shape[2], 1))

# Process MFCC
newtrain_mfcc = np.empty((mfcc_train.shape[0], 120, 600))
newtest_mfcc = np.empty((mfcc_test.shape[0], 120, 600))

for i in range(mfcc_train.shape[0]):
    curr = cv2.resize(mfcc_train[i], (600, 120))
    newtrain_mfcc[i] = curr

for i in range(mfcc_test.shape[0]):
    curr = cv2.resize(mfcc_test[i], (600, 120))
    newtest_mfcc[i] = curr

mfcc_train = newtrain_mfcc.reshape((newtrain_mfcc.shape[0], 120, 600, 1))
mfcc_test = newtest_mfcc.reshape((newtest_mfcc.shape[0], 120, 600, 1))

# Normalize MFCC
mean_data = np.mean(mfcc_train)
std_data = np.std(mfcc_train)
mfcc_train = (mfcc_train - mean_data) / std_data
mfcc_test = (mfcc_test - mean_data) / std_data

# Process Mel-Spectrogram
maximum2 = np.amax(mel_train)
mel_train /= maximum2
mel_test /= maximum2

mel_train = mel_train.reshape((mel_train.shape[0], mel_train.shape[1], mel_train.shape[2], 1))
mel_test = mel_test.reshape((mel_test.shape[0], mel_test.shape[1], mel_test.shape[2], 1))

# Save Spectrogram train-test
np.savez_compressed(os.getcwd() + "/new_spectrogram_train_test.npz", S_train=S_train, S_test=S_test, y_train=y_train, y_test=y_test)

# Save MFCC train-test
np.savez_compressed(os.getcwd() + "/new_mfcc_train_test.npz", mfcc_train=mfcc_train, mfcc_test=mfcc_test, y_train=y_train, y_test=y_test)

# Save Mel-Spectrogram train-test
np.savez_compressed(os.getcwd() + "/new_mel_train_test.npz", mel_train=mel_train, mel_test=mel_test, y_train=y_train, y_test=y_test)
