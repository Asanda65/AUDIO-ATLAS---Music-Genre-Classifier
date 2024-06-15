import numpy as np
import os
import tensorflow as tf

# Function to calculate the majority vote
def get_majority(preds):
    final_pred = np.array(preds)
    # Transpose to shape [samples, models] for easier majority voting
    final_pred = np.transpose(final_pred)
    # Apply majority voting on each sample
    majority_vote = np.array([np.argmax(np.bincount(sample)) for sample in final_pred])
    return majority_vote

# Load individual models
model_spectrogram = tf.keras.models.load_model(os.getcwd() + "/models/new_spec_model_spectrogram1.h5")
model_mel_spectrogram = tf.keras.models.load_model(os.getcwd() + "/models/final_melspectrogram_model.h5")
model_mfcc1 = tf.keras.models.load_model(os.getcwd() + "/models/normalized_new_ensemble_classifier_mfcc1.h5")
model_mfcc2 = tf.keras.models.load_model(os.getcwd() + "/models/normalized_new_ensemble_classifier_mfcc2.h5")
model_mfcc3 = tf.keras.models.load_model(os.getcwd() + "/models/normalized_new_ensemble_classifier_mfcc3.h5")

# Load data for spectrogram and mel-spectrogram models
spec_file = np.load(os.getcwd()+"/new_spectrogram_train_test.npz")
S_test = spec_file['S_test']

mel_file = np.load(os.getcwd() + "/new_mel_train_test.npz")
mel_test = mel_file['mel_test']

# Load data and labels for MFCC model (used across all models for simplicity in this example)
mfcc_file = np.load(os.getcwd() + "/new_mfcc_train_test.npz")
mfcc_test = mfcc_file['mfcc_test']
y_test = mfcc_file['y_test']  # Assuming the same y_test can be used to evaluate all models

# Make predictions with each model
y_pred_spectrogram = np.argmax(model_spectrogram.predict(S_test), axis=1)
y_pred_mel_spectrogram = np.argmax(model_mel_spectrogram.predict(mel_test), axis=1)
y_pred_mfcc1 = np.argmax(model_mfcc1.predict(mfcc_test), axis=1)
y_pred_mfcc2 = np.argmax(model_mfcc2.predict(mfcc_test), axis=1)
y_pred_mfcc3 = np.argmax(model_mfcc3.predict(mfcc_test), axis=1)

# Combine predictions from all models for ensemble decision
y_pred_combined = [y_pred_spectrogram, y_pred_mel_spectrogram, y_pred_mfcc1, y_pred_mfcc2, y_pred_mfcc3]

# Calculate ensemble prediction using majority voting
y_pred_ensemble = get_majority(y_pred_combined)

# Calculate ensemble accuracy
y_true = np.argmax(y_test, axis=1)  # Assuming the same y_test is used for all models
ensemble_accuracy = np.mean(y_pred_ensemble == y_true)
print(f"Ensemble Model Accuracy: {ensemble_accuracy * 100:.2f}%")
