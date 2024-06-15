import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold

# Ensure the models directory exists
models_dir = os.path.join(os.getcwd(), "models")
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# Load MFCC file
mfcc_file = np.load(os.getcwd() + "/new_mfcc_train_test.npz")
mfcc_train = mfcc_file['mfcc_train']
mfcc_test = mfcc_file['mfcc_test']
y_train = mfcc_file['y_train']
y_test = mfcc_file['y_test']

# Define model for MFCC
def get_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), input_shape=mfcc_train[0].shape, activation='tanh', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((4, 6), padding='same'),
        tf.keras.layers.Conv2D(32, (3, 3), activation='tanh', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((4, 6), padding='same'),
        tf.keras.layers.Conv2D(64, (3, 3), activation='tanh', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((4, 6), padding='same'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='tanh'),
        tf.keras.layers.Dense(64, activation='tanh'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Function to get the majority vote
def get_majority(preds):
    majority_vote = []
    for i in range(len(preds[0])):
        # Count the votes for each class
        votes = [pred[i] for pred in preds]
        # Choose the class with the most votes
        majority_vote.append(np.bincount(votes).argmax())
    return np.array(majority_vote)

# Train and save three models using KFold cross-validation
for i in range(1, 4):
    model = get_model()
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    for train_index, val_index in kf.split(mfcc_train):
        kf_mfcc_train, kf_X_val = mfcc_train[train_index], mfcc_train[val_index]
        kf_y_train, kf_y_val = y_train[train_index], y_train[val_index]
        model.fit(kf_mfcc_train, kf_y_train, validation_data=(kf_X_val, kf_y_val), epochs=30, batch_size=30, verbose=1)
    model_path = os.path.join(models_dir, f"normalized_new_ensemble_classifier_mfcc{i}.h5")
    model.save(model_path)

# Load the trained models
models = [tf.keras.models.load_model(os.path.join(models_dir, f"normalized_new_ensemble_classifier_mfcc{i}.h5")) for i in range(1, 4)]

# Making predictions on the training set
y_preds_train = [np.argmax(model.predict(mfcc_train), axis=-1) for model in models]
y_pred_train_majority = get_majority(y_preds_train)
y_true_train = np.argmax(y_train, axis=1)
train_accuracy = np.sum(y_pred_train_majority == y_true_train) / len(y_true_train)
print(f"Training Accuracy: {train_accuracy * 100:.2f}%")

# Making predictions on the test set
y_preds_test = [np.argmax(model.predict(mfcc_test), axis=-1) for model in models]
y_pred_test_majority = get_majority(y_preds_test)
y_true_test = np.argmax(y_test, axis=1)
test_accuracy = np.sum(y_pred_test_majority == y_true_test) / len(y_true_test)
print(f"Testing Accuracy: {test_accuracy * 100:.2f}%")
