import numpy as np
import os
import tensorflow as tf

# Load npz file of Mel-Spectrogram
file = np.load(os.getcwd() + "/new_mel_train_test.npz")
mel_train = file['mel_train']
mel_test = file['mel_test']
y_train = file['y_train']
y_test = file['y_test']

# Define the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(8, (3, 3), activation='relu', input_shape=mel_train[0].shape, padding='same'),
    tf.keras.layers.MaxPooling2D((4, 4), padding='same'),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((4, 4), padding='same'),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((4, 4), padding='same'),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((4, 4), padding='same'),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

# EarlyStopping callback definition
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# ModelCheckpoint callback definition
checkpoint_path = os.getcwd() + "/models/melspectrogram_model_{epoch:03d}.h5"
checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_freq='epoch', save_weights_only=False, period=5)

# Train Model with both ModelCheckpoint and EarlyStopping
model.fit(mel_train, y_train, validation_split=0.2, epochs=200, batch_size=32, verbose=1,
          callbacks=[checkpoint, early_stopping])

# Save the final model
model.save(os.getcwd() + "/models/final_melspectrogram_model.h5")

# Evaluate training accuracy
train_loss, train_accuracy = model.evaluate(mel_train, y_train, verbose=0)
print(f"Train Accuracy: {train_accuracy * 100:.2f}%")

# Evaluate testing accuracy
test_loss, test_accuracy = model.evaluate(mel_test, y_test, verbose=0)
print(f"Testing Accuracy: {test_accuracy * 100:.2f}%")
