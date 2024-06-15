import numpy as np
import os
import tensorflow as tf

# Load Spectrogram Train-test data
spec_file = np.load(os.getcwd() + "/new_spectrogram_train_test.npz")

# Model 1 for Spectrogram
S_train = spec_file['S_train']
S_test = spec_file['S_test']
y_train = spec_file['y_train']
y_test = spec_file['y_test']

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(8, (3, 3), activation='relu', input_shape=S_train[0].shape, padding='same'))
model.add(tf.keras.layers.MaxPooling2D((4, 4), padding='same'))
model.add(tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(tf.keras.layers.MaxPooling2D((4, 4), padding='same'))
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(tf.keras.layers.MaxPooling2D((4, 4), padding='same'))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(tf.keras.layers.MaxPooling2D((4, 4), padding='same'))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(tf.keras.layers.MaxPooling2D((4, 4), padding='same'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

#adam to reduce loss
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

# Train Model 1
checkpoint_path = os.getcwd() + "/models/new_spec_model_spectrogram1_{epoch:03d}.h5"
checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_freq='epoch', period=5)

history = model.fit(S_train, y_train, epochs=100, callbacks=[checkpoint], batch_size=32, verbose=1)
model.save(os.getcwd() + "/models/new_spec_model_spectrogram1.h5")

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(S_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")