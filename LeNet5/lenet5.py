import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense
from tensorflow.keras.losses import sparse_categorical_crossentropy
import numpy as np

(train_x, train_y), (test_x, test_y) = keras.datasets.mnist.load_data()
train_x = train_x / 255.0
test_x = test_x / 255.0

train_x = tf.expand_dims(train_x, 3)
test_x = tf.expand_dims(test_x, 3)

val_x = train_x[:5000]
val_y = train_y[:5000]

# LeNet5 Architecture
lenet_5_model = keras.models.Sequential([
  Conv2D(6, kernel_size=5, strides=1, activation='tanh', input_shape=train_x[0].shape, padding='same'),
  AveragePooling2D(),
  Conv2D(16, kernel_size=5, strides=1, activation='tanh', padding='valid'),
  AveragePooling2D(),
  Conv2D(120, kernel_size=5, strides=1, activation='tanh', padding='valid'),
  Flatten(),
  Dense(84, activation='tanh'),
  Dense(10, activation='softmax'),
])


# Compiling Model
lenet_5_model.compile(optimizer='adam', loss=sparse_categorical_crossentropy, metrics=['accuracy'])

# Training Model
lenet_5_model.fit(train_x, train_y, epochs=5, validation_data=(val_x, val_y))

# Evaluating model
predictions = lenet_5_model.evaluate(test_x, test_y)

print('-'*50)
print('Predictions: ')
print("Accuracy :", predictions[1])
print("Loss :", predictions[0])
print('-'*50)
