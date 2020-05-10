from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from init import plotAccAndLoss


def to_bit_map(digit):
  arr = np.zeros(10)
  arr[digit] = 1
  return arr

def to_digit(bit_map):
  for x in range(len(bit_map)):
    if bit_map[x]==1: return x
  return -1

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.expand_dims(x_train, 3)
x_test = np.expand_dims(x_test, 3)
y_train = np.array([to_bit_map(n) for n in y_train])
y_test = np.array([to_bit_map(n) for n in y_test])

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

inputs = keras.Input(shape=(28, 28, 1), name='input_digit')
x = layers.Conv2D(activation="relu", filters=2, strides=1, kernel_size=4, name='conv1')(inputs)
x = layers.Conv2D(activation="relu", filters=3, strides=1, kernel_size=4, name='conv2')(x)
x = layers.Conv2D(activation="relu", filters=4, strides=1, kernel_size=4, name='conv3')(x)
x = layers.Conv2D(activation="relu", filters=5, strides=1, kernel_size=4, name='conv4')(x)
x = layers.Conv2D(activation="relu", filters=6, strides=1, kernel_size=4, name='conv5')(x)
x = layers.Conv2D(activation="relu", filters=7, strides=1, kernel_size=4, name='conv6')(x)
x = layers.Conv2D(activation="relu", filters=8, strides=1, kernel_size=4, name='conv7')(x)
x = layers.Conv2D(activation="relu", filters=9, strides=1, kernel_size=4, name='conv8')(x)
x = layers.Conv2D(activation="softmax", filters=10, strides=2, kernel_size=4, name='conv_out')(x)
outputs = layers.Flatten()(x)

model = keras.Model(inputs=inputs, outputs=outputs)
model.summary()
tf.keras.utils.plot_model(model, to_file='visuals/classification/model_1.png', show_shapes=True)
model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['accuracy'])
early_stop = tf.keras.callbacks.EarlyStopping(
  monitor="val_loss",
  patience=3,
  verbose=1,
  restore_best_weights=True,
)
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=40, callbacks=[early_stop])
plotAccAndLoss(history, "visuals/classification/training_process.png")

test_in = x_test[:10]
actual = y_test[:10]
test_out = model.predict(test_in)
test_in_reduced = np.reshape(test_in, (10, 28, 28))

# Plot results
for imgID in range(len(test_in)):
  plt.subplot(len(test_in)+1, 1, imgID + 1)
  plt.xlabel('predict: '+np.array2string(test_out[imgID], precision=2, separator=', ', suppress_small=True)+', actual: '+str(to_digit(actual[imgID])))
  plt.imshow(test_in_reduced[imgID])
  plt.show()

# model.save("models/classification.h5")