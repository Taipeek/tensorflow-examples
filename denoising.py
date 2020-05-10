from init import *
from tensorflow import keras
from tensorflow.keras import layers

allImagesDict = loadPicturesFromFiles(basePath, numFaces=1500)

[originalTrainingImages, trainingLabels, originalTestingImages, testingLabels] = splitImagesToTrainingAndTestingSets(allImagesDict)

[expandedTrainingImages, expandedTestingImages] = expandAllImages(originalTrainingImages, originalTestingImages)

[trainingImages, testingImages] = zeroOneScaleColoursInImages(expandedTrainingImages, expandedTestingImages)

noisedTrainingImages = addRandomNoiseToAllImages(trainingImages, 0.03)
noisedTestingImages = addRandomNoiseToAllImages(testingImages, 0.03)

inputs = keras.Input(shape=(256, 256, 3), name='noised_input')
x = layers.Conv2DTranspose(activation="relu", filters=16, strides=2, kernel_size=3, name='deconv1', padding="same")(inputs)
x = layers.Conv2DTranspose(activation="relu", filters=32, strides=2, kernel_size=3, name='deconv2', padding="same")(x)
x = layers.Conv2D(activation="relu", filters=16, strides=2, kernel_size=3, name='conv1', padding="same")(x)
outputs = layers.Conv2D(activation="relu", filters=3, strides=2, kernel_size=3, name='conv2', padding="same")(x)

model = keras.Model(inputs=inputs, outputs=outputs)
# model = keras.models.load_model("models/_.h5")
model.summary()
model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['accuracy'])

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    verbose=1,
    restore_best_weights=True,
)
tf.keras.utils.plot_model(model, to_file='visuals/denoising/model_1.png', show_shapes=True)
history = model.fit(noisedTrainingImages, trainingImages, validation_data=(noisedTestingImages, testingImages), epochs=50, callbacks=[early_stop], batch_size=8)
plotAccAndLoss(history, "visuals/denoising/training_process.png")

outputTstImgs = model.predict(noisedTestingImages[:12])
outputTstImgs = backScaleColoursInImages(outputTstImgs)
inputTstImgs = backScaleColoursInImages(noisedTestingImages[:12])

# Plot results
plt.figure(figsize=(5, 2.5 * len(outputTstImgs)))
for imgID in range(len(outputTstImgs)):
  plt.subplot(len(outputTstImgs)+1, 2, imgID * 2 + 1)
  plt.imshow(inputTstImgs[imgID])

  plt.subplot(len(outputTstImgs)+1, 2, imgID * 2 + 2)
  plt.imshow(outputTstImgs[imgID])
plt.show()

# model.save("models/denoise_two_deconv_.h5")