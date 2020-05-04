from init import *
from tensorflow import keras
from tensorflow.keras import layers

basePath = "./lfw/"
numFaces = 50
allImagesDict = loadPicturesFromFiles(basePath)

[originalTrainingImages, trainingLabels, originalTestingImages, testingLabels] = splitImagesToTrainingAndTestingSets(allImagesDict)

[expandedTrainingImages, expandedTestingImages] = expandAllImages(originalTrainingImages, originalTestingImages)


numImageRows = 1 + (len(expandedTrainingImages) // 10)
plt.figure(figsize=(50, 5 * numImageRows))
for imgID in range(len(expandedTrainingImages)):
  plt.subplot(numImageRows, 10, imgID + 1)
  plt.imshow(expandedTrainingImages[imgID])
plt.show()

numImageRows = 1 + (len(expandedTestingImages) // 10)
plt.figure(figsize=(50, numImageRows * 5))
for imgID in range(len(expandedTestingImages)):
  plt.subplot(numImageRows, 10, imgID + 1)
  plt.imshow(expandedTestingImages[imgID])
plt.show()

[trainingImages, testingImages] = zeroOneScaleColoursInImages(expandedTrainingImages, expandedTestingImages)

noisedTrainingImages = addRandomNoiseToAllImages(trainingImages, 0.03)
noisedTestingImages = addRandomNoiseToAllImages(testingImages, 0.03)

numImages = len(trainingImages[:5])
plt.figure(figsize=(5, 2.5 * numImages))
for imgID in range(numImages):
  plt.subplot(numImages+1, 2, imgID * 2 + 1)
  plt.imshow(trainingImages[imgID])

  plt.subplot(numImages+1, 2, imgID * 2 + 2)
  plt.imshow(noisedTrainingImages[imgID])
plt.show()

numImages = len(testingImages[:5])
plt.figure(figsize=(5, 2.5 * numImages))
for imgID in range(numImages):
  plt.subplot(numImages+1, 2, imgID * 2 + 1)
  plt.imshow(testingImages[imgID])

  plt.subplot(numImages+1, 2, imgID * 2 + 2)
  plt.imshow(noisedTestingImages[imgID])
plt.show()

inputs = keras.Input(shape=(256, 256, 3), name='noised_input')
x = layers.Conv2DTranspose(activation="relu", filters=64, strides=2, kernel_size=5, name='deconv1', padding="same")(inputs)
# x = layers.Conv2DTranspose(filters=32, strides=2, kernel_size=5, name='deconv2', padding="same")(x)
# x = layers.Conv2D(filters=16, strides=2, kernel_size=5, name='conv1', padding="same")(x)
outputs = layers.Conv2D(activation="relu", filters=3, strides=2, kernel_size=5, name='conv2', padding="same")(x)

model = keras.Model(inputs=inputs, outputs=outputs)
model.summary()
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
noisedTrainingImages = numpy.array(noisedTrainingImages)
trainingImages = numpy.array(trainingImages)
noisedTestingImages = numpy.array(noisedTestingImages)
testingImages = numpy.array(testingImages)
model.fit(noisedTrainingImages, trainingImages, validation_data=(noisedTestingImages, testingImages), epochs=20)

outputTstImgs = model.predict(noisedTestingImages)
outputTstImgs = backScaleColoursInImages(outputTstImgs)
inputTstImgs = backScaleColoursInImages(noisedTestingImages)

plt.figure(figsize=(5, 2.5 * len(inputTstImgs)))
for imgID in range(len(outputTstImgs)):
  plt.subplot(len(outputTstImgs)+1, 2, imgID * 2 + 1)
  plt.imshow(inputTstImgs[imgID])

  plt.subplot(len(outputTstImgs)+1, 2, imgID * 2 + 2)
  plt.imshow(outputTstImgs[imgID])
plt.show()

# model.save("models/denoise_one_deconv_40_epochs.h5")