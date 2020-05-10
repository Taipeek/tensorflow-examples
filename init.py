# Import all the libraries
import os as os

import imageio
import numpy as numpy
import random as random
import itertools as itertools
from skimage.util import random_noise

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.keras.layers import Add, Conv2D, Input, Lambda
from tensorflow.python.keras.models import Model
from skimage.transform import resize

basePath = "./lfw/"
numFaces = 50

# Load the images and store them if necessary.
def readSingleImage(fileName):
  return imageio.imread(fileName)


def loadPicturesFromFiles(basePath, numFaces = numFaces, filter_names=None):
  allNames = sorted(os.listdir(basePath))
  allNames = allNames[1:numFaces] if numFaces > 0 else allNames
  allNames = filter_names if filter_names else allNames
  allPictures = {}
  for personName in allNames:
    personPictureDirectory = os.path.join(basePath, personName)
    if (not (personName[0] == ".")) & os.path.isdir(personPictureDirectory):
      print("Reading faces of " + personName + "...", end="")
      pictureFiles = sorted(os.listdir(personPictureDirectory))
      pictureFiles = list(map(os.path.join, [personPictureDirectory] * len(pictureFiles), pictureFiles))
      pictures = list(map(readSingleImage, pictureFiles))
      print(" DONE (" + str(len(pictures)) + " read)")
      allPictures[personName] = pictures
  return allPictures

def splitImagesToTrainingAndTestingSets(allImagesDict, trainingPortion = 0.75):
  trainingImages = []
  trainingLabels = []
  testingImages  = []
  testingLabels  = []
  for personNames, pictures in allImagesDict.items():
    if (random.uniform(0,1) < trainingPortion):
      trainingLabels.append([personNames] * len(pictures))
      trainingImages.append(pictures)
    else:
      testingLabels.append([personNames] * len(pictures))
      testingImages.append(pictures)
  return((
    list(itertools.chain(*trainingImages)),
    list(itertools.chain(*trainingLabels)),
    list(itertools.chain(*testingImages)),
    list(itertools.chain(*testingLabels))))


def expandSingleImage(img):
  expandedImage = numpy.zeros((256, 256, 3), dtype=numpy.uint8)
  expandedImage[:img.shape[0], :img.shape[1], :] = img
  return(expandedImage)

def expandAllImages(trainingImages, testingImages):
  trainingImages = numpy.array([expandSingleImage(img) for img in trainingImages])
  testingImages = numpy.array([expandSingleImage(img) for img in testingImages])
  return((trainingImages, testingImages))

def zeroOneScaleColoursInImages(trainingImages, testingImages):
  trainingImages = numpy.array(trainingImages) / 255.0
  testingImages = numpy.array(testingImages) / 255.0
  return((trainingImages, testingImages))

def backScaleColoursInImages(imageList):
  imageList = numpy.array(imageList)
  imageList = numpy.clip(imageList*255, 0, 255).astype(numpy.uint8)
  return(imageList)


def showProgressOnTestingImages(inputTstImgs, outputTstImgs):

  inputTstImgs  = backScaleColoursInImages(inputTstImgs)
  outputTstImgs = backScaleColoursInImages(outputTstImgs)

  plt.figure(figsize=(7.5, 2.5 * len(inputTstImgs)))
  for imgID in range(len(inputTstImgs)):
    plt.subplot(len(inputTstImgs)+1, 3, imgID * 3 + 1)
    plt.imshow(inputTstImgs[imgID])

    plt.subplot(len(inputTstImgs)+1, 3, imgID * 3 + 3)
    plt.imshow(outputTstImgs[imgID])
  plt.show()

def addRandomNoiseToSingleImage(img, noiseLevel):
  noise_img = random_noise(img, mode='s&p', amount=noiseLevel)
  return noise_img

def addRandomNoiseToAllImages(allImages, noiseLevel):
  return numpy.array(list(map(lambda img: addRandomNoiseToSingleImage(img, noiseLevel), allImages)))




def edsr(scale, num_filters=64, num_res_blocks=8, res_block_scaling=None):
  x_in = Input(shape=(None, None, 3))
  x = Lambda(normalize, name="normalize")(x_in)

  x = b = Conv2D(num_filters, 3, padding='same')(x)
  for i in range(num_res_blocks):
    b = res_block(b, num_filters, res_block_scaling)
  b = Conv2D(num_filters, 3, padding='same')(b)
  x = Add()([x, b])

  x = upsample(x, scale, num_filters)
  x = Conv2D(3, 3, padding='same')(x)

  x = Lambda(denormalize, name="denormalize")(x)
  return Model(x_in, x, name="edsr")


def res_block(x_in, filters, scaling):
  x = Conv2D(filters, 3, padding='same', activation='relu')(x_in)
  x = Conv2D(filters, 3, padding='same')(x)
  if scaling:
    x = Lambda(lambda t: t * scaling)(x)
  x = Add()([x_in, x])
  return x


def upsample(x, scale, num_filters):
  def upsample_1(x, factor, **kwargs):
    x = Conv2D(num_filters * (factor ** 2), 3, padding='same', **kwargs)(x)
    return Lambda(pixel_shuffle(scale=factor))(x)

  if scale == 2:
    x = upsample_1(x, 2, name='conv2d_1_scale_2')
  elif scale == 3:
    x = upsample_1(x, 3, name='conv2d_1_scale_3')
  elif scale == 4:
    x = upsample_1(x, 2, name='conv2d_1_scale_2')
    x = upsample_1(x, 2, name='conv2d_2_scale_2')

  return x


# ---------------------------------------
#  Normalization
# ---------------------------------------

dataset_mean = numpy.array([106.70164812, 93.1222307, 83.21076972])

def normalize(x, rgb_mean=dataset_mean):
  return (x - rgb_mean) / 127.5


def denormalize(x, rgb_mean=dataset_mean):
  return tf.clip_by_value(
    x * 127.5 + rgb_mean, 0, 255
  )



def pixel_shuffle(scale):
  return lambda x: tf.nn.depth_to_space(x, scale)

def downscale(image, scale=4):
  small_img = resize(image, (image.shape[0] / float(scale), image.shape[1] / float(scale)), preserve_range=True)
  return small_img.astype(numpy.uint8)

def visualizeInputOutput(input, output, real=None, path_to_save=False):
  rows = 3 if real is not None else 2
  plt.figure(dpi=200)
  plt.subplot(1, rows, 1)
  plt.imshow(input)
  plt.xticks(())
  plt.yticks(())
  plt.subplot(1, rows, 2)
  plt.imshow(output)
  plt.xticks(())
  plt.yticks(())
  if real is not None:
    plt.subplot(1, rows, 3)
    plt.imshow(real)
    plt.xticks(())
    plt.yticks(())
  if(path_to_save):
    plt.savefig(path_to_save, bbox_inches='tight')
  plt.show()

def plotAccAndLoss(history, path_to_save=None):
  acc = history.history['accuracy']
  val_acc = history.history['val_accuracy']

  loss = history.history['loss']
  val_loss = history.history['val_loss']

  epochs_range = history.epoch

  plt.figure(figsize=(8, 8))
  plt.subplot(1, 2, 1)
  plt.plot(epochs_range, acc, label='Training Accuracy')
  plt.plot(epochs_range, val_acc, label='Validation Accuracy')
  plt.legend(loc='lower right')
  plt.title('Training and Validation Accuracy')

  plt.subplot(1, 2, 2)
  plt.plot(epochs_range, loss, label='Training Loss')
  plt.plot(epochs_range, val_loss, label='Validation Loss')
  plt.legend(loc='upper right')
  plt.title('Training and Validation Loss')
  if(path_to_save):
    plt.savefig(path_to_save, bbox_inches='tight')

  plt.show()
