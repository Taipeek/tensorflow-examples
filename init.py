# Import all the libraries
import os as os

import imageio
import numpy as numpy
from scipy import ndimage
from six.moves import cPickle as pickle
import random as random
import itertools as itertools
import tensorflow as tf
import matplotlib.pyplot as plt
import math as math

basePath = "./lfw/"
numFaces = 50

# Load the images and store them if necessary.
def readSingleImage(fileName):
  return imageio.imread(fileName)


def loadPicturesFromFiles(basePath):
  allNames = os.listdir(basePath)[1:numFaces]
  allPictures = {}
  for personName in allNames:
    personPictureDirectory = os.path.join(basePath, personName)
    if (not (personName[0] == ".")) & os.path.isdir(personPictureDirectory):
      print("Reading faces of " + personName + "...", end="")
      pictureFiles = os.listdir(personPictureDirectory)
      pictureFiles = list(map(os.path.join, [personPictureDirectory] * len(pictureFiles), pictureFiles))
      pictures = list(map(readSingleImage, pictureFiles))
      print(" DONE (" + str(len(pictures)) + " read)")
      allPictures[personName] = pictures
  return allPictures

allImagesDict = loadPicturesFromFiles(basePath)


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
  scale = numpy.vectorize(lambda x: x/255.0)
  trainingImages = scale(numpy.array(trainingImages))
  testingImages = scale(numpy.array(testingImages))
  return((trainingImages, testingImages))

def backScaleColoursInImages(imageList):
  scale = numpy.vectorize(lambda x: 0 if x < 0 else x*255 if x <= 1 else 255)
  imageList = numpy.array(imageList)
  imageList = scale(imageList).astype(numpy.uint8)
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
  imageDimensions = img.shape
  noise = numpy.random.rand(imageDimensions[0], imageDimensions[1], imageDimensions[2])
  noiseIter = numpy.nditer([noise, None])
  for i, out in noiseIter:
    if i > noiseLevel:
      out[...] = 1
    else:
      out[...] = 0
  imageMask = noiseIter.operands[1]
  return(img * imageMask)

def addRandomNoiseToAllImages(allImages, noiseLevel):
  return(list(map(lambda img : addRandomNoiseToSingleImage(img, noiseLevel), allImages)))