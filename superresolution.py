from init import *
import numpy as np

allImagesDict = loadPicturesFromFiles(basePath, numFaces=0)
allFunImagesDict = loadPicturesFromFiles(basePath, numFaces=0, filter_names=['Vladimir_Putin', 'Donald_Trump', 'Harrison_Ford'])

[originalTrainingImages, trainingLabels, originalTestingImages, testingLabels] = splitImagesToTrainingAndTestingSets(allImagesDict)
[funTrainingImages, _, _, _] = splitImagesToTrainingAndTestingSets(allFunImagesDict, 2)

[y_train, y_test] = expandAllImages(originalTrainingImages, originalTestingImages)
[y_fun, _] = expandAllImages(funTrainingImages, funTrainingImages)

x_train = np.array([downscale(img, scale=4) for img in y_train])
x_test = np.array([downscale(img, scale=4) for img in y_test])
x_fun = np.array([downscale(img, scale=4) for img in y_fun])
visualizeInputOutput(x_test[0], y_test[0])

model = tf.keras.models.load_model("models/superres_3.1459val_loss.h5")
model = edsr(4)
model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['accuracy'])

tf.keras.utils.plot_model(model, to_file='visuals/superresolution/model_1.png', show_shapes=True)
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    verbose=1,
    restore_best_weights=True,
)
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=30, callbacks=[early_stop])
# model.save("models/superres_3.1459val_loss.h5")

plotAccAndLoss(history, "visuals/superresolution/training_process.png")
model.summary()

# Plot results
predictCount = 10
index = np.random.choice(x_test.shape[0], predictCount)
test_in = x_test[index]
test_real = y_test[index]
test_out = model.predict(test_in).astype(np.uint8)
for i in range(predictCount):
    visualizeInputOutput(test_in[i], test_out[i], test_real[i], "visuals/superresolution/pic_"+str(index[i])+".png")

y_fun = model.predict(x_fun).astype(np.uint8)
for i in range(len(x_fun)):
    visualizeInputOutput(x_fun[i], y_fun[i], path_to_save="visuals/superresolution/fun_pic_"+str(i)+".png")
