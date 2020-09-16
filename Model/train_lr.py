import matplotlib
matplotlib.use('Agg')

# import the necessary packages
from LRfinder import LRFinder
from Sudoku import SudokuModel
from clr_callback import CyclicLR
import config
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD,Adam
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2 as cv
import sys
import os
import sys
import inspect

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--lr-find", type=int, default=0,
  help="whether or not to find optimal learning rate")
# ap.add_argument("-m", "--model", required=True,
#   help="path to output model after training")
args = vars(ap.parse_args())



print("Loading & Processing Data ...")

((train_X,train_Y),(test_X,test_Y)) = mnist.load_data()
# train_X = train_X[:]
# print(len(train_X))

train_X = train_X.reshape((train_X.shape[0],28,28,1))
test_X = test_X.reshape((test_X.shape[0],28,28,1))

#Rescale
train_X = train_X.astype("float32")/255.0
test_X = test_X.astype("float32")/255.0

le = LabelBinarizer()
train_Y = le.fit_transform(train_Y)
test_Y = le.transform(test_Y)

opt = Adam(lr=config.MIN_LR)
model = SudokuModel.build(width=28, height=28, depth=1, classes=10)
model.compile(loss="categorical_crossentropy",optimizer=opt,metrics=["accuracy"])


# construct the image generator for data augmentation
aug = ImageDataGenerator(width_shift_range=0.1,
  height_shift_range=0.1, horizontal_flip=False,
  fill_mode="nearest")


# check to see if we are attempting to find an optimal learning rate
# before training for the full number of epochs
if args["lr_find"] > 0:

  # initialize the learning rate finder and then train with learning
  # rates ranging from 1e-10 to 1e+1
  print("[INFO] finding learning rate...")
  lrf = LRFinder(model)
  lrf.find(
    aug.flow(train_X, train_Y, batch_size=config.BATCH_SIZE),
    1e-10, 1e+1,
    stepsPerEpoch=np.ceil((len(train_X) / float(config.BATCH_SIZE))),
    batchSize=config.BATCH_SIZE)

  # plot the loss for the various learning rates and save the
  # resulting plot to disk
  lrf.plot_loss()

  plt.savefig(config.LRFIND_PLOT_PATH)
  # gracefully exit the script so we can adjust our learning rates
  # in the config and then train the network for our full set of
  # epochs
  print("[INFO] learning rate finder complete")
  print("[INFO] examine plot and adjust learning rates before training")
  sys.exit(0)

#Make the folder for training
#Create specific folder for this training
if not os.path.exists(config.TRAINING_PATH):
  os.makedirs(config.TRAINING_PATH)
  os.makedirs(config.PLOT_PATH)
  os.makedirs(config.TMP_PATH)
else:
  sys.exit(f"{config.TRAINING_PATH} already exists, exiting")


with open(config.SUMMARY_PATH,'w') as f:
  model.summary(print_fn=lambda x: f.write(x + '\n'))
  model_code = inspect.getsourcelines(SudokuModel.build) #tuple (list_of_lines,nb_lines)
  f.write('\n')
  for line in model_code[0]:
    f.write(line + '\n')


stepSize = config.STEP_SIZE * (train_X.shape[0] // config.BATCH_SIZE)
clr = CyclicLR(
  mode=config.CLR_METHOD,
  base_lr=config.MIN_LR,
  max_lr=config.MAX_LR,
  step_size=stepSize)

model_checkpoint_callback = ModelCheckpoint(
    filepath=config.CP_PATH,
    monitor='val_loss',
    save_weights_only=False,
    save_best_only=True,
    save_frequency=1)

# train the network
print("[INFO] training network...")
H = model.fit(
  x=aug.flow(train_X, train_Y, batch_size=config.BATCH_SIZE),
  validation_data=(test_X, test_Y),
  steps_per_epoch=train_X.shape[0] // config.BATCH_SIZE,
  epochs=config.NUM_EPOCHS,
  callbacks=[clr,model_checkpoint_callback],
  verbose=1)
# evaluate the network and show a classification report
print("[INFO] evaluating network...")
predictions = model.predict(x=test_X, batch_size=config.BATCH_SIZE)
print(classification_report(test_Y.argmax(axis=1),
  predictions.argmax(axis=1), target_names=config.CLASSES))

# construct a plot that plots and saves the training history
N = np.arange(0, config.NUM_EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["accuracy"], label="train_acc")
plt.plot(N, H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(config.TRAINING_PLOT_PATH)

# plot the learning rate history
N = np.arange(0, len(clr.history["lr"]))
plt.figure()
plt.plot(N, clr.history["lr"])
plt.title("Cyclical Learning Rate (CLR)")
plt.xlabel("Training Iterations")
plt.ylabel("Learning Rate")
plt.savefig(config.CLR_PLOT_PATH)
