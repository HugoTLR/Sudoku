from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from Sudoku import SudokuModel

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

import argparse

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
  help="path to output model after training")
args = vars(ap.parse_args())


INIT_LR = 1e-3
EPOCHS = 5
BATCH_SIZE = 128

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
test_Y = le.fit_transform(test_Y)


print("Compiling Model ...")

opt = Adam(lr=INIT_LR)
model = SudokuModel.build(width=28, height=28, depth=1, classes=10)
model.compile(loss="categorical_crossentropy",optimizer=opt,metrics=["accuracy"])

print("Training Model ...")


H = model.fit(train_X,train_Y,
              validation_data=(test_X,test_Y),
              batch_size=BATCH_SIZE,
              epochs=EPOCHS,
              verbose=1,)

print("Evaluating Model ...")
predictions = model.predict(test_X)
print(classification_report(test_Y.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names =[str(x) for x in le.classes_]))

print("Saving Model ...")
model.save(args["model"],save_format="h5")
