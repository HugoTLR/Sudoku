import matplotlib

# import the necessary packages
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist

import glob

from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import numpy as np
import cv2 as cv

import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns 
import argparse
import sys


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-r", "--real", type=int, default=0,
  help="0 test on mnist, 1 test on private folder")
# ap.add_argument("-m", "--model", required=True,
#   help="path to output model after training")
args = vars(ap.parse_args())




model_date = "20200916_122327"
model_name = "cp-66-0.0508.h5"
model_path = f"./Output/{model_date}/tmp/{model_name}"
# model_path = f"./Output/Models/cp-38-0.0509.h5"
model = load_model(model_path)
print(model.summary())
print(type(model))

# cmap = cm.get_cmap('viridis') # Colour map (there are many others)
# colors = ['black','gray','red','blue','orange','teal','peru','crimson','pink','maroon','yellow']
colors = "bgrcmykw"
# print(type(cmap),cmap)
print("Loading & Processing Data ...")

if args['real'] == 0:

  ((train_X,train_Y),(test_X,test_Y)) = mnist.load_data()
  print(train_X,train_X.shape)
  print(type(train_X[0]),train_X[0].shape)

  train_X = train_X.reshape((train_X.shape[0],28,28,1))
  test_X = test_X.reshape((test_X.shape[0],28,28,1))

  #Rescale
  train_X = train_X.astype("float32")/255.0
  test_X = test_X.astype("float32")/255.0

  print(train_X,train_X.shape)
  print(type(train_Y),train_Y.shape)
else:

  collect_folder = "./Data/test/"

  nb_files = len(glob.glob(f"{collect_folder}*.jpg"))
  images = [[cv.imread(f,cv.IMREAD_GRAYSCALE) for f in glob.glob(f'{collect_folder}{g}/*.jpg')] for g in range(10)]
  min_class_num = min([len(im) for im in images])
  print(min_class_num)
  data = []
  for i,classes in enumerate(images):
    for im in classes[:min_class_num]:
      im = im.astype("float") / 255.0
      im = img_to_array(im)
      # im = np.expand_dims(im,axis=0)
      data.append((im,i))
  X,Y = [],[]
  for d in data:
    X.append(d[0])
    Y.append(d[1])
  X = np.array(X)
  Y = np.array(Y,dtype=np.uint8)
  train_X,test_X,train_Y,test_Y = train_test_split(X,Y,test_size=.2)


# pred
pred_X = model.predict(test_X)
print(len(pred_X))
confusion = confusion_matrix(test_Y,np.argmax(pred_X,axis=1))
print(confusion)

ax = sns.heatmap(confusion, annot=True, fmt="d")
plt.show()


# roc curve for classes
colors = ['orange','blue','yellow',]
fpr = {}
tpr = {}
thresh ={}
n_class = 10
for i in range(n_class):    
    fpr[i], tpr[i], thresh[i] = roc_curve(test_Y, pred_X[:,i], pos_label=i)
    plt.plot(fpr[i], tpr[i], linestyle='--',c=np.random.random(3), label=f'Class {i} vs Rest')
plt.title('Multiclass ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.legend(loc='best')
plt.show()