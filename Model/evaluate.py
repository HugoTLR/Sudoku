import matplotlib

# import the necessary packages
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist

import glob

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns 
import argparse


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-r", "--real", type=int, default=0,
  help="0 test on mnist, 1 test on private folder")
# ap.add_argument("-m", "--model", required=True,
#   help="path to output model after training")
args = vars(ap.parse_args())


model_date = "20200915_182309"
model_name = "cp-04-0.0603.h5"
# model_path = f"./Output/{model_date}/tmp/{model_name}"
model_path = f"./Output/Models/cp-38-0.0509.h5"
model = load_model(model_path)
print(model.summary())
print(type(model))

cmap = cm.get_cmap('viridis') # Colour map (there are many others)
print(type(cmap),cmap)
print("Loading & Processing Data ...")

((train_X,train_Y),(test_X,test_Y)) = mnist.load_data()

train_X = train_X.reshape((train_X.shape[0],28,28,1))
test_X = test_X.reshape((test_X.shape[0],28,28,1))

#Rescale
train_X = train_X.astype("float32")/255.0
test_X = test_X.astype("float32")/255.0

# pred
pred_X = model.predict(test_X)
print(len(pred_X))
confusion = confusion_matrix(test_Y,np.argmax(pred_X,axis=1))
print(confusion)

ax = sns.heatmap(confusion, annot=True, fmt="d")



# roc curve for classes
colors = ['orange','blue','yellow',]
fpr = {}
tpr = {}
thresh ={}
ax1 = plt.figure
n_class = 10
for i in range(n_class):    
    fpr[i], tpr[i], thresh[i] = roc_curve(test_Y, pred_X[:,i], pos_label=i)
     = plt.plot(fpr[i], tpr[i], linestyle='--',color=cmap.colors[i], label=f'Class {i} vs Rest')
plt.title('Multiclass ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.legend(loc='best')
plt.show()