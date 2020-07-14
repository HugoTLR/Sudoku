import random
import time
import cv2 as cv
import glob
import numpy as np
import tensorflow as tf

class Model:
  def __init__(self,width,height,w_filename):
    self.w = width
    self.h = height
    self.weights = w_filename
    self.initiateModel()
    self.model.load_weights(self.weights)

  def initiateModel(self):
    self.model = tf.keras.models.Sequential()

    self.model.add(tf.keras.layers.Flatten(input_shape=(self.w,self.h,1)))#))  #Input layer
    self.model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))  #Hidden Layer 1
    self.model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))  #Hidden Layer 1
    self.model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))  #Output Layer

    self.model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])


  def run_test(self,file_names):
    imgs = [cv.cvtColor(cv.resize(cv.imread(file),(32,32)),\
            cv.COLOR_BGR2GRAY) for file in file_names]

    for img in imgs:
      X = np.array(img).reshape(-1,self.w,self.h,1)
      X = tf.keras.utils.normalize(X,axis=1)
      preds = self.model.predict(X)
      val = np.argmax(preds[0])
      conf = preds[0][np.argmax(preds[0])] * 100
      print(f"Mod predict ( {val} ) with {conf} % confidence")
      cv.imshow("img",img)
      cv.waitKey(0)
    cv.destroyAllWindows


# if __name__ == "__main__":
#   model_name = "pkmn_casino_digit_reader.h5"
#   model = Model(32,32,model_name)

#   imgs = glob.glob("./imgs/*.png")
#   model.run_test(imgs)del')