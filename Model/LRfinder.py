from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import tempfile

class LRFinder:
  iterClasses = ["NumpyArrayIterator", "DirectoryIterator", "DataFrameIterator", "Iterator", "Sequence"]
  def __init__(self,model,stopFactor=4, beta=0.98):
    self.model = model
    self.stopFactor = stopFactor
    self.beta = beta

    self.lrs = []
    self.losses = []


    self.lrMultiplier = 1
    self.avgLoss = 0
    self.bestLoss = 1e9
    self.batchNum = 0
    self.weightsFile = None

  def reset(self):
    self.lrs = []
    self.losses = []
    self.lrMultiplier = 1
    self.avgLoss = 0
    self.bestLoss = 1e9
    self.batchNum = 0
    self.weightsFile = None

  def is_data_iter(self,data):
    #Is our data a generator ?
    return data.__class__.__name__ in LRFinder.iterClasses



  def on_batch_end(self,batch,logs):
    #Process every time we end a batch
    #Change lr, compute loss ...
    lr = K.get_value(self.model.optimizer.lr)
    self.lrs.append(lr)



    l = logs['loss']
    self.batchNum += 1
    self.avgLoss = (self.beta * self.avgLoss) + ((1 - self.beta) * l)
    smooth = self.avgLoss / (1 - (self.beta ** self.batchNum) )
    self.losses.append(smooth)

    stopLoss = self.stopFactor * self.bestLoss

    if self.batchNum > 1 and smooth > stopLoss:
      self.model.stop_training = True
      return
    if self.batchNum == 1 or smooth < self.bestLoss:
      self.bestLoss = smooth

    lr *= self.lrMultiplier
    K.set_value(self.model.optimizer.lr, lr)

  def find(self, trainData, startLR, endLR, epochs=None, stepsPerEpoch=None, batchSize=32, sampleSize=2048, verbose=1):
    # reset our class-specific variables
    self.reset()

    # determine if we are using a data generator or not
    useGen = self.is_data_iter(trainData)

    # if we're using a generator and the steps per epoch is not
    # supplied, raise an error
    if useGen and stepsPerEpoch is None:
      msg = "Using generator without supplying stepsPerEpoch"
      raise Exception(msg)
    # if we're not using a generator then our entire dataset must
    # already be in memory
    elif not useGen:
      # grab the number of samples in the training data and
      # then derive the number of steps per epoch
      numSamples = len(trainData[0])
      stepsPerEpoch = np.ceil(numSamples / float(batchSize))

    # if no number of training epochs are supplied, compute the
    # training epochs based on a default sample size
    if epochs is None:
      epochs = int(np.ceil(sampleSize / float(stepsPerEpoch)))

    # compute the total number of batch updates that will take
    # place while we are attempting to find a good starting
    # learning rate
    numBatchUpdates = epochs * stepsPerEpoch

    # derive the learning rate multiplier based on the ending
    # learning rate, starting learning rate, and total number of
    # batch updates
    self.lrMultiplier = (endLR / startLR) ** (1.0 / numBatchUpdates)
    

    # create a temporary file path for the model weights and
    # then save the weights (so we can reset the weights when we
    # are done)
    self.weightsFile = tempfile.mkstemp()[1]
    self.model.save_weights(self.weightsFile)

    # grab the *original* learning rate (so we can reset it
    # later), and then set the *starting* learning rate
    origLR = K.get_value(self.model.optimizer.lr)
    K.set_value(self.model.optimizer.lr, startLR)

    # construct a callback that will be called at the end of each
    # batch, enabling us to increase our learning rate as training
    # progresses
    callback = LambdaCallback(on_batch_end=lambda batch, logs:self.on_batch_end(batch, logs))

    # check to see if we are using a data iterator
    if useGen:
      self.model.fit(
        x=trainData,
        steps_per_epoch=stepsPerEpoch,
        epochs=epochs,
        verbose=verbose,
        callbacks=[callback])
    # otherwise, our entire training data is already in memory
    else:
      # train our model using Keras' fit method
      self.model.fit(
        x=trainData[0], y=trainData[1],
        batch_size=batchSize,
        epochs=epochs,
        callbacks=[callback],
        verbose=verbose)
    # restore the original model weights and learning rate
    self.model.load_weights(self.weightsFile)
    K.set_value(self.model.optimizer.lr, origLR)

  def plot_loss(self, skipBegin=10, skipEnd=1, title=""):
    # grab the learning rate and losses values to plot
    lrs = self.lrs[skipBegin:-skipEnd]
    losses = self.losses[skipBegin:-skipEnd]

    # print(f"LRS : {lrs}")
    # print(f"losses : {losses}")

    # plot the learning rate vs. loss
    plt.plot(lrs, losses)
    plt.xscale("log")
    plt.xlabel("Learning Rate (Log Scale)")
    plt.ylabel("Loss")

    # if the title is not empty, add it to the plot
    if title != "":
      plt.title(title)