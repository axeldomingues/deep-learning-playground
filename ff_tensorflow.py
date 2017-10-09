from sklearn.utils import shuffle
import sklearn as sk
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import numpy as np
from numpy import inf
from numpy import nan
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from time import monotonic as timer # or time.time if it is not available
import dask.dataframe as dd
import dask.array as da
from dask import compute 

def to_dask_chunks(df, dtype):
  partitions = df.to_delayed()
  shapes = [part.values.shape for part in partitions]

  shapes = compute(*shapes)  # trigger computation to find shape

  chunks = [da.from_delayed(part.values, shape, dtype) for part, shape in zip(partitions, shapes)]
  return chunks

#Weight Initialization

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.01, dtype=tf.float32)
  return tf.Variable(initial, dtype=tf.float32)

def bias_variable(shape):
  initial = tf.constant(0.01, shape=shape, dtype=tf.float32)
  return tf.Variable(initial, dtype=tf.float32)
  
def y2indicator(y):
    N = len(y)
    K = len(set(y))
    ind = np.zeros((N, K))
    for i in range(N):
        ind[i, y[i]] = 1
    return ind
  
def main():
  
  train_costs = []
  validation_costs = []
  
  train_df = dd.read_csv('_TrainningData.csv', header=None, assume_missing=True, dtype=np.float32)
  validation_df = pd.read_csv('TrainningData.csv', header=None, dtype=np.float32)
  
  train_dk_chunks = to_dask_chunks(train_df, np.float32)
  
  #Prepare validation data set
  
  validationData = validation_df.values.astype(np.float32)
  
  del validation_df
  
  Xvalid, Yvalid = validationData[:,1:], validationData[:,0]
  
  del validationData
  
  K = len(set(Yvalid)) # won't work later b/c we turn it into indicator
  
  Yvalid = y2indicator(Yvalid.astype(np.int)).astype(np.float32)
  
  train_dk_array = da.concatenate(train_dk_chunks, axis=0)   
  
  mean = train_dk_array.mean(axis=0).compute()
  sigma = train_dk_array.std(axis=0).compute()
  
  mean = mean[1:]
  sigma = sigma[1:]

  Xvalid = (Xvalid - mean)/sigma
  Xvalid[Xvalid == inf] = 0
  Xvalid[Xvalid == -inf] = 0
  where_are_NaNs = np.isnan(Xvalid)
  Xvalid[where_are_NaNs] = 0
  Xvalid = Xvalid.astype(np.float32)  
  
  # initialize hidden layers
  N, D = train_dk_array.shape
  
  D -= 1 # we have the y record there
  
  x = tf.placeholder(tf.float32, shape=[None, D])
  y_ = tf.placeholder(tf.float32, shape=[None, K])
  
  # network weights
  W_fc1 = weight_variable([D, 1024])
  b_fc1 = bias_variable([1024])
	
  W_fc2 = weight_variable([1024, 512])
  b_fc2 = bias_variable([512])
	
  W_fc3 = weight_variable([512, 256])
  b_fc3 = bias_variable([256])
	
  W_fc4 = weight_variable([256, 128])
  b_fc4 = bias_variable([128])
	
  W_fc5 = weight_variable([128, 64])
  b_fc5 = bias_variable([64])

  W_fc6 = weight_variable([64, K])
  b_fc6 = bias_variable([K])

  # input layer
  # x is the input layer

  # hidden layers
  h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)

  h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
	
  h_fc3 = tf.nn.relu(tf.matmul(h_fc2, W_fc3) + b_fc3)
	
  h_fc4 = tf.nn.relu(tf.matmul(h_fc3, W_fc4) + b_fc4)
	
  h_fc5 = tf.nn.relu(tf.matmul(h_fc4, W_fc5) + b_fc5)

  # Output layer
  y_ff = tf.matmul(h_fc5, W_fc6) + b_fc6  

  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_ff))
  train_step = tf.train.AdamOptimizer(10e-4).minimize(cross_entropy)
  
  argmax_prediction = tf.argmax(y_ff, 1)
  argmax_y = tf.argmax(y_, 1)
  
  batch_sz = 200
  partition_sz = 1000000 #1 million records will be loaded at each time for train
  
  n_partitions = N // partition_sz
  n_batches = partition_sz // batch_sz
  
  target_names = ['Neutral', 'Back', 'Lay']

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    start_time = timer()
    for i in range(30):# 150 epochs
      train_dk_chunks = shuffle(train_dk_chunks)
      train_dk_array = da.concatenate(train_dk_chunks, axis=0)
      for p in range(n_partitions):
	    #Preparing the training partition
        print('Partition %g' % (p))
        trainingData = train_dk_array[p*partition_sz:(p*partition_sz+partition_sz)].compute()
        np.random.shuffle(trainingData)
        Xtrain, Ytrain= trainingData[:,1:], trainingData[:,0]
        del trainingData
        Ytrain = y2indicator(Ytrain.astype(np.int)).astype(np.float32)
        Xtrain = (Xtrain - mean)/sigma
        Xtrain[Xtrain == inf] = 0
        Xtrain[Xtrain == -inf] = 0
        where_are_NaNs = np.isnan(Xtrain)
        Xtrain[where_are_NaNs] = 0
        Xtrain = Xtrain.astype(np.float32)
		#Preparing the aprox data
        Xvalid, Yvalid = shuffle(Xvalid, Yvalid)
        XvalidAprox, YvalidAprox = Xvalid[:100000], Yvalid[:100000]
        XtrainAprox, YtrainAprox = Xtrain[:100000], Ytrain[:100000]
		#Running over the this partition batches
        for j in range(n_batches):
          Xbatch = Xtrain[j*batch_sz:(j*batch_sz+batch_sz)]
          Ybatch = Ytrain[j*batch_sz:(j*batch_sz+batch_sz)]
          if j == 0 and p == 0:
            train_cost = cross_entropy.eval(feed_dict={x: XtrainAprox, y_: YtrainAprox})
            test_cost = cross_entropy.eval(feed_dict={x: XvalidAprox, y_: YvalidAprox})
            train_costs.append(train_cost)
            validation_costs.append(test_cost)
            labels = argmax_y.eval(feed_dict={x: XtrainAprox, y_: YtrainAprox})
            predictions = argmax_prediction.eval(feed_dict={x: XtrainAprox, y_: YtrainAprox})
            print('---Training---')
            print(sk.metrics.confusion_matrix(labels, predictions))
            print(sk.metrics.classification_report(labels, predictions, target_names=target_names))
            labels = argmax_y.eval(feed_dict={x: XvalidAprox, y_: YvalidAprox})
            predictions = argmax_prediction.eval(feed_dict={x: XvalidAprox, y_: YvalidAprox})
            print('---Test---')
            print(sk.metrics.confusion_matrix(labels, predictions))
            print(sk.metrics.classification_report(labels, predictions, target_names=target_names))
		
          if j % 125 == 0:
            train_cost = cross_entropy.eval(feed_dict={x: XtrainAprox, y_: YtrainAprox})		  
            test_cost = cross_entropy.eval(feed_dict={x: XvalidAprox, y_: YvalidAprox})
            print('epoch %d, training cost %g, test cost %g' % (i, train_cost, test_cost))
          train_step.run(feed_dict={x: Xbatch, y_: Ybatch})
	
  print('Elapsed time %g' % (timer()-start_time))	
  plt.plot(train_costs, label="training")
  plt.plot(validation_costs, label="validation")
  plt.legend()
  plt.show()

if __name__ == '__main__':
  main()
