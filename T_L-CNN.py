#!/usr/bin/env python
# coding: utf-8

# # Importing Necessary Packages


import csv
import pandas as pd
from IPython.display import display

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential,Model
from keras.models import Model
from keras.layers import Input
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import Dense, Activation
from keras.layers import Dropout, BatchNormalization
from keras.callbacks import ReduceLROnPlateau,EarlyStopping
from matplotlib import pyplot
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.merge import concatenate
from keras.utils import np_utils
from keras import backend as K
#from keras.utils.vis_utils import plot_model
K.set_image_dim_ordering('th')


# ## Load Array files


# Load pp -> w jet image
files = 10
X_ppwj = []
for i in range(files):
    for j in range(5):
        Xtemp = np.load("/afs/crc.nd.edu/user/t/tkim12/Work/MadGraph/TRR_clustering/pTwindow400-500/pp2wj/jet_img-singevnt/run_"+str(i+1)+"-"+str(j)+"_jet_img_pTmax-Norm-single-W-events.npy")
        for k in range(Xtemp.shape[0]):
            X_ppwj.append(Xtemp[k])


# Load pp -> h -> ww image
X_pphww = []
for i in range(files):
    for j in range(5):
        Xtemp = np.load("/afs/crc.nd.edu/user/t/tkim12/Work/MadGraph/TRR_clustering/pTwindow400-500/pp2h2ww/jet_img-singevnt/run_"+str(i+1)+"-"+str(j)+"_jet_img_pTmax-Norm-single-W-events.npy")
        for k in range(Xtemp.shape[0]):
            X_pphww.append(Xtemp[k])


# Load pp -> j j image
#X_ppjj = []
#for i in range(files):
#    for j in range(5):
#        Xtemp = np.load("/afs/crc.nd.edu/user/t/tkim12/Work/MadGraph/TRR_clustering/pTwindow400-500/pp2jj/jet_img-singevnt/run_"+str(i+1)+"-"+str(j)+"_dijet_img_pTmax-Norm-single-QCD-events.npy")
#        for k in range(Xtemp.shape[0]):
#            X_ppjj.append(Xtemp[k])


X_ppwj = np.array(X_ppwj)
X_pphww = np.array(X_pphww)
#X_ppjj = np.array(X_ppjj)

ppwj_num = X_ppwj.shape[0]
pphww_num = X_pphww.shape[0]
#ppjj_num = X_ppjj.shape[0]

minnum = min([ppwj_num,pphww_num])
print minnum

X_data = []
for i in range(minnum):
    X_data.append((X_ppwj[i])/np.amax(X_ppwj[i]))
    
for i in range(minnum):
    X_data.append((X_pphww[i])/np.amax(X_pphww[i]))

#for i in range(minnum):
#    X_data.append((X_ppjj[i])/np.amax(X_ppjj[i]))
    

Y_data = []
for i in range(minnum):
    Y_data.append(0)  # 0 for pp -> w jet

for i in range(minnum):
    Y_data.append(1)  # 1 for pp -> h -> ww

#for i in range(minnum):
#    Y_data.append(2)  # 2 for pp -> dijet
    
X_data = np.array(X_data)
Y_data = np.array(Y_data)


print X_data.shape



# Data Splitting
ts1 = 0.2
rs1 = 42
X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size = ts1, random_state = rs1)



X_train = X_train.reshape(X_train.shape[0], 1, 20, 20).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 20, 20).astype('float32')

#Y_train = np_utils.to_categorical(Y_train)
#Y_test = np_utils.to_categorical(Y_test)
#num_classes = Y_test.shape[1]
#print num_classes

print "Training sample : "+str(X_train.shape[0])+" , Validation sample : "+str(X_test.shape[0])



# ## Neutral Network Building


# Channel 1
model = Sequential()
model.add(Conv2D(20, kernel_size=4,padding="same",input_shape=(1, 20, 20), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.3))
model.add(Conv2D(40, kernel_size=4, padding="same", activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.3))
model.add(Flatten())

model.add(Dense(units=300, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(units=100, activation='relu'))
model.add(Dense(units=100, activation='relu'))
model.add(Dense(units=1, activation = 'sigmoid'))

model.summary()
#plot_model(model, show_shapes=True, show_layer_names=True)


model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy','binary_crossentropy'])

# enable early stopping based on mean_squared_error
earlystopping = EarlyStopping(monitor="binary_crossentropy", patience=20, verbose=1, mode='auto')


# Reduce learning rate(lr) when a metric has stopped improving
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1,
                             patience=10, min_lr = 0.0001) 

history = model.fit(X_train, Y_train, epochs=400, verbose=2, 
                      batch_size=32, callbacks=[reduce_lr,earlystopping], validation_data=(X_test, Y_test))



# ## Final evaluation of the model

scores = model.evaluate(X_test, Y_test, verbose=0)
print("Large CNN Error: %.2f%%" % (100-scores[1]*100))



# plot training curve for categorical_crossentropy
crossent, ax = plt.subplots(1,1)
ax.plot(history.history['binary_crossentropy'])
ax.plot(history.history['val_binary_crossentropy'])
ax.set_title('model binary_crossentropy')
ax.set_ylabel('binary_crossentropy')
ax.set_xlabel('epoch')
ax.legend(['train', 'validation'], loc='lower right')
crossent.savefig('/afs/crc.nd.edu/user/t/tkim12/Work/MadGraph/TRR_clustering/pTwindow400-500/CNN_v2/results/binary_crossentropy.pdf')


# plot training curve for accuracy
acc, ax = plt.subplots(1,1)
ax.plot(history.history['acc'])
ax.plot(history.history['val_acc'])
ax.set_title('accuracy')
ax.set_ylabel('accuracy')
ax.set_xlabel('epoch')
ax.legend(['train', 'validation'], loc='upper left')
acc.savefig('/afs/crc.nd.edu/user/t/tkim12/Work/MadGraph/TRR_clustering/pTwindow400-500/CNN_v2/results/accuracy.pdf')


# Save the model
model.save('/afs/crc.nd.edu/user/t/tkim12/Work/MadGraph/TRR_clustering/pTwindow400-500/CNN_v2/results/single-evnt-saved_model.h5')



#for layer in model.layers:
#    weights = layer.get_weights() # list of numpy arrays
#    print weights
