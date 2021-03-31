import numpy as np #
import librosa
import matplotlib.pyplot as plt
import os
from scipy.io import wavfile
from collections import defaultdict, Counter
from scipy import signal
import random
import sklearn
from unicodedata import normalize
import tensorflow as tf
from keras.layers import Dense
from keras import Input
from keras.engine import Model
from keras.utils import to_categorical
from keras.layers import Dense, TimeDistributed, Dropout, Bidirectional, GRU, BatchNormalization, Activation, LeakyReLU, LSTM, Flatten, RepeatVector, Permute, Multiply, Conv2D, MaxPooling2D
import keras.backend as K
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from keras.callbacks import EarlyStopping


dataset = np.load('padded.npz')
train_mfccs = dataset['rm']
train_y = dataset['ry']
test_mfccs = dataset['em']
test_y = dataset['ey']

print('train_mfccs:', train_mfccs.shape)
print('train_y:', train_y.shape)
# print(train_y)

print('test_mfccs:', test_mfccs.shape)
print('test_y:', test_y.shape)
# print(test_y)

train_X_ex = np.expand_dims(train_mfccs, -1)
test_X_ex = np.expand_dims(test_mfccs, -1)
print('train X shape:', train_X_ex.shape)
print('test X shape:', test_X_ex.shape)

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

ip = Input(shape=train_X_ex[0].shape)

m = Conv2D(32, kernel_size=(4,4), activation='relu')(ip)
m = MaxPooling2D(pool_size=(4,4))(m)

m = Conv2D(32*2, kernel_size=(4,4), activation='relu')(ip)
m = MaxPooling2D(pool_size=(4,4))(m)

m = Conv2D(32*3, kernel_size=(4,4), activation='relu')(ip)
m = MaxPooling2D(pool_size=(4,4))(m)

m = Flatten()(m)

m = Dense(64, activation='relu')(m)

m = Dense(36, activation='relu')(m)

op = Dense(3, activation='softmax')(m)

model = Model(ip, op)

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
#               metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
        
early_stop = EarlyStopping(monitor='loss', patience=1, verbose=1)


history = model.fit(train_X_ex,
                    train_y,
                    epochs=20,
                    batch_size=16,
                    verbose=1,
                    validation_data=(test_X_ex, test_y),
                    callbacks=[early_stop])

# Plotting

plt.figure(figsize=(8,8))
plt.subplot(2, 1, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend(loc='upper right')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.show()

model.save('/home/mintai/capstone/src/cnn_model.h5')