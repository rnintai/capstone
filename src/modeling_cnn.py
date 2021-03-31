import glob
import librosa
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import joblib

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical


from keras import models
from keras import layers
from keras.layers import *
from keras import optimizers

from keras.layers import LSTM
from keras.models import Sequential
from keras.layers import Dense
import keras.backend as K
from keras.callbacks import EarlyStopping

class AcousticSoundModel(tf.keras.Model):
    def __init__(self):
        super(AcousticSoundModel, self).__init__()
        self.conv1 = Conv2D(filters=64, kernel_size=[3, 3], padding='SAME', activation=tf.nn.relu)
        self.drop1 = Dropout(rate=0.2)
        self.pool1 = MaxPool2D(padding='SAME') ###### pooling 2x2. stride는 표기 x, 확인 ######
        
        self.conv2 = Conv2D(filters=64, kernel_size=[3, 3], padding='SAME', activation=tf.nn.relu)
        self.drop2 = Dropout(rate=0.2) #20% dropout
        self.pool2 = MaxPool2D(padding='SAME')
        
        self.conv3 = Conv2D(filters=64, kernel_size=[3, 3], padding='SAME', activation=tf.nn.relu)
        self.drop3 = Dropout(rate=0.2) #20% dropout
        self.pool3 = MaxPool2D(padding='SAME')
        
        self.pool3_flat = keras.layers.Flatten()
        self.dense4 = Dense(units=128, activation=tf.nn.relu)
        self.dense5 = Dense(units=5, activation=tf.nn.sigmoid) ### 일단 5. class 개수 추가되는 대로 변경
        
    def call(self, inputs, training=False):
        net = self.conv1(inputs)
        net = self.drop1(net)
        net = self.pool1(net)
        
        net = self.conv2(net)
        net = self.drop2(net)
        net = self.pool2(net)
        
        net = self.conv3(net)
        net = self.drop3(net)
        net = self.pool3(net)
        
        net = self.pool3_flat(net)
        net = self.dense4(net)
        net = self.dense5(net) #
        return net

# 알림, 차량 엔진, 차량 경적, 지하철 트리거 순
sound_data = np.load('data3D.npz')
X_train = sound_data['X']
y_train = sound_data['y']

print(X_train.shape, y_train.shape)

# X_train.shape, y_train.shape

K.clear_session()

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[2], 40,1)

model = AcousticSoundModel()
model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adam(lr=0.001),
              metrics=['acc'])

history = model.fit(
      X_train,y_train,
      epochs=100,
      batch_size=20
      )





# 모델 pkl로 저장하기
# joblib.dump(model, '/home/mintai/capstone/src/model/pkl/model_1.pkl')

# 모델 json으로 저장하기
# model_1 = model.to_json()
# model = model_from_json(json_string)

# 모델 h5로 저장하기
# from keras.models import load_model
# model.save('model/h5/model_1')
# model.save('model/h5/model_1.h5')

# 모델 pb로 저장하기
# model = keras.models.load_model('model/h5/model_1', compile=False)
# model.save('model/pb/',save_format=tf)

#모델 tflite 로 저장하기
# saved_model_dir='model/pb/'
# converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
# converter.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS,
#                                      tf.lite.OpsSet.SELECT_TF_OPS]
# tfilte_mode=converter.convert()
# open('model/tflite/model_1.tflite','wb').write(ftlite_model)
