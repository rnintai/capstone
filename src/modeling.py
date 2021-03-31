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

# 알림, 차량 엔진, 차량 경적, 지하철 트리거 순
sound_data = np.load('updated_3D.npz')
X_train = sound_data['X']
y_train = sound_data['y']

print(X_train.shape, y_train.shape)

# X_train.shape, y_train.shape

K.clear_session()

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

model = Sequential() # Sequential Model
model.add(LSTM(20, input_shape=(100, 400))) # (timestep, feature)
model.add(Dense(1)) # output = 1
model.summary()
model.compile(loss='mean_squared_error', optimizer='adam')

#X_train = X_train.values
# X_train = X_train.reshape(X_train.shape[0], 40, 1)


early_stop = EarlyStopping(monitor='loss', patience=1, verbose=1)

model.fit(X_train, y_train, epochs=100,
          batch_size=20, verbose=1, callbacks=[early_stop])


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
