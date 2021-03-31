import numpy as np
import glob
import librosa
from sklearn.preprocessing import MinMaxScaler
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

sample_rate = 16000
n_mfcc = 100
n_fft = 400
hop_length = 160
my_path = "/home/mintai/capstone/snd/snd2/"

# Data set list, include (raw data, mfcc data, y data)
trainset = []
testset = []

# split each set into raw data, mfcc data, and y data
# STFT 한 것, CNN 분석하기 위해 Spectogram으로 만든 것, MF한 것, mel-spectogram 한 것
train_X = []
train_mfccs = []
train_y = []

test_X = []
test_mfccs = []
test_y = []

# 모든 음성파일의 길이가 같도록 후위에 padding 처리
pad1d = lambda a, i: a[0: i] if a.shape[0] > i else np.hstack((a, np.zeros(i-a.shape[0])))
pad2d = lambda a, i: a[:, 0:i] if a.shape[1] > i else np.hstack((a, np.zeros((a.shape[0], i-a.shape[1]))))

frame_length = 0.025
frame_stride = 0.0010


for filename in os.listdir(my_path + 'train/'):
    filename = normalize('NFC', filename)
    try:
        # wav 포맷 데이터만 사용
        if '.wav' not in filename in filename:
            continue
        
        wav, sr = librosa.load(my_path + 'train/' + filename, sr=16000)

        mfccs = librosa.feature.mfcc(wav, sr=16000, n_mfcc=100, n_fft=400, hop_length=160)
        mms = MinMaxScaler()
        print(mms.fit(mfccs))
        train_data_mmsed = mms.transform(mfccs)
        print(train_data_mmsed)
        padded_mfcc = pad2d(mfccs, 40)

        # 소리 별로 dataset에 추가 int(fn.split('-')[1])
        if filename.split('-')[1] == '1':
            trainset.append((padded_mfcc, 0))
        elif filename.split('-')[1] == '8':
            trainset.append((padded_mfcc, 1))
    except Exception as e:
        print(filename, e)
        raise

# 학습 데이터를 무작위로 섞는다.
random.shuffle(trainset)

for filename in os.listdir(my_path + 'test/'):
    filename = normalize('NFC', filename)
    try:
        # wav 포맷 데이터만 사용
        if '.wav' not in filename in filename:
            continue
        
        wav, sr = librosa.load(my_path + 'test/' + filename, sr=16000)
        
        mfccs = librosa.feature.mfcc(wav, sr=16000, n_mfcc=100, n_fft=400, hop_length=160)
        mms = MinMaxScaler()
        print(mms.fit(mfccs))
        test_data_mmsed = mms.transform(mfccs)
        print(test_data_mmsed)
        padded_mfcc = pad2d(mfccs, 40)

        # 소리 별로 dataset에 추가 int(fn.split('-')[1])
        if filename.split('-')[1] == '1':
            testset.append((padded_mfcc, 0))
        elif filename.split('-')[1] == '8':
            testset.append((padded_mfcc, 1))
    except Exception as e:
        print(filename, e)
        raise

# 학습 데이터를 무작위로 섞는다.
random.shuffle(testset)

train_mfccs = [a for (a,b) in trainset]
train_y = [b for (a,b) in trainset]

test_mfccs = [a for (a,b) in testset]
test_y = [b for (a,b) in testset]


train_mfccs = np.array(train_mfccs)
train_y = to_categorical(np.array(train_y))
train_y = np.column_stack([train_y,np.zeros(train_y.shape[0])]) # check

test_mfccs = np.array(test_mfccs)
test_y = to_categorical(np.array(test_y))
test_y = np.column_stack([test_y,np.zeros(test_y.shape[0])])

np.savez('small_example.npz', rm=train_mfccs, ry=train_y, em=test_mfccs, ey=test_y)



