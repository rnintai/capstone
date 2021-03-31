import numpy as np
#wav 파일들의 피처 생성
#librosa 사용
#사용 특성은 mfcc, chroma_stft, melspectorgram, spectral_contrast, tonnetz로 총193
#딥러닝 모델만 사용할 예정 -> 피처 축소 생략
import glob
import librosa
from sklearn.preprocessing import MinMaxScaler
import os.path

sample_rate = 16000
n_mfcc = 100
n_fft = 400
hop_length = 160
my_path = "/home/mintai/capstone/snd/snd2"

# 오디오 불러오기 + 피쳐 생성
# 피쳐 193개
# row 통일 안시킴
def extract_feature(file_name):
    X = librosa.load(file_name, sr = sample_rate)[0]
    # F = np.abs(librosa.stft(X,n_fft=512,win_length=512, hop_length = 128))
    # mfccs = librosa.feature.mfcc(S=librosa.power_to_db(F),sr=sample_rate,n_mfcc=n_mfcc)
    # mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=n_mfcc).T,axis=0)
    mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    # mfccs = sklearn.preprocessing.scale(mfccs, axis=1)
    mms = MinMaxScaler()
    print(mms.fit(mfccs))
    train_data_mmsed = mms.transform(mfccs)
    print(train_data_mmsed)
    pad2d = lambda a, i: a[:, 0:i] if a.shape[1] > i else np.hstack((a, np.zeros((a.shape[0], i-a.shape[1]))))
    padded_mfcc = pad2d(mfccs, 400)

    print('audio shape: ', padded_mfcc.shape)
    print('length: ', X.shape[0]/float(sample_rate), 'secs')
    # librosa.display.specshow(padded_mfcc, sr=16000, x_axis='time')
    return padded_mfcc

#데이터 가공
#행렬로 변환
def parse_audio_files(filenames):
    rows = len(filenames)
    print('rows: ',rows)
    # feature는 각 파일 별 row(window) * 피처 의 2차원 행렬
    # labels은 파일 별 카테고리 int 값
    features, labels = np.zeros((rows,n_mfcc,400)), np.zeros((rows,n_mfcc,1))
    i = 0
    for fn in filenames:
        try:
            mfccs = extract_feature(fn)
            # ext_features = np.hstack([mfccs])
            # print(mfccs.shape)
            y_col = int(fn.split('-')[1])
        except:
            print("error : "+fn)
        else:
            features[i] = mfccs
            labels[i] = y_col
            print(y_col)
            i += 1
    return features, labels

# files = []
# file = glob.glob(os.path.join(my_path, '*.wav'))
# files.extend(file)
# for a in files:
#     print(a)

audio_files = []
#0 : 사이렌
#1 : 자동차가 다가오는 소리(엔진소리)
#2 : 자동차 경적소리
#4 : 환승역 안내음
audio_files.extend(glob.glob(os.path.join(my_path,'*.wav')))

print(len(audio_files))

files = audio_files
X, y = parse_audio_files(files)

#?.npz
# np.savez('updated_3D', X=X, y=y)