import numpy as np
#wav 파일들의 피처 생성
#librosa 사용
#사용 특성은 mfcc, chroma_stft, melspectorgram, spectral_contrast, tonnetz로 총193
#딥러닝 모델만 사용할 예정 -> 피처 축소 생략
import glob
import librosa

sample_rate = 8000
n_mels = 128
n_mfcc = 40
num_files = 3

# 오디오 불러오기 + 피쳐 생성
# 피쳐 193개
# row 통일 안시킴
def extract_feature(file_name):
    X = librosa.load(file_name, sr = sample_rate)[0]
    S = librosa.feature.melspectrogram(X, sr=sample_rate, n_mels=128)
    log_S = librosa.power_to_db(S, ref=np.max)
    mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=20)
    # abs values & db to get magnitude
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    return delta2_mfcc

#데이터 가공
#행렬로 변환
def parse_audio_files(filenames):
    rows = len(filenames)
    # feature는 각 파일 별 row(window) * 피처 의 2차원 행렬
    # labels은 파일 별 카테고리 int 값
    features, labels = np.zeros((rows,num_files)), np.zeros((rows, 1))
    i = 0
    for fn in filenames:
        # try:
            mfccs = extract_feature(fn)
            print(features.shape)
            y_col = int(fn.split('-')[1])
        # except:
            # print("error : "+fn)
        # else:
            features[i] = mfccs
            labels[i] = y_col
            print(y_col)
            i += 1
    return features, labels

audio_files = []
#0 : 사이렌
#1 : 자동차가 다가오는 소리(엔진소리)
#2 : 자동차 경적소리
#4 : 환승역 안내음
audio_files.extend(glob.glob('*.wav'))

print(len(audio_files))

files = audio_files
X, y = parse_audio_files(files)

#?.npz
np.savez('data', X=X, y=y)