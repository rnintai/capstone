import numpy as np
#wav 파일들의 피처 생성
#librosa 사용
#사용 특성은 mfcc, chroma_stft, melspectorgram, spectral_contrast, tonnetz로 총193
#딥러닝 모델만 사용할 예정 -> 피처 축소 생략
import glob
import librosa
import os.path

sample_rate = 8000
n_mels = 128
n_mfcc = 40
n_fft = 2048
my_path = "/home/mintai/capstone/snd"

# 오디오 불러오기 + 피쳐 생성
# 피쳐 40개
# row 통일 안시킴
def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name)
    stft = np.abs(librosa.stft(X,n_fft = n_fft))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate,n_fft=n_fft, n_mfcc=n_mfcc).T,axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate, n_fft=n_fft).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate, n_fft=n_fft).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate,n_fft=n_fft).T,axis=0)
    # tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
    return mfccs,chroma,mel,contrast
    # return mfccs,chroma,mel,contrast,tonnetz

#데이터 가공
#행렬로 변환
def parse_audio_files(filenames):
    rows = len(filenames)
    print(rows)
    # feature는 각 파일 별 row(window) * 피처 의 2차원 행렬
    # labels은 파일 별 카테고리 int 값
    features, labels = np.zeros((rows,n_mfcc)), np.zeros((rows, 1))
    i = 0
    for fn in filenames:
        # try:
            # mfccs, chroma, mel, contrast, tonnetz = extract_feature(fn)
            mfccs, chroma, mel, contrast = extract_feature(fn)
            
            # print(mfccs.shape, chroma.shape, mel.shape, contrast.shape, tonnetz)
            print(mfccs.shape, chroma.shape, mel.shape, contrast.shape)

            # ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
            ext_features = np.hstack([mfccs,chroma,mel,contrast])
            y_col = int(fn.split('-')[1])
        # except:
        #     print("error : "+fn)
        # else:
            # features[i] = mfccs
            # labels[i] = y_col
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
np.savez('prep_all_siren_horn', X=X, y=y)