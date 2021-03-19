import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

FIG_SIZE = (12,4)
file = "/home/mintai/cap/snd/24074-1-0-2.wav"

sr = 8000
hop_length = 512  # interval, 낮으면 깔끔
n_fft = 1024      # ?? 높으면 깔끔

x = librosa.load(file, sr = 8000)[0]
# print(sample_rate)
# n_fft_duration = float(n_fft)/sample_rate
# hop_length_duration = float(hop_length)/sample_rate

# STFT & abs
# stft = np.abs(librosa.stft(signal, n_fft=n_fft, hop_length = hop_length))
S = librosa.feature.melspectrogram(x, sr=sr, n_mels=128)
log_S = librosa.power_to_db(S, ref=np.max)
mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=20)
# abs values & db to get magnitude
delta2_mfcc = librosa.feature.delta(mfcc, order=2)
print(delta2_mfcc.shape)
print(delta2_mfcc)


# display
plt.figure(figsize=FIG_SIZE)

librosa.display.specshow(delta2_mfcc)
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.colorbar()
# plt.colorbar(format="%+2.0f dB")
plt.title("Spectogram (dB) of " + file)

fig = plt.gcf()

plt.show()
fig.savefig('3.png')
