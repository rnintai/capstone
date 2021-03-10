import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

FIG_SIZE = (15,10)
file = "/home/mintai/cap/snd/24074-1-0-2.wav"

sr = 8000
hop_length = 512  # interval, 낮으면 깔끔
n_fft = 1024      # ?? 높으면 깔끔

signal, sample_rate = librosa.load(file, sr = sr)
# print(sample_rate)
n_fft_duration = float(n_fft)/sample_rate
hop_length_duration = float(hop_length)/sample_rate

# STFT
stft = librosa.stft(signal, n_fft=n_fft, hop_length = hop_length)

# abs values & db to get magnitude
spectogramDB = librosa.amplitude_to_db(np.abs(stft))
print(spectogramDB.shape)
print(spectogramDB)
# display
plt.figure(figsize=FIG_SIZE)

librosa.display.specshow(spectogramDB,sr = sample_rate, hop_length = hop_length)
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.colorbar(format="%+2.0f dB")
plt.title("Spectogram (dB) of " + file)

fig = plt.gcf()

plt.show()
fig.savefig('3.png')
