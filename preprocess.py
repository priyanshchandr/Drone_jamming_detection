import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt

def generate_spectrogram(signal, sampling_rate=1000):
    """Converts the signal to a spectrogram."""
    S = librosa.feature.melspectrogram(y=signal, sr=sampling_rate, n_mels=128)
    return librosa.power_to_db(S, ref=np.max)

def plot_spectrogram(signal):
    """Plots the spectrogram of the signal."""
    spectrogram = generate_spectrogram(signal)
    librosa.display.specshow(spectrogram, sr=1000, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title("Spectrogram")
    plt.show()

# Load the jammed signal from CSV (or replace with normal_signal.csv if you want to test the normal signal)
df_jammed = pd.read_csv('jammed_signal.csv')
jammed_signal = df_jammed['Jammed Signal'].values  # Extract the signal data

# Example usage: plot spectrogram for the jammed signal
plot_spectrogram(jammed_signal)
