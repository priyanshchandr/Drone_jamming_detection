import numpy as np
import librosa
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('anti_drone_model.h5')

# Generate a synthetic signal (e.g., a sine wave)
sample_rate = 44100  # Adjust based on your signal's sample rate
duration = 2  # 2 seconds
frequency = 1000  # Frequency of the sine wave (adjust as needed)
t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
synthetic_signal = 0.5 * np.sin(2 * np.pi * frequency * t)  # A sine wave

# Generate spectrogram for the synthetic signal
def generate_spectrogram(signal, sample_rate=44100):
    """Generate spectrogram of a signal."""
    padded_signal = np.pad(signal, (0, max(0, 2048 - len(signal))), mode='constant')
    S = librosa.feature.melspectrogram(y=padded_signal, sr=sample_rate, n_fft=512, n_mels=128)
    return S

new_spectrogram = generate_spectrogram(synthetic_signal)
new_spectrogram = np.expand_dims(new_spectrogram, axis=-1)  # Add channel dimension
new_spectrogram = np.resize(new_spectrogram, (64, 64, 1))  # Resize to match model input

# Make a prediction with the synthetic signal
prediction = model.predict(np.expand_dims(new_spectrogram, axis=0))
print("Prediction (0 = Normal, 1 = Jammed):", prediction)
