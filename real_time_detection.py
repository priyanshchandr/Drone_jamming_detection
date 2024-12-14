import numpy as np
import tensorflow as tf
import librosa

# Load the trained model
model = tf.keras.models.load_model('anti_drone_model.h5')

def generate_spectrogram(signal, sample_rate=22050, n_fft=2048, hop_length=512, n_mels=128):
    """Generate a Mel spectrogram from the audio signal."""
    S = librosa.feature.melspectrogram(y=signal, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    # Convert to log scale (dB)
    log_S = librosa.power_to_db(S, ref=np.max)
    
    # Print the shape of the spectrogram to debug input size
    print(f"Generated Spectrogram Shape: {log_S.shape}")
    
    return log_S

def detect_jamming(new_signal):
    """Detects if the signal is jammed or normal."""
    # Generate the spectrogram from the signal
    spectrogram = generate_spectrogram(new_signal)
    
    # Ensure the spectrogram has the correct dimensions: (height, width, channels)
    spectrogram = spectrogram[..., np.newaxis]  # Adds a new axis to make it (height, width, 1)
    
    # Resize the spectrogram to the expected input shape (64, 64) using TensorFlow's resize function
    spectrogram_resized = tf.image.resize(spectrogram, [64, 64])
    
    # Ensure the spectrogram has the correct dimensions for the model (batch size, height, width, channels)
    spectrogram_resized = spectrogram_resized.numpy().reshape(1, 64, 64, 1)  # Add batch dimension

    # Predict the class of the signal (Jammed or Normal)
    prediction = model.predict(spectrogram_resized)
    
    # Return the result based on the prediction
    return "Jammed" if np.argmax(prediction) == 1 else "Normal"

# Example usage:
# Replace `jammed_signal` with the actual signal data for testing
# Let's generate a sample normal signal using a sine wave for testing
duration = 2.0  # 2 seconds
sample_rate = 22050  # Standard sample rate
t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
normal_signal = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave (A4 note)

# Detect jamming for a sample normal signal
result_normal = detect_jamming(normal_signal)
print(f"Normal Signal Status: {result_normal}")

# Example of a jammed signal (this is just an example, you need real jammed signal data)
# For demo, you could simulate a jammed signal with noise or other disturbances
jammed_signal = np.random.normal(0, 1, size=normal_signal.shape)  # Simulated random noise

# Detect jamming for the sample jammed signal
result_jammed = detect_jamming(jammed_signal)
print(f"Jammed Signal Status: {result_jammed}")
