import numpy as np
from tensorflow.keras.models import load_model
import librosa

# Load signal from CSV
def load_signal_from_csv(file_path):
    # Load the CSV and skip the header row
    data = np.genfromtxt(file_path, delimiter=',', skip_header=1)
    # The signal data is in the second column, we return it as a float32 array
    return data[:, 1].astype(np.float32)  # Ensure the signal is in float32 format

# Generate spectrogram
def generate_spectrogram(signal, sample_rate=44100):
    """Generate spectrogram of a signal."""
    # Padding signal to prevent size issues with n_fft
    padded_signal = np.pad(signal, (0, max(0, 2048 - len(signal))), mode='constant')
    
    # Adjust n_fft for shorter signals and generate mel spectrogram
    S = librosa.feature.melspectrogram(y=padded_signal, sr=sample_rate, n_fft=512, n_mels=128)
    
    return S

# Load and prepare test data
def prepare_test_data():
    # Test signal files (you should specify paths to actual test files)
    test_normal_file = 'normal_signal.csv'  # Replace with actual file path
    test_jammed_file = 'jammed_signal.csv'  # Replace with actual file path

    # Load the signals
    test_normal_signal = load_signal_from_csv(test_normal_file)
    test_jammed_signal = load_signal_from_csv(test_jammed_file)

    # Generate spectrograms for both signals
    test_normal_spectrogram = generate_spectrogram(test_normal_signal)
    test_jammed_spectrogram = generate_spectrogram(test_jammed_signal)

    # Expand dimensions (add channel dimension)
    test_normal_spectrogram = np.expand_dims(test_normal_spectrogram, axis=-1)
    test_jammed_spectrogram = np.expand_dims(test_jammed_spectrogram, axis=-1)

    # Resize to match the input shape of the model
    test_normal_spectrogram = np.resize(test_normal_spectrogram, (test_normal_spectrogram.shape[0], 64, 64, 1))
    test_jammed_spectrogram = np.resize(test_jammed_spectrogram, (test_jammed_spectrogram.shape[0], 64, 64, 1))

    # Combine both signals for evaluation
    X_test = np.concatenate((test_normal_spectrogram, test_jammed_spectrogram), axis=0)
    y_test = np.concatenate((np.zeros(len(test_normal_spectrogram)), np.ones(len(test_jammed_spectrogram))), axis=0)

    return X_test, y_test

# Main function for evaluation
if __name__ == "__main__":
    # Load the trained model
    model = load_model('anti_drone_model.h5')

    # Prepare the test data
    X_test, y_test = prepare_test_data()

    # Evaluate the model on the test data
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Make predictions on the test data
    predictions = model.predict(X_test)
    print("Predictions:", predictions)
    
    # Optionally, you can print the first few predictions to inspect
    for i in range(5):
        print(f"Test Sample {i+1}: Prediction = {predictions[i][0]}, True Label = {y_test[i]}")
