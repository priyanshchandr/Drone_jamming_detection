import numpy as np
import librosa
from tensorflow.keras import layers, models

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

# Generate features for training
def generate_features():
    normal_signal_file = 'normal_signal.csv'  # Update with actual file path
    jammed_signal_file = 'jammed_signal.csv'  # Update with actual file path

    # Load signals
    normal_signal = load_signal_from_csv(normal_signal_file)
    jammed_signal = load_signal_from_csv(jammed_signal_file)

    # Generate spectrograms
    normal_spectrogram = generate_spectrogram(normal_signal)
    jammed_spectrogram = generate_spectrogram(jammed_signal)

    # Ensure spectrograms have valid dimensions
    if normal_spectrogram.shape[1] == 0 or jammed_spectrogram.shape[1] == 0:
        raise ValueError("Spectrogram has invalid dimensions (width is 0).")

    # Expand dimensions for the model (e.g., adding channel dimension)
    normal_spectrogram = np.expand_dims(normal_spectrogram, axis=-1)
    jammed_spectrogram = np.expand_dims(jammed_spectrogram, axis=-1)

    # Reshape if necessary to match the model input (e.g., (None, 64, 64, 1))
    normal_spectrogram = np.resize(normal_spectrogram, (normal_spectrogram.shape[0], 64, 64, 1))
    jammed_spectrogram = np.resize(jammed_spectrogram, (jammed_spectrogram.shape[0], 64, 64, 1))

    # Return the features
    return np.array([normal_spectrogram, jammed_spectrogram])

# Define the CNN model
def create_model():
    model = models.Sequential([
        layers.InputLayer(input_shape=(64, 64, 1)),  # Ensure input shape is correct
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Output layer for binary classification
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Main function to train the model
if __name__ == "__main__":
    features = generate_features()

    # Separate features and labels (assuming binary classification: 0 for normal, 1 for jammed)
    normal_features = features[0]
    jammed_features = features[1]
    X_train = np.concatenate((normal_features, jammed_features), axis=0)
    y_train = np.concatenate((np.zeros(len(normal_features)), np.ones(len(jammed_features))), axis=0)

    # Create the model
    model = create_model()

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32)
    # After training the model
    model.save('./anti_drone_model.h5')  # Save the trained model to a file
    print("Model saved successfully.")
