# Signal Jamming Detection Using CNN

This project focuses on detecting normal and jammed signals using **Convolutional Neural Networks (CNN)**. The CNN-based approach provides robust classification of signals by learning intricate patterns and features directly from the data. The goal is to classify signals into **normal** or **jammed** categories, enabling better monitoring and response mechanisms.

---

## Key Features

- **Signal Feature Extraction**: Utilizes CNNs to automatically extract relevant features from signal data.
- **Robust Classification**: Employs a trained CNN model to distinguish between normal and jammed signals with high accuracy.
- **Real-Time Detection**: Processes input signals and provides instant classification results.
- **Logging and Visualization**: Captures detailed logs of the classification process and offers visual insights into signal patterns.

---

## Installation and Setup

### Prerequisites
- **Python 3.x**
- Required Libraries:
  - `numpy`
  - `tensorflow` / `keras`
  - `matplotlib`
  - `scipy`

Install the dependencies with:

```bash
pip install numpy tensorflow matplotlib scipy
```

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/priyanshchandr/Drone_jamming_detection.git
   cd Drone_jamming_detection
   ```
2. Open the main Python script in your IDE or text editor.
3. Run the script:
   ```bash
   python <script_name>.py
   ```
4. Provide the required inputs (e.g., signal frequency and noise levels).

---

## Example Interaction

```plaintext
Enter the signal frequency (Hz, e.g., 50): 50
Enter the noise level (e.g., 0.1 for low noise, 1.5 for high noise): 1.5

Analyzing the signal using CNN...
Prediction: The signal is normal.

Would you like to visualize the signal plot? (yes/no): yes
```

---

## Machine Learning Approach

1. **Dataset Preparation**:
   - The dataset consists of signal samples with labels (`normal` or `jammed`).
   - Signals are preprocessed into a format suitable for CNN input.

2. **Model Architecture**:
   - The CNN architecture includes:
     - Convolutional layers for feature extraction.
     - Pooling layers for dimensionality reduction.
     - Dense layers for classification.

3. **Training**:
   - The model was trained on a dataset of signal samples with varying frequencies and noise levels.
   - Validation ensures high performance on unseen data.

4. **Real-Time Classification**:
   - The trained CNN model is used to classify new signals during runtime.

---

## Visualization

- Generate signal plots to analyze frequency and amplitude patterns.
- Visualize the CNN model's accuracy and loss during training (optional).

---

## Future Enhancements

- Incorporation of additional datasets for enhanced generalization.
- Integration with real-time hardware for signal collection and processing.
- Optimization of the CNN architecture for faster inference.

