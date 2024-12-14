# generate_signals.py

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  # Import pandas for saving to CSV

def generate_signal(frequency=10, duration=1, sampling_rate=1000):
    """Generates a pure sine wave signal."""
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    signal = np.sin(2 * np.pi * frequency * t)  # Pure sine wave
    return t, signal

def add_jamming(signal, noise_level=0.5):
    """Adds noise to the signal to simulate jamming."""
    noise = noise_level * np.random.normal(size=len(signal))
    return signal + noise

# Generate normal and jammed signals
t, normal_signal = generate_signal()
_, jammed_signal = generate_signal()
jammed_signal = add_jamming(jammed_signal)

# Save normal and jammed signals to CSV
df_normal = pd.DataFrame({'Time': t, 'Normal Signal': normal_signal})
df_jammed = pd.DataFrame({'Time': t, 'Jammed Signal': jammed_signal})

# Save to CSV files
df_normal.to_csv('normal_signal.csv', index=False)
df_jammed.to_csv('jammed_signal.csv', index=False)

# Plot the signals
plt.plot(t, normal_signal, label='Normal Signal')
plt.plot(t, jammed_signal, label='Jammed Signal')
plt.legend()
plt.title("Normal vs Jammed Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.show()

print("Signals saved as 'normal_signal.csv' and 'jammed_signal.csv'.")
