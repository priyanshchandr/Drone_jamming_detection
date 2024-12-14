import numpy as np
import matplotlib.pyplot as plt

# Sample parameters
duration = 1.0  # seconds
sample_rate = 44100  # samples per second
frequency = 1000  # Hz

# Generate time array
t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

# Generate sine wave signal
normal_signal = np.sin(2 * np.pi * frequency * t)

# Plot the normal signal
plt.plot(t, normal_signal)
plt.title("Normal Signal (Sine Wave)")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.show()

# Generate noisy signal (Jammed signal)
jammed_signal = normal_signal + 0.5 * np.random.randn(len(normal_signal))

# Plot the jammed signal
plt.plot(t, jammed_signal)
plt.title("Jammed Signal (Noisy Signal)")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.show()

# Save normal signal to CSV
normal_signal_data = np.column_stack((t, normal_signal))
np.savetxt('normal_signal.csv', normal_signal_data, delimiter=',', header='Time,Normal Signal', comments='')

# Save jammed signal to CSV
jammed_signal_data = np.column_stack((t, jammed_signal))
np.savetxt('jammed_signal.csv', jammed_signal_data, delimiter=',', header='Time,Normal Signal', comments='')
