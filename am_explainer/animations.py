import librosa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Palatino', 'Georgia'],  # This list is in order of preference
    'text.color': 'white',
    'figure.facecolor': '1e293b',
})

def sampling(out_path):
    # Basic parameters
    fs_low = 2  # Low frequency signal frequency in Hz
    fs_high = 10  # High frequency signal frequency in Hz
    t = np.linspace(0, 2, 1000)  # Time vector

    # Generate signals
    signal_low = np.sin(2 * np.pi * fs_low * t)
    signal_high = np.sin(2 * np.pi * fs_high * t)

    # Initial sample rate
    sample_rate = 1  # Starting sample rate in Hz

    # Sampling function
    def sample_signal(signal, sample_rate, t):
        sample_points = np.arange(0, t[-1], 1/sample_rate)
        sampled_signal = np.interp(sample_points, t, signal)
        return sample_points, sampled_signal

    # Initial sampling
    sample_points_low, sampled_signal_low = sample_signal(signal_low, sample_rate, t)
    sample_points_high, sampled_signal_high = sample_signal(signal_high, sample_rate, t)

    # Interpolation
    interp_low = np.interp(t, sample_points_low, sampled_signal_low, left=sampled_signal_low[0], right=sampled_signal_low[-1])
    interp_high = np.interp(t, sample_points_high, sampled_signal_high, left=sampled_signal_low[0], right=sampled_signal_low[-1])

    # Plot
    fig, axs = plt.subplots(2, 1, figsize=(8, 8))
    plt.ylim = ([-1,1])
    axs[0].set_title(f'Sample Rate: {sample_rate:.1f} Hz', fontsize=24)
    axs[0].axis('off')
    axs[1].axis('off')

    # Low frequency signal plot
    axs[0].plot(t, signal_low, '#6cbdf2', linewidth=3, alpha=0.5)
    axs[0].plot(t, interp_low, '--', color='#dd8453', linewidth=2)
    axs[0].plot(sample_points_low, sampled_signal_low, 'o', color='#ff8e08', markersize=8)

    # High frequency signal plot
    axs[1].plot(t, signal_high, '#6cbdf2', linewidth=3, alpha=0.5)
    axs[1].plot(t, interp_high, '--', color='#dd8453', linewidth=2)
    axs[1].plot(sample_points_high, sampled_signal_high, 'o', color='#ff8e08', markersize=8)

    plt.tight_layout()
    # plt.show()
    # Function to update the plots for each frame in the animation
    def update(frame):
        # Clear the previous contents of the axes
        axs[0].clear()
        axs[1].clear()

        # Update sample rate
        sample_rate = 1 + frame / 50  # Increasing sample rate with each frame

        # Resample signals with new sample rate
        sample_points_low, sampled_signal_low = sample_signal(signal_low, sample_rate, t)
        sample_points_high, sampled_signal_high = sample_signal(signal_high, sample_rate, t)

        # Re-interpolate
        interp_low = np.interp(t, sample_points_low, sampled_signal_low, left=sampled_signal_low[0], right=sampled_signal_low[-1])
        interp_high = np.interp(t, sample_points_high, sampled_signal_high, left=sampled_signal_low[0], right=sampled_signal_low[-1])

        # Plot low frequency signal and its approximation
        axs[0].plot(t, signal_low, '#6cbdf2', linewidth=3, alpha=0.5)
        axs[0].plot(t, interp_low, '--', color='#dd8453', linewidth=2)
        axs[0].plot(sample_points_low, sampled_signal_low, 'o', color='#ff8e08', markersize=8)

        # Plot high frequency signal and its approximation
        axs[1].plot(t, signal_high, '#6cbdf2', linewidth=3, alpha=0.5)
        axs[1].plot(t, interp_high, '--', color='#dd8453', linewidth=2)
        axs[1].plot(sample_points_high, sampled_signal_high, 'o', color='#ff8e08', markersize=8)

        axs[0].set_title(f'Sample Rate: {sample_rate:.1f} Hz', fontsize=24)
        axs[0].axis('off')
        axs[1].axis('off')

    # Create the animation
    ani = FuncAnimation(fig, update, frames=np.arange(1, 1500), interval=14)
    ani.save(out_path, dpi=150)

def am(out_path):
    # Load sample audio
    audio, sr_aud = librosa.load('static/example.wav', sr=None)
    # Amplitude-Modulation animation
    fig, ax = plt.subplots(figsize=(8,4))
    ax.set_ylim([-1, 3])
    ax.grid(True)
    ax.axis('off')
    line, = ax.plot([], [], color="#6cbdf2")
    time_axis = np.arange(len(audio)) / sr_aud
    ax.set_xlim(time_axis[0], time_axis[-1])
    # Generate a non-linear array of steps
    fps = 60
    num_steps = fps * 3
    carrier_freq = (1 / (1 + np.exp(-0.8*np.linspace(-6, 6, num_steps)))) * np.pi / 2  # logistic funciton
    carrier_freq = np.append(carrier_freq, carrier_freq[::-1]) # reverse back to 0
    carrier_freq = carrier_freq - carrier_freq[0] # make sure frequency starts/ends at zero
    carrier_freq = np.append(carrier_freq, np.zeros(fps)) # extra delay between loops

    def animate(f):
        carrier = 1 + np.sin(2 * np.pi * f * np.arange(len(audio)) / sr_aud)
        modulated = ((1 + audio) * carrier)
        # modulated = modulated / np.max(np.abs(modulated))
        line.set_data(time_axis, modulated)
        return line,
    def init():
        line.set_data([], [])
        return line,
    plt.tight_layout()
    anim = FuncAnimation(fig, animate, frames=carrier_freq, init_func=init, blit=True)
    anim.save(out_path, dpi=150, fps=fps)

print('Generating sampling.mp4...')
sampling('static/sampling.mp4')
print('Generating am.mp4...')
am('static/am.mp4')