import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import wave

plt.rcParams.update({
    'font.size': 22,  # Adjust font size here
    'font.family': 'serif',
    'font.serif': ['Palatino', 'Georgia'],  # This list is in order of preference
    'text.color': 'white',
    'figure.facecolor': '1e293b',
    'figure.labelsize': 22,  # Adjust axis label size if different from general font size
    'axes.titlesize': 24,
    'axes.labelsize': 22,  # Adjust axis label size if different from general font size
    'xtick.labelsize': 18,  # Adjust for x-axis tick label size
    'ytick.labelsize': 18,  # Adjust for y-axis tick label size
})

outdir = ''
def set_outdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    global outdir
    outdir = path

def save_audio(filename, audio_data, sample_rate):
    if not isinstance(audio_data, np.ndarray):
        if isinstance(audio_data, torch.Tensor) and audio_data.device != 'cpu':
            audio_data = audio_data.cpu().numpy()
    if audio_data.dtype != np.int16:
        audio_data = np.int16(audio_data / np.max(np.abs(audio_data)) * 32767)
    with wave.open(os.path.join(outdir, filename), 'wb') as wav_file:
        wav_file.setnchannels(1)
        # Set the sample width to 2 bytes (16 bits)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.setnframes(len(audio_data))
        wav_file.writeframes(audio_data.tobytes())

def plot_waveform(signal, sample_rate, title='Signal Waveform', filename=None):
    time_axis = np.arange(len(signal)) / sample_rate
    plt.figure(figsize=(16, 8))
    plt.plot(time_axis, signal, color="#6cbdf2")
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    if filename:
        save_fig(filename)  # Save the plot to the file
        plt.close()  # Close the figure to prevent it from being displayed in this case
    else:
        plt.show()  # Display the plot

def animate_waveform(start, end, sample_rate, title=None, ylim=None, filename="animaion.mp4"):
    # Animate a waveform plot, linearly transitioning from start to end
    fig, ax = plt.subplots(figsize=(16, 8), facecolor='#1e293b')
    if title is not None:
        plt.title(title)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.grid(True)
    # ax.axis('off')
    ax.set_xticks([], [])
    time_axis = np.arange(len(start)) / sample_rate
    ax.set_xlim(time_axis[0], time_axis[-1])
    line, = ax.plot([], [], color="#6cbdf2")
    fps = 60
    num_steps = fps * 6

    def animate(i):
        # Visual transition from start to end and back
        if i < num_steps // 2:
            transition_phase = i / (num_steps / 2)
        else:
            transition_phase = 1 - ((i - num_steps / 2) / (num_steps / 2))
        waveform = start * (1 - transition_phase) + end * transition_phase
        line.set_data(time_axis, waveform.real)
        return line,

    def init():
        line.set_data([], [])
        return line,
    plt.tight_layout()
    plt.gca().spines['bottom'].set_color('white')
    plt.gca().spines['top'].set_color('white') 
    plt.gca().spines['right'].set_color('white')
    plt.gca().spines['left'].set_color('white')
    plt.gca().tick_params(axis='x', colors='white', labelcolor='white')
    plt.gca().tick_params(axis='y', colors='white', labelcolor='white')
    plt.gca().xaxis.label.set_color('white')
    plt.gca().yaxis.label.set_color('white')
    plt.gca().title.set_color('white')
    plt.gca().set_facecolor('none')
    anim = FuncAnimation(fig, animate, frames=num_steps, init_func=init, blit=True)
    anim.save(os.path.join(outdir, filename), dpi=150, fps=fps)

def plot_waterfall_spectrum(signal, sample_rate, window_size=1024, hop_size=512, title='Waterfall Spectrum', include_negative_frequencies=True, filename=None):
    """
    Plots a waterfall spectrum of a signal. Optionally includes negative frequencies.

    Args:
        signal (np.ndarray): The signal to plot.
        sample_rate (float): The sample rate of the signal in Hz.
        window_size (int): The size of the window for FFT calculation.
        hop_size (int): The hop size between windows.
        title (str): The title of the plot.
        include_negative_frequencies (bool): Whether to include negative frequencies in the plot.
    """
    n_segments = int(np.ceil(len(signal) / hop_size))
    segment_times = np.linspace(0, len(signal) / sample_rate, n_segments, endpoint=False)

    # Adjust waterfall array initialization based on include_negative_frequencies
    if include_negative_frequencies:
        waterfall = np.zeros((window_size, n_segments))
    else:
        waterfall = np.zeros((window_size // 2, n_segments))  # Only half the size for positive frequencies

    for i in range(n_segments):
        start = i * hop_size
        end = start + window_size
        segment = signal[start:end]
        if len(segment) < window_size:
            segment = np.pad(segment, (0, window_size - len(segment)), mode='constant')
        fft = np.fft.fft(segment, window_size)
        if include_negative_frequencies:
            fft_shifted = np.fft.fftshift(fft)
            waterfall[:, i] = 20 * np.log10(np.abs(fft_shifted))
            freqs = np.fft.fftshift(np.fft.fftfreq(window_size, 1 / sample_rate))
        else:
            # Use only the first half of FFT results for positive frequencies
            waterfall[:, i] = 20 * np.log10(np.abs(fft[:window_size // 2]))
            freqs = np.fft.fftfreq(window_size, 1 / sample_rate)[:window_size // 2]

    plt.figure(figsize=(16, 8))
    if include_negative_frequencies:
        extent = (segment_times[0], segment_times[-1], freqs[0], freqs[-1])
    else:
        extent = (segment_times[0], segment_times[-1], 0, freqs[-1])
    mappable = plt.imshow(waterfall, extent=extent, aspect='auto', cmap='viridis', origin='lower', vmin=-50, vmax=50)
    colorbar = plt.colorbar(label='Magnitude (dB)')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title(title)
    if filename:
        save_fig(filename, colorbar=colorbar, colorbar_mappable=mappable)  # Save the plot to the file
        plt.close()  # Close the figure to prevent it from being displayed in this case
    else:
        plt.show()  # Display the plot

def save_fig(filename, colorbar=None, colorbar_mappable=None):
    """Save two versions of the figure for light and dark mode"""
    plt.tight_layout()  # Adjust layout to make room for elements
    plt.gca().spines['bottom'].set_color('black')
    plt.gca().spines['top'].set_color('black') 
    plt.gca().spines['right'].set_color('black')
    plt.gca().spines['left'].set_color('black')
    plt.gca().tick_params(axis='x', colors='black', labelcolor='black')
    plt.gca().tick_params(axis='y', colors='black', labelcolor='black')
    plt.gca().xaxis.label.set_color('black')
    plt.gca().yaxis.label.set_color('black')
    plt.gca().title.set_color('black')
    plt.gca().set_facecolor('none')
    plt.savefig(os.path.join(outdir, filename), transparent=True)  # Save light mode
    # Config for Dark Mode
    plt.gca().spines['bottom'].set_color('white')
    plt.gca().spines['top'].set_color('white') 
    plt.gca().spines['right'].set_color('white')
    plt.gca().spines['left'].set_color('white')
    plt.gca().tick_params(axis='x', colors='white', labelcolor='white')
    plt.gca().tick_params(axis='y', colors='white', labelcolor='white')
    plt.gca().xaxis.label.set_color('white')
    plt.gca().yaxis.label.set_color('white')
    plt.gca().title.set_color('white')
    
    if colorbar:
        colorbar.ax.yaxis.label.set_color('white')  # Set colorbar label color for dark mode
        colorbar.ax.yaxis.set_tick_params(color='white')  # Change tick colors for dark mode
        # Ensure tick labels are updated for dark mode
        for label in colorbar.ax.get_yticklabels():
            label.set_color('white')

    # 4. Save Dark Theme Version
    import pathlib
    dark_filename = f"{os.path.splitext(filename)[0]}_dark{os.path.splitext(filename)[1]}"
    plt.savefig(os.path.join(outdir, dark_filename), facecolor='none', transparent=True)

