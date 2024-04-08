import os
import numpy as np
import torch
import IPython.display
from IPython.display import Audio
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 22,  # Adjust font size here
    'font.family': 'serif',
    'font.serif': ['Palatino', 'Georgia'],  # This list is in order of preference
    'figure.facecolor': 'none',
    'figure.labelsize': 22,  # Adjust axis label size if different from general font size
    'axes.titlesize': 24,
    'axes.labelsize': 22,  # Adjust axis label size if different from general font size
    'xtick.labelsize': 18,  # Adjust for x-axis tick label size
    'ytick.labelsize': 18,  # Adjust for y-axis tick label size
})

sr_aud = 8e3
def play_audio(data, rate=sr_aud):
    if not isinstance(data, np.ndarray):
        if isinstance(data, torch.Tensor) and data.device != 'cpu':
            data = data.cpu().numpy()
    norm = np.float32(data / np.max(np.abs(data)))
    return Audio(norm, rate=rate)

outdir = ''
def set_outdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    global outdir
    outdir = path

def plot_spectrum(signal, sample_rate, title='Signal Spectrum', zoom_range=None, filename=None):
    fft = np.fft.fftshift(np.fft.fft(signal))
    freqs = np.fft.fftshift(np.fft.fftfreq(len(signal), 1/sample_rate))
    plt.figure(figsize=(12, 6))
    plt.plot(freqs, 20*np.log10(np.abs(fft)), color="#09f")
    # Zoom into the specified frequency range if provided
    if zoom_range is not None:
        plt.xlim(zoom_range)
    plt.title(title)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.grid(True)
    if filename:
        save_fig(filename)  # Save the plot to the file
        plt.close()  # Close the figure to prevent it from being displayed in this case
    else:
        plt.show()  # Display the plot

def plot_waveform(signal, sample_rate, title='Signal Waveform', filename=None):
    time_axis = np.arange(len(signal)) / sample_rate
    plt.figure(figsize=(12, 4))
    plt.plot(time_axis, signal, color="#09f")
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    if filename:
        save_fig(filename)  # Save the plot to the file
        plt.close()  # Close the figure to prevent it from being displayed in this case
    else:
        plt.show()  # Display the plot
    
def analyze_spectrum(iq_samples, sample_rate, offset, zoom_range=None, peak_threshold=0.6, filename=None):
    # Compute the FFT and shift the result
    fft_vals = np.fft.fftshift(np.fft.fft(iq_samples))
    
    # Compute shifted frequency bins
    fft_freq = np.fft.fftshift(np.fft.fftfreq(len(iq_samples), 1/sample_rate))

    # Adjust the frequency bins to account for the offset
    fft_freq += offset
    
    # Compute magnitude spectrum and normalize
    mag_spectrum = np.abs(fft_vals)
    mag_spectrum = mag_spectrum / np.max(mag_spectrum)

    # Plot the magnitude spectrum with shifted frequencies
    plt.figure(figsize=(10, 6))
    plt.plot(fft_freq, mag_spectrum, color="#09f")
    plt.title('Magnitude Spectrum with Offset Tuning')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Normalized Magnitude')

    # Zoom into the specified frequency range if provided
    if zoom_range is not None:
        plt.xlim(zoom_range)

    # Find and mark peaks above a certain threshold
    peaks, _ = find_peaks(mag_spectrum, height=peak_threshold)
    peak_freqs = fft_freq[peaks]
    plt.scatter(peak_freqs, mag_spectrum[peaks], color='red')

    if filename:
        save_fig(filename)
        plt.close()  # Close the figure to prevent it from being displayed in this case
    else:
        plt.show()  # Display the plot

    # Print identified peak frequencies
    print("Identified peak frequencies (Hz):", peak_freqs)

def plot_waterfall_spectrum(signal, sample_rate, window_size=1024, hop_size=512, title='Waterfall Spectrum', include_negative_frequencies=True):
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

    plt.figure(figsize=(15, 6))
    if include_negative_frequencies:
        extent = (segment_times[0], segment_times[-1], freqs[0], freqs[-1])
    else:
        extent = (segment_times[0], segment_times[-1], 0, freqs[-1])
    plt.imshow(waterfall, extent=extent, aspect='auto', cmap='viridis', origin='lower')
    plt.colorbar(label='Magnitude (dB)')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title(title)
    plt.show()

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.cuda.amp import autocast
class TestDataset(Dataset):
    def __init__(self, data, window_size, step_size=None):
        """
        Initializes the dataset with windowed segments of the input signal data.
        
        Args:
        data (torch.Tensor): The input signal data.
        window_size (int): The size of each windowed segment.
        step_size (int, optional): The step size between windows for overlapping. If None, it equals window_size (non-overlapping).
        """
        super(TestDataset, self).__init__()
        
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.complex64)
        
        self.window_size = window_size
        self.step_size = step_size if step_size is not None else window_size
        
        self.windows = self._create_windows(data)
    
    def _create_windows(self, data):
        n = data.shape[0]
        windows = []
        for start in range(0, n - self.window_size + 1, self.step_size):
            end = start + self.window_size
            windows.append(data[start:end])
        
        # Stack all windowed segments along a new dimension
        return torch.stack(windows)
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        # Return the idx-th window
        out = self.windows[idx]
        return out

def pad_collate(batch):
    rx_batch = [torch.view_as_real(rx).T for rx in batch]
    rx_padded = pad_sequence(rx_batch, batch_first=True, padding_value=0)
    return rx_padded

@torch.no_grad()
def apply_model(model, data, device='cpu'):
    num_samp = 256
    test_dataset = TestDataset(data=data, window_size=num_samp)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, collate_fn=pad_collate, 
                            num_workers=0, pin_memory=False, drop_last=False)
    output = []
    model.eval()
    for batch in test_loader:
        x = batch.to(device)
        with autocast():
            a, _ = model(x)
            output.append(a.view(-1))
    output = torch.cat(output).cpu().numpy()
    return output

def save_fig(filename, colorbar=None, colorbar_mappable=None, legend=False):
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
    if legend:
        plt.legend(facecolor='none', edgecolor='black', labelcolor='black')
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
    if legend:
        plt.legend(facecolor='none', edgecolor='white', labelcolor='white')
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
