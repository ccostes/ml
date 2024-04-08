import math
import numpy as np
import torch
import torch.nn.functional as F

def amplitude_modulate(audio_signal, carrier_freq, original_rate, desired_rate, modulation_index=1.0, snr_db=None, device='cpu'):
    """
    Perform amplitude modulation on an audio signal with upsampling to a desired sample rate
    and add noise based on a specified signal-to-noise ratio in dB.

    Args:
    audio_signal (torch.Tensor): The original audio signal tensor, shape [length].
    carrier_freq (float): The carrier frequency in Hz.
    original_rate (int): The original sampling rate of the audio signal in Hz.
    desired_rate (int): The desired sampling rate of the modulated signal in Hz.
    modulation_index (float): The modulation index (default is 1.0 for 100% modulation).
    snr_db (float, optional): The desired signal-to-noise ratio in decibels.
    device (str): The device to perform calculations on ('cuda' or 'cpu').

    Returns:
    torch.Tensor: The amplitude modulated signal with added noise based on the specified SNR, upsampled to the desired sample rate.
    """
    # Ensure audio_signal is on the correct device
    if not isinstance(audio_signal, torch.Tensor):
        audio_signal = torch.tensor(audio_signal, dtype=torch.float32)
    audio_signal = audio_signal.to(device)
    # Add a channel dimension and a batch dimension to the audio signal for interpolate
    audio_signal = audio_signal.unsqueeze(0).unsqueeze(0)

    # Upsample
    upsampling_factor = desired_rate / original_rate
    audio_signal_upsampled = F.interpolate(audio_signal, scale_factor=upsampling_factor, mode='linear', align_corners=False)

    # Remove the added dimensions to get back to 1D
    audio_signal_upsampled = audio_signal_upsampled.squeeze()
    num_samples_upsampled = audio_signal_upsampled.size(-1)

    if carrier_freq == 0:
        # Perform baseband amplitude modulation on the upsampled signal
        # In baseband AM, the carrier frequency is 0, so we can omit multiplying by a carrier signal
        modulated_signal = 1 + modulation_index * audio_signal_upsampled
        modulated_signal = modulated_signal + 0j # make it complex
    else:
        # Amplitude modulate
        t = torch.linspace(0, num_samples_upsampled / desired_rate, num_samples_upsampled, device=device)
        carrier_signal = torch.exp(2j * torch.pi * carrier_freq * t)
        modulated_signal = (1 + modulation_index * audio_signal_upsampled) * carrier_signal

    # Normalize the modulated signal to be within the range [-1, 1]
    modulated_signal = modulated_signal / modulated_signal.abs().max()
    # Adjust if there's a significant DC offset
    modulated_signal = modulated_signal - modulated_signal.real.mean()

    # Optionally add noise
    if snr_db is not None:
        # Calculate signal power
        signal_power = torch.mean(torch.abs(modulated_signal) ** 2)
        # Calculate noise power based on SNR
        noise_power = signal_power / (10 ** (snr_db / 10))
        # Generate noise with calculated noise power
        noise = torch.randn_like(modulated_signal) * torch.sqrt(noise_power)
        # Add noise to the modulated signal
        modulated_signal = modulated_signal + noise

    return modulated_signal

def fir_low_pass_filter(signal, kernel, padding='same'):
    # Convert kernel to complex if signal is complex
    if torch.is_complex(signal):
        kernel = kernel.to(dtype=torch.complex64)
    # Ensure the signal and kernel are 3D (batch, channel, length) as expected by conv1d
    if signal.dim() == 1:
        signal = signal.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    elif signal.dim() == 2:
        signal = signal.unsqueeze(1)  # Add channel dimension
    if kernel.dim() == 1:
        kernel = kernel.unsqueeze(0).unsqueeze(0)
    elif kernel.dim() == 2:
        kernel = kernel.unsqueeze(1)

    # Calculate padding for 'same'
    if padding == 'same':
        padding = kernel.size(-1) // 2
    # Ensure the kernel and signal are on the same device
    kernel = kernel.to(signal.device)
    # Perform 1D convolution (filtering)
    filtered_signal = F.conv1d(signal, kernel, padding=padding)
    return filtered_signal.squeeze()  # Remove added batch and channel dimensions for output
def design_band_pass_filter(low_cut, high_cut, sample_rate, kernel_size, device='cpu'):
    nyquist_rate = sample_rate / 2
    low = low_cut / nyquist_rate
    high = high_cut / nyquist_rate
    t = torch.linspace(-kernel_size // 2, kernel_size // 2, kernel_size, device=device)
    sinc_func_low = torch.sin(2 * math.pi * low * t) / (math.pi * t)
    sinc_func_high = torch.sin(2 * math.pi * high * t) / (math.pi * t)
    sinc_func = sinc_func_high - sinc_func_low
    sinc_func[kernel_size // 2] = 2 * (high - low)  # Correct division by zero at center
    window = torch.hamming_window(kernel_size, device=device)
    fir_kernel = sinc_func * window
    fir_kernel /= fir_kernel.sum()
    return fir_kernel
def design_low_pass_filter(cutoff_freq, sample_rate, kernel_size, device='cpu'):
    nyquist_rate = sample_rate / 2
    normalized_cutoff = cutoff_freq / nyquist_rate
    t = torch.linspace(-kernel_size // 2, kernel_size // 2, kernel_size, device=device)
    sinc_func = torch.sin(2 * math.pi * normalized_cutoff * t) / (math.pi * t)
    sinc_func[kernel_size // 2] = 2 * normalized_cutoff  # Correct division by zero at center
    window = torch.hamming_window(kernel_size, device=device)
    fir_kernel = sinc_func * window
    fir_kernel /= fir_kernel.sum()
    return fir_kernel

def demodulate_am(modulated_signal, carrier_freq, sample_rate, audio_sample_rate=8e3, kernel_size=101, low_cut_freq=300, high_cut_freq=4000, device='cpu'):
    # Ensure modulated_signal is a PyTorch tensor and on GPU
    if not isinstance(modulated_signal, torch.Tensor):
        modulated_signal = torch.tensor(modulated_signal)
    modulated_signal = modulated_signal.to(device)
    # Downconvert to basebad
    if carrier_freq != 0:
        num_samples = modulated_signal.size(0)
        t = torch.linspace(0, num_samples / sample_rate, num_samples, device=device)
        downconversion_signal = torch.exp(-2j * math.pi * carrier_freq * t)
        baseband_signal = modulated_signal * downconversion_signal
    else:
        baseband_signal = modulated_signal
    # Design and apply anti-aliasing low-pass filter
    anti_aliasing_cutoff = audio_sample_rate / 2
    anti_aliasing_kernel = design_low_pass_filter(anti_aliasing_cutoff, sample_rate, kernel_size, device)
    baseband_filtered = fir_low_pass_filter(baseband_signal.unsqueeze(0).unsqueeze(0), anti_aliasing_kernel.unsqueeze(0).unsqueeze(0), padding='same')
    # Decimate to desired sample rate
    decimation_factor = int(sample_rate // audio_sample_rate)
    decimated_signal = baseband_filtered[::decimation_factor]
    # Demodulate - Rectify and filter
    rectified_signal = torch.abs(decimated_signal)
    fir_kernel = design_band_pass_filter(low_cut_freq, high_cut_freq, audio_sample_rate, kernel_size, device)
    demodulated_signal = fir_low_pass_filter(rectified_signal.unsqueeze(0).unsqueeze(0), fir_kernel.unsqueeze(0).unsqueeze(0), padding='same')
    return demodulated_signal

def load_baseband(filename, offset=None, duration=None, sr_in=1.04e6, device='cpu'):
    # Read the raw IQ bytes
    with open(filename, 'rb') as f:
        raw_bytes = np.fromfile(f, dtype=np.uint8, 
                                count= -1 if duration is None else int(sr_in * 2 * duration), 
                                offset= 0 if offset is None else int(sr_in * offset * 2))
    # Convert raw bytes to IQ components (-1.0 to 1.0 range)
    iq_samples = (raw_bytes - 127.5) / 127.5
    i_samples = iq_samples[0::2]
    q_samples = iq_samples[1::2]
    complex_samples = i_samples + 1j * q_samples
    # Create a pattern [1, 1j, -1, -1j] and repeat it to match the length of buf/2
    pattern = np.array([1, 1j, -1, -1j], dtype=np.complex64)
    pattern_repeated = np.tile(pattern, len(complex_samples) // len(pattern))
    shifted_complex = complex_samples * pattern_repeated
    baseband = torch.tensor(shifted_complex, dtype=torch.complex64, device=device)
    return baseband

def demod_baseband(baseband, sr_in=1.04e6, sr_aud=2**13, device='cpu'):
    kernel_size=101
    # Design and apply anti-aliasing low-pass filter
    anti_aliasing_cutoff = sr_aud / 2
    anti_aliasing_kernel = design_low_pass_filter(anti_aliasing_cutoff, sr_in, kernel_size, device)
    filtered = fir_low_pass_filter(baseband.unsqueeze(0).unsqueeze(0), anti_aliasing_kernel.unsqueeze(0).unsqueeze(0), padding='same')
    # plot_waterfall_spectrum(filtered.cpu(), sr_aud, title="Filtered")
    # plot_spectrum(filtered.cpu(), sr_aud, title="Filtered")
    # Decimate to desired sample rate
    decimation_factor = int(sr_in // sr_aud)
    decimated = filtered[::decimation_factor]
    # plot_waterfall_spectrum(decimated.cpu(), sr_aud, title="Decimated")
    # util.plot_spectrum(decimated.cpu(), sr_aud, title="Decimated", zoom_range=[-100,100])
    low_cut_freq = 300
    high_cut_freq = 3000
    # Demodulate - Rectify and filter
    rectified = torch.abs(decimated)
    fir_kernel = design_band_pass_filter(low_cut_freq, high_cut_freq, sr_aud, kernel_size, device)    
    demodulated = fir_low_pass_filter(rectified.unsqueeze(0).unsqueeze(0), fir_kernel.unsqueeze(0).unsqueeze(0), padding='same')    
    return demodulated.cpu()

def downsample(samples, sr_in, sr_out, device='cpu'):
    # Downsample SDR to model input sample rate sr_mod
    kernel_size=101
    # Design and apply anti-aliasing low-pass filter
    anti_aliasing_cutoff = sr_out / 2
    anti_aliasing_kernel = design_low_pass_filter(anti_aliasing_cutoff, sr_in, kernel_size, device)
    filtered = fir_low_pass_filter(samples.unsqueeze(0).unsqueeze(0), anti_aliasing_kernel.unsqueeze(0).unsqueeze(0), padding='same')
    # Decimate to desired sample rate
    decimation_factor = int(sr_in // sr_out)
    return filtered[::decimation_factor]