# ---
# date: 2024-03-26
# draft: false
# title: "ML In Practice Part 1: AM Radio"
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: gpu11
#     language: python
#     name: python3
# ---
# %% [markdown]
# This is the first in a series of articles detailing how I trained a machine learning model to demodulate AM radio (specifically air-traffic control voice audio) while removing noise. My goal was not to find the best solution to this problem, rather get experience applying machine learning to a real problem from start to finish. I wasn't even sure how feasible of a problem this was to begin with, but was quite impressed with the results.
#
# Given a noisy AM signal processed with standard demodulation code vs. the model you can see the much stronger and more distinct signal in audio spectrum, and hear the difference below.
# ![0dB Signal Comparison](comparison.png)
# <!--more-->
# **Standard Processing:**
# <audio controls src="0dB_standard.wav"></audio>
# 
# **Model output:**
# <audio controls src="0dB_model.wav"></audio>
# 
# 

# %% [markdown]
# The goal of these articles is to show my overall process for getting this result in hopes it might help others who have a foothold on understanding the machine learning fundamentals but aren't sure how to apply it in practice. Part one (you are here) introduces the problem space with a quick primer on AM radio demodulation. In part two we'll devise a model architecture and create synthetic training data, and finally train the model and analyze the results in part three.
#
# Disclaimer: I am not an expert. This is not legal advice. Ask your doctor if machine learning is right for you.

# %% [markdown]
# ## AM Demodulation Primer
# > *If you don't understand the problem you're trying to apply machine learning to, you're gonna have a bad time.*
#
# Amplitude Modulation (AM) is the simplest way to transmit audio signals via radio, and is widely used for aircraft and maritime communication. Transmitting audio signals directly is undesirable for a variety of reasons (the antenna would need to hundreds of kilometers long, for one), so the next simplest thing is to add the audio signal to a higher frequency sine wave "carrier", transmit that signal, and then remove the carrier on the receiving end.
#
# The animation below shows what this looks like as the carrier frequency increases from zero, but actual signals will use a much higher carrier frequency (VHF aircraft radio is around 120 MHz).
# <video controls="" autoplay="" loop="">
#     <source src="am.mp4" type="video/mp4">
# </video>

# %% [markdown]
# ## Software-Defined Radio
# Traditionally this process of receiving radio signals and turning them into audio could only be implemented with analog electronics, but modern processing power enables us to do it in software, called "software-defined radio" (SDR). By turning the analog signal from the antenna into a digital one we can receive a wide variety of signals, from audio, to satellite weather imagery, to TV broadcasts, with the same hardware device.
#
# SDR receivers capture a "window" of frequencies, the size of which is called the "bandwidth" of the receiver, which we can tune so that the frequency we are interested in is in the window. This also means that we can receive multiple signals simultaneously if they fall within that window; neat!
#
# ## AM Demodulation Process
# Now that you understand the basics of the AM signal and how we are receiving it, lets go through each step of the process to turn a raw SDR signal into audio. 
#
# ### Input signal
# Starting with the raw input signal, we'll use a spectrogram to visualize the SDR data. This illustrates the window of frequencies captured by the SDR, showing signal magnitude at each frequency over time. Frequencies here are relative to where the receiver was tuned. For example if it was tuned to 500 kHz, then -200,000 Hz on this plot would correspond to an actual signal frequency of 300 kHz.

# %%
sr_in = 1.04e6  # SDR sampling rate
carrier_in = (
    int(sr_in / 4) * -1
)  # negative b/c tuner adds positive offset, so signal appears at negative offset
sr_aud = 8192  # Desired audio sampling rate
downsample = round(sr_in / sr_aud)  # Downsampling factor
print(
    f"Input Rate: {sr_in/1e3} kHz\nAudio Rate: {sr_aud/1e3} kHz\nDownsampling factor: {downsample}"
)

# Read the raw IQ bytes
import numpy as np
filename = "twr_rx.bin"
with open(filename, "rb") as f:
    raw_bytes = np.fromfile(f, dtype=np.uint8)

# Convert raw bytes to IQ components (-1.0 to 1.0 range)
iq_samples = (raw_bytes - 127.5) / 127.5
i_samples = iq_samples[0::2]
q_samples = iq_samples[1::2]
complex_samples = i_samples + 1j * q_samples

from util import save_audio, plot_waveform, plot_waterfall_spectrum, set_outdir
set_outdir('public')
plot_waterfall_spectrum(complex_samples, sr_in, title="SDR Signal", filename='sdr_data.png')

# %% [markdown]
# ![SDR Signal](sdr_data.png)
# You can see we captured a few different signals, shown by the bright horizontal lines. The signal we want to demodulate is the continuous line just below 200,000 Hz, which corresponds to a real frequency of 120.9 MHz since the receiver was tuned to 121.16 MHz.
#
# ### Shift to Baseband
# The first step is to remove the carrier signal which, for fun signal processing reasons we don't need to get into, we can do by shifting the frequency of the data by the frequency of that carrier. On the spectrogram, this literally just looks like shifting the whole thing up or down so that our signal of interest is centered at 0 Hz.
#
# We also don't need to get into complex values or why multiplying the signal by `1, 1j, -1, -1j` performs the frequency shift we need, but if you're interested and want a clue, the receiver tuning was offset from the signal frequency by 1/4 of the sample rate.

# %%
pattern = np.array([1, 1j, -1, -1j], dtype=np.complex64)
pattern_repeated = np.tile(pattern, len(complex_samples) // len(pattern))
shifted_complex = complex_samples * pattern_repeated
plot_waterfall_spectrum(shifted_complex, sr_in, title="Baseband Signal", filename='baseband.png')

# %% [markdown]
# ![Baseband Signal](baseband.png)
# ### Downsample
# Now that we have removed the high-frequency carrier from our signal we can reduce the number of data points per second, called sample rate, we use to represent it. This concept is illustrated in the animation below, which samples two signals at increasing rates and plots an interpolated signal based on those points. You can see that the lower frequency signal is approximated with much lower sampling rate than the higher frequency signal.
#
# By downsampling we need to process much less data: our input sampling rate was ~1 MHz which is 2MB of data per second, and we downsample to ~8 kHz which is only 16 kB per second.
#
# <video controls="" autoplay="" loop="">
#     <source src="sampling.mp4" type="video/mp4">
# </video>
#
# Notice that the frequency range of the spectrogram is much lower after downsampling. We showed previously that a lower sampling rate can be used to represent lower-frequency signals, which also means that lowering the sample rate limits the maximum frequency that can be represented - specifically half the sampling rate. This is why the spectrogram only goes to about 4 kHz with our new sampling rate of about 8 kHz. 
#
# Anything outside that range will be folded back, or "aliased", within the range, which you can see in the animation as lower-frequency sine waves on the high-frequency signal at various sample rates. See [Nyquist–Shannon sampling theorem](https://en.wikipedia.org/wiki/Nyquist%E2%80%93Shannon_sampling_theorem) for more.
#
# To avoid aliasing we first filter out any signals at frequencies above the max of our new sampling rate.

# %%
import torch
import modulation
baseband = torch.tensor(
    shifted_complex, dtype=torch.complex64
)  # Convert to complex torch tensor

# Design and apply anti-aliasing low-pass filter
kernel_size = 101
anti_aliasing_cutoff = sr_aud / 2
anti_aliasing_kernel = modulation.design_low_pass_filter(
    anti_aliasing_cutoff, sr_in, kernel_size
)
filtered = modulation.fir_low_pass_filter(
    baseband.unsqueeze(0).unsqueeze(0),
    anti_aliasing_kernel.unsqueeze(0).unsqueeze(0),
    padding="same",
)
plot_waterfall_spectrum(filtered, sr_in, title="Anti-alias Filtered", filename='baseband_filtered.png')

# %% [markdown]
# ![Anti-Alias-Filtered Baseband](baseband_filtered.png)
# The filtered signal is then downsampled by simply picking out samples at our new sampling rate and throwing the rest away. Eg. if the original sampling rate were 1 kHz and our desired rate was 100 Hz we would save every 10th sample.

# %%
# Decimate to desired sample rate
decimation_factor = int(sr_in // sr_aud)
decimated = filtered[::decimation_factor]
save_audio('decimated.wav', decimated, sr_aud)
plot_waterfall_spectrum(decimated, sr_aud, include_negative_frequencies=False, title="Decimated", filename='decimated.png')

# %% [markdown]
# ![Decimated Signal](decimated.png)
# ### Rectify and filter
# With just these two steps were are actually quite close to the desired audio, which you can start to hear at this stage, but there's clearly something wrong. Look at the previous spectrogram and see if you can guess what the issue is.
# <audio controls src="decimated.wav"></audio>
#
# You can see that the carrier signal is still present, albeit shifted to a very low frequency, as the bright continuous line near the bottom of the spectrogram; not quite at 0, but well outside of the audio frequency range.
#
# To get rid fo this we apply another filter, this time a band-pass filter which filters out anything outside of a given frequency band that we set to the audio frequency range of 300 Hz to 3 kHz.

# %%
# Demodulate: Rectify and filter
low_cut_freq = 300
high_cut_freq = 3000
rectified = torch.abs(decimated)
fir_kernel = modulation.design_band_pass_filter(
    low_cut_freq, high_cut_freq, sr_aud, kernel_size
)
demodulated = modulation.fir_low_pass_filter(
    rectified.unsqueeze(0).unsqueeze(0),
    fir_kernel.unsqueeze(0).unsqueeze(0),
    padding="same",
)
save_audio('demodulated.wav', demodulated, sr_aud)
plot_waveform(demodulated, sr_aud, title="Demodulated Audio", filename='demodulated_waveform.png')
plot_waterfall_spectrum(demodulated, sr_aud, include_negative_frequencies=False, title="Demodulated Audio", filename='demodulated_waterfall.png')

# %% [markdown]
# ![Demodulated Signal Spectrum](demodulated_waterfall.png)
# The carrier signal is gone, and playing the signal as audio you can hear that we have success!
#
# <audio controls src="demodulated.wav"></audio>
# ![Demodulated Signal Waveform](demodulated_waveform.png)
# You've now seen all of the steps required to demodulate an AM radio signal, and hopefully have an intuition for what each step is doing.
#
# In the next part we will devise a machine learning architecture for this problem and a plan for creating synthetic training data. In the final installment we will use those peices to train a model and analyze the results!
