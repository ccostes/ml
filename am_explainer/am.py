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
# Would you guess that it's possible to build a program that turns AM radio into audio and reduces noise, without actually knowing how radio or noise reduction works? It turns out you can; I know because I did it! (you can probably guess how based on the title)
# 
# Okay I lied a bit, I do know a thing or two about radio, but the results still impressed me! You can see that the audio (lower half of the plot below) is much stronger (brighter) and more distinct in the model output, and can hear the difference below.
# ![Noise Comparison](comparison.png)
# <!--more-->
# **Standard Processing:**
# <audio controls src="0dB_standard.wav"></audio>
# **Model Output:**
# <audio controls src="0dB_model.wav"></audio>
# In this series of articles, I'll take you through the process of how I achieved these results, from understanding the fundamentals of AM radio to designing and training a machine learning model.
# 
# Part one (you are here) is a quick primer on AM radio and how these signals are traditionally processed. In the next part, we will design a model architecture and generate synthetic training data. Finally, the third installment will cover model training and analyze how the results change depending on our choices.
# 
# AM is one of the oldest radio technologies, while still being widely used for aviation and maritime communication. However, the point of this series isn't actually about making a better radio. Rather, the goal is to show my process of applying machine learning to a real-world problem from start to finish (though I hope you find the radio concepts as interesting as I do).
# 
# Disclaimer: I am not an expert. This is the way I figured out how to do it, but there could certainly be better ways of doing things (and if you know of any I'd love to hear about them). 
# 
# Not legal advice. Check with your doctor if machine learning is right for you.
# 

# %% [markdown]
# ## AM Demodulation Primer
# > *If you don't understand the problem you're trying to apply machine learning to, you're gonna have a bad time.*
#
# Amplitude Modulation (AM) is the simplest way to transmit audio signals via radio. You might wonder why we don't just transmit the audio signals directly, but this is undesirable for a variety of reasons (your antenna would need to hundreds of kilometers long, for one). Higher frequencies are generally easier to transmit, so the next simplest thing is to combine the audio signal with a higher frequency sine wave "carrier", transmit that signal, and then remove the carrier on the receiving end.
# ```
# carrier = 1 + sin(2 * pi * frequency)
# modulated = ((1 + audio) * carrier)
# ```
# The animation below shows what this looks like as the carrier frequency increases from zero, but actual signals will use a much higher carrier frequency (aircraft radio is around 120 MHz).
# <video controls="" autoplay="" loop="">
#     <source src="am.mp4" type="video/mp4">
# </video>

# %% [markdown]
# ## Software-Defined Radio
# An important detail is how we will receive the signals we wish to process. Traditionally, radios could only be implemented with analog electronics, but with modern processing power we can replace a lot of that signal-specific circuitry with software; this is called *Software-Defined Radio (SDR)*. This allows us to receive a wide variety of signals, from  satellite weather imagery to TV broadcasts, with the a single hardware device. Processing the signal in software is a lot more flexible, and enables entirely new capabilities (like using a machine learning model).
#
# SDR receivers capture a "window" of frequencies, and we can tune this window so that it includes the frequency we are interested in. This also means that we can receive multiple signals simultaneously if they fall within that window, e.g. simultaneously listening to multiple radio stations; neat!
#
# ## AM Demodulation Process
# Now that you understand what an AM signal is and how we are receiving it, lets look at each step of the process to turn a raw SDR signal containing an AM signal into audio.
#
# ### Input signal
# We will start by taking a look at our input signal using what's called a spectrogram. This shows the magnitude of a signal at various frequencies on the vertical axis, over time on the horizontal axis. Note that these frequencies are relative to where the receiver was tuned, e.g. tuned to 500 kHz, a 300 kHz signal would show up on the plot at -200 kHz.
# The code below reads recorded SDR data, stored as pairs of bytes representing a complex signal, and plots the spectrogram. Complex values aren't as scary as they sound, and for our purposes all you really need to know is that it is a two-dimensional signal where the first is referred to as `I` (or *real*), and the second is `Q` (or *imaginary*).

# %%
sr_in = 1.04e6  # SDR sampling rate
carrier_in = int(sr_in / 4) * -1  # SDR tuned to freq + (sampling rate / 4)
sr_aud = 8192  # Desired audio sampling rate
downsample = round(sr_in / sr_aud)  # Downsampling factor
print(f"Input Rate: {sr_in/1e3} kHz\nAudio Rate: {sr_aud/1e3} kHz\nDownsampling factor: {downsample}")
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
# You can see in the spectrogram that we captured a few different signals, shown by the bright horizontal lines. The signal we want to demodulate is at 120.9 MHz, which shows here as the continuous line just below 200 kHz since the receiver was tuned to 121.16 MHz.
#
# ### Shift to Baseband
# Recall that an AM signal mixes the audio with a high-frequency carrier, in this case at a frequency of 120.9 MHz, which effectively shifts the frequency of the audio up by that amount. Naturally, the first step to recover the audio is to reverse that and shift it back down, specifically by the frequency of the carrier
#
# This is implemented by the code below. For our purposes it isn't necessary to understand how or why it works, as long as you understand that the result (shown in the spectrogram below) has moved our signal of interest to 0 Hz, or *Baseband* (or at least pretty close).

# %%
pattern = np.array([1, 1j, -1, -1j], dtype=np.complex64)
pattern_repeated = np.tile(pattern, len(complex_samples) // len(pattern))
shifted_complex = complex_samples * pattern_repeated
plot_waterfall_spectrum(shifted_complex, sr_in, title="Baseband Signal", filename='baseband.png')

# %% [markdown]
# ![Baseband Signal](baseband.png)
# ### Downsample
# Looking at the spectrogram after shifting the input so that our AM signal is at baseband, the audio portion we are interested in (~300 Hz to 3 kHz) represents a tiny portion of the over one-million-Hz frequency range in the data. We can get rid of all that extra information with what's called *downsampling*. 
# Fundamentally, lower-frequency signals can be represented by fewer data points per second, called the *sampling rate*. This concept is illustrated in the animation below, showing two signals sampled at increasing rates, with a reconstructed signal based on the sampled points. You can see that the low frequency signal is approximated with much lower sampling rate than the high frequency signal.
#
# <video controls="" autoplay="" loop="">
#     <source src="sampling.mp4" type="video/mp4">
# </video>
#
# The corollary to this is that the *maximum* frequency a given sample rate can represent is twice the sampling rate (see [Nyquist-Shannon Theorem](https://en.m.wikipedia.org/wiki/Nyquist%E2%80%93Shannon_sampling_theorem) for details). Any signals outside that range will be "folded back" into the range, called *aliasing*. You can see this effect on the high-frequency signal in the animation as lower-frequency sine waves that appear at various sampling rates.
#
# Applying this to our data, we can get rid of the signals we're not interested in, *and* have a lot less data to process, by lowering the sampling rate to roughly double the highest frequency we're interested in. Concretely, an ~8 kHz sampling rate is sufficient for voice audio which requires only 16 kB/s of data, compared to our input signal at over 2MB per second!
# To avoid the higher-frequency signals in our data from aliasing and interfering with the audio we first apply a filter to remove any signals at frequencies above the max of our new sampling rate.

# %%
import torch
import modulation
baseband = torch.tensor(shifted_complex, dtype=torch.complex64)  # Convert to complex torch tensor
# Apply anti-aliasing low-pass filter
anti_aliasing_cutoff = sr_aud / 2
anti_aliasing_kernel = modulation.design_low_pass_filter(
    anti_aliasing_cutoff, sample_rate=sr_in, kernel_size = 101
)
filtered = modulation.fir_low_pass_filter(
    baseband.unsqueeze(0).unsqueeze(0),
    anti_aliasing_kernel.unsqueeze(0).unsqueeze(0),
    padding="same",
)
plot_waterfall_spectrum(filtered, sr_in, title="Anti-alias Filtered", filename='baseband_filtered.png')

# %% [markdown]
# ![Anti-Alias-Filtered Baseband](baseband_filtered.png)
# You can see that the filtering has removed all of the other signals from our data, leaving a thin bright line in the middle that contains our audio.
# This filtered signal can now be downsampled by simply picking out samples at the new rate and throwing the rest away. E.g. if the original sampling rate were 1 kHz and our desired rate was 100 Hz we would save every 10th sample.

# %%
# Decimate to desired sample rate
decimation_factor = int(sr_in // sr_aud)
decimated = filtered[::decimation_factor]
save_audio('decimated.wav', decimated, sr_aud)
plot_waterfall_spectrum(decimated, sr_aud, include_negative_frequencies=False, title="Decimated", filename='decimated.png')

# %% [markdown]
# ![Decimated Signal](decimated.png)
# The spectrogram of the decimated signal shows the new frequency range of up to about 4 kHz, and you can see the audio data.
# ### Rectify and filter
# With just these two steps we are actually quite close to the desired audio, which you can start to hear at this stage, but there's clearly something wrong. Look at the previous spectrogram and see if you can guess what the issue is.
# <audio controls src="decimated.wav"></audio>
#
# You can see that the carrier signal is still present as the bright continuous line near the bottom of the spectrogram; not quite at 0 Hz. The reason it is not at exactly 0 Hz is because to the SDR hardware used here isn't super-precise and has some tuning inaccuracy (which can even change over time due to things like temperature). Thus the original spectrogram wasn't tuned quite to where we thought, and the shift didn't bring the AM signal to exactly 0 Hz.
#
# Luckily this is relatively easy to fix by applying another filter, this time a band-pass which only allows frequencies within a given band to pass, which we set to the audio frequency range of 300 Hz to 3 kHz.

# %%
# Demodulate: Rectify and filter
low_cut_freq = 300
high_cut_freq = 3000
rectified = torch.abs(decimated)
save_audio('demodulated.wav', rectified, sr_aud)
fir_kernel = modulation.design_band_pass_filter(
    low_cut_freq, high_cut_freq, sr_aud, kernel_size=101
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
# The carrier signal is gone, and playing the signal as audio, you can hear that we have successfully demodulated the AM signal!
#
# <audio controls src="rectified.wav"></audio>
# <audio controls src="demodulated.wav"></audio>
# ![Demodulated Signal Waveform](demodulated_waveform.png)
# You've now seen all of the steps required to demodulate an AM radio signal, and hopefully have an intuition for what is happening in each.
#
# The next installment will introduce machine learning, designing a model architecture for this problem, and devising a plan for creating synthetic training data. Stay tuned!
