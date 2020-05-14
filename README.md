# Musical-Instrument-Recognition
Musical Instrument Recognition DS Project

Dataset to be used - [IRMAS](https://www.upf.edu/web/mtg/irmas)

## Instruments
1. First iteration: gac, pia, sax
2. Added gel

## Preprocessing
Filtering: tracks from the dataset which have some noise or other instruments on the background, were removed (i.e. for gac, pia, sax).

### Chromagram
Main property of chromagram features is that they capture harmonic and melodic characteristics of music, while being robust to changes in timbre and instrumentation.

### Constant-Q chromagram
The transform is well suited to musical data, and this can be seen in some of its advantages compared to the fast Fourier transform. As the output of the transform is effectively amplitude/phase against log frequency, fewer frequency bins are required to cover a given range effectively, and this proves useful where frequencies span several octaves. As the range of human hearing covers approximately ten octaves from 20 Hz to around 20 kHz, this reduction in output data is significant.

### “Chroma Energy Normalized” (CENS)
To compute CENS features, following steps are taken after obtaining chroma vectors using chroma_cqt:
1. L-1 normalization of each chroma vector
2. Quantization of amplitude based on “log-like” amplitude thresholds
3. (Optional) Smoothing with sliding window. Default window length = 41 frames 4

CENS features are robust to dynamics, timbre and articulation, thus these are commonly used in audio matching and retrieval applications.

### Mel Spectogram
The Mel Spectrogram is the result of the following pipeline:
1. Separate to windows: Sample the input with windows of size n_fft=2048, making hops of size hop_length=512 each time to sample the next window.
2. Compute FFT (Fast Fourier Transform) for each window to transform from time domain to frequency domain.
3. Generate a Mel scale: Take the entire frequency spectrum, and separate it into n_mels=128 evenly spaced frequencies. And what do we mean by evenly spaced? not by distance on the frequency dimension, but distance as it is heard by the human ear.
4. Generate Spectrogram: For each window, decompose the magnitude of the signal into its components, corresponding to the frequencies in the mel scale.

The formula for mel frequency scale is 

$mel = 2595 * log_{10} (1 + hertz / 700)$

### RMS
Compute root-mean-square (RMS) value for each frame of the audiosignal.

### Spectral Bandwidth
The spectral bandwidth at frame t is computed by

$(sum_k S[k, t] * (freq[k, t] - centroid[t])^p)^{(1/p)}$

The spectral bandwidth is related to the resolution capabilities of the instrument.
https://www.analiticaweb.com.br/newsletter/02/AN51721_UV.pdf

### Spectral Centroid
Each frame of a magnitude spectrogram is normalized and treated as a distribution over frequency bins, from which the mean (centroid) is extracted per frame.
More precisely, the centroid at frame t is defined as

$centroid[t] = sum_k S[k, t] * freq[k] / (sum_j S[j, t])$

The spectral centroid is a good predictor of the "brightness" of a sound.

### Spectral Contrast
Each frame of a spectrogram S is divided into sub-bands. For each sub-band, the energy contrast is estimated by comparing the mean energy in the top quantile (peak energy) to that of the bottom quantile (valley energy). High contrast values generally correspond to clear, narrow-band signals, while low contrast values correspond to broad-band noise.

### Spectral Flatness
Spectral flatness (or tonality coefficient) is a measure to quantify how much noise-like a sound is, as opposed to being tone-like. A high spectral flatness (closer to 1.0) indicates the spectrum is similar to white noise. It is often converted to decibel.

### Spectral Rolloff
The roll-off frequency is defined for each frame as the center frequency for a spectrogram bin such that at least roll_percent (0.85 by default) of the energy of the spectrum in this frame is contained in this bin and the bins below.

The spectrall rolloff can be useful to distinguish voiced from unvoiced signals, i.e. determine some noise.

### Poly Features
Get coefficients of fitting an nth-order polynomial to the columns of a spectrogram.

### Tonnetz
Computes the tonal centroid features (tonnetz). The Tonnetz is often thought of as an approximate visualization of four important harmonic relationships – the parallel (e.g., C maj–C min), leittonwechsel (e.g., C maj–E min), relative (e.g., C maj–A min), and dominant (e.g., C maj–G maj).

### Zero Crossing Rate
The zero-crossing rate is the rate of sign-changes along a signal, i.e., the rate at which the signal changes from positive to zero to negative or from negative to zero to positive.

Zero-crossing rate can be interpreted as a measure of the noisiness of a signal.

### Tempogram
It is the local autocorrelation of the onset strength envelope. Tempo refers to the rate of the musical beat and is given by the reciprocal of the beat period. Tempo is often defined in units of beats per minute (BPM)

### Fourier Tempogram
The Fourier Tempogram is basically the short-time Fourier transform of the onset strength envelope.

## One feature accuracy (logistic regression)
1. melspectogram - 0.59
2. tempogram - 0.56
3. fourier_tempogram - 0.55 (almost the same as tempogram meaning)
4. spectral_contrast - 0.53
5. zero_crossing_rate - 0.46
6. tonnetz - 0.44
7. chroma_cens - 0.43
8. chroma_cqt - 0.43
9. chroma_stft - 0.41
10. spectral_flatness - 0.40
11. spectral_bandwidth - 0.40
12. spectral_centroid - 0.39
13. spectral_rolloff - 0.39
14. rms - 0.38
15. poly_features - 0.38
    
## Variants
1. melspectogram, rms, spectral_bandwidth, spectral_centroid, spectral_rolloff, zero_crossing_rate - 0.58
2. melspectogram, rms, spectral_bandwidth, spectral_centroid, zero_crossing_rate - 0.54
3. melspectogram, rms, spectral_bandwidth, spectral_centroid, zero_crossing_rate, tonnetz - 0.53
4. melspectogram, tempogram, spectral_contrast, zero_crossing_rate, tonnetz, chroma_cens, chroma_cqt, chroma_stft, spectral_flatness, spectral_bandwidth - 0.62

## Random Forest
After reading some articles I decided to try random forest to recognize the instruments and accuracy became much higher - 0.77


1. поэкспериментировать с фичами
2. Выбрать лучший
3. На каких треках падает модель??
4. С чем это связано?

Almir Mullanurov
