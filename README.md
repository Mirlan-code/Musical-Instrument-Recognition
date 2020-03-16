# Musical-Instrument-Recognition
Musical Instrument Recognition DS Project

Dataset to be used - [IRMAS](https://www.upf.edu/web/mtg/irmas)

## Instruments
1. First iteration: gac, pia, sax
2. Added gel

## Preprocessing
Filtering: tracks from the dataset which have some noise or other instruments on the background, were removed (i.e. for gac, pia, sax).

### Mel Spectogram:
The Mel Spectrogram is the result of the following pipeline:
1. Separate to windows: Sample the input with windows of size n_fft=2048, making hops of size hop_length=512 each time to sample the next window.
2. Compute FFT (Fast Fourier Transform) for each window to transform from time domain to frequency domain.
3. Generate a Mel scale: Take the entire frequency spectrum, and separate it into n_mels=128 evenly spaced frequencies. And what do we mean by evenly spaced? not by distance on the frequency dimension, but distance as it is heard by the human ear.
4. Generate Spectrogram: For each window, decompose the magnitude of the signal into its components, corresponding to the frequencies in the mel scale.

The formula for mel frequency scale is 

**mel = 2595 * log10 (1 + hertz / 700)**

### RMS
Compute root-mean-square (RMS) value for each frame of the audiosignal.

### Spectral Bandwidth
The spectral bandwidth at frame t is computed by

**(sum_k S[k, t] * (freq[k, t] - centroid[t])\*\*p)\*\*(1/p)**

The spectral bandwidth is related to the resolution capabilities of the instrument.
https://www.analiticaweb.com.br/newsletter/02/AN51721_UV.pdf

### Spectral Centroid
Each frame of a magnitude spectrogram is normalized and treated as a distribution over frequency bins, from which the mean (centroid) is extracted per frame.
More precisely, the centroid at frame t is defined as

**centroid[t] = sum_k S[k, t] * freq[k] / (sum_j S[j, t])**

The spectral centroid is a good predictor of the "brightness" of a sound.

### Spectral Rolloff
The roll-off frequency is defined for each frame as the center frequency for a spectrogram bin such that at least roll_percent (0.85 by default) of the energy of the spectrum in this frame is contained in this bin and the bins below.
The spectrall rolloff can be useful to distinguish voiced from unvoiced signals, i.e. erase determine some noise.

### Zero Crossing Rate
The zero-crossing rate is the rate of sign-changes along a signal, i.e., the rate at which the signal changes from positive to zero to negative or from negative to zero to positive.
Zero-crossing rate can be interpreted as a measure of the noisiness of a signal.

Almir Mullanurov
