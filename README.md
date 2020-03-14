# Musical-Instrument-Recognition
Musical Instrument Recognition DS Project

Dataset to be used - [IRMAS](https://www.upf.edu/web/mtg/irmas)

## Preprocessing
Filtering: tracks from the dataset which have some noise or other instruments on the background, were removed (i.e. for gac, pia, sax).

### Mel Spectogram:
The Mel Spectrogram is the result of the following pipeline:
1. Separate to windows: Sample the input with windows of size n_fft=2048, making hops of size hop_length=512 each time to sample the next window.
2. Compute FFT (Fast Fourier Transform) for each window to transform from time domain to frequency domain.
3. Generate a Mel scale: Take the entire frequency spectrum, and separate it into n_mels=128 evenly spaced frequencies. And what do we mean by evenly spaced? not by distance on the frequency dimension, but distance as it is heard by the human ear.
4. Generate Spectrogram: For each window, decompose the magnitude of the signal into its components, corresponding to the frequencies in the mel scale.

### RMS
Compute root-mean-square (RMS) value for each frame of the audiosignal.

Almir Mullanurov
