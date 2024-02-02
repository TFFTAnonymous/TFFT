# Tailored FFT

This repository is the official implementation of 
[Tailored FFT: Compact Frequency Representation with Discrete Taylor Transform and FFT](https://openreview.net/forum?id=8Fa78aHkO9) 
(submitted to ICML 2024).

## Abstract

Fast Fourier transform (FFT) has garnered considerable attention in a variety of 
research fields due to its widespread applicability including data compression, 
anomaly detection, and feature extraction. Given a signal in the time domain, 
applying FFT to it typically yields a skewed frequency representation with larger 
amplitudes distributed at lower frequencies, and by removing the less-significant 
high-frequency components, one can reduce the amount of data while still 
preserving the essential characteristics of the signal. Clearly, this approach 
achieves greater efficiency for more compact representations concentrated towards 
the low frequencies. However, the FFT and its variants have limitations in obtaining 
compact frequency representations because their frequency coefficients are widely 
spread throughout the entire frequency spectrum. In this paper, we propose 
Tailored FFT (T-FFT), a simple yet effective model for a compact frequency 
representation of a signal. The key of T-FFT is to combine the FFT with our novel 
operator, Discrete Taylor Transform (DTT), which is a discrete analogue of the 
continuous Taylor expansion. DTT adaptively redistributes the Fourier coefficients 
to prioritize the lower frequencies and provides shorter representations while 
retaining important information. Furthermore, DTT has an inverse function compatible 
with the inverse FFT, enabling perfect reconstruction of the original signal. Experiments 
show that T-FFT provides more compact frequency representations than existing 
methods, such as MDCT and STFT, achieving the lowest reconstruction error with 
the same number of coefficients.

## Datasets
The seven datasets used in our paper are included in `data/`.

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Evaluate Models

```
python main.py --model {method} --data {dataset} --cr {compression_ratio}
```
- {method} : Method to evaluate. One of TFFT, FFT, LAFFT, DCT, STFT, MDCT.
- {dataset} : Dataset to use. One of synth-low, synth-high, har, power, pressure, aircond, stocknet.
- {compression_ratio}: Compression ratio (e.g., 2, 4).

### Demo

To run a demo of our proposed method, simply execute this command:

```run demo
python main.py --model TFFT --data synth-low --cr 2
```

