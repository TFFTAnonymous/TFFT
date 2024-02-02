"""
baselines.py

Implementation of baseline methods:
Fast Fourier Transform (FFT),
FFT with Linear Adjustment (LA+FFT),
Discrete Cosine Transform of type 2 (DCT-II),
Short-Time Fourier Transform (STFT),
and Modified Discrete Cosine Transform (MDCT).
"""

import tensorflow as tf
import numpy as np
import torch


def recons_fft(x, compression_ratio):
    """
    Use FFT for reconstruction
    :param x: input array
    :param compression_ratio: compression_ratio
    :return: RMSE of reconstruction
    """
    x = torch.tensor(x, dtype=torch.float)
    N = len(x)
    # Apply FFT
    fx = torch.fft.rfft(x)
    # Drop coefficients
    fx[N // (2 * compression_ratio):] = 0.0
    # Apply Inverse FFT
    y = torch.fft.irfft(fx, N)
    error = ((x - y) ** 2).mean().sqrt().item()
    return error


def recons_lafft(x, compression_ratio, frame_length=128):
    """
    Use FFT with linear adjustment for reconstruction
    :param x: input array
    :param compression_ratio: compression_ratio
    :param frame_length: frame_length for each window
    :return: RMSE of reconstruction
    """
    x_matrix = x.reshape(-1, frame_length)
    error = 0.0
    for frame in x_matrix:
        frame = torch.tensor(frame, dtype=torch.float)
        # Linear adjustment
        frame = frame - (frame[frame_length - 1] - frame[0]) / (frame_length - 1) * torch.arange(frame_length)
        # Apply FFT
        fframe = torch.fft.rfft(frame)
        # Drop coefficients
        fframe[frame_length // (2 * compression_ratio) - 1:] = 0.0
        # Apply Inverse FFT
        y = torch.fft.irfft(fframe, frame_length)
        error += ((frame - y) ** 2).mean().item()
    error = (error / len(x_matrix)) ** 0.5
    return error


def recons_dct(x, compression_ratio, frame_length=128):
    """
    Use Type II DCT for reconstruction
    :param x: input array
    :param compression_ratio: compression_ratio
    :param frame_length: frame_length for each window
    :return: RMSE of reconstruction
    """
    x_matrix = x.reshape(-1, frame_length)
    error = 0.0
    for frame in x_matrix:
        frame = torch.tensor(frame, dtype=torch.float)
        # Apply DCT-II
        dct_result = tf.signal.dct(frame, type=2)
        # Drop coefficients
        drop_rate = 1.0 - 1.0 / compression_ratio
        num_zeros = int(drop_rate * dct_result.shape[-1])
        mask = tf.concat([tf.ones(dct_result.shape[0] - num_zeros, dtype=tf.float32),
                          tf.zeros(num_zeros, dtype=tf.float32)], axis=0)
        masked_dct_result = dct_result * mask
        # Apply Inverse DCT-II
        reconstructed_data = tf.signal.idct(masked_dct_result, type=2) * (0.5 / frame_length)
        error += tf.reduce_mean((frame - reconstructed_data) ** 2).numpy()
    error = (error / len(x_matrix)) ** 0.5
    return error


def recons_stft(x, compression_ratio, frame_length=128):
    """
    Use STFT for reconstruction
    :param x: input array
    :param compression_ratio: compression_ratio
    :param frame_length: frame_length for each window
    :return: RMSE of reconstruction
    """
    step_size = frame_length // 2
    # Apply STFT
    time_series = np.pad(x, (step_size, step_size), mode='edge')
    stft_result = tf.signal.stft(tf.cast(time_series, tf.float32), frame_length=frame_length,
                                 frame_step=step_size, fft_length=frame_length)
    # Drop coefficients
    rows, cols = stft_result.shape
    drop_rate = 1.0 - 1.0 / (2 * compression_ratio)
    num_zeros = int(drop_rate * stft_result.shape[-1])
    mask = tf.concat([tf.ones((rows, cols - num_zeros), dtype=tf.complex64),
                      tf.zeros((rows, num_zeros), dtype=tf.complex64)], axis=1)
    masked_stft_result = stft_result * mask
    # Apply Inverse STFT
    istft_result = tf.signal.inverse_stft(masked_stft_result, frame_length=frame_length,
                                          frame_step=step_size, fft_length=frame_length,
                                          window_fn=tf.signal.inverse_stft_window_fn(step_size))
    reconstructed_time_series = istft_result.numpy()
    # Compute RMSE
    error = np.sqrt(np.mean((time_series[step_size:-step_size] -
                             reconstructed_time_series[step_size:-step_size]) ** 2))
    return error


def recons_mdct(x, compression_ratio, frame_length=128):
    """
    Use MDCT for reconstruction
    :param x: input array
    :param compression_ratio: compression_ratio
    :param frame_length: frame_length for each window
    :return: RMSE of reconstruction
    """
    step_size = frame_length // 2
    # Apply MDCT
    time_series = np.pad(x, (step_size, step_size), mode='edge')
    mdct_result = tf.signal.mdct(tf.cast(time_series, tf.float32), frame_length=frame_length,
                                 norm=None, window_fn=tf.signal.kaiser_bessel_derived_window)
    # Drop coefficients
    rows, cols = mdct_result.shape
    drop_rate = 1.0 - 1.0 / compression_ratio
    num_zeros = int(drop_rate * mdct_result.shape[-1])
    mask = tf.concat([tf.ones((rows, cols - num_zeros), dtype=tf.float32),
                      tf.zeros((rows, num_zeros), dtype=tf.float32)], axis=1)
    masked_mdct_result = mdct_result * mask
    # Apply Inverse MDCT
    imcct_result = tf.signal.inverse_mdct(masked_mdct_result, norm=None,
                                          window_fn=tf.signal.kaiser_bessel_derived_window)
    reconstructed_time_series = imcct_result.numpy()
    # Compute RMSE
    error = np.sqrt(np.mean((time_series[step_size:-step_size] -
                             reconstructed_time_series[step_size:-step_size]) ** 2))
    return error

