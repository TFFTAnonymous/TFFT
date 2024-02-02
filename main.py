"""
main.py

This is an implementation of "Tailored FFT: Compact Frequency Representation
with Discrete Taylor Transform and FFT" (submitted to ICML 2024).
See README.md for the usage.
"""

import argparse
import models


def parse_args():
    """
    parser arguments to run program in cmd
    :return: arguments
    """
    parser = argparse.ArgumentParser()
    # Hyperparameter for dataset
    parser.add_argument('--data', type=str, default="synth-low",
                        help='One of synth-low, synth-high, har, power, pressure, aircond, stocknet')
    # Hyperparameter for model
    parser.add_argument('--model', type=str, default="TFFT",
                        help='One of TFFT, FFT, LAFFT, DCT, STFT, MDCT')
    # Hyperparameter for compression ratio
    parser.add_argument('--cr', type=int, default=2,
                        help='Compression ratio (e.g., 2, 4)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    models.run(data=args.data, model=args.model, cr=args.cr)
