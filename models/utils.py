"""
utils.py

Utility functions for evaluating each method.
"""

from .baselines import *
from .model import *
from torch import optim


def mse_loss(output, target):
    """
    Compute MSE loss between output and target
    :param output: output
    :param target: target
    :return: MSE loss
    """
    return ((output - target) ** 2).mean()


def train_tfft(x, in_size, hidden_size, epochs=301, iters=5):
    """
    Train T-FFT and return encoded hidden representations and learned parameters
    :param x: input array
    :param in_size: input size
    :param hidden_size: hidden representation size
    :param epochs: number of epochs
    :param iters: number of evaluations, among which the best result is returned
    :return: encoded hidden representations and learned parameters
    """
    encoded_best, loss_best = None, torch.inf
    for it in range(iters):
        tfft = TFFT(in_size=in_size, hidden_size=hidden_size)
        criterion = mse_loss
        optimizer = optim.Adam(tfft.parameters(), lr=5e-3)
        for epoch in range(epochs):
            encoded, y = tfft(x)
            loss = criterion(x, y)
            if loss < loss_best:
                loss_best = loss.item()
                encoded_best = (encoded[0], encoded[1].clone(),
                                encoded[2].clone(), encoded[3].clone(),
                                encoded[4].clone(), encoded[5].clone())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return encoded_best


def recons_tfft(x, compression_ratio, frame_length=128):
    """
    Use T-FFT for reconstruction
    :param x: input array
    :param compression_ratio: compression_ratio
    :param frame_length: frame_length for each window
    :return: RMSE of reconstruction
    """
    x_matrix = x.reshape(-1, frame_length)
    hidden = frame_length // (2 * compression_ratio) - 4
    error = 0.0
    for i, frame in enumerate(x_matrix):
        frame = torch.tensor(frame, dtype=torch.float)
        encoded = train_tfft(frame, frame_length, hidden)
        decoder = DECODE(frame_length)
        y = decoder(encoded)
        error += mse_loss(frame, y).item()
    error = (error / len(x_matrix)) ** 0.5
    return error


def run(data, model, cr):
    """
    Evaluate a method
    :param data: dataset file name
    :param model: model for evalutation
    :param cr: compression ratio
    :return: average RMSE
    """
    file_path = f"data/{data}.csv"
    dataset = np.genfromtxt(file_path, delimiter=',', skip_header=1,
                            dtype=float, encoding='utf-8')
    eval_model = {'TFFT': recons_tfft, 'FFT': recons_fft,
                  'LAFFT': recons_lafft, 'DCT': recons_dct,
                  'STFT': recons_stft, 'MDCT': recons_mdct}[model]
    list_rmse = []
    for i, x in enumerate(dataset.transpose()):
        rmse = eval_model(x, compression_ratio=cr)
        list_rmse.append(rmse)
        print(f'RMSE of {i}th instance = {rmse:.4f}')
    list_rmse = np.array(list_rmse)[~np.isnan(list_rmse)]
    avg_rmse = list_rmse.mean()
    print(f'Average RMSE = {avg_rmse:.4f}')
