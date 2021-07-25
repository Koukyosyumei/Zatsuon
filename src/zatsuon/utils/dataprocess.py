import glob
import struct
import wave

import librosa
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset


def np2wav(np_wave, path, sampling_rate):
    """Save numpy array as wav file

    Args:
        np_wave: target np array
        path: path to the saved file
        sampling_rate: sampling rate
    """
    max_num = 32767.0 / np.max(np.abs(np_wave))  # int16は-32768~32767の範囲

    wave16 = [int(x * max_num) for x in np_wave]  # 16bit符号付き整数に変換

    # バイナリ化，'h'が2byte整数のフォーマット
    bi_wave = struct.pack("h" * len(wave16), *wave16)

    # サイン波をwavファイルとして書き出し
    w = wave.Wave_write(path)
    p = (1, 2, sampling_rate, len(bi_wave), "NONE", "not compressed")
    w.setparams(p)
    w.writeframes(bi_wave)
    w.close()


def pad(np_wave, sampling_rate):
    """paddin numpy array with zero to make its length equalt to sampling rate

    Args:
        np_wave: target audio (np.array)
        sampling_rate: sampling rate

    Returns:
        np_wave_padded: padded np.array (whose length is equal to sampling rate)
    """
    np_wave_padded = np.concatenate(
        [np_wave, np.zeros(int(sampling_rate) - np_wave.shape[0] % int(sampling_rate))]
    )
    return np_wave_padded


def convert_wav_to_training_data(
    path_to_folder, sampling_rate, split_sec=1.0, noise_amp=0.0
):
    """Convert wav files within given folder to training data
       This function creates dataset by spliting wav files with equal length and addin random noise (~ N())

    Args:
        path_to_folder: target folder
        sampling_rate: sampling rate
        split_sec: length of one record [sec]
        noise_amp: mean amplitude of noise
    """
    path_list = glob.glob(path_to_folder + "/*.wav")

    raw_data = []
    X = []
    y = []

    for p in path_list:
        raw, _ = librosa.load(p, sr=sampling_rate)
        raw_data.append(raw)

    for raw_wave in raw_data:

        wave_padded = pad(raw_wave, sampling_rate)
        wave_padded_split = np.array(
            np.split(wave_padded, wave_padded.shape[0] / int(sampling_rate) / split_sec)
        )
        y.append(wave_padded_split)

        wave_padded_split_noised = wave_padded_split + np.random.uniform(
            -noise_amp, noise_amp, wave_padded_split.shape
        )
        X.append(wave_padded_split_noised)

    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)

    return X, y


def create_train_and_test_dataset(X, y, test_size=1 / 3, random_state=42):
    """Split given data and conver them to Torch Dataset

    Args:
        X: numpy array
        y: numpy array
        test_size: split size
        random_state: used for random split
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))

    return train_dataset, test_dataset
