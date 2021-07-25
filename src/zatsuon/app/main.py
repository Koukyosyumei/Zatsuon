import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt

from ..dae import DAE_Module, DenoisedAutoEncoder
from ..utils import convert_wav_to_training_data, create_train_and_test_dataset
from .args import add_args


def main():
    parser = add_args(argparse.ArgumentParser(description="denoise audio data"))
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = DAE_Module(sampling_rate=int(args.sampling_rate * args.split_sec))
    model.to(device)

    if args.pretrained is not None:
        model.load_state_dict(torch.load(args.pretrained, map_location=device))

    if args.task == "train":

        X, y = convert_wav_to_training_data(
            args.datadir,
            args.sampling_rate,
            split_sec=args.split_sec,
            noise_amp=args.noise_amp,
        )
        train_dataset, val_dataset = create_train_and_test_dataset(
            X, y, test_size=args.partition_ratio
        )

        criterion = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

        denoised_autoencoder = DenoisedAutoEncoder(
            model,
            criterion=criterion,
            optimizer=optimizer,
            sampling_rate=args.sampling_rate,
            split_sec=args.split_sec,
            batch_size=args.batch_size,
            epochs=args.epochs,
            log_interval=args.log_interval,
        )

        train_log, val_log = denoised_autoencoder.train(train_dataset, val_dataset)
        if args.path_to_loss is not None:
            plt.plot(train_log, label="train")
            plt.plot(val_log, label="validation")
            plt.legend()
            plt.savefig(args.path_to_loss)

        denoised_autoencoder.save_model(args.saved_model_path)

    elif args.task == "denoise":

        denoised_autoencoder = DenoisedAutoEncoder(
            model,
            sampling_rate=args.sampling_rate,
            split_sec=args.split_sec,
        )
        denoised_autoencoder.denoise(args.noisy_wav, args.denoised_wav)

    else:
        raise ValueError(f"{args.task} is not supported, it should be train or denoise")
