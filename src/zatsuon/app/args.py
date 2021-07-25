def add_args(parser):
    """Add arguments to the argparse.ArgumentParser
    Args:
        parser: argparse.ArgumentParser
    Returns:
        parser: a parser added with args
    """

    # Training settings
    parser.add_argument(
        "--task",
        type=str,
        default="train",
        metavar="T",
        help="the type of task: train or denoise",
    )

    parser.add_argument(
        "--datadir",
        type=str,
        metavar="DD",
        help="data directory for training",
    )

    parser.add_argument(
        "--noisy_wav",
        type=str,
        metavar="NW",
        help="path to noisy wav",
    )

    parser.add_argument(
        "--denoised_wav",
        type=str,
        default="denoised_sample.wav",
        metavar="DW",
        help="path to denoised wav",
    )

    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        metavar="PT",
        help="path to pre-trainedmodel",
    )

    parser.add_argument(
        "--saved_model_path",
        type=str,
        default="model.pth",
        metavar="SMP",
        help="path to trained model",
    )

    parser.add_argument(
        "--partition_ratio",
        type=float,
        default=1 / 3,
        metavar="PR",
        help="partition ratio for trainig (default: 1/3)",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=5,
        metavar="BS",
        help="input batch size for training (default: 5)",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        metavar="LR",
        help="learning rate (default: 0.3)",
    )

    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        metavar="M",
        help="momentum (default: 0.9)",
    )

    parser.add_argument(
        "--noise_amp",
        type=float,
        default=0.01,
        metavar="NA",
        help="amplitude of added noise for trainign (default: 0.01)",
    )

    parser.add_argument(
        "--split_sec",
        type=float,
        default=1.0,
        metavar="SS",
        help="interval for splitting [sec]",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        metavar="EP",
        help="how many epochs will be trained",
    )

    parser.add_argument(
        "--sampling_rate",
        type=int,
        default=16000,
        metavar="SR",
        help="sampling rate",
    )

    parser.add_argument(
        "--log_interval",
        type=int,
        default=2,
        metavar="LI",
        help="log interval",
    )

    parser.add_argument(
        "--path_to_loss",
        type=str,
        default=None,
        metavar="PL",
        help="path to png filw which shows the transtion of loss",
    )

    return parser
