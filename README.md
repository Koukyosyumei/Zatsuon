# Zatsuon
CLI Tool to remove audio noise with auto encoder

## Usage

train

    zatsuon --task train --datadir ./train --saved_model_path "dae.pth" --partition_ratio 0.3 --batch_size 5 --lr 0.01 --momentum 0.9 --noise_amp 0.01 --split_sec 1.0 --epochs 10 --sampling_rate 16000 --log_interval 2 --path_to_loss loss.png

denoise

    zatsuon --task denoise --noisy_wav ./train/meian_0018.wav --denoised_wav denoised-sample.wav --pretrained ./dae.pth

optional arguments

    -h, --help            show this help message and exit
    --task T              the type of task: train or denoise
    --datadir DD          data directory for training
    --noisy_wav NW        path to noisy wav
    --denoised_wav DW     path to denoised wav
    --pretrained PT       path to pre-trainedmodel
    --saved_model_path SMP path to trained model
    --partition_ratio PR  partition ratio for trainig (default: 1/3)
    --batch_size BS       input batch size for training (default: 5)
    --lr LR               learning rate (default: 0.3)
    --momentum M          momentum (default: 0.9)
    --noise_amp NA        amplitude of added noise for trainign (default: 0.01)
    --split_sec SS        interval for splitting [sec]
    --epochs EP           how many epochs will be trained
    --sampling_rate SR    sampling rate
    --log_interval LI     log interval
    --path_to_loss PL     path to png filw which shows the transtion of loss