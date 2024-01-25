# Long Video Generation with Time-Agnostic VQGAN and Time-Sensitive Transformer (ECCV 2022)

[Project Website](https://songweige.github.io/projects/tats) | [Video](https://youtu.be/WZj7vW2mTJo) | [Paper](https://arxiv.org/abs/2204.03638)

<p align="center">
    <img src=assets/tats-ucf101.gif width="852" height="284" />
</p>
<p align="center">
    <img src=assets/tats_sky_long_high_res.gif width="852" height="210" />
</p>


**tl;dr** We propose TATS, a long video generation framework that is trained on videos with tens of frames while it is able to generate videos with thousands of frames using sliding window.

## Setup
```
  conda create -n tats python=3.8
  conda activate tats
  conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
  pip install pytorch-lightning==1.5.4
  pip install einops ftfy h5py imageio imageio-ffmpeg regex scikit-video tqdm
```
#### Datasets and trained models
UCF-101: [official data](https://www.crcv.ucf.edu/data/UCF101.php), [VQGAN](https://drive.google.com/file/d/15Otpyr7v6Wnyw2HfQr_cuaRsBiSMd7Rh/view?usp=sharing), [TATS-base](https://drive.google.com/file/d/1Nxt35mmBDuNANxHP0p8WBMWXOQ-YPkus/view?usp=sharing), [TATS-base-uncond](https://drive.google.com/file/d/1pQsMdO2b84m7asp_lNg44c_UE5DUC6pa/view?usp=sharing) <br>
Sky-Timelapse: [official data](https://github.com/weixiong-ur/mdgan), [VQGAN](https://drive.google.com/file/d/1ExV0XdJKlGP4lzn0X2W9307X-DE240iW/view?usp=sharing), [TATS-base](https://drive.google.com/file/d/1mtd_mC0ZEvImlPXAdda2-4CvE-10Ljci/view?usp=sharing) <br>
Taichi-HD: [official data](https://github.com/AliaksandrSiarohin/first-order-model/blob/master/data/taichi-loading/README.md), [VQGAN](https://drive.google.com/file/d/1hcWIADkDsm916Xkxfz1YbljHU2ZAQFpQ/view?usp=sharing), [TATS-base](https://drive.google.com/file/d/10j0p4PlkZwqQd7CmZmk9-4_ZboW4r03R/view?usp=sharing) <br>
MUGEN: [official data](https://mugen-org.github.io/download), [VQGAN](https://drive.google.com/file/d/1H-l80_zOYtIj2g7IonjRej00B9LeleRi/view?usp=sharing), [TATS-base](https://drive.google.com/file/d/1uZICD3DgbRlto__mEEY6i0i9n20eRLcV/view?usp=sharing) <br>
AudioSet-Drums: [official data](https://www.dropbox.com/s/7ykgybrc8nb3lgf/AudioSet_Drums.zip?dl=0), [Video-VQGAN](https://drive.google.com/file/d/1ZpQkVUoGWShL2M8phTzyO6W9-uYPJ40J/view?usp=sharing), [STFT-VQGAN](https://drive.google.com/file/d/1CNQLfgazb7OI_c69RWUdhO56mg-9HN-o/view?usp=sharing), [TATS-base](https://drive.google.com/file/d/1eHP0eSv3s-NGAv60HbF3AZsp3SBvR70u/view?usp=sharing) <br>

## Synthesis

1. **Short videos:** To sample the videos of the same length with the training data, use the code under `scripts/` with the following flags:

- `gpt_ckpt`: path to the trained transformer checkpoint.
- `vqgan_ckpt`: path to the trained VQGAN checkpoint.
- `save`: path to the save the generation results.
- `save_videos`: indicates that videos will be saved.
- `class_cond`: indicates that class labels are used as conditional information.

To compute the FVD, these flags are required:

- `compute_fvd`: indicates that FVD will be calculated.
- `data_path`: path to the dataset folder.
- `dataset`: dataset name.
- `image_folder`: should be used when the dataset contains frames instead of videos, e.g. Sky Time-lapse.
- `sample_every_n_frames`: number of frames to skip in the real video data, e.g. please set it to 4 when using Taichi-HD dataset.
- `resolution`: the resolution of real videos to compute FVD, e.g. 128 for UCF, Sky, and Taichi, and 256 for MUGEN.

```
python sample_vqgan_transformer_short_videos.py \
    --gpt_ckpt {GPT-CKPT} --vqgan_ckpt {VQGAN-CKPT} --class_cond \
    --save {SAVEPATH} --data_path {DATAPATH} --batch_size 16 --resolution 128 \
    --top_k 2048 --top_p 0.8 --dataset {DATANAME} --compute_fvd --save_videos
```

2. **Long videos:** To sample the videos with a length longer than the training length with a sliding window, use the following script.

- `sample_length`: number of latent frames to be generated.
- `temporal_sample_pos`: position of the frame that the sliding window approach generates.

```
python sample_vqgan_transformer_long_videos.py \
    --gpt_ckpt {GPT-CKPT} --vqgan_ckpt {VQGAN-CKPT} \
    --dataset ucf101 --class_cond --sample_length 16 --temporal_sample_pos 1 --batch_size 5 --n_sample 5 --save_videos
```

3. **Text to video:** To sample MUGEN videos conditioned on the text, check this [colab notebook](https://colab.research.google.com/drive/1yblr4IolH91ZA61FfZyk2n8rvndCIFmm?usp=sharing) for an example!


4. **Audio to video:** To sample drum videos conditioned on the audio, use the following script.

- `stft_vqgan_ckpt`: path to the trained VQGAN checkpoint for STFT features.

```
python sample_vqgan_transformer_audio_cond.py \
    --gpt_ckpt {GPT-CKPT} --vqgan_ckpt {VQGAN-CKPT} --stft_vqgan_ckpt {STFT-CKPT} \
    --dataset drum --n_sample 10
```

5. **Hierarchical sampling:** To sample the videos with a length longer than the training length with the hierarchical models, first an AR transformer and then an interpolation transformer, use the following checkpoints and scripts.

- Sky checkpoints: [VQGAN](https://drive.google.com/file/d/1ExV0XdJKlGP4lzn0X2W9307X-DE240iW/view?usp=sharing), [AR-transformer](https://drive.google.com/file/d/15ay8Lrra6opBU4o__CnrHfR2Caq2G-1G/view?usp=sharing), [Interpolation-transformer](https://drive.google.com/file/d/1WJiH3qh1_fe3PLbzKdRKR-h6iFudtAfr/view?usp=sharing) <br>
- Taichi checkpoints: [VQGAN](https://drive.google.com/file/d/15zujnX54wfqhhIToocvyHhdpbsxcTIv0/view?usp=sharing), [AR-transformer](https://drive.google.com/file/d/1TrjbpGwCkFhvsiQrnl5EH76HYZoWszbY/view?usp=sharing), [Interpolation-transformer](https://drive.google.com/file/d/1EQnV3IqmwFsEdza7gz4PUeUqr9bINPkz/view?usp=sharing) <br>

- `top_k_init`, `top_p_init`: these allows different `top_k` and `top_p` parameters when sampling the initial frames. For example, one can use `--top_k_init 16384 --top_p_init 0.99` when sampling the sky videos to increase the diversity.

```
python sample_vqgan_transformer_hierarchical.py \
    --ckpt1 {AR-CKPT} --CKPT2 {Interpolation-CKPT} --vqgan {VQGAN-CKPT} \
    --dataset sky --top_k_init 2048 --top_p_init 0.8 --top_k 2048 --top_p 0.8 --temporal_sample_pos 1
```

## Training

Example usages of training the VQGAN and transformers are shown below. Explanation of the flags that are opted to change according to different settings:

- `data_path`: path to the dataset folder.
- `default_root_dir`: path to save the checkpoints and the tensorboard logs.
- `vqvae`: path to the trained VQGAN checkpoint.
- `resolution`: the resolution of the training video clips.
- `sequence_length`: frame number of the training video clips.
- `discriminator_iter_start`: the step id to start the GAN losses.
- `image_folder`: should be used when the dataset contains frames instead of videos, e.g. Sky Time-lapse.
- `unconditional`: when no conditional information is available, e.g. Sky Time-lapse, use this flag.
- `sample_every_n_frames`: number of frames to skip in the real video data, e.g. please set it to 4 when training on the Taichi-HD dataset.
- `downsample`: sample rate in the dimensions of time, height, and width.
- `no_random_restart`: whether to re-initialize the codebook tokens.

### VQGAN
```
python train_vqgan.py --embedding_dim 256 --n_codes 16384 --n_hiddens 32 --downsample 4 8 8 --no_random_restart \
                      --gpus 8 --sync_batchnorm --batch_size 2 --num_workers 32 --accumulate_grad_batches 6 \
                      --progress_bar_refresh_rate 500 --max_steps 2000000 --gradient_clip_val 1.0 --lr 3e-5 \
                      --data_path {DATAPATH} --default_root_dir {CKPTPATH} \
                      --resolution 128 --sequence_length 16 --discriminator_iter_start 10000 --norm_type batch \
                      --perceptual_weight 4 --image_gan_weight 1 --video_gan_weight 1  --gan_feat_weight 4
```

### Transformer

#### TATS-base Transformer

```
python train_transformer.py --num_workers 32 --val_check_interval 0.5 --progress_bar_refresh_rate 500 \
                        --gpus 8 --sync_batchnorm --batch_size 3 --unconditional \
                        --vqvae {VQGAN-CKPT} --data_path {DATAPATH} --default_root_dir {CKPTPATH} \
                        --vocab_size 16384 --block_size 1024 --n_layer 24 --n_head 16 --n_embd 1024  \
                        --resolution 128 --sequence_length 16 --max_steps 2000000
```

To train a conditional transformer, remove the `--unconditional` flag and use the following flags

- `cond_stage_key`: what kind of conditional information to be used. It can be `label`, `text`, or `stft`.
- `stft_vqvae`: path to the trained VQGAN checkpoint for STFT features.
- `text_cond`: use this flag to indicate BPE encoded text.


#### TATS-hierarchical Transformer
```
python train_transformer.py --num_workers 32 --val_check_interval 0.5 --progress_bar_refresh_rate 500 \
                        --gpus 8 --sync_batchnorm --batch_size 3 --unconditional \
                        --vqvae {VQGAN-CKPT} --data_path {DATAPATH} --default_root_dir {CKPTPATH} \
                        --vocab_size 16384 --block_size 1280 --n_layer 24 --n_head 16 --n_embd 1024  \
                        --resolution 128 --sequence_length 20 --spatial_length 128 --n_unmasked 256 --max_steps 2000000

python train_transformer.py --num_workers 32 --val_check_interval 0.5 --progress_bar_refresh_rate 500 \
                        --gpus 8 --sync_batchnorm --batch_size 4 --unconditional \
                        --vqvae {VQGAN-CKPT} --data_path {DATAPATH} --default_root_dir {CKPTPATH} \
                        --vocab_size 16384 --block_size 1024 --n_layer 24 --n_head 16 --n_embd 1024  \
                        --resolution 128 --sequence_length 64 --sample_every_n_latent_frames 4 --spatial_length 128 --max_steps 2000000
```


## Acknowledgments
Our code is partially built upon [VQGAN](https://github.com/CompVis/taming-transformers) and
[VideoGPT](https://github.com/wilson1yan/VideoGPT).


## Citation
```
@article{ge2022long,
  title={Long video generation with time-agnostic vqgan and time-sensitive transformer},
  author={Ge, Songwei and Hayes, Thomas and Yang, Harry and Yin, Xi and Pang, Guan and Jacobs, David and Huang, Jia-Bin and Parikh, Devi},
  journal={arXiv preprint arXiv:2204.03638},
  year={2022}
}
```

## License

TATS is licensed under the MIT license, as found in the LICENSE file.
