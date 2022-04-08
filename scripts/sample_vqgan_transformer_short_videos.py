# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import os
import tqdm
import time
import torch
import argparse
import numpy as np
import pytorch_lightning as pl
from einops import repeat

from tats import VideoData, Net2NetTransformer, load_transformer, load_vqgan
from tats.utils import save_video_grid
from tats.data import preprocess
from tats.modules.gpt import sample_with_past
from tats.utils import shift_dim

parser = argparse.ArgumentParser()
parser = VideoData.add_data_specific_args(parser)
parser.add_argument('--gpt_ckpt', type=str, default='')
parser.add_argument('--vqgan_ckpt', type=str, default='')
parser.add_argument('--save', type=str, default='./results/tats')
parser.add_argument('--run', type=int, default=0)
parser.add_argument('--top_k', type=int, default=2048)
parser.add_argument('--top_p', type=float, default=0.92)
parser.add_argument('--n_sample', type=int, default=2048)
parser.add_argument('--dataset', type=str, default='ucf101', choices=['ucf101', 'sky', 'taichi'])
parser.add_argument('--class_cond', action='store_true')
parser.add_argument('--save_videos', action='store_true')
parser.add_argument('--compute_fvd', action='store_true')
args = parser.parse_args()

gpt = load_transformer(args.gpt_ckpt, vqgan_ckpt=args.vqgan_ckpt).cuda().eval()

@torch.no_grad()
def sample(model, batch_size, class_label, steps=256, temperature=None, top_k=None, callback=None,
                            verbose_time=False, top_p=None, latent_shape=(4, 16, 16), n_cond=0):
    log = dict()
    assert type(class_label) == int, f'expecting type int but type is {type(class_label)}'
    c_indices = repeat(torch.tensor([class_label]), '1 -> b 1', b=batch_size).to(model.device)  # class token
    t1 = time.time()
    index_sample = sample_with_past(c_indices, model.transformer, steps=steps,
                                    sample_logits=True, top_k=top_k, callback=callback,
                                    temperature=temperature, top_p=top_p)
    if verbose_time:
        sampling_time = time.time() - t1
        print(f"Full sampling takes about {sampling_time:.2f} seconds.")
    index = index_sample.reshape([batch_size, *latent_shape])
    index = torch.clamp(index-n_cond, min=0, max=model.first_stage_model.n_codes-1)
    x_sample = model.first_stage_model.decode(index)
    log["samples"] = torch.clamp(x_sample, -0.5, 0.5) + 0.5
    log["class_label"] = c_indices
    return log

if args.top_k:
    save_dir = '%s/videos/%s/topp%.2f_topk%d_run%d'%(args.save, args.dataset, args.top_p, args.top_k, args.run)
    save_np = '%s/numpy_files/%s/topp%.2f_topk%d_run%d_eval.npy'%(args.save, args.dataset, args.top_p, args.top_k, args.run)
else:
    save_dir = '%s/videos/%s/toppNA_topkNA_run%d'%(args.save, args.dataset, args.run)
    save_np = '%s/numpy_files/%s/toppNA_topkNA_run%d_eval.npy'%(args.save, args.dataset, args.run)

print('generating and saving video to %s...'%save_dir)
os.makedirs(save_dir, exist_ok=True)

steps = np.prod(gpt.first_stage_model.latent_shape)
all_data = []
n_row = int(np.sqrt(args.batch_size))

if args.class_cond:
    print('number of classes: %d'%gpt.class_cond_dim)
    n_sample_class = args.n_sample//gpt.class_cond_dim+1
    n_batch_class = n_sample_class//args.batch_size+1
    with torch.no_grad():
        for class_label in range(gpt.class_cond_dim):
            for sample_id in range(n_batch_class):
                class_label = np.random.randint(gpt.class_cond_dim) if not args.compute_fvd else class_label
                logs = sample(gpt, batch_size=args.batch_size, class_label=class_label, steps=steps, n_cond=gpt.class_cond_dim, 
                              temperature=1., top_k=args.top_k, top_p=args.top_p, verbose_time=False, latent_shape=gpt.first_stage_model.latent_shape)
                if args.save_videos:
                    save_video_grid(logs['samples'], os.path.join(save_dir, 'generation_%d_%d.avi'%(class_label, sample_id)), n_row)
                all_data.append(logs['samples'].cpu().data.numpy()) # 256*4 x 8 x 3 x 16 x 128 x 128
else:
    n_batch = args.n_sample//args.batch_size+1
    with torch.no_grad():
        for sample_id in range(n_batch):
            logs = sample(gpt, batch_size=args.batch_size, class_label=0, steps=steps, 
                          temperature=1., top_k=args.top_k, top_p=args.top_p, verbose_time=False, latent_shape=gpt.first_stage_model.latent_shape)
            if args.save_videos:
                save_video_grid(logs['samples'], os.path.join(save_dir, 'generation_%d.avi'%(sample_id)), n_row)
            all_data.append(logs['samples'].cpu().data.numpy()) # 256*4 x 8 x 3 x 16 x 128 x 128



print('saving numpy file to %s...'%save_np)
os.makedirs(os.path.dirname(save_np), exist_ok=True)
all_data_np = np.array(all_data)
all_data_np = np.transpose(all_data_np.reshape(-1, 3, 16, 128, 128), (0, 2, 3, 4, 1))
n_total = all_data_np.shape[0]
all_data_np = (all_data_np*255).astype(np.uint8)[np.random.permutation(n_total)[:args.n_sample]]
np.save(save_np, all_data_np)


if args.compute_fvd:
    from tats.fvd.fvd import FVD_SAMPLE_SIZE, MAX_BATCH, get_fvd_logits, frechet_distance, \
        load_fvd_model, preprocess, TARGET_RESOLUTION, polynomial_mmd
    device = torch.device('cuda')
    i3d = load_fvd_model(device)
    data = VideoData(args)
    loader = data.train_dataloader()
    real_embeddings = []
    print('computing fvd embeddings for real videos')
    for batch in loader:
        real_embeddings.append(get_fvd_logits(shift_dim((batch['video']+0.5)*255, 1, -1).byte().data.numpy(), i3d=i3d, device=device))
        if len(real_embeddings)*args.batch_size >=2048: break
    print('caoncat fvd embeddings for real videos')
    real_embeddings = torch.cat(real_embeddings, 0)[:2048]
    print('computing fvd embeddings for fake videos')
    fake_embeddings = []
    n_batch = all_data_np.shape[0]//args.batch_size
    for i in range(n_batch):
        fake_embeddings.append(get_fvd_logits(all_data_np[i*args.batch_size:(i+1)*args.batch_size], i3d=i3d, device=device))
    print('caoncat fvd embeddings for fake videos')
    fake_embeddings = torch.cat(fake_embeddings, 0)[:2048]
    print('FVD = %.2f'%(frechet_distance(fake_embeddings, real_embeddings)))
    print('KVD = %.2f'%(polynomial_mmd(fake_embeddings.cpu(), real_embeddings.cpu())))
