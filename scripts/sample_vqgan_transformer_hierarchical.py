import os
import tqdm
import time
import torch
import argparse
import numpy as np
import pytorch_lightning as pl
from einops import repeat

from tats import load_transformer, load_vqgan
from tats.utils import save_video_grid
from tats.modules.gpt import sample_with_past, sample_with_past_and_future

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt1', type=str, default='')
parser.add_argument('--ckpt2', type=str, default='')
parser.add_argument('--save', type=str, default='./results')
parser.add_argument('--vqgan', type=str, default='')
parser.add_argument('--batch_size', type=int, default=12)
parser.add_argument('--top_k_init', type=int, default=2048)
parser.add_argument('--top_p_init', type=float, default=0.8)
parser.add_argument('--top_k', type=int, default=2048)
parser.add_argument('--top_p', type=float, default=0.8)
parser.add_argument('--n_sample', type=int, default=2048)
parser.add_argument('--sample_length', type=int, default=64)
parser.add_argument('--sample_resolution', type=int, default=16)
parser.add_argument('--temporal_sample_pos', type=int, default=1)
parser.add_argument('--run', type=int, default=0)
parser.add_argument('--dataset', type=str, default='taichi', choices=['sky', 'taichi'])
parser.add_argument('--class_cond', action='store_true')
parser.add_argument('--compute_fvd', action='store_true')
parser.add_argument('--save_videos', action='store_true')
args = parser.parse_args()

gpt1 = load_transformer(args.ckpt1, vqgan_ckpt=args.vqgan).cuda().eval()
gpt2 = load_transformer(args.ckpt2, vqgan_ckpt=args.vqgan).cuda().eval()
vqgan = load_vqgan(args.vqgan).cuda().eval()

if args.top_k:
    save_dir = '%s/videos/%s_hierarchical_video/%s_%s_topp%.2f_topk%d_%d'%(args.save, args.dataset, args.ckpt1.split('/')[7], args.ckpt1.split('/')[8], args.top_p, args.top_k, args.sample_length)
    save_np = '%s/numpy_files/%s_hierarchical_video/%s_%s_topp%.2f_topk%d_run%d_eval.npy'%(args.save, args.dataset, args.ckpt1.split('/')[7], args.ckpt1.split('/')[8], args.top_p, args.top_k, args.run)
else:
    save_dir = '%s/videos/%s_hierarchical_video/%s_%s_%s_%s_toppNA_topkNA_%d'%(args.save, args.dataset, args.ckpt1.split('/')[7], args.ckpt1.split('/')[8], args.ckpt2.split('/')[7], args.ckpt2.split('/')[8], args.sample_length)
    save_np = '%s/numpy_files/%s_hierarchical_video/%s_%s_%s_%s_toppNA_topkNA_run%d_eval.npy'%(args.save, args.dataset, args.ckpt1.split('/')[7], args.ckpt1.split('/')[8], args.ckpt2.split('/')[7], args.ckpt2.split('/')[8], args.run)


args.save_videos = True
os.makedirs(save_dir, exist_ok=True)
if args.save_videos:
    print('generating and saving video to %s...'%save_dir)
    os.makedirs(save_dir, exist_ok=True)

assert gpt1.args.spatial_length == gpt2.args.spatial_length
total_temporal_length = args.sample_length + 1
all_data = []
n_row = int(np.sqrt(args.batch_size))

@torch.no_grad()
def sample_long_fast(gpt, temporal_infer, spatial_infer, temporal_sample_pos, batch_size, class_label, args, temperature=1.):
    steps = slice_n_code = spatial_infer**2
    with torch.no_grad():
        index_sample_all = torch.zeros([batch_size, temporal_infer*spatial_infer*spatial_infer]).long().cuda()
        c_indices = repeat(torch.tensor([class_label]), '1 -> b 1', b=batch_size).to(gpt.device)  # class token
        index_sample_all[:,:temporal_sample_pos*steps] = sample_with_past(c_indices, gpt.transformer, steps=temporal_sample_pos*steps,
                                        sample_logits=True, top_k=args.top_k_init, temperature=temperature, top_p=args.top_p_init)
        for t_id in range(temporal_infer-temporal_sample_pos):
            i_start = t_id*slice_n_code
            i_end = (temporal_sample_pos+t_id)*slice_n_code
            x_past = index_sample_all[:,i_start:i_end]
            index_sample_all[:,i_end:i_end+steps] = sample_with_past(torch.cat([c_indices, x_past], dim=1), gpt.transformer, steps=steps,
                                            sample_logits=True, top_k=args.top_k, temperature=temperature, top_p=args.top_p)
        torch.cuda.empty_cache()
        index_sample_all = index_sample_all.reshape([batch_size, temporal_infer, spatial_infer, spatial_infer])
    return index_sample_all


@torch.no_grad()
def sample_hierarchical(gpt1, gpt2, batch_size, class_label, args, verbose_time=True):
    steps = slice_n_code = args.sample_resolution**2
    total_temporal_length = args.sample_length + 1
    # hard code to assume that the gpt is trained to fill 3 frames in the middle
    temporal_length2 = 5
    total_temporal_length_AR = args.sample_length // 4 + 1
    with torch.no_grad():
        log = dict()
        c_indices = repeat(torch.tensor([class_label]), '1 -> b 1', b=batch_size).to(gpt1.device)  # class token
        t1 = time.time()
        index_lv1 = sample_long_fast(gpt1, total_temporal_length_AR, args.sample_resolution, 
                                     temporal_sample_pos=args.temporal_sample_pos, batch_size=batch_size, 
                                     class_label=class_label, args=args)
        index_sample_lv2 = torch.zeros([batch_size, total_temporal_length*slice_n_code]).long().cuda()
        if verbose_time:
            sampling_time = time.time() - t1
            t1 = time.time()
            print(f"Full sampling lv1 takes about {sampling_time:.2f} seconds.")
        for i in range(total_temporal_length_AR):
            index_sample_lv2[:, slice_n_code*i*(temporal_length2-1):slice_n_code*(i*(temporal_length2-1)+1)] = index_lv1.reshape(batch_size, -1)[:, slice_n_code*i:slice_n_code*(i+1)].clone()
        for t_id in tqdm.tqdm(range(total_temporal_length_AR-1)):
            i_start = t_id*(temporal_length2-1)*slice_n_code
            i_end = (1+t_id)*(temporal_length2-1)*slice_n_code+slice_n_code
            x_past = index_sample_lv2[:,i_start:i_start+slice_n_code]
            x_future = index_sample_lv2[:,i_end-slice_n_code:i_end]
            index_lv2 = sample_with_past_and_future(torch.cat([c_indices, x_past], dim=1), 
                                    x_future, gpt2.transformer, steps=steps*3,
                                    sample_logits=True, top_k=args.top_k, 
                                    temperature=1., top_p=args.top_p)
            index_sample_lv2[:,i_start+slice_n_code:i_end-slice_n_code] = index_lv2
        index_lv2 = torch.clamp(index_sample_lv2.reshape([batch_size, total_temporal_length, args.sample_resolution, args.sample_resolution])-gpt1.cond_stage_vocab_size, min=0, max=vqgan.n_codes-1)
        x_sample_lv2 = []
        for i in range(batch_size):
            x_sample_lv2.append(vqgan.decode(index_lv2[i:i+1]))
        x_sample_lv2 = torch.cat(x_sample_lv2, 0)
        if verbose_time:
            sampling_time = time.time() - t1
            print(f"Full sampling lv2 takes about {sampling_time:.2f} seconds.")
        log["samples_lv2"] = torch.clamp(x_sample_lv2, -0.5, 0.5) + 0.5
    return log


args.batch_size = 4
all_data = []
if args.class_cond:
    print('number of classes: %d'%gpt1.class_cond_dim)
    n_batch = args.n_sample//args.batch_size+1
    with torch.no_grad():
        for sample_id in tqdm.tqdm(range(n_batch)):
            class_label = np.random.randint(101)
            logs = sample_hierarchical(gpt1, gpt2, batch_size=args.batch_size, class_label=class_label, args=args, verbose_time=True)
            if args.save_videos:
                save_video_grid(logs['samples_lv2'], os.path.join(save_dir, 'generation_%d.avi'%(sample_id+args.run*args.n_sample)), n_row)
            all_data.append(logs['samples_lv2'].cpu().data.numpy())
            if len(all_data) > n_batch: break
else:
    n_batch = args.n_sample//args.batch_size+1
    with torch.no_grad():
        for sample_id in tqdm.tqdm(range(4, n_batch)):
            logs = sample_hierarchical(gpt1, gpt2, batch_size=args.batch_size, class_label=0, args=args, verbose_time=True)
            if args.save_videos:
                save_video_grid(logs['samples_lv2'], os.path.join(save_dir, 'generation_%d.avi'%(sample_id+args.run*args.n_sample)), n_row)
            all_data.append(logs['samples_lv2'].cpu().data.numpy())
            if len(all_data) > n_batch: break
