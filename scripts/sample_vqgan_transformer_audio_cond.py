import os
import tqdm
import time
import torch
import argparse
import numpy as np

from tats import VideoData, load_transformer
from tats.utils import save_video_grid
from tats.modules.gpt import sample_with_past

from shutil import copyfile

parser = argparse.ArgumentParser()
parser.add_argument('--gpt_ckpt', type=str, default='')
parser.add_argument('--vqgan_ckpt', type=str, default='')
parser.add_argument('--stft_vqgan_ckpt', type=str, default='')
parser.add_argument('--save', type=str, default='./results/tats')
parser.add_argument('--data_path', type=str, default='./dataset/AudioSet_Dataset/stft')
parser.add_argument('--top_k', type=int, default=2048)
parser.add_argument('--top_p', type=float, default=0.92)
parser.add_argument('--n_sample', type=int, default=2048)
parser.add_argument('--dataset', type=str, default='drum', choices=['drum'])
args = parser.parse_args()

gpt = load_transformer(args.gpt_ckpt, args.vqgan_ckpt, args.stft_vqgan_ckpt).cuda().eval()

gpt.args.batch_size = 1
gpt.args.data_path = args.data_path
data = VideoData(gpt.args)
loader = data.test_dataloader()

@torch.no_grad()
def sample(model, batch_size, cond, steps=256, temperature=None, top_k=None, callback=None,
                            verbose_time=False, top_p=None, latent_shape=(4, 16, 16), n_cond=0):
    log = dict()
    t1 = time.time()
    index_sample = sample_with_past(cond, model.transformer, steps=steps,
                                    sample_logits=True, top_k=top_k, callback=callback,
                                    temperature=temperature, top_p=top_p)
    if verbose_time:
        sampling_time = time.time() - t1
        print(f"Full sampling takes about {sampling_time:.2f} seconds.")
    index = index_sample.reshape([batch_size, *latent_shape])
    # x_sample = torch.cat([model.first_stage_model.decode(index[:batch_size//2]-n_cond), model.first_stage_model.decode(index[batch_size//2:]-n_cond)])
    x_sample = model.first_stage_model.decode(index-n_cond)
    log["samples"] = torch.clamp(x_sample, -0.5, 0.5) + 0.5
    return log

save_dir = '%s/videos/%s/topp%.2f_topk%d'%(args.save, args.dataset, args.top_p, args.top_k)
print('generating and saving video to %s...'%save_dir)
os.makedirs(save_dir, exist_ok=True)


steps = np.prod(gpt.first_stage_model.latent_shape)
all_data = []

with torch.no_grad():
    for sample_id in tqdm.tqdm(range(min(args.n_sample, len(loader.dataset)))):
        batch = loader.dataset.__getitem__(sample_id)
        x, c = gpt.get_xc(batch)
        _, c_indices = gpt.encode_to_c(c.unsqueeze(0).cuda())
        logs = sample(gpt, batch_size=1, cond=c_indices, steps=steps, n_cond=gpt.cond_stage_vocab_size, 
                      temperature=1., top_k=args.top_k, top_p=args.top_p, verbose_time=True, latent_shape=gpt.first_stage_model.latent_shape)
        save_video_grid(logs['samples'], os.path.join(save_dir, 'generation_%d.avi'%(sample_id)), 1)
        save_video_grid(torch.clamp(x.unsqueeze(0), -0.5, 0.5) + 0.5, os.path.join(save_dir, 'groundtruth_%d.avi'%(sample_id)), 1)
        copyfile(batch['path'], os.path.join(save_dir, 'groundtruth_%s_%d.avi'%(os.path.basename(batch['path'])[:-len('.mp4')], sample_id)))
