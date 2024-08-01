import numpy as np
import os,sys
import argparse
from tqdm import tqdm
from einops import rearrange, repeat

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
# from ptflops import get_model_complexity_info
from torchvision import utils as vutils

dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name,'../dataset/'))
sys.path.append(os.path.join(dir_name,'..'))

import scipy.io as sio
from dataset.dataset_denoise import *
import utils
import math
from model import UNet,Uformer
from glob import glob

from skimage import img_as_float32, img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss
from pdb import set_trace

parser = argparse.ArgumentParser(description='Image denoising evaluation on SIDD')
parser.add_argument('--input_dir', default='./outputs/jietuV9/demo_frames/',
    type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./outputs/jietuV9/denoise/',
    type=str, help='Directory for results')
parser.add_argument('--weights', default='logs/denoising/car/Uformer_B_0531/models/model_best.pth',
    type=str, help='Path to weights')
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--arch', default='Uformer_B', type=str, help='arch')
parser.add_argument('--batch_size', default=1, type=int, help='Batch size for dataloader')
parser.add_argument('--save_images', action='store_true', help='Save denoised images in result directory')
parser.add_argument('--embed_dim', type=int, default=32, help='number of data loading workers')    
parser.add_argument('--win_size', type=int, default=8, help='number of data loading workers')
parser.add_argument('--token_projection', type=str,default='linear', help='linear/conv token projection')
parser.add_argument('--token_mlp', type=str,default='leff', help='ffn/leff token mlp')
parser.add_argument('--dd_in', type=int, default=3, help='dd_in')

# args for vit
parser.add_argument('--vit_dim', type=int, default=256, help='vit hidden_dim')
parser.add_argument('--vit_depth', type=int, default=12, help='vit depth')
parser.add_argument('--vit_nheads', type=int, default=8, help='vit hidden_dim')
parser.add_argument('--vit_mlp_dim', type=int, default=512, help='vit mlp_dim')
parser.add_argument('--vit_patch_size', type=int, default=16, help='vit patch_size')
parser.add_argument('--global_skip', action='store_true', default=False, help='global skip connection')
parser.add_argument('--local_skip', action='store_true', default=False, help='local skip connection')
parser.add_argument('--vit_share', action='store_true', default=False, help='share vit module')

parser.add_argument('--train_ps', type=int, default=128, help='patch size of training sample')
args = parser.parse_args()


# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("using device: ", device)

# if args.save_images:
utils.mkdir(args.result_dir)

# test_dataset = get_validation_data(args.input_dir)
# test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=False)

model_restoration= utils.get_arch(args)

utils.load_checkpoint(model_restoration,args.weights)
print("===>Testing using weights: ", args.weights)

model_restoration.to(device)
model_restoration.eval()

# Process data
filepaths = glob(os.path.join(args.input_dir, '*.png'))
filepaths += glob(os.path.join(args.input_dir, '*.jpg'))

print(len(filepaths))
for filepath in tqdm(filepaths):
    basename = os.path.basename(filepath)
    rgba = cv2.imread(filepath, -1)
    img = rgba
    if img.shape[-1] == 4:
        mask = img[...,-1:] / 255.
        img = (img[...,:-1] * mask).astype('uint8')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)/255.
    noisy = cv2.resize(img, (600, 400), cv2.INTER_AREA)
    ps = [x // 2 for x in [768, 1024]]
    H, W = noisy.shape[:2]
    top = (H - ps[0]) // 2
    left= (W - ps[1]) // 2
    crop = noisy[top:top+ps[0], left:left+ps[1]]
    input = torch.from_numpy(crop).permute(2,0,1).unsqueeze(0).to(device)
    top *= 2
    left *= 2
    with torch.no_grad():
        restored = model_restoration(input)
        input = restored.clone()

    restored = torch.clamp(restored, 0, 1)
    cat = torch.cat([input, restored], dim=0)
    # set_trace()
    save_file = os.path.join(args.result_dir, basename)
    # vutils.save_image(cat, save_file)
    restored = restored[0].permute(1,2,0).cpu().numpy()*255
    restored = cv2.resize(restored, (1024, 768), cv2.INTER_LINEAR)
    
    rgba = np.zeros((800, 1200, 4), dtype='uint8')
    rgba[top:top+768,left:left+1024,:3] = restored[...,::-1]
    rgba[top:top+768,left:left+1024,3] = (restored.max(axis=-1) > 1) * 255
    cv2.imwrite(save_file[:-4]+'.png', rgba.astype("uint8"))

    """
    for i in range(10):
        with torch.no_grad():
            restored = model_restoration(input)
            input = restored.clone()
    
        restored = torch.clamp(restored, 0, 1)
        cat = torch.cat([input, restored], dim=0)
        # set_trace()
        save_file = os.path.join(args.result_dir, basename)
        # vutils.save_image(cat, save_file)
        restored = restored[0].permute(1,2,0).cpu().numpy()*255
        restored = cv2.resize(restored, (1024, 768), cv2.INTER_LINEAR)
        
        rgba = np.zeros((800, 1200, 4), dtype='uint8')
        rgba[top:top+768,left:left+1024,:3] = restored[...,::-1]
        rgba[top:top+768,left:left+1024,3] = (restored.max(axis=-1) > 1) * 255
        cv2.imwrite(save_file[:-4]+'_%d.png' % i, rgba.astype("uint8"))
"""

# save denoised data
# sio.savemat(os.path.join(result_dir_mat, 'Idenoised.mat'), {"Idenoised": restored,})
