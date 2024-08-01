import os
import sys

# add dir
dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name,'../dataset/'))
sys.path.append(os.path.join(dir_name,'..'))
print(sys.path)
print(dir_name)

import argparse
import options
######### parser ###########
opt = options.Options().init(argparse.ArgumentParser(description='Image denoising')).parse_args()
print(opt)

import utils
from dataset.dataset_denoise import *
######### Set GPUs ###########
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
import torch
torch.backends.cudnn.benchmark = True

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import utils as vutils
from natsort import natsorted
import glob
import random
import time
import numpy as np
from einops import rearrange, repeat
import datetime

from losses import CharbonnierLoss

from tqdm import tqdm 
from warmup_scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import StepLR
from timm.utils import NativeScaler
# from utils.loader import  get_training_data,get_validation_data
from pdb import set_trace
from dcgan import Discriminator
from thop import profile


# Hinge Loss: maximum dis_real, minimum dis_fake
def loss_hinge_dis(dis_fake, dis_real):
  loss_real = torch.mean(F.relu(1. - dis_real))
  loss_fake = torch.mean(F.relu(1. + dis_fake))
  return loss_real, loss_fake

# maximum dis_fake
def loss_hinge_gen(dis_fake):
  loss = -torch.mean(dis_fake)
  return loss

######### Logs dir ########### ./logs/denoising/car/Uformer_B_0516
log_dir = os.path.join(opt.save_dir, 'denoising', opt.dataset, opt.arch+opt.env)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
logname = os.path.join(log_dir, datetime.datetime.now().isoformat()+'.txt') 
print("Now time is : ",datetime.datetime.now().isoformat())
result_dir = os.path.join(log_dir, 'results')
model_dir  = os.path.join(log_dir, 'models')
utils.mkdir(result_dir)
utils.mkdir(model_dir)

# ######### Set Seeds ###########
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

######### Model ###########
model_restoration = utils.get_arch(opt)
inputx = torch.randn(1, 3, 384, 512)
# flops, params = profile(model_restoration.cuda(), (inputx.cuda(),))
# print("model_restoration Gflops: %.3f G %.3d M" % (flops / 2**30, params / 2**20))
# Unet: 43.2G UformerB: 239.6G

with open(logname,'a') as f:
    f.write(str(opt)+'\n')
    f.write(str(model_restoration)+'\n')

######### Optimizer ###########
start_epoch = 1
if opt.optimizer.lower() == 'adam':
    optimizer = optim.Adam(model_restoration.parameters(), lr=opt.lr_initial, betas=(0.9, 0.999),eps=1e-8, weight_decay=opt.weight_decay)
elif opt.optimizer.lower() == 'adamw': # True
        optimizer = optim.AdamW(model_restoration.parameters(), lr=opt.lr_initial, betas=(0.9, 0.999),eps=1e-8, weight_decay=opt.weight_decay)
else:
    raise Exception("Error optimizer...")


######### DataParallel ########### 
model_restoration = torch.nn.DataParallel(model_restoration) 
model_restoration.cuda()

config = {'model': 'BigGAN', 
          'D_param': 'SN',
          'D_fp16': True,
          'D_ch': 32}   # 64 input3x384x512, flops=35G, 32=4.45G
model_discriminator = Discriminator(**config)
model_discriminator.cuda()
lr = 2e-4
B1, B2 =0.0, 0.999
adam_eps = 1e-8
optimD = optim.AdamW(params=model_discriminator.parameters(), lr=lr, 
                            betas=(B1, B2), weight_decay=0, eps=adam_eps)
        
# flops, params = profile(model_discriminator.cuda(), (inputx.cuda(),))
# print("model_discriminator Gflops: %.3fG %.3dM" % (flops / 2**30, params / 2**20))

######### Scheduler ###########
if opt.warmup:  # True
    print("Using warmup and cosine strategy!")
    warmup_epochs = opt.warmup_epochs   # 3
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.nepoch-warmup_epochs, eta_min=1e-6)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
    scheduler.step()
else:
    step = 50
    print("Using StepLR,step={}!".format(step))
    scheduler = StepLR(optimizer, step_size=step, gamma=0.5)
    scheduler.step()

######### Resume ########### 
if opt.resume: # False
    path_chk_rest = opt.pretrain_weights # ./log/Uformer_B/models/model_best.pth
    print("Resume from "+path_chk_rest)
    utils.load_checkpoint(model_restoration,path_chk_rest) 
    start_epoch = utils.load_start_epoch(path_chk_rest) + 1 
    lr = utils.load_optim(optimizer, path_chk_rest) 

    # for p in optimizer.param_groups: p['lr'] = lr 
    # warmup = False 
    # new_lr = lr 
    # print('------------------------------------------------------------------------------') 
    # print("==> Resuming Training with learning rate:",new_lr) 
    # print('------------------------------------------------------------------------------') 
    for i in range(1, start_epoch):
        scheduler.step()
    new_lr = scheduler.get_lr()[0]
    print('------------------------------------------------------------------------------')
    print("==> Resuming Training with learning rate:", new_lr)
    print('------------------------------------------------------------------------------')

    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.nepoch-start_epoch+1, eta_min=1e-6) 

######### Loss ###########
criterion = CharbonnierLoss().cuda()

######### DataLoader ###########
print('===> Loading datasets')
img_options_train = {'patch_size':opt.train_ps} # 128
train_dataset = get_training_data(opt.train_dir, img_options_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True, 
        num_workers=opt.train_workers, pin_memory=False, drop_last=False)
        # num_workers=0, pin_memory=False, drop_last=False)
val_dataset = get_validation_data(opt.val_dir)
val_loader = DataLoader(dataset=val_dataset, batch_size=opt.batch_size, shuffle=False, 
        num_workers=opt.eval_workers, pin_memory=False, drop_last=False)
        # num_workers=0, pin_memory=False, drop_last=False)

len_trainset = train_dataset.__len__()  # 30
len_valset = val_dataset.__len__()
print("Sizeof training set: ", len_trainset,", sizeof validation set: ", len_valset)
######### validation ###########
with torch.no_grad():
    model_restoration.eval()
    psnr_dataset = []
    psnr_model_init = []
    for ii, data_val in enumerate((val_loader), 0):
        target = data_val[0].cuda()
        input_ = data_val[1].cuda()
        with torch.cuda.amp.autocast():
            restored = model_restoration(input_)
            restored = torch.clamp(restored,0,1)  
        psnr_dataset.append(utils.batch_PSNR(input_, target, False).item())
        psnr_model_init.append(utils.batch_PSNR(restored, target, False).item())
    psnr_dataset = sum(psnr_dataset)/len_valset
    psnr_model_init = sum(psnr_model_init)/len_valset
    print('Input & GT (PSNR) -->%.4f dB'%(psnr_dataset), ', Model_init & GT (PSNR) -->%.4f dB'%(psnr_model_init))

######### train ###########
print('===> Start Epoch {} End Epoch {}'.format(start_epoch,opt.nepoch))
best_psnr = 0
best_epoch = 0
best_iter = 0
eval_now = len(train_loader)//4
print("\nEvaluation after every {} Iterations !!!\n".format(eval_now))

loss_scaler = NativeScaler()
torch.cuda.empty_cache()
model_restoration.train()
model_discriminator.train()
for epoch in range(start_epoch, opt.nepoch + 1):
    epoch_start_time = time.time()
    epoch_loss, gan_loss, d_real_loss, d_fake_loss = 0, 0, 0, 0
    train_id = 1

    for i, data in enumerate(tqdm(train_loader), 0): 
        # zero_grad
        optimizer.zero_grad()
        optimD.zero_grad()

        target = data[0].cuda()
        input_ = data[1].cuda()

        with torch.cuda.amp.autocast():
            restored = model_restoration(input_)

            D_fake = model_discriminator(restored)
            g_loss = loss_hinge_gen(D_fake) / len(target)

            crit_loss = criterion(restored, target)
            loss = crit_loss + g_loss * 0.5 * (epoch > start_epoch+1)

        loss_scaler(
                loss, optimizer,parameters=model_restoration.parameters())
                        
        with torch.cuda.amp.autocast():
            real_fake = torch.cat([target, restored.detach()], dim=0)
            D_out = model_discriminator(real_fake)
            D_real, D_fake = torch.split(D_out, [target.shape[0], restored.shape[0]])            
            
            D_loss_real, D_loss_fake = loss_hinge_dis(D_fake, D_real)
            D_loss = (D_loss_real + D_loss_fake) / len(target)

        loss_scaler(
            D_loss, optimD, parameters=model_discriminator.parameters())

        epoch_loss += crit_loss.item()
        gan_loss += g_loss.item()
        d_real_loss += D_loss_real.item()
        d_fake_loss += D_loss_fake.item()
        
        #### Evaluation ####
        if (i+1)%eval_now==0 and i>0:
            with torch.no_grad():
                model_restoration.eval()
                psnr_val_rgb = []
                for ii, data_val in enumerate((val_loader), 0):
                    target = data_val[0].cuda()
                    input_ = data_val[1].cuda()
                    filenames = data_val[2]
                    with torch.cuda.amp.autocast():
                        restored = model_restoration(input_)
                    restored = torch.clamp(restored,0,1)  
                    psnr_val_rgb.append(utils.batch_PSNR(restored, target, False).item())  
                    cat = torch.cat([input_, target, restored], dim=0)
                    filename = os.path.join(result_dir, '%06d.png' % ii)
                    try:
                        vutils.save_image(cat, filename, nrow=len(target))
                    except:
                        print("save image failed")
                        continue

                psnr_val_rgb = sum(psnr_val_rgb)/len_valset
                
                if psnr_val_rgb > best_psnr:
                    best_psnr = psnr_val_rgb
                    best_epoch = epoch
                    best_iter = i 
                    torch.save({'epoch': epoch, 
                                'state_dict': model_restoration.state_dict(),
                                'optimizer' : optimizer.state_dict()
                                }, os.path.join(model_dir,"model_best.pth"))

                print("[Ep %d it %d\t PSNR SIDD: %.4f\t] ----  [best_Ep_SIDD %d best_it_SIDD %d Best_PSNR_SIDD %.4f] " % (epoch, i, psnr_val_rgb,best_epoch,best_iter,best_psnr))
                with open(logname,'a') as f:
                    f.write("[Ep %d it %d\t PSNR SIDD: %.4f\t] ----  [best_Ep_SIDD %d best_it_SIDD %d Best_PSNR_SIDD %.4f] " \
                        % (epoch, i, psnr_val_rgb,best_epoch,best_iter,best_psnr)+'\n')
                model_restoration.train()
                torch.cuda.empty_cache()
    scheduler.step()
    gan_loss /= len(train_loader)
    d_real_loss /= len(train_loader)
    d_fake_loss /= len(train_loader)
    
    print("------------------------------------------------------------------")
    print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tGLoss: {:.4f}\tRLoss: {:.4f}\tFLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time()-epoch_start_time,epoch_loss, gan_loss, d_real_loss, d_fake_loss, scheduler.get_lr()[0]))
    print("------------------------------------------------------------------")
    with open(logname,'a') as f:
        f.write("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tGLoss: {:.4f}\tRLoss: {:.4f}\tFLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time()-epoch_start_time,epoch_loss, gan_loss, d_real_loss, d_fake_loss, scheduler.get_lr()[0])+'\n')

    torch.save({'epoch': epoch, 
                'state_dict': model_restoration.state_dict(),
                'optimizer' : optimizer.state_dict()
                }, os.path.join(model_dir,"model_latest.pth"))   

    if epoch%opt.checkpoint == 0:
        torch.save({'epoch': epoch, 
                    'state_dict': model_restoration.state_dict(),
                    'optimizer' : optimizer.state_dict()
                    }, os.path.join(model_dir,"model_epoch_{}.pth".format(epoch))) 
print("Now time is : ",datetime.datetime.now().isoformat())
