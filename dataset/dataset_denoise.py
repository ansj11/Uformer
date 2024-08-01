import numpy as np
import os
import cv2
from torch.utils.data import Dataset
import torch
from utils import is_png_file, load_img, Augment_RGB_torch
import torch.nn.functional as F
import random
from PIL import Image
import torchvision.transforms.functional as TF
from natsort import natsorted
from glob import glob
augment   = Augment_RGB_torch()
transforms_aug = [method for method in dir(augment) if callable(getattr(augment, method)) if not method.startswith('_')] 
from pdb import set_trace


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])
    
##################################################################################################
class DataLoaderTrain(Dataset):
    def __init__(self, rgb_dir, img_options=None, target_transform=None):
        super(DataLoaderTrain, self).__init__()

        self.target_transform = target_transform    # None
        with open(rgb_dir, 'r') as f:
            lines = f.readlines()

        self.noisy_filenames = [x.strip().split()[0] for x in lines]
        self.clean_filenames = [x.strip().split()[1] for x in lines]
        # gt_dir = os.path.join(rgb_dir.replace('outputs/', ''), 'images')
        # input_dir = os.path.join(rgb_dir, 'render')
        
        # noisy_files = sorted(os.listdir(input_dir))
        
        # self.clean_filenames = [os.path.join(gt_dir, x[:3]+'jpg') for x in noisy_files if is_image_file(x)]
        # self.noisy_filenames = [os.path.join(input_dir, x) for x in noisy_files if is_png_file(x)]
        
        self.img_options=img_options

        self.tar_size = len(self.clean_filenames)  # get the size of target
        print("loading test images: ", self.tar_size)

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index   = index % self.tar_size
        clean = np.float32(load_img(self.clean_filenames[tar_index]))
        noisy = np.float32(load_img(self.noisy_filenames[tar_index]))
        
        clean = cv2.resize(clean, (600, 400), cv2.INTER_AREA)
        noisy = cv2.resize(noisy, (600, 400), cv2.INTER_AREA)
        
        clean = torch.from_numpy(clean).permute(2,0,1)
        noisy = torch.from_numpy(noisy).permute(2,0,1)

        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1]
        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]

        #Crop Input and Target
        ps = [x // 2 for x in [768, 1024]] # self.img_options['patch_size']
        H = clean.shape[1]
        W = clean.shape[2]

        if H-ps[0]==0:
            r=0
            c=0
        else:
            r = np.random.randint(0, H - ps[0])
            c = np.random.randint(0, W - ps[1])
        clean = clean[:, r:r + ps[0], c:c + ps[1]]
        noisy = noisy[:, r:r + ps[0], c:c + ps[1]]

        apply_trans = transforms_aug[random.getrandbits(1)]

        clean = getattr(augment, apply_trans)(clean)
        noisy = getattr(augment, apply_trans)(noisy)

        return clean, noisy, clean_filename, noisy_filename


##################################################################################################
class DataLoaderVal(Dataset):
    def __init__(self, rgb_dir, target_transform=None):
        super(DataLoaderVal, self).__init__()

        self.target_transform = target_transform
        with open(rgb_dir, 'r') as f:
            lines = f.readlines()

        self.noisy_filenames = [x.strip().split()[0] for x in lines]
        self.clean_filenames = [x.strip().split()[1] for x in lines]
        # gt_dir = os.path.join(rgb_dir.replace('outputs/', ''), 'images')
        # input_dir = os.path.join(rgb_dir, 'render')
        
        # noisy_files = sorted(os.listdir(input_dir))
        
        # self.clean_filenames = [os.path.join(gt_dir, x[:3]+'jpg') for x in noisy_files if is_image_file(x)]
        # self.noisy_filenames = [os.path.join(input_dir, x) for x in noisy_files if is_image_file(x)]        

        self.tar_size = len(self.clean_filenames)  
        print("loading test images: ", self.tar_size)

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index   = index % self.tar_size
        # set_trace()
        clean = np.float32(load_img(self.clean_filenames[tar_index]))[16:-16, 88:-88]
        noisy = np.float32(load_img(self.noisy_filenames[tar_index]))[16:-16, 88:-88]

        clean = cv2.resize(clean, (512, 384), cv2.INTER_AREA)
        noisy = cv2.resize(noisy, (512, 384), cv2.INTER_AREA)

        clean = torch.from_numpy(clean)
        noisy = torch.from_numpy(noisy)
                
        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1]
        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]

        clean = clean.permute(2,0,1)
        noisy = noisy.permute(2,0,1)

        return clean, noisy, clean_filename, noisy_filename

##################################################################################################

class DataLoaderTest(Dataset):
    def __init__(self, inp_dir, img_options):
        super(DataLoaderTest, self).__init__()

        inp_files = sorted(os.listdir(inp_dir))
        self.inp_filenames = [os.path.join(inp_dir, x) for x in inp_files if is_image_file(x)]

        self.inp_size = len(self.inp_filenames)
        self.img_options = img_options

    def __len__(self):
        return self.inp_size

    def __getitem__(self, index):

        path_inp = self.inp_filenames[index]
        filename = os.path.splitext(os.path.split(path_inp)[-1])[0]
        inp = Image.open(path_inp)

        inp = TF.to_tensor(inp)
        return inp, filename


def get_training_data(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    return DataLoaderTrain(rgb_dir, img_options, None)


def get_validation_data(rgb_dir):
    assert os.path.exists(rgb_dir)
    return DataLoaderVal(rgb_dir, None)

def get_test_data(rgb_dir, img_options=None):
    assert os.path.exists(rgb_dir)
    return DataLoaderTest(rgb_dir, img_options)