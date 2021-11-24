
import os
import time
import torch
import datetime
import numpy as np

import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image
from torchvision import transforms
import glob
import cv2
import PIL
from unet import unet
from utils import *
from PIL import Image

def transformer(resize, totensor, normalize, centercrop, imsize):
    options = []
    if centercrop:
        options.append(transforms.CenterCrop(160))
    if resize:
        options.append(transforms.Resize((imsize,imsize), interpolation=PIL.Image.NEAREST))
    if totensor:
        options.append(transforms.ToTensor())
    if normalize:
        options.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    transform = transforms.Compose(options)
    
    return transform

def make_dataset(dir):
    images = []
    # assert os.path.isdir(dir), '%s is not a valid directory' % dir
    # images = glob.glob(os.path.join(dir, "*.jpg")) + glob.glob(os.path.join(dir, "*.png"))
    i = 0
    fin = open(dir, "r")
    for line in fin.readlines():
        line = line.strip().split()
        images.append(line[0])
        images.append(line[1])
        i+=1
    return images

class Tester(object):
    def __init__(self, config):
        # exact model and loss
        self.model = config.model

        # Model hyper-parameters
        self.imsize = config.imsize
        self.parallel = config.parallel

        self.total_step = config.total_step
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.g_lr = config.g_lr
        self.lr_decay = config.lr_decay
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.pretrained_model = config.pretrained_model

        self.img_path = config.img_path
        self.label_path = config.label_path 
        self.log_path = config.log_path
        self.model_save_path = config.model_save_path
        self.sample_path = config.sample_path
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.version = config.version

        # Path
        self.log_path = os.path.join(config.log_path, self.version)
        self.sample_path = os.path.join(config.sample_path, self.version)
        self.model_save_path = os.path.join(config.model_save_path, self.version)
        self.test_label_path = config.test_label_path
        self.test_color_label_path = config.test_color_label_path
        self.test_image_path = config.test_image_path

        # Test size and model
        self.test_size = config.test_size
        self.model_name = config.model_name

        self.build_model()

    def test(self):
        # transform = transformer(True, True, True, False, self.imsize) 
        transform = transformer(False, True, True, False, self.imsize) 
        test_paths = make_dataset(self.test_image_path)
        # make_folder(self.test_label_path, '')
        # make_folder(self.test_color_label_path, '') 
        self.G.load_state_dict(torch.load(os.path.join(self.model_save_path, self.model_name)))
        self.G.eval() 
        self.test_size = len(test_paths)
        batch_num = int((self.test_size-1) / self.batch_size) + 1
        for i in range(batch_num):
            print (i)
            cur_batch_size = min(self.batch_size, self.test_size - i*self.batch_size)
            imgs = []
            label_paths = []
            label_show_paths = []
            for j in range(cur_batch_size):
                path = test_paths[i * self.batch_size + j]
                img = transform(Image.open(path).convert('RGB'))

                fpath, fname = os.path.split(path)
                if "train_frame" in self.test_image_path:
                    label_show_root = fpath.replace("iphoneX_face", "FaceFlowData/train") + "/parsing_show/"
                elif "val_frame" in self.test_image_path:
                    label_show_root = fpath.replace("iphoneX_face", "FaceFlowData/val") + "/parsing_show/"
                elif "test_frame" in self.test_image_path:
                    label_show_root = fpath.replace("iphoneX_face", "FaceFlowData/test") + "/parsing_show/"

                if os.path.exists(label_show_root + fname.replace("color.jpg", "parsing_label.png").replace("color.png", "parsing_label.png")) and os.path.exists(label_show_root + fname.replace("color.jpg", "parsing.png").replace("color.png", "parsing.png")):
                    continue
                if not os.path.exists(label_show_root):
                    os.makedirs(label_show_root)

                imgs.append(img)
                label_paths.append(label_show_root + fname.replace("color.jpg", "parsing_label.png").replace("color.png", "parsing_label.png"))
                label_show_paths.append(label_show_root + fname.replace("color.jpg", "parsing.png").replace("color.png", "parsing.png"))
            if len(imgs)==0:
                continue
            cur_batch_size = len(imgs)
            imgs = torch.stack(imgs) 
            imgs = imgs.cuda()
            labels_predict = self.G(imgs)
            labels_predict_plain = generate_label_plain(labels_predict, self.imsize)
            labels_predict_color = generate_label(labels_predict, self.imsize)
            for k in range(cur_batch_size):
                cv2.imwrite(label_paths[k], labels_predict_plain[k])
                save_image(labels_predict_color[k], label_show_paths[k])



        # for i in range(batch_num):
        #     print (i)
        #     imgs = []
        #     name = []
        #     for j in range(self.batch_size):
        #         path = test_paths[i * self.batch_size + j]
        #         img = transform(Image.open(path).convert('RGB'))
        #         imgs.append(img)
        #         name.append(os.path.basename(path))
        #     imgs = torch.stack(imgs) 
        #     imgs = imgs.cuda()
        #     labels_predict = self.G(imgs)
        #     labels_predict_plain = generate_label_plain(labels_predict, self.imsize)
        #     labels_predict_color = generate_label(labels_predict, self.imsize)
        #     for k in range(self.batch_size):
        #         cv2.imwrite(os.path.join(self.test_label_path, name[k]), labels_predict_plain[k])
        #         save_image(labels_predict_color[k], os.path.join(self.test_color_label_path, name[k]))

    def build_model(self):
        self.G = unet().cuda()
        if self.parallel:
            self.G = nn.DataParallel(self.G)

        # print networks
        print(self.G)
