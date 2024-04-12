"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

modified by Yihao Zhao
"""
from __future__ import print_function

import time

from utils import get_config, get_data_loader_folder, pytorch03_to_pytorch04, load_inception
from trainer import aclgan_Trainer
from torch import nn
from scipy.stats import entropy
import torch.nn.functional as F
import argparse
from torch.autograd import Variable
from data import ImageFolder
import numpy as np
import torchvision.utils as vutils

try:
    from itertools import izip as zip
except ImportError:  # will be 3.x series
    pass
import sys
import torch
import os

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/edges2handbags_folder', help='Path to the config file.')
parser.add_argument('--input_folder_A', type=str, help="input image folder A ")
parser.add_argument('--input_folder_B', type=str, help="input image folder B ")
parser.add_argument('--output_folder', type=str, help="output image folder")
parser.add_argument('--checkpoint', type=str, help="checkpoint of autoencoders")
parser.add_argument('--a2b', type=int, help="1 for a2b and 0 for b2a", default=1)
parser.add_argument('--seed', type=int, default=1, help="random seed")
parser.add_argument('--num_style', type=int, default=10, help="number of styles to sample")
parser.add_argument('--synchronized', action='store_true', help="whether use synchronized style code or not")
parser.add_argument('--output_only', action='store_true',
                    help="whether only save the output images or also save the input images")
parser.add_argument('--output_path', type=str, default='.', help="path for logs, checkpoints, and VGG model weight")
parser.add_argument('--trainer', type=str, default='aclgan', help="aclgan")
parser.add_argument('--compute_IS', action='store_true', help="whether to compute Inception Score or not")
parser.add_argument('--compute_CIS', action='store_true', help="whether to compute Conditional Inception Score or not")
parser.add_argument('--inception_a', type=str, default='.',
                    help="path to the pretrained inception network for domain A")
parser.add_argument('--inception_b', type=str, default='.',
                    help="path to the pretrained inception network for domain B")

opts = parser.parse_args()

torch.manual_seed(opts.seed)
torch.cuda.manual_seed(opts.seed)

# Load experiment setting
config = get_config(opts.config)
input_dim = config['input_dim_a'] if opts.a2b else config['input_dim_b']

# Load the inception networks if we need to compute IS or CIIS
if opts.compute_IS or opts.compute_IS:
    inception = load_inception(opts.inception_b) if opts.a2b else load_inception(opts.inception_a)
    # freeze the inception models and set eval mode
    inception.eval()
    for param in inception.parameters():
        param.requires_grad = False
    inception_up = nn.Upsample(size=(299, 299), mode='bilinear')

# Setup model and data loader
# A
image_names = ImageFolder(opts.input_folder_A, transform=None, return_paths=True)
data_loader = get_data_loader_folder(opts.input_folder_A, 1, False, new_size=config['new_size'], crop=False)

# B
image_names_B = ImageFolder(opts.input_folder_B, transform=None, return_paths=True)
data_loader_B = get_data_loader_folder(opts.input_folder_B, 1, False, new_size=config['new_size'], crop=False)

config['vgg_model_path'] = opts.output_path
if opts.trainer == 'aclgan':
    style_dim = config['gen']['style_dim']
    trainer = aclgan_Trainer(config)
else:
    sys.exit("Only support aclgan")


def focus_translation(x_fg, x_bg, x_focus):
    x_map = (x_focus + 1) / 2
    x_map = x_map.repeat(1, 3, 1, 1)
    return (torch.mul((x_fg + 1) / 2, x_map) + torch.mul((x_bg + 1) / 2, 1 - x_map)) * 2 - 1


if opts.trainer == 'aclgan':
    try:
        state_dict = torch.load(opts.checkpoint)
        trainer.gen_AB.load_state_dict(state_dict['AB'])
        trainer.gen_BA.load_state_dict(state_dict['BA'])
    except:
        state_dict = pytorch03_to_pytorch04(torch.load(opts.checkpoint), opts.trainer)
        trainer.gen_AB.load_state_dict(state_dict['AB'])
        trainer.gen_BA.load_state_dict(state_dict['BA'])

    trainer.cuda()
    trainer.eval()
    Gab = trainer.gen_AB.encode if opts.a2b else trainer.gen_BA.encode  # encode function
    Dab = trainer.gen_AB.decode if opts.a2b else trainer.gen_BA.decode  # decode functions
    Gba = trainer.gen_BA.encode if opts.a2b else trainer.gen_BA.encode  # encode function
    Dba = trainer.gen_BA.decode if opts.a2b else trainer.gen_BA.decode  # decode functions

if opts.compute_IS:
    IS = []
    all_preds = []
if opts.compute_CIS:
    CIS = []

# save real B
for i, (imagesB, namesB) in enumerate(zip(data_loader_B, image_names_B)):
    basename = os.path.basename(namesB[1])
    image = imagesB
    path = os.path.join(opts.output_folder, "realB", basename)
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    vutils.save_image(image.data, path, padding=0, normalize=True)

if opts.trainer == 'aclgan':
    count_max = 0
    style_fixed = Variable(torch.randn(opts.num_style * 3, style_dim, 1, 1).cuda(), volatile=True)
    for i, (images, names) in enumerate(zip(data_loader, image_names)):
        if i >= 5000:
            break
        if opts.compute_CIS:
            cur_preds = []
        print(names[1])
        images = Variable(images.cuda(), volatile=True)
        start_time = time.time()
        content, _ = Gab(images)
        content_til, _ = Gba(images)
        style = style_fixed * 2 if opts.synchronized else Variable(
            torch.randn(opts.num_style * 3, style_dim, 1, 1).cuda(), volatile=True) * 2
        for j in range(opts.num_style):
            s = style[j * 3].unsqueeze(0)
            outputs = Dab(content, s)
            if config['focus_loss'] > 0:
                img, mask = outputs.split(3, 1)
                outputs = focus_translation(img, images, mask)
                outputs_mask = mask.expand(-1, 3, -1, -1)
            content_hat, _ = Gba(outputs)
            s2 = style[j * 3 + 1].unsqueeze(0)
            outputs_hat = Dba(content_hat, s2)
            if config['focus_loss'] > 0:
                img, mask = outputs_hat.split(3, 1)
                outputs_hat = focus_translation(img, outputs, mask)
            s3 = style[j * 3 + 2].unsqueeze(0)
            outputs_til = Dba(content_til, s3)
            if config['focus_loss'] > 0:
                img, mask = outputs_til.split(3, 1)
                outputs_til = focus_translation(img, images, mask)
            cnt = torch.mean(outputs_til - images)
            outputs = (outputs + 1) / 2.
            outputs_hat = (outputs_hat + 1) / 2.
            outputs_til = (outputs_til + 1) / 2.

            basename = os.path.basename(names[1])
            path_bar = os.path.join(opts.output_folder + "/_%02d_fakeB" % j, basename)

            if not os.path.exists(os.path.dirname(path_bar)):
                os.makedirs(os.path.dirname(path_bar))

            vutils.save_image(outputs.data, path_bar, padding=0, normalize=True)
    print(count_max)
else:
    pass
