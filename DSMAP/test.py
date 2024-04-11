"""
Copyright (C) 2020 Hsin-Yu Chang <acht7111020@gmail.com>
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import argparse
import os
import time
import torch
import torchvision.utils as vutils

from utils import get_config, get_test_data_loaders
from trainer import DSMAP_Trainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help="Path to the config file")
    parser.add_argument('--test_path', type=str,
                        help="Path to the root folder of testing images (ex. ROOT/testA, testB)")
    parser.add_argument('--output_path', type=str, default='./tmp', help="Path to save results")
    parser.add_argument('--checkpoint', type=str, help="Path to load checkpoint")
    parser.add_argument('--a2b', type=int, default=1, help="1 for a2b and others for b2a")
    opts = parser.parse_args()

    if not os.path.exists(opts.output_path):
        os.makedirs(opts.output_path)

    # Load experiment setting
    config = get_config(opts.config)
    model = DSMAP_Trainer(config)

    state_dict = torch.load(opts.checkpoint)
    model.gen_a.load_state_dict(state_dict['a'])
    model.gen_b.load_state_dict(state_dict['b'])

    model.cuda()
    model.eval()
    encode = model.gen_a.encode if opts.a2b else model.gen_b.encode
    style_encode = model.gen_b.encode if opts.a2b else model.gen_a.encode
    decode = model.gen_b.decode if opts.a2b else model.gen_a.decode

    # load all images in test folders
    loader_content, loader_style = get_test_data_loaders(opts.test_path, opts.a2b, new_size=256)

    for idx1, (img1, name) in enumerate(loader_content):
        img1 = img1.cuda()

        name = os.path.basename(name[0])
        print(name)
        test_saver_path = os.path.join(opts.output_path)
        if not os.path.exists(test_saver_path):
            os.mkdir(test_saver_path)

        _, share_content, content, _, _, _ = encode(img1)

        vutils.save_image(img1.data, os.path.join(test_saver_path, 'input_' + name), padding=0, normalize=True)

        for idx2, (img2, _) in enumerate(loader_style):
            if idx2 >= 10: break
            img2 = img2.cuda()
            _, _, _, style, _, _ = style_encode(img2)
            with torch.no_grad():
                outputs = decode(share_content, content, style)
            outputs = (outputs + 1) / 2.

            path = os.path.join(test_saver_path, 'output_' + name.split('.')[0] + '_{}.png'.format(idx2))
            vutils.save_image(outputs.data, path, padding=0, normalize=True)
