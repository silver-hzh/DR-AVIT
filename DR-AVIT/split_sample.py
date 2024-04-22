import os
import random
import shutil
import argparse

parser = argparse.ArgumentParser('split DR-AVIT image')
parser.add_argument('--input', help='input directory for src image')
args = parser.parse_args()


def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def copy_file(src_path, target_path):
    """
    this function aims to copy file from src_path to target_path
    :param src_path:  source path
    :param target_path: target path
    :return: None
    """
    shutil.copy(src=src_path, dst=target_path)


def remove_file(path):
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        os.remove(file_path)


fake_path = os.path.join(args.input + '_fake')
if os.path.exists(fake_path):
    remove_file(fake_path)
else:
    makedir(fake_path)
real_path = os.path.join(args.input + '_real')
if os.path.exists(real_path):
    remove_file(real_path)
else:
    makedir(real_path)

input_img_path = os.path.join(args.input)
filename_list = [x for x in os.listdir(input_img_path)]
sample_list = []
for name in filename_list:
    if 'output_' not in name:
        copy_file(src_path=os.path.join(input_img_path, name), target_path=os.path.join(real_path, name))
    else:
        name_ = name.split('_')[1]
        random_ = ['_0.png', '_1.png', '_2.png', '_3.png', '_4.png', '_5.png', '_6.png', '_7.png', '_8.png',
                   '_9.png']
        number = random.sample(random_, 1)
        new_name = name_ + '.png'
        copy_file(src_path=os.path.join(input_img_path, 'output_' + name_ + number[0]),
                  target_path=os.path.join(fake_path, new_name))
