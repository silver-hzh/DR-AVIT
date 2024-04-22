import argparse
import os
import random
from skimage import metrics
from PIL import Image
import numpy as np

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dir', type=str)
parser.add_argument('--out', type=str)
parser.add_argument('-N', type=int, default=5)
opt = parser.parse_args()


def calculate_ssim(img1, img2):
    img1 = np.array(img1).astype(np.float64)
    img2 = np.array(img2).astype(np.float64)

    ssim_value = metrics.structural_similarity(img1, img2)

    return ssim_value


def sample_pairs(filename, number):
    sample_list = []
    # DR-AVIT
    random_ = ['_0.png', '_1.png', '_2.png', '_3.png', '_4.png', '_5.png', '_6.png', '_7.png', '_8.png',
               '_9.png']
    for name in filename:
        for i in range(number):
            random_sample = random.sample(random_, 2)
            sample_list.append([name + random_sample[0], name + random_sample[1]])
    return sample_list


# sample pairs

file_name = [x for x in os.listdir(opt.dir)]
all_names = []
for name in file_name:
    # DR-AVIT
    all_names.append('output_' + name.split('_')[1])
    # all_names.append(name.split('_')[0])
#     print(name.split('_')[0])
#     all_names.append(name.split('_')[0])
all_names = list(set(all_names))
sample_list = sample_pairs(all_names, opt.N)

dists = []
for i in range(len(sample_list)):
    pair = sample_list[i]
    image0 = Image.open(os.path.join(opt.dir, pair[0])).convert('L')
    image1 = Image.open(os.path.join(opt.dir, pair[1])).convert('L')
    ssim_value = calculate_ssim(image0, image1)
    print('([%s/%s] %s,%s): %.4f' % (i, len(sample_list), pair[0], pair[1], ssim_value))
    dists.append(ssim_value)
avg_dist = np.mean(np.array(dists))
print('Avg:{:.6f}'.format(avg_dist))
