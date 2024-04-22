import argparse
import os
import random

import lpips
import numpy as np

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d', '--dir', type=str)
parser.add_argument('-o', '--out', type=str)
parser.add_argument('-v', '--version', type=str, default='0.1')
parser.add_argument('-N', type=int, default=5)
parser.add_argument('--use_gpu', default='True', help='turn on flag to use GPU')

opt = parser.parse_args()

## Initializing the model
loss_fn = lpips.LPIPS(net='alex', version=opt.version)
if (opt.use_gpu):
    loss_fn.cuda()


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
    all_names.append('output_' + name.split('_')[1])
all_names = list(set(all_names))
sample_list = sample_pairs(all_names, opt.N)

dists = []
for i in range(len(sample_list)):
    pair = sample_list[i]
    img0 = lpips.im2tensor(lpips.load_image(os.path.join(opt.dir, pair[0])))
    img1 = lpips.im2tensor(lpips.load_image(os.path.join(opt.dir, pair[1])))

    if opt.use_gpu:
        img0 = img0.cuda()
        img1 = img1.cuda()
    distance = loss_fn.forward(img0, img1)
    print('(%s,%s): %.4f' % (pair[0], pair[1], distance))
    dists.append(distance.item())
avg_dist = np.mean(np.array(dists))
print('Avg:{:.6f}'.format(avg_dist))
