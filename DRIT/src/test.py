import time

import torch
from options import TestOptions
from dataset import dataset_single
from model import DRIT
from saver import save_imgs
import os


def main():
    # parse options
    parser = TestOptions()
    opts = parser.parse()

    # data loader
    print('\n--- load dataset ---')
    dataset_A = dataset_single(opts, 'A', opts.input_dim_a)
    dataset_B = dataset_single(opts, 'B', opts.input_dim_b)
    loader_A = torch.utils.data.DataLoader(dataset_A, batch_size=1, num_workers=opts.nThreads)
    loader_B = torch.utils.data.DataLoader(dataset_B, batch_size=1, num_workers=opts.nThreads)

    # model
    print('\n--- load model ---')
    model = DRIT(opts)

    model.resume(opts.resume, train=False)
    model.setgpu(opts.gpu)
    model.eval()

    # directory
    result_dir = os.path.join(opts.result_dir, opts.name)
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    # test
    print('\n--- testing ---')
    for idx2, (img2, name) in enumerate(loader_B):
        print('{}/{}'.format(idx2, len(loader_B)))
        img = img2.cuda(opts.gpu)
        img = [img]
        name = [name[0].split('.')[0]]
        save_imgs(img, name, result_dir)

    for idx1, (img1, name) in enumerate(loader_A):
        print('{}/{}'.format(idx1, len(loader_A)))
        img1 = img1.cuda(opts.gpu)
        imgs = []
        # print(name)
        names = []
        for idx2 in range(opts.num):
            with torch.no_grad():
                img = model.test_forward(img1, a2b=opts.a2b)
            imgs.append(img)
            names.append('output_{}_{}'.format(name[0].split('.')[0], idx2))
        save_imgs(imgs, names, result_dir)

    return


if __name__ == '__main__':
    main()

