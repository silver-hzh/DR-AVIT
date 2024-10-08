# DR-AVIT-Towards-Diverse-and-Realistic-Aerial-Visible-to-Infrared-Image-Translation
This repository contains the availability of two new benchmark datasets: Day-DroneVehicle and Night-DroneVehicle for aerial visible-to-infrared image translation tasks, and the training codes of our proposed method.
# Availability of Datasets
The datasets can be downloaded from <[https://pan.baidu.com/s/1D4l3wXmAVSG2ywL6QLGURw?pwd=hrqf](https://pan.baidu.com/s/18CI6cbg4tRK7h7CdmapruQ?pwd=feqh)>, the code is feqh. 
# Code of DR-AVIT
## Requirements
- Python 3.7 or higher 
- Pytorch 1.8.0, torchvison 0.9.0 
- Tensorboard, TensorboardX, Pyyaml, Pillow, dominate, visdom
## Usage
Download the DR_AVIT code. Make the `Datasets` folder and put the downloaded datasets in the `Datasets` folder. Making the `outputs`, `results`, and  `logs` folders to save checkpoints and translation results. 
### Training:
```
cd src/
python train.py --dataroot ../Datasets/NightDrone --name NightDrone_DR_AVIT_0  --n_ep 100 --n_ep_decay 50 --gpu 0  
```
The training results are stored in the `./results/NightDrone_DR_AVIT_0` folder.

### Testing:
```
python test.py --dataroot ../Datasets/NightDrone --name NightDrone_DR_AVIT_0  --resume ../results/NightDrone_DR_AVIT_0/00099.pth --gpu 0
```
The translation results are saved in the `./outputs/NightDrone_DR_AVIT_0` folder.

## Evaluation
### Sample division
```
python split_sample --input ./outputs/NightDrone_DR_AVIT_0 
```
The real images are saved in `./outputs/NightDrone_DR_AVIT_0_real` and the translated images are saved in `./outputs/NightDrone_DR_AVIT_0_fake`.
### Realism  Evaluation  
We use torch-fidelity (https://github.com/toshas/torch-fidelity) to evaluate the realism of the translated results.
#### FID
```
fidelity --gpu 0 --fid --input1  ./outputs/NightDrone_DR_AVIT_0_fake --input2 ./outputs/NightDrone_DR_AVIT_0_real
```
#### KID
```
fidelity --gpu 0 --kid --input1  ./outputs/NightDrone_DR_AVIT_0_fake --input2 ./outputs/NightDrone_DR_AVIT_0_real
```

### Diversity Evaluation
We use the mean LPIPS distance (https://github.com/richzhang/PerceptualSimilarity) and mean SSIM to evaluate the diversity of the translation results.
#### LPIPS
```
cd Metric/MLPIPS/
python mlpips.py --dir ../../DR-AVIT/outputs/NightDrone_DR_AVIT_0_fake
```
#### SSIM
```
cd Metric/
python MSSIM.py --dir ../DR-AVIT/outputs/NightDrone_DR_AVIT_0_fake
```

# Other Methods
We also support MUNIT, DRIT, DSMAP, and ACLGAN:
## MUNIT：
The code of the MUNIT is followed by https://github.com/NVlabs/MUNIT. Download the MUNIT code. Make the `Datasets` folder and put the downloaded datasets in the `Datasets` folder. Making the `outputs` and `results` folders to save checkpoints and translation results.
### Training:
```  
CUDA_VISIBLE_DEVICES=0 python train.py --config ./configs/NightDrone_MUNIT.yaml --task 0
```
The training results are stored in the `./outputs/NightDrone_MUNIT_0` folder.
### Testing:
```
CUDA_VISIBLE_DEVICES=0 python test_batch.py --config ./configs/NightDrone_MUNIT.yaml --input_folder_A ./Datasets/NightDrone/testA --input_folder_B ./Datasets/NightDrone/testB --output_folder ./results/NightDrone_MUNIT_0 --checkpoint ./outputs/NightDrone_MUNIT_0/checkpoints/gen_00200000.pt
```
The translation results are saved in the `./results/NightDrone_MUNIT_0` folder.


## DSMAP：
The code of the DSMAP is followed by https://github.com/acht7111020/DSMAP. Download the DSMAP code. Make the `Datasets` folder and put the downloaded datasets in the `Datasets` folder. Making the `outputs` and `results` folders to save checkpoints and translation results.
### Training:
```  
CUDA_VISIBLE_DEVICES=0 python train.py --config ./configs/NightDrone_DSMAP.yaml --save_name NightDrone_DSMAP_0
```
The training results are stored in the `./outputs/NightDrone_DSMAP` folder.
### Testing:
```
CUDA_VISIBLE_DEVICES=0 python test.py --config ./configs/NightDrone_DSMAP.yaml --test_path ./Datasets/NightDrone --output_path ./results/NightDrone_DSMAP_0 --checkpoint ./outputs/NightDrone_DSMAP/NightDrone_DSMAP_0/ckpts/gen_00200000.pt
```
The translation results are saved in the `./results/NightDrone_DSMAP_0` folder.



## ACLGAN：
The code of the DSMAP is followed by https://github.com/hyperplane-lab/ACL-GAN. Download the ACLGAN code. Make the `Datasets` folder and put the downloaded datasets in the `Datasets` folder. Making the `outputs` and `results` folders to save checkpoints and translation results.
### Training:
```  
CUDA_VISIBLE_DEVICES=0 python train.py --config ./configs/NightDrone_ACLGAN.yaml --task 0
```
The training results are stored in the `./outputs/NightDrone_ACLGAN_0` folder.
### Testing:
```
CUDA_VISIBLE_DEVICES=0 python test_batch.py --config ./configs/NightDrone_ACLGAN.yaml --input_folder_A ../Datasets/NightDrone/testA --input_folder_B ../Datasets/NightDrone/testB --output_folder ./results/NightDrone_ACLGAN_0 --checkpoint ./outputs/NightDrone_ACLGAN_0/checkpoints/gen_00200000.pt
```
The translation results are saved in the `./results/NightDrone_ACLGAN_0` folder.



## DRIT：
The code of the DRIT is followed by https://github.com/HsinYingLee/DRIT. Download the DRIT code. Make the `Datasets` folder and put the downloaded datasets in the `Datasets` folder. Making the `outputs`, `results`, and  `logs` folders to save checkpoints and translation results.
### Training:
```
cd src/
python train.py --dataroot ../Datasets/NightDrone --name NightDrone_DRIT_0 --gpu 0 --no_ms 
```
The training results are stored in the `./results/NightDrone_DRIT_0` folder.
### Testing:
```
python test.py --dataroot ../Datasets/NightDrone --resume ../results/NightDrone_DRIT_0/00099.pth  --name NightDrone_DRIT_0 --no_ms
```
The translation results are saved in the `./outputs/NightDrone_DRIT_0` folder.

Citation
If you find this code useful for your research, please cite our paper.
```
@article{0DR,
  title={DR-AVIT: Toward Diverse and Realistic Aerial Visible-to-Infrared Image Translation},
  author={ Han, Zonghao  and  Zhang, Shun  and  Su, Yuru  and  Chen, Xiaoning  and  Mei, Shaohui },
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  volume={62},
}
```
Our dataset was created based on DroneVehicle, please also cite this paper.
```
@article{sun2022drone,
  title={Drone-based RGB-infrared cross-modality vehicle detection via uncertainty-aware learning},
  author={Sun, Yiming and Cao, Bing and Zhu, Pengfei and Hu, Qinghua},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  volume={32},
  number={10},
  pages={6700--6713},
  year={2022},
  publisher={IEEE}
}
```

