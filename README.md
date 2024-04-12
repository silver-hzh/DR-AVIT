# DR-AVIT-Towards-Diverse-and-Realistic-Aerial-Visible-to-Infrared-Image-Translation
This repository contains the availability of two new benchmark datasets: Day-DroneVehicle and Night-DroneVehicle for aerial visible-to-infrared image translation tasks, and the training codes of our proposed method.
# Availability of Datasets
The datasets can be downloaded from <https://pan.baidu.com/s/1q7hxGzAaR97L8T36NKFqpw>, the code is xbhm.
# Code of DR-AVIT
## Requirements
- Python 3.7 or higher 
- Pytorch 1.8.0 or higher, torchvison 0.9.0 or higher
- Tensorboard, TensorboardX, Pyyaml, Pillow, dominate, visdom

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




