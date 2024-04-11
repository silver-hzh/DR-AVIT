# DR-AVIT-Towards-Diverse-and-Realistic-Aerial-Visible-to-Infrared-Image-Translation
This repository contains the availability of two new benchmark datasets: Day-DroneVehicle and Night-DroneVehicle for aerial visible-to-infrared image translation tasks, and the training codes of our proposed method.
# Availability of Datasets
The datasets can be downloaded from <https://pan.baidu.com/s/1q7hxGzAaR97L8T36NKFqpw>, the code is xbhm.
# Code of DR-AVIT


# Other Methods
We also support MUNIT, DRIT, DSMAP, and DCLGAN:
## MUNIT：
The code of the MUNIT is followed by https://github.com/NVlabs/MUNIT. Download the MUNIT code. Make the `Datasets` folder and put the downloaded datasets in the `Datasets` folder. Making the `outputs` and `results` folders to save checkpoints and translation results.
### Training:
```  
CUDA_VISIBLE_DEVICES=0 python train.py --config ./configs/NightDrone_MUNIT.yaml --task 0
```
### Testing:
```
CUDA_VISIBLE_DEVICES=0 python test_batch.py 
```
