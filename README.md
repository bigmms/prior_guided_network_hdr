# Deep Prior Guided Network for High-quality Image Fusion
Implementation of deep prior guided network for high-quality image fusion, ICME 2020.

**Paper**: [PDF](https://ieeexplore.ieee.org/document/9102832)

## Introduction
In this repository, we provide
* Our model architecture description (Prior Guided Network)
* Demo code
* Trained models
* Fusion examples

## Architecture

![](./demo/framework.png)

## Dependencies
* Python 3.6
* [Tensorflow >= 1.14.0](https://www.tensorflow.org/) (CUDA version >= 10.0 if installing with CUDA. [More details](https://www.tensorflow.org/install/gpu/))
* Python packages:  `pip install -r requirement.txt`

Our code is tested under Windows 10 environment with NVIDIA TITAN RTX GPU (24GB VRAM). Might work under others, but didn't get to test any other OSs just yet.


## Test models
1. Clone this github repo. 
```
git clone https://github.com/bigmms/prior_guided_network_hdr
cd prior_guided_network_hdr
```
2. Place your own **LDR images** in `./HDR` folder. (There are several sample images there).
3. Download pretrained models from [Google Drive](https://drive.google.com/file/d/19lT7K_Ea0qYsEIBI44tS8D76tHUhDoxU/view?usp=sharing). Place the trained model in `./saved_models`. 
4. Run test. We provide the trained model and you can config in the `PGNet_test.py`.You can run different models and testing datasets by changing input arguments.
```
# To run with different models, set --model as your model path.
# To run for different testing dataset, you need to set --testing_dir as your data path.

cd $makeReposit/prior_guided_network_hdr

# Test model
python PGNet_test.py --model ./saved_model/generator.h5 --testing_dir ./HDR/
```
    

5. The results are in `./Results` folder.


## Results
![](./demo/results-1.png)

## License + Attribution
This code is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Commercial usage is not permitted. If you use this code in a scientific publication, please cite the following [paper](https://ieeexplore.ieee.org/document/9102832):
```
@INPROCEEDINGS{YinICME20, 
author={Yin, Jia-Li and Chen, Bo-Hao and Peng, Yan-Tsung and Tsai, Chung-Chi}, 
booktitle={IEEE International Conference on Multimedia and Expo (ICME)},
title={Deep Prior Guided Network For High-Quality Image Fusion},  
year={2020},  
volume={}, 
number={},  
pages={1-6}}
```
