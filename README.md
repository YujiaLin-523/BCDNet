# BCDNet

## Introduction
Invasive Ductal Carcinoma (IDC) is the most common subtype of all breast cancers. To assign an aggressiveness grade to a whole mount sample, pathologists typically focus on the 
regions which contain the IDC. As a result, one of the common pre-processing steps for automatic aggressiveness grading is to delineate the exact regions of IDC instead of a whole 
mount slide. We proposed a Convolutional Neural Network (CNN) for Breast Cancer Detection (BCDNet), which is capable of assisting the doctors to diagnose this type of cancer by 
training on a large scale of Breast Histopathology Images.

## Usage
You can access the code by
```
git clone https://github.com/404-UnknownUsername/BCDNet
```
PyTorch and Python(3.8.19) are required, as well as conda environment is recommended to be installed to manage the packages.  
  
You can install the dependencies by
```
conda create -n bcdnet python==3.8.19
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```
Attention: If your CUDA version is earlier than 12.1, please visit [PyTorch](https://pytorch.org/get-started/locally/) to find the corresponding installation command.

## Dataset
The Breast Histopathology Images Dataset can be downloaded from [kaggle](https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images/data). Then, you should store it in the `data` folder under the `BCDNet` folder.

## Train
To train BCDNet on your devices, you can use
```
python train.py
```
Remember to change the `root_dir` in the `train.py` to your own data path. 
In addition, if you want to train on Windows, you should change the `num_workers` to 0 in the `train.py` file and change `'/'` to `'\\'` in the line 26 of`data_loader.py` file.

## Test
To test the model, you can use
```
python test.py
```
