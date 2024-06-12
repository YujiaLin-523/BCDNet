# BCDNet

## Introduction
Previous research has established that breast cancer is a prevalent cancer type, with Invasive Ductal Carcinoma (IDC) being the most common subtype. The incidence of this dangerous cancer continues to rise, making accurate and rapid diagnosis, particularly in the early stages, critically important. While modern Computer-Aided Diagnosis (CAD) systems can address most cases, medical professionals still face challenges in quickly adapting CAD systems or using them in the field without powerful computing resources. In this paper, we enhance the traditional Convolutional Neural Network (CNN) architecture by integrating Batch Normalization and Dropout layers, tailoring the model to meet the specific demands of IDC detection. Furthermore, we introduce a novel CNN called BCDNet, which effectively detects IDC in histopathological images with an accuracy of up to 89.5% and reduces training time by up to 82.1%.

## Install
Download the code and install the dependencies, for which conda environment is recommended.
```
git clone https://github.com/404-UnknownUsername/BCDNet
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
Remember to change the configuration in `train.py` based on your requirements and devices.

## Test
To test the model, you can use
```
python test.py
```
Our test results are as follows:   
BCDNet:
![](https://github.com/404-UnknownUsername/BCDNet/blob/main/logs/BCDNet.png)
ResNet 50:
![](https://github.com/404-UnknownUsername/BCDNet/blob/main/logs/resnet.png)
ViT-B-16:
![](https://github.com/404-UnknownUsername/BCDNet/blob/main/logs/ViT.png)
