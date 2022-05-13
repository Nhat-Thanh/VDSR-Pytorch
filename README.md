# [Pytorch] Very Deep Super-Resolution Network

Implementation of VDSR model in **Accurate Image Super-Resolution Using Very Deep Convolutional Networks** paper with Pytorch.

I used Adam with optimize tuned hyperparameters instead of SGD + Momentum with clipping gradient.


## Contents
- [Train](#train)
- [Test](#test)
- [Demo](#demo)
- [Evaluate](#evaluate)
- [References](#references)


## Train
You run this command to begin the training:
```
python train.py  --epochs=80             \
                 --batch_size=64         \
                 --save-best-only=1      \
		 --save-log=0		 \
                 --ckpt-dir="checkpoint/"
```
- **--save-best-only**: if it's equal to **0**, model weights will be saved every epoch.
- **--save-log**: if it's equal to 1, **train loss, train metrics, validation losses, validation metrics** will be saved every save-every steps.


**NOTE**: if you want to re-train a new model, you should delete all files in **checkpoint** directory. Your checkpoint will be saved when above command finishs and can be used for the next times, so you can train a model on Google Colab without taking care of GPU time limit.

I trained the model on Google Colab in 80 epochs:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nhat-Thanh/VDSR-Pytorch/blob/main/VDSR-Pytorch.ipynb)

You can get the models here: [VDSR.pt](checkpoint/VDSR.pt)


## Test
I use **Set5** as the test set. After Training, you can test models with scale factors **x2, x3, x4**, the result is calculated by compute average PSNR of all images.
```
python test.py --scale=2 --ckpt-path="default"
```

**--ckpt-path="default"** means you are using default model path, aka **checkpoint/VDSR.pt**. If you want to use your trained model, you can pass yours to **--ckpt-path**.

## Demo 
After Training, you can test models with this command, the result is the **sr.png**.
```
python demo.py --image-path="dataset/test1.png" \
	       --ckpt-path="default" 			\
               --scale=2
```

**--ckpt-path** is the same as in [Test](#test)

## Evaluate

I evaluated models with Set5, Set14, BSD100 and Urban100 dataset by PSNR. I use Set5's Butterfly to show my result:

<div align="center">

|  Dataset  |   Set5  |  Set14  |  BSD100 | Urban100 |
|:---------:|:-------:|:-------:|:-------:|:--------:|
|     x2    | 36.9849 | 33.3692 | 33.4341 | 30.5529  |
|     x3    | 34.2582 | 31.0208 | 32.0901 |     X    |
|     x4    | 31.9323 | 29.3366 | 29.6939 | 26.9200  |

  <br/>

  <img src="./README/example.png" width="1000"/><br/>
  <b>Bicubic (left), VDSR x2 (center), High Resolution (right).</b>
</div>

## References
- Accurate Image Super-Resolution Using Very Deep Convolutional Networks: https://arxiv.org/abs/1511.04587
- T91, BSD200: http://vllab.ucmerced.edu/wlai24/LapSRN/results/SR_training_datasets.zip
- Set5: https://filebox.ece.vt.edu/~jbhuang/project/selfexsr/Set5_SR.zip
- Set14: https://filebox.ece.vt.edu/~jbhuang/project/selfexsr/Set14_SR.zip
- BSD100: https://filebox.ece.vt.edu/~jbhuang/project/selfexsr/BSD100_SR.zip
- Urban100: https://filebox.ece.vt.edu/~jbhuang/project/selfexsr/Urban100_SR.zip
