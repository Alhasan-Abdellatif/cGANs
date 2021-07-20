# GANs
This repository provides a pytorch implementation of a variety of GANs models including
[SN-GANs](https://arxiv.org/abs/1802.05957) - with both DCGAN and Residual architecture.
[SA-GANs](https://arxiv.org/abs/1805.08318) - which uses attention mechanism on intermeidate layers of G and D.

For Conditional-GANs models, in addition to the standard concatenation methods, conditional-batch normalization (as in [SN-GANs](https://arxiv.org/abs/1802.05957) and [SA-GANs](https://arxiv.org/abs/1805.08318)) is implemented in the Generator and [projection method](https://arxiv.org/abs/1802.05637) in the discriminator.

The losses implemented are the standard adverserial and Hinge which work quite well with spectral normalization [SN-GANs](https://arxiv.org/abs/1802.05957)

### Prerequisites

* PyTorch, version 1.0.1
* tqdm, numpy, scipy, matplotlib
* A Training set (e.g. MNIST) should be added in the datasets folder



## Running

To run unconditional GAN on images in ```datasets/folder ```  and save models in ```results``` :
```
python train.py --data_path datasets/folder --data_ext txt  --img_ch 1  --zdim 128 --x_fake_GD  --batch_size 32  --epochs  160  --save_rate 2  --ema --dev_num 1  --att  --fname results 
```

```
python train.py --data channels --data_path datasets/Channel_train --img_ch 1 --model cnn_GAN --G_ch 512 --D_ch 64 --leak_D 0.1 --save_rate 5 --epoch 100 --fname Exps/channels_files/cnn_GAN_512_snD --batch_size 32
```
To run a conditional GAN :


```
python train.py --data propchannels --data_path datasets/prop_channels_train --cgan  --img_ch 1 --model residual_GAN  --G_ch 52  --D_ch 32   --att --save_rate 2 --epoch 80   --fname Exps/propchannels_files/res_adv_snD --max_label 0.35 --min_label 0.25   --batch_size 32 --dev_num 1 --D_cond_method concat
```

```
python train.py --data propchannels --data_path datasets/prop_channels_train --cgan  --img_ch 1 --model residual_GAN  --G_ch 52  --D_ch 32   --att --save_rate 2 --epoch 80   --fname Exps/propchannels_files/res_adv_snD --ohe   --batch_size 32 --dev_num 1 --D_cond_method proj
```
To sample from a model :

```
python sample.py --figure images --img_ch 1 --G_cp Exps/channels_files/res_GAN_G52_SND_D32/100_100.pth  --model residual_GAN --G_ch 52 --num_imgs 5000 --out_path out
```

```
python sample.py --figure grid --img_ch 1 --G_cp Exps/channels_files/res_GAN_G52_SND_D32/100_100.pth  --model residual_GAN --G_ch 52 --out_path out
```


To run FID on a folder of images :

```
python pytorch-fid/fid_score.py --gpu 0 --test_path datasets/Channel_test --path out ```
```



