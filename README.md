# GANs
This repository provides a pytorch implementation of recent GANs models including
[SN-GANs](https://arxiv.org/abs/1802.05957) - with both DCGAN and Residual architecture.
[SA-GANs](https://arxiv.org/abs/1805.08318) - which uses attention mechanism on an intermeidate layer of G and D.

For Conditional-GANs models, conditional-batch normalization (as in [SN-GANs](https://arxiv.org/abs/1802.05957) and [SA-GANs](https://arxiv.org/abs/1805.08318)) is used in the Generator while in the discriminator you can choose between the [projection method](https://arxiv.org/abs/1802.05637) and [concatenation method](https://arxiv.org/abs/1605.05396).

The losses implemented are the standard adverserial, Hinge  and WGAN losses. The first two works quite well with spectral normalization [SN-GANs](https://arxiv.org/abs/1802.05957)

The FID implementation is adapted from [mseitzer/pytorch-fid](https://github.com/mseitzer/pytorch-fid)

### Prerequisites

* PyTorch, version 1.0.1
* tqdm, numpy, scipy, matplotlib
* A Training set (e.g. ImageNet) should be added in the datasets folder (the current model uses simple geological channels images with a plan to extend it to more datasets )
* For FID computation, a testing set should be added to datasets and its path should be added to the  file pytorch-fid/fid_score.py


## Running

To run unconditional GAN (note the difference G_ch and D_ch for cnn_GAN and residual_GAN)  :
```
python train.py --data channels --data_path datasets/Channel_train --img_ch 1 --model residual_GAN  --G_ch 52  --D_ch 32  --leak_D 0  --save_rate 5 --epoch 100 --fname Exps/channels_files/res_GAN_G52_SND_D32  --batch_size 32
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



