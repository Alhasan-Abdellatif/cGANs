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
(For more documentation on the paramters, see utils.py)

To run unconditional GAN on images in ```datasets/trainfolder ```  and save models in ```results``` :
```
python train.py --data_path datasets/folder --data_ext txt  --img_ch 1  --zdim 128 --spec_norm_D --x_fake_GD  --batch_size 32  --epochs  160 --smooth --save_rate 2  --ema --dev_num 1  --att  --fname results 
```
To run a conditional GAN :

```
python train.py --data_path datasets/trainfolder --labels_path datasets/label.csv  --data_ext txt  --img_ch 1  --zdim 128 --spec_norm_D --y_real_GD --x_fake_GD  --n_cl 4 --cgan --batch_size 32  --epochs  160 --smooth --save_rate 2  --ema --dev_num 1  --att  --fname results 
```





