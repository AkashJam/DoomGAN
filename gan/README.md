# GAN Models
There are 2 gan networks that are used to derive the 3 models for the generation of DOOM Levels. The 3 models consist of a simple Wasserstien GAN with Gradient penality and 2 hybrid architecture with a conditional GAN using either a traditional generator loss or the modified counterpart. These can be accessed through a series of flags when creating an instance of the class. Shown below is the recommended setting
```
from wgan import WGAN_GP
wgan = WGAN_GP(use_hybrid = True)
wgan.train(epochs = 100)

from cgan import cGAN
cgan = cGAN(use_mod = True)
cgan.train(epochs = 100)
```

Once the systems have been trained you can generate samples by simply calling the generator and providing it an appropriate seed to obtain the feature maps