# Complex-Valued Neural Networks in Keras with Tensorflow
[![Documentation](https://readthedocs.org/projects/keras-complex/badge/?version=latest)](https://keras-complex.readthedocs.io/) [![Build Status](https://github.com/JesperDramsch/keras-complex/actions/workflows/test_python.yml/badge.svg)](https://github.com/JesperDramsch/keras-complex/actions/) [![PyPI Versions](https://img.shields.io/pypi/pyversions/keras-complex.svg)](https://pypi.python.org/pypi/keras-complex) ![Tensorflow 2+](https://img.shields.io/badge/tensorflow-%3E2.0-orange) ![PyPI - Downloads](https://img.shields.io/pypi/dm/keras-complex)  [![PyPI Status](https://img.shields.io/pypi/status/keras-complex.svg)](https://pypi.python.org/pypi/keras-complex) [![PyPI License](https://img.shields.io/badge/License-MIT-green)](LICENSE.md)

## Status

keras-complex works with current TensorFlow and Keras 3. It is maintained as time allows: it gets fixes when TensorFlow or Keras change, and issues and pull requests are welcome. New features are not planned.

CI tests every change against the latest TensorFlow with Keras 3 on Python 3.11 to 3.13, and against Keras 2 through TensorFlow 2.15 and through `tf-keras`.

Not working in this release: spectral pooling, `spectral_parametrization=True`, `WeightNorm_Conv`, `SqrtInit`, the `fft` module, and training `ComplexConv1D` with `transposed=True`. They fail on Keras 2 as well, so they were already broken before the Keras 3 update.

[Complex-valued convolutions](https://en.wikipedia.org/wiki/Convolution#Domain_of_definition) could provide some interesting results in signal processing-based deep learning. A simple(-ish) idea is including explicit phase information of time series in neural networks. This code enables complex-valued convolution in convolutional neural networks in [keras](https://keras.io) with the [TensorFlow](https://tensorflow.org/) backend. This makes the network modular and interoperable with standard keras layers and operations.

This repository is based on the code which reproduces experiments presented in the paper [Deep Complex Networks](https://arxiv.org/abs/1705.09792). It is a port to Keras with Tensorflow-backend.

Requirements
------------

- Python 3.11 or later
- TensorFlow 2.15 or later

Keras 3 (TensorFlow 2.16 and later) and Keras 2 both work. To use Keras 2 with a newer TensorFlow, install `tf-keras` and set `TF_USE_LEGACY_KERAS=1` before TensorFlow is first imported.


Installation
------------

```
pip install keras-complex
```

Usage
-----
Build your neural networks with the help of keras. 

``` 
import complexnn

import keras
from keras import models
from keras import layers
from keras import optimizers

model = models.Sequential()

model.add(complexnn.conv.ComplexConv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 2)))
model.add(complexnn.bn.ComplexBatchNormalization())
model.add(layers.MaxPooling2D((2, 2), padding='same'))

model.compile(optimizer=optimizers.Adam(), loss='mse')

```

An example working implementation of an autoencoder can be found [here](https://github.com/JesperDramsch/Complex-CNN-Seismic/).

Saving and loading
------------------

Models with complexnn layers save to the `.keras` format and load again without `custom_objects`. The layers are registered with Keras when `complexnn` is imported, so import it before loading:

```
import complexnn
import tensorflow as tf

model.save("model.keras")
model = tf.keras.models.load_model("model.keras")
```

Complex Format of Tensors
-------------------------

This library assumes that complex values are split into two real-valued parts. The real-valued and complex-valued complement, also seen [in the Docs](https://keras-complex.readthedocs.io/math.html).

The tensors for a 2D complex tensor of 3x3, the look like:

```
[[[r r r],
  [r r r],
  [r r r]],
  [i,i,i],
  [i,i,i],
  [i,i,i]]]
```

So multiple samples should then be arranged into `[r,r,r,i,i,i]`, which is also documented [in the Docs](https://keras-complex.readthedocs.io/math.html#implementation).

Citation
--------

Find the [CITATION file](/CITATION.cff) or cite this software version as:
```
@misc{dramsch2019complex, 
    title     = {Complex-Valued Neural Networks in Keras with Tensorflow}, 
    url       = {https://figshare.com/articles/Complex-Valued_Neural_Networks_in_Keras_with_Tensorflow/9783773/1}, 
    DOI       = {10.6084/m9.figshare.9783773}, 
    publisher = {figshare}, 
    author    = {Dramsch, Jesper S{\"o}ren and Contributors}, 
    year      = {2019}
}
```

Please cite the original work as: 

```
@ARTICLE {Trabelsi2017,
    author  = "Chiheb Trabelsi, Olexa Bilaniuk, Ying Zhang, Dmitriy Serdyuk, Sandeep Subramanian, João Felipe Santos, Soroush Mehri, Negar Rostamzadeh, Yoshua Bengio, Christopher J Pal",
    title   = "Deep Complex Networks",
    journal = "arXiv preprint arXiv:1705.09792",
    year    = "2017"
}
```
