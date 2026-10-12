# Data layout

The layers in keras-complex work on real-valued tensors. Each complex number is stored as two real numbers, its real part and its imaginary part, and the two parts are stacked along one axis, usually the channel axis:

- the first half of that axis holds the real parts,
- the second half holds the imaginary parts.

An input with `n` complex channels therefore has `2n` channels, and that axis must have an even size of at least 2.

## Example

A batch of 3x3 images with two complex channels, `z0` and `z1`, has the shape `(batch, 3, 3, 4)` with `data_format="channels_last"`. Its four channels hold, in this order:

| Channel | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| Content | real part of `z0` | real part of `z1` | imaginary part of `z0` | imaginary part of `z1` |

The parts are not interleaved. Channels ordered as `[Re z0, Im z0, Re z1, Im z1]` would be read as two other complex channels, `Re z0 + i Re z1` and `Im z0 + i Im z1`.

## Converting complex data

Concatenate the real and imaginary parts of a complex NumPy array along the channel axis:

```python
import numpy as np

# 8 images of 28x28 pixels with 2 complex channels each.
z = np.random.randn(8, 28, 28, 2) + 1j * np.random.randn(8, 28, 28, 2)

x = np.concatenate([z.real, z.imag], axis=-1)  # shape (8, 28, 28, 4)
```

With `data_format="channels_first"`, concatenate along axis 1 instead.

The complex layers return their outputs in the same layout. A `ComplexConv2D` with `filters=f` returns `2 * f` channels, the real parts of its `f` complex feature maps followed by their imaginary parts, and a `ComplexDense` with `units=u` returns `2 * u` features. The layers can therefore be stacked directly. Keras layers that treat each channel on its own, such as activations and pooling, act on the real and imaginary parts separately.

To turn an output back into complex numbers, split its channel axis in half:

```python
y = model.predict(x)
half = y.shape[-1] // 2
z_out = y[..., :half] + 1j * y[..., half:]
```

Inside a model, `GetReal`, `GetImag` and `GetAbs` return the real parts, the imaginary parts and the magnitudes.

## The axis each layer uses

| Layer | Axis that holds the real and imaginary parts |
|---|---|
| `ComplexConv1D`, `ComplexConv2D`, `ComplexConv3D` | The channel axis: the last axis, or axis 1 with `data_format="channels_first"`. When `data_format` is not given, the layers use the Keras image data format, which is `channels_last` unless configured otherwise. |
| `ComplexDense` | The last axis of its 2D input `(batch_size, features)`. |
| `ComplexBatchNormalization`, `ComplexLayerNorm` | The axis given by their `axis` argument, the last axis by default. |
| `GetReal`, `GetImag`, `GetAbs` | The last axis, except for 4D and 5D inputs when the Keras image data format is `channels_first`, where they use axis 1. |

## Errors

A layer that is built on an input whose complex axis has an odd size, or a size smaller than 2, raises a `ValueError` naming the axis, its size and this layout.

A common cause is keeping the real and imaginary parts on an axis of their own. For example, a `ComplexConv3D` with `data_format="channels_first"` cannot take inputs of shape `(1, 32, 32, 2)` that hold one channel with the two parts on the last axis. Move the parts to the channel axis, so that the inputs have the shape `(2, 32, 32, 1)`.
