#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from tensorflow.keras import activations, constraints, initializers, regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    InputSpec,
    Layer,
)
from tensorflow.python.keras.layers.convolutional import Conv
from tensorflow.python.keras.utils import conv_utils

from .bn import ComplexBN as complex_normalization
from .bn import sqrt_init
from .fft import fft, fft2, ifft, ifft2
from .init import ComplexIndependentFilters, ComplexInit


def conv1d_transpose(
    inputs,
    filter,
    kernel_size=None,
    filters=None,
    strides=(1,),
    padding="SAME",
    output_padding=None,
    data_format="channels_last",
):
    """Compatibility layer for nn.conv1d_transpose

    Take a filter defined for forward convolution and adjusts it for a
    transposed convolution."""
    if isinstance(kernel_size, tuple):
        kernel_size = kernel_size[0]
    input_shape = inputs.shape
    batch_size = input_shape[0]
    if data_format == "channels_first":
        w_axis = 2
        d_format = "NCW"
    else:
        w_axis = 1
        d_format = "NWC"

    width = input_shape[w_axis]

    # Infer the dynamic output shape:
    out_width = conv_utils.deconv_output_length(
        input_length=width,
        filter_size=kernel_size,
        padding=padding,
        output_padding=output_padding,
        stride=strides,
    )

    if data_format == "channels_first":
        output_shape = (batch_size, filters, out_width)
    else:
        output_shape = (batch_size, out_width, filters)

    filter = K.permute_dimensions(filter, (0, 2, 1))
    return tf.nn.conv1d_transpose(
        inputs,
        filter,
        output_shape,
        strides,
        padding=padding.upper(),
        data_format=d_format,
    )


def conv2d_transpose(
    inputs,
    filter,
    kernel_size=None,
    filters=None,
    strides=(1, 1),
    padding="SAME",
    output_padding=None,
    data_format="channels_last",
):
    """Compatibility layer for K.conv2d_transpose

    Take a filter defined for forward convolution and adjusts it for a
    transposed convolution."""
    input_shape = inputs.shape
    batch_size = input_shape[0]
    if data_format == "channels_first":
        h_axis, w_axis = 2, 3
    else:
        h_axis, w_axis = 1, 2

    height, width = input_shape[h_axis], input_shape[w_axis]
    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = strides

    # Infer the dynamic output shape:
    out_height = conv_utils.deconv_output_length(
        input_length=height,
        filter_size=kernel_h,
        padding=padding,
        output_padding=output_padding,
        stride=stride_h,
    )
    out_width = conv_utils.deconv_output_length(
        input_length=width,
        filter_size=kernel_w,
        padding=padding,
        output_padding=output_padding,
        stride=stride_w,
    )

    if data_format == "channels_first":
        output_shape = (batch_size, filters, out_height, out_width)
    else:
        output_shape = (batch_size, out_height, out_width, filters)

    filter = K.permute_dimensions(filter, (0, 1, 3, 2))
    return K.conv2d_transpose(
        inputs, filter, output_shape, strides, padding=padding, data_format=data_format
    )


def ifft(f):
    """Stub"""
    raise NotImplementedError(str(f))


def ifft2(f):
    """Stub"""
    raise NotImplementedError(str(f))


def conv_transpose_output_length(
    input_length, filter_size, padding, stride, dilation=1, output_padding=None
):
    """Rearrange arguments for compatibility with conv_output_length."""
    if dilation != 1:
        msg = f"Dilation must be 1 for transposed convolution. "
        msg += f"Got dilation = {dilation}"
        raise ValueError(msg)
    # return conv_utils.deconv_length(
    #     input_length,  # dim_size
    #     stride,  # stride_size
    #     filter_size,  # kernel_size
    #     padding,  # padding
    #     output_padding, # output_padding
    # )
    return conv_utils.deconv_output_length(
        input_length,
        filter_size,
        padding,
        output_padding=output_padding,
        stride=stride,
        dilation=dilation,
    )


def sanitizedInitGet(init):
    """sanitizedInitGet"""
    if init in ["sqrt_init"]:
        return sqrt_init
    elif init in ["complex", "complex_independent", "glorot_complex", "he_complex"]:
        return init
    else:
        return initializers.get(init)


def sanitizedInitSer(init):
    """sanitizedInitSer"""
    if init in [sqrt_init]:
        return "sqrt_init"
    elif init == "complex" or isinstance(init, ComplexInit):
        return "complex"
    elif init == "complex_independent" or isinstance(init, ComplexIndependentFilters):
        return "complex_independent"
    else:
        return initializers.serialize(init)


class ComplexConv(Layer):
    """Abstract nD complex convolution layer.

    This layer creates a complex convolution kernel that is convolved with the
    layer input to produce a tensor of outputs. If `use_bias` is True, a bias
    vector is created and added to the outputs. Finally, if `activation` is not
    `None`, it is applied to the outputs as well.

    Arguments:
        rank: Integer, the rank of the convolution, e.g., "2" for 2D
            convolution.
        filters: Integer, the dimensionality of the output space, i.e., the
            number of complex feature maps. It is also the effective number of
            feature maps for each of the real and imaginary parts. (I.e., the
            number of complex filters in the convolution) The total effective
            number of filters is 2 x filters.
        kernel_size: An integer or tuple/list of n integers, specifying the
            dimensions of the convolution window.
        strides: An integer or tuple/list of n integers, specifying the strides
            of the convolution. Specifying any stride value != 1 is
            incompatible with specifying any `dilation_rate` value != 1.
        padding: One of `"valid"` or `"same"` (case-insensitive).
        data_format: A string, one of `channels_last` (default) or
            `channels_first`. The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, ..., channels)` while `channels_first` corresponds to
            inputs with shape `(batch, channels, ...)`. It defaults to the
            `image_data_format` value found in your Keras config file at
            `~/.keras/keras.json`. If you never set it, then it will be
            "channels_last".
        dilation_rate: An integer or tuple/list of n integers, specifying
            the dilation rate to use for dilated convolution. Currently,
            specifying any `dilation_rate` value != 1 is incompatible with
            specifying any `strides` value != 1.
        activation: Activation function to use (see keras.activations). If you
            don't specify anything, no activation is applied (i.e., "linear"
            activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        normalize_weight: Boolean, whether the layer normalizes its complex
            weights before convolving the complex input. The complex
            normalization performed is similar to the one for the batchnorm.
            Each of the complex kernels is centred and multiplied by the
            inverse square root of the covariance matrix. Then a complex
            multiplication is performed as the normalized weights are
            multiplied by the complex scaling factor gamma.
        kernel_initializer: Initializer for the complex `kernel` weights
            matrix. By default it is 'complex'. The 'complex_independent'
            and the usual initializers could also be used. (See
            keras.initializers and init.py).
        bias_initializer: Initializer for the bias vector
            (see keras.initializers).
        kernel_regularizer: Regularizer function applied to the `kernel`
            weights matrix (see keras.regularizers).
        bias_regularizer: Regularizer function applied to the bias vector
            (see keras.regularizers).
        activity_regularizer: Regularizer function applied to the output of the
            layer (its "activation"). (See keras.regularizers).
        kernel_constraint: Constraint function applied to the kernel matrix
            (see keras.constraints).
        bias_constraint: Constraint function applied to the bias vector
            (see keras.constraints).
        spectral_parametrization: Boolean, whether or not to use a spectral
            parametrization of the parameters.
        transposed: Boolean, whether or not to use transposed convolution
    """

    def __init__(
        self,
        rank,
        filters,
        kernel_size,
        strides=1,
        padding="valid",
        data_format=None,
        dilation_rate=1,
        activation=None,
        use_bias=True,
        normalize_weight=False,
        kernel_initializer="complex",
        bias_initializer="zeros",
        gamma_diag_initializer=sqrt_init,
        gamma_off_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        gamma_diag_regularizer=None,
        gamma_off_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        gamma_diag_constraint=None,
        gamma_off_constraint=None,
        init_criterion="he",
        seed=None,
        spectral_parametrization=False,
        transposed=False,
        epsilon=1e-7,
        **kwargs,
    ):
        super(ComplexConv, self).__init__(**kwargs)
        self.rank = rank
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, rank, "kernel_size")
        self.strides = conv_utils.normalize_tuple(strides, rank, "strides")
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = (
            "channels_last"
            if rank == 1
            else conv_utils.normalize_data_format(data_format)
        )
        self.dilation_rate = conv_utils.normalize_tuple(
            dilation_rate, rank, "dilation_rate"
        )
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.normalize_weight = normalize_weight
        self.init_criterion = init_criterion
        self.spectral_parametrization = spectral_parametrization
        self.transposed = transposed
        self.epsilon = epsilon
        self.kernel_initializer = sanitizedInitGet(kernel_initializer)
        self.bias_initializer = sanitizedInitGet(bias_initializer)
        self.gamma_diag_initializer = sanitizedInitGet(gamma_diag_initializer)
        self.gamma_off_initializer = sanitizedInitGet(gamma_off_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.gamma_diag_regularizer = regularizers.get(gamma_diag_regularizer)
        self.gamma_off_regularizer = regularizers.get(gamma_off_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.gamma_diag_constraint = constraints.get(gamma_diag_constraint)
        self.gamma_off_constraint = constraints.get(gamma_off_constraint)
        if seed is None:
            self.seed = np.random.randint(1, 10e6)
        else:
            self.seed = seed
        self.input_spec = InputSpec(ndim=self.rank + 2)

        # The following are initialized later
        self.kernel_shape = None
        self.kernel = None
        self.gamma_rr = None
        self.gamma_ii = None
        self.gamma_ri = None
        self.bias = None

    def build(self, input_shape):
        """build"""
        if self.data_format == "channels_first":
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError(
                "The channel dimension of the inputs "
                "should be defined. Found `None`."
            )
        # Divide by 2 for real and complex input.
        input_dim = input_shape[channel_axis] // 2
        if False and self.transposed:
            self.kernel_shape = self.kernel_size + (self.filters, input_dim)
        else:
            self.kernel_shape = self.kernel_size + (input_dim, self.filters)
        # The kernel shape here is a complex kernel shape:
        #   nb of complex feature maps = input_dim;
        #   nb of output complex feature maps = self.filters;
        #   imaginary kernel size = real kernel size
        #                         = self.kernel_size
        #                         = complex kernel size
        if self.kernel_initializer in {"complex", "complex_independent"}:
            kls = {
                "complex": ComplexInit,
                "complex_independent": ComplexIndependentFilters,
            }[self.kernel_initializer]
            kern_init = kls(
                kernel_size=self.kernel_size,
                input_dim=input_dim,
                weight_dim=self.rank,
                nb_filters=self.filters,
                criterion=self.init_criterion,
            )
        else:
            kern_init = self.kernel_initializer

        # Fix for 'ValueError: The initial value's shape ((3, 3, 1, 8)) is not compatible with the explicitly supplied `shape` argument'
        actual_kernel_shape = list(self.kernel_shape)
        actual_kernel_shape[-1] *= 2

        self.kernel = self.add_weight(
            name="kernel",
            shape=tuple(actual_kernel_shape),
            initializer=kern_init,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )

        if self.normalize_weight:
            gamma_shape = (input_dim * self.filters,)
            self.gamma_rr = self.add_weight(
                shape=gamma_shape,
                name="gamma_rr",
                initializer=self.gamma_diag_initializer,
                regularizer=self.gamma_diag_regularizer,
                constraint=self.gamma_diag_constraint,
            )
            self.gamma_ii = self.add_weight(
                shape=gamma_shape,
                name="gamma_ii",
                initializer=self.gamma_diag_initializer,
                regularizer=self.gamma_diag_regularizer,
                constraint=self.gamma_diag_constraint,
            )
            self.gamma_ri = self.add_weight(
                shape=gamma_shape,
                name="gamma_ri",
                initializer=self.gamma_off_initializer,
                regularizer=self.gamma_off_regularizer,
                constraint=self.gamma_off_constraint,
            )
        else:
            self.gamma_rr = None
            self.gamma_ii = None
            self.gamma_ri = None

        if self.use_bias:
            bias_shape = (2 * self.filters,)
            self.bias = self.add_weight(
                name="bias",
                shape=bias_shape,
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )

        else:
            self.bias = None

        # Set input spec.
        self.input_spec = InputSpec(
            ndim=self.rank + 2, axes={channel_axis: input_dim * 2}
        )
        self.built = True

    def call(self, inputs):
        if self.data_format == "channels_first":
            channel_axis = 1
        else:
            channel_axis = -1
        input_dim = K.shape(inputs)[channel_axis] // 2
        if False and self.transposed:
            if self.rank == 1:
                f_real = self.kernel[:, : self.filters, :]
                f_imag = self.kernel[:, self.filters :, :]
            elif self.rank == 2:
                f_real = self.kernel[:, :, : self.filters, :]
                f_imag = self.kernel[:, :, self.filters :, :]
            elif self.rank == 3:
                f_real = self.kernel[:, :, :, : self.filters, :]
                f_imag = self.kernel[:, :, :, self.filters :, :]
        else:
            if self.rank == 1:
                f_real = self.kernel[:, :, : self.filters]
                f_imag = self.kernel[:, :, self.filters :]
            elif self.rank == 2:
                f_real = self.kernel[:, :, :, : self.filters]
                f_imag = self.kernel[:, :, :, self.filters :]
            elif self.rank == 3:
                f_real = self.kernel[:, :, :, :, : self.filters]
                f_imag = self.kernel[:, :, :, :, self.filters :]

        convArgs = {
            "strides": self.strides[0] if self.rank == 1 else self.strides,
            "padding": self.padding,
            "data_format": self.data_format,
            "dilation_rate": self.dilation_rate[0]
            if self.rank == 1
            else self.dilation_rate,
        }
        if self.transposed:
            convArgs.pop("dilation_rate", None)
            convArgs["kernel_size"] = self.kernel_size
            convArgs["filters"] = 2 * self.filters
            convFunc = {1: conv1d_transpose, 2: conv2d_transpose}[self.rank]
        else:
            convFunc = {1: K.conv1d, 2: K.conv2d, 3: K.conv3d}[self.rank]

        # processing if the weights are assumed to be represented in the
        # spectral domain

        if self.spectral_parametrization:
            if self.rank == 1:
                f_real = K.permute_dimensions(f_real, (2, 1, 0))
                f_imag = K.permute_dimensions(f_imag, (2, 1, 0))
                f = K.concatenate([f_real, f_imag], axis=0)
                fshape = K.shape(f)
                f = K.reshape(f, (fshape[0] * fshape[1], fshape[2]))
                f = ifft(f)
                f = K.reshape(f, fshape)
                f_real = f[: fshape[0] // 2]
                f_imag = f[fshape[0] // 2 :]
                f_real = K.permute_dimensions(f_real, (2, 1, 0))
                f_imag = K.permute_dimensions(f_imag, (2, 1, 0))
            elif self.rank == 2:
                f_real = K.permute_dimensions(f_real, (3, 2, 0, 1))
                f_imag = K.permute_dimensions(f_imag, (3, 2, 0, 1))
                f = K.concatenate([f_real, f_imag], axis=0)
                fshape = K.shape(f)
                f = K.reshape(f, (fshape[0] * fshape[1], fshape[2], fshape[3]))
                f = ifft2(f)
                f = K.reshape(f, fshape)
                f_real = f[: fshape[0] // 2]
                f_imag = f[fshape[0] // 2 :]
                f_real = K.permute_dimensions(f_real, (2, 3, 1, 0))
                f_imag = K.permute_dimensions(f_imag, (2, 3, 1, 0))

        # In case of weight normalization, real and imaginary weights are
        # normalized

        if self.normalize_weight:
            ker_shape = self.kernel_shape
            nb_kernels = ker_shape[-2] * ker_shape[-1]
            kernel_shape_4_norm = (np.prod(self.kernel_size), nb_kernels)
            reshaped_f_real = K.reshape(f_real, kernel_shape_4_norm)
            reshaped_f_imag = K.reshape(f_imag, kernel_shape_4_norm)
            reduction_axes = list(range(2))
            del reduction_axes[-1]
            mu_real = K.mean(reshaped_f_real, axis=reduction_axes)
            mu_imag = K.mean(reshaped_f_imag, axis=reduction_axes)

            broadcast_mu_shape = [1] * 2
            broadcast_mu_shape[-1] = nb_kernels
            broadcast_mu_real = K.reshape(mu_real, broadcast_mu_shape)
            broadcast_mu_imag = K.reshape(mu_imag, broadcast_mu_shape)
            reshaped_f_real_centred = reshaped_f_real - broadcast_mu_real
            reshaped_f_imag_centred = reshaped_f_imag - broadcast_mu_imag
            Vrr = K.mean(reshaped_f_real_centred**2, axis=reduction_axes) + self.epsilon
            Vii = K.mean(reshaped_f_imag_centred**2, axis=reduction_axes) + self.epsilon
            Vri = (
                K.mean(
                    reshaped_f_real_centred * reshaped_f_imag_centred,
                    axis=reduction_axes,
                )
                + self.epsilon
            )

            normalized_weight = complex_normalization(
                K.concatenate([reshaped_f_real, reshaped_f_imag], axis=-1),
                Vrr,
                Vii,
                Vri,
                beta=None,
                gamma_rr=self.gamma_rr,
                gamma_ri=self.gamma_ri,
                gamma_ii=self.gamma_ii,
                scale=True,
                center=False,
                axis=-1,
            )

            normalized_real = normalized_weight[:, :nb_kernels]
            normalized_imag = normalized_weight[:, nb_kernels:]
            f_real = K.reshape(normalized_real, self.kernel_shape)
            f_imag = K.reshape(normalized_imag, self.kernel_shape)

        # Performing complex convolution

        f_real._keras_shape = self.kernel_shape
        f_imag._keras_shape = self.kernel_shape

        cat_kernels_4_real = K.concatenate([f_real, -f_imag], axis=-2)
        cat_kernels_4_imag = K.concatenate([f_imag, f_real], axis=-2)
        cat_kernels_4_complex = K.concatenate(
            [cat_kernels_4_real, cat_kernels_4_imag], axis=-1
        )
        if False and self.transposed:
            cat_kernels_4_complex._keras_shape = self.kernel_size + (
                2 * self.filters,
                2 * input_dim,
            )
        else:
            cat_kernels_4_complex._keras_shape = (
                self.kernel_size + 2 * input_dim,
                2 * self.filters,
            )

        output = convFunc(inputs, cat_kernels_4_complex, **convArgs)

        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format=self.data_format)

        if self.activation is not None:
            output = self.activation(output)

        return output

    def compute_output_shape(self, input_shape):
        if self.transposed:
            outputLengthFunc = conv_transpose_output_length
        else:
            outputLengthFunc = conv_utils.conv_output_length
        if self.data_format == "channels_last":
            space = input_shape[1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = outputLengthFunc(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i],
                )
                new_space.append(new_dim)
            return (input_shape[0],) + tuple(new_space) + (2 * self.filters,)
        if self.data_format == "channels_first":
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = outputLengthFunc(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i],
                )
                new_space.append(new_dim)
            return (input_shape[0],) + (2 * self.filters,) + tuple(new_space)

    def get_config(self):
        config = {
            "rank": self.rank,
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "data_format": self.data_format,
            "dilation_rate": self.dilation_rate,
            "activation": activations.serialize(self.activation),
            "use_bias": self.use_bias,
            "normalize_weight": self.normalize_weight,
            "kernel_initializer": sanitizedInitSer(self.kernel_initializer),
            "bias_initializer": sanitizedInitSer(self.bias_initializer),
            "gamma_diag_initializer": sanitizedInitSer(self.gamma_diag_initializer),
            "gamma_off_initializer": sanitizedInitSer(self.gamma_off_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "gamma_diag_regularizer": regularizers.serialize(
                self.gamma_diag_regularizer
            ),
            "gamma_off_regularizer": regularizers.serialize(self.gamma_off_regularizer),
            "activity_regularizer": regularizers.serialize(self.activity_regularizer),
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            "bias_constraint": constraints.serialize(self.bias_constraint),
            "gamma_diag_constraint": constraints.serialize(self.gamma_diag_constraint),
            "gamma_off_constraint": constraints.serialize(self.gamma_off_constraint),
            "init_criterion": self.init_criterion,
            "spectral_parametrization": self.spectral_parametrization,
            "transposed": self.transposed,
        }
        base_config = super(ComplexConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ComplexConv1D(ComplexConv):
    """1D complex convolution layer.
    This layer creates a complex convolution kernel that is convolved
    with a complex input layer over a single complex spatial (or temporal)
    dimension
    to produce a complex output tensor.
    If `use_bias` is True, a bias vector is created and added to the complex
    output.
    Finally, if `activation` is not `None`,
    it is applied each of the real and imaginary parts of the output.
    When using this layer as the first layer in a model,
    provide an `input_shape` argument
    (tuple of integers or `None`, e.g.
    `(10, 128)` for sequences of 10 vectors of 128-dimensional vectors,
    or `(None, 128)` for variable-length sequences of 128-dimensional vectors.
    # Arguments
        filters: Integer, the dimensionality of the output space, i.e,
            the number of complex feature maps. It is also the effective number
            of feature maps for each of the real and imaginary parts.
            (i.e. the number of complex filters in the convolution)
            The total effective number of filters is 2 x filters.
        kernel_size: An integer or tuple/list of n integers, specifying the
            dimensions of the convolution window.
        strides: An integer or tuple/list of a single integer,
            specifying the stride length of the convolution.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        padding: One of `"valid"`, `"causal"` or `"same"` (case-insensitive).
            `"causal"` results in causal (dilated) convolutions, e.g. output[t]
            does not depend on input[t+1:]. Useful when modeling temporal data
            where the model should not violate the temporal order.
            See [WaveNet: A Generative Model for Raw Audio, section 2.1]
            (https://arxiv.org/abs/1609.03499).
        dilation_rate: an integer or tuple/list of a single integer, specifying
            the dilation rate to use for dilated convolution.
            Currently, specifying any `dilation_rate` value != 1 is
            incompatible with specifying any `strides` value != 1.
        activation: Activation function to use
            (see keras.activations).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        normalize_weight: Boolean, whether the layer normalizes its complex
            weights before convolving the complex input.
            The complex normalization performed is similar to the one
            for the batchnorm. Each of the complex kernels are centred and
            multiplied by
            the inverse square root of covariance matrix.
            Then, a complex multiplication is perfromed as the normalized
            weights are
            multiplied by the complex scaling factor gamma.
        kernel_initializer: Initializer for the complex `kernel` weights
        matrix.
            By default it is 'complex'. The 'complex_independent'
            and the usual initializers could also be used.
            (see keras.initializers and init.py).
        bias_initializer: Initializer for the bias vector
            (see keras.initializers).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see keras.regularizers).
        bias_regularizer: Regularizer function applied to the bias vector
            (see keras.regularizers).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see keras.regularizers).
        kernel_constraint: Constraint function applied to the kernel matrix
            (see keras.constraints).
        bias_constraint: Constraint function applied to the bias vector
            (see keras.constraints).
        spectral_parametrization: Whether or not to use a spectral
            parametrization of the parameters.
        transposed: Boolean, whether or not to use transposed convolution
    # Input shape
        3D tensor with shape: `(batch_size, steps, input_dim)`
    # Output shape
        3D tensor with shape: `(batch_size, new_steps, 2 x filters)`
        `steps` value might have changed due to padding or strides.
    """

    def __init__(
        self,
        filters,
        kernel_size,
        strides=1,
        padding="valid",
        dilation_rate=1,
        activation=None,
        use_bias=True,
        kernel_initializer="complex",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        seed=None,
        init_criterion="he",
        spectral_parametrization=False,
        transposed=False,
        **kwargs,
    ):
        super(ComplexConv1D, self).__init__(
            rank=1,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format="channels_last",
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            init_criterion=init_criterion,
            spectral_parametrization=spectral_parametrization,
            transposed=transposed,
            **kwargs,
        )

    def get_config(self):
        config = super(ComplexConv1D, self).get_config()
        config.pop("rank")
        return config


class ComplexConv2D(ComplexConv):
    """2D Complex convolution layer (e.g. spatial convolution over images).
    This layer creates a complex convolution kernel that is convolved
    with a complex input layer to produce a complex output tensor. If
    `use_bias`
    is True, a complex bias vector is created and added to the outputs.
    Finally, if `activation` is not `None`, it is applied to both the
    real and imaginary parts of the output.
    When using this layer as the first layer in a model,
    provide the keyword argument `input_shape`
    (tuple of integers, does not include the sample axis),
    e.g. `input_shape=(128, 128, 3)` for 128x128 RGB pictures
    in `data_format="channels_last"`.
    # Arguments
        filters: Integer, the dimensionality of the complex output space
            (i.e, the number complex feature maps in the convolution). The
            total effective number of filters or feature maps is 2 x filters.
        kernel_size: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and
            height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        padding: one of `"valid"` or `"same"` (case-insensitive).
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, height, width, channels)` while `channels_first`
            corresponds to inputs with shape
            `(batch, channels, height, width)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
        dilation_rate: an integer or tuple/list of 2 integers, specifying
            the dilation rate to use for dilated convolution.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Currently, specifying any `dilation_rate` value != 1 is
            incompatible with specifying any stride value != 1.
        activation: Activation function to use
            (see keras.activations).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        normalize_weight: Boolean, whether the layer normalizes its complex
            weights before convolving the complex input.
            The complex normalization performed is similar to the one
            for the batchnorm. Each of the complex kernels are centred and
            multiplied by
            the inverse square root of covariance matrix.
            Then, a complex multiplication is perfromed as the normalized
            weights are
            multiplied by the complex scaling factor gamma.
        kernel_initializer: Initializer for the complex `kernel` weights
        matrix.
            By default it is 'complex'. The 'complex_independent'
            and the usual initializers could also be used.
            (see keras.initializers and init.py).
        bias_initializer: Initializer for the bias vector
            (see keras.initializers).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see keras.regularizers).
        bias_regularizer: Regularizer function applied to the bias vector
            (see keras.regularizers).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see keras.regularizers).
        kernel_constraint: Constraint function applied to the kernel matrix
            (see keras.constraints).
        bias_constraint: Constraint function applied to the bias vector
            (see keras.constraints).
        spectral_parametrization: Whether or not to use a spectral
            parametrization of the parameters.
        transposed: Boolean, whether or not to use transposed convolution
    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if data_format='channels_last'.
    # Output shape
        4D tensor with shape:
        `(samples, 2 x filters, new_rows, new_cols)` if
        data_format='channels_first' or 4D tensor with shape:
        `(samples, new_rows, new_cols, 2 x filters)` if
        data_format='channels_last'. `rows` and `cols` values might have
        changed due to padding.
    """

    def __init__(
        self,
        filters,
        kernel_size,
        strides=(1, 1),
        padding="valid",
        data_format=None,
        dilation_rate=(1, 1),
        activation=None,
        use_bias=True,
        kernel_initializer="complex",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        seed=None,
        init_criterion="he",
        spectral_parametrization=False,
        transposed=False,
        **kwargs,
    ):
        super(ComplexConv2D, self).__init__(
            rank=2,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            init_criterion=init_criterion,
            spectral_parametrization=spectral_parametrization,
            transposed=transposed,
            **kwargs,
        )

    def get_config(self):
        config = super(ComplexConv2D, self).get_config()
        config.pop("rank")
        return config


class ComplexConv3D(ComplexConv):
    """3D convolution layer (e.g. spatial convolution over volumes).
    This layer creates a complex convolution kernel that is convolved
    with a complex layer input to produce a complex output tensor.
    If `use_bias` is True,
    a complex bias vector is created and added to the outputs. Finally, if
    `activation` is not `None`, it is applied to each of the real and imaginary
    parts of the output.
    When using this layer as the first layer in a model,
    provide the keyword argument `input_shape`
    (tuple of integers, does not include the sample axis),
    e.g. `input_shape=(2, 128, 128, 128, 3)` for 128x128x128 volumes
    with 3 channels,
    in `data_format="channels_last"`.
    # Arguments
        filters: Integer, the dimensionality of the complex output space
            (i.e, the number complex feature maps in the convolution). The
            total effective number of filters or feature maps is 2 x filters.
        kernel_size: An integer or tuple/list of 3 integers, specifying the
            width and height of the 3D convolution window.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        strides: An integer or tuple/list of 3 integers, specifying
            the strides of the convolution along each spatial dimension.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        padding: one of `"valid"` or `"same"` (case-insensitive).
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
            while `channels_first` corresponds to inputs with shape
            `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
        dilation_rate: an integer or tuple/list of 3 integers, specifying
            the dilation rate to use for dilated convolution.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Currently, specifying any `dilation_rate` value != 1 is
            incompatible with specifying any stride value != 1.
        activation: Activation function to use
            (see keras.activations).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        normalize_weight: Boolean, whether the layer normalizes its complex
            weights before convolving the complex input.
            The complex normalization performed is similar to the one
            for the batchnorm. Each of the complex kernels are centred and
            multiplied by
            the inverse square root of covariance matrix.
            Then, a complex multiplication is perfromed as the normalized
            weights are
            multiplied by the complex scaling factor gamma.
        kernel_initializer: Initializer for the complex `kernel` weights
            matrix.
            By default it is 'complex'. The 'complex_independent'
            and the usual initializers could also be used.
            (see keras.initializers and init.py).
        bias_initializer: Initializer for the bias vector
            (see keras.initializers).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see keras.regularizers).
        bias_regularizer: Regularizer function applied to the bias vector
            (see keras.regularizers).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see keras.regularizers).
        kernel_constraint: Constraint function applied to the kernel matrix
            (see keras.constraints).
        bias_constraint: Constraint function applied to the bias vector
            (see keras.constraints).
        spectral_parametrization: Whether or not to use a spectral
            parametrization of the parameters.
        transposed: Boolean, whether or not to use transposed convolution
    # Input shape
        5D tensor with shape:
        `(samples, channels, conv_dim1, conv_dim2, conv_dim3)` if
        data_format='channels_first'
        or 5D tensor with shape:
        `(samples, conv_dim1, conv_dim2, conv_dim3, channels)` if
        data_format='channels_last'.
    # Output shape
        5D tensor with shape:
        `(samples, 2 x filters, new_conv_dim1, new_conv_dim2, new_conv_dim3)`
        if data_format='channels_first'
        or 5D tensor with shape:
        `(samples, new_conv_dim1, new_conv_dim2, new_conv_dim3, 2 x filters)`
        if data_format='channels_last'.
        `new_conv_dim1`, `new_conv_dim2` and `new_conv_dim3` values might have
        changed due to padding.
    """

    def __init__(
        self,
        filters,
        kernel_size,
        strides=(1, 1, 1),
        padding="valid",
        data_format=None,
        dilation_rate=(1, 1, 1),
        activation=None,
        use_bias=True,
        kernel_initializer="complex",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        seed=None,
        init_criterion="he",
        spectral_parametrization=False,
        transposed=False,
        **kwargs,
    ):
        super(ComplexConv3D, self).__init__(
            rank=3,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            init_criterion=init_criterion,
            spectral_parametrization=spectral_parametrization,
            transposed=transposed,
            **kwargs,
        )

    def get_config(self):
        config = super(ComplexConv3D, self).get_config()
        config.pop("rank")
        return config


class WeightNorm_Conv(Conv):
    """WeightNorm_Conv"""

    # Real-valued Convolutional Layer that normalizes its weights
    # before convolving the input.
    # The weight Normalization performed the one
    # described in the following paper:
    # Weight Normalization: A Simple Reparameterization to Accelerate Training
    # of Deep Neural Networks
    # (see https://arxiv.org/abs/1602.07868)

    def __init__(
        self,
        gamma_initializer="ones",
        gamma_regularizer=None,
        gamma_constraint=None,
        epsilon=1e-07,
        **kwargs,
    ):
        super(WeightNorm_Conv, self).__init__(**kwargs)
        if self.rank == 1:
            self.data_format = "channels_last"
        self.gamma_initializer = sanitizedInitGet(gamma_initializer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.gamma_constraint = constraints.get(gamma_constraint)
        self.epsilon = epsilon
        self.gamma = None

    def build(self, input_shape):
        super(WeightNorm_Conv, self).build(input_shape)
        if self.data_format == "channels_first":
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError(
                "The channel dimension of the inputs "
                "should be defined. Found `None`."
            )
        input_dim = input_shape[channel_axis]
        gamma_shape = (input_dim * self.filters,)
        self.gamma = self.add_weight(
            shape=gamma_shape,
            name="gamma",
            initializer=self.gamma_initializer,
            regularizer=self.gamma_regularizer,
            constraint=self.gamma_constraint,
        )

    def call(self, inputs):
        input_shape = K.shape(inputs)
        if self.data_format == "channels_first":
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError(
                "The channel dimension of the inputs "
                "should be defined. Found `None`."
            )
        input_dim = input_shape[channel_axis]
        ker_shape = self.kernel_size + (input_dim, self.filters)
        nb_kernels = ker_shape[-2] * ker_shape[-1]
        kernel_shape_4_norm = (np.prod(self.kernel_size), nb_kernels)
        reshaped_kernel = K.reshape(self.kernel, kernel_shape_4_norm)
        normalized_weight = K.l2_normalize(
            reshaped_kernel, axis=0, epsilon=self.epsilon
        )
        normalized_weight = (
            K.reshape(self.gamma, (1, ker_shape[-2] * ker_shape[-1]))
            * normalized_weight
        )
        shaped_kernel = K.reshape(normalized_weight, ker_shape)
        shaped_kernel._keras_shape = ker_shape

        convArgs = {
            "strides": self.strides[0] if self.rank == 1 else self.strides,
            "padding": self.padding,
            "data_format": self.data_format,
            "dilation_rate": self.dilation_rate[0]
            if self.rank == 1
            else self.dilation_rate,
        }
        convFunc = {1: K.conv1d, 2: K.conv2d, 3: K.conv3d}[self.rank]
        output = convFunc(inputs, shaped_kernel, **convArgs)

        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format=self.data_format)

        if self.activation is not None:
            output = self.activation(output)

        return output

    def get_config(self):
        config = {
            "gamma_initializer": sanitizedInitSer(self.gamma_initializer),
            "gamma_regularizer": regularizers.serialize(self.gamma_regularizer),
            "gamma_constraint": constraints.serialize(self.gamma_constraint),
            "epsilon": self.epsilon,
        }
        base_config = super(WeightNorm_Conv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# Aliases
ComplexConvolution1D = ComplexConv1D
ComplexConvolution2D = ComplexConv2D
ComplexConvolution3D = ComplexConv3D
