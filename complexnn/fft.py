#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import tensorflow.keras.engine as KE
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as KB
import tensorflow.keras.layers as KL
import tensorflow.keras.optimizers as KO

#
# FFT functions:
#
#  fft():   Batched 1-D FFT  (Input: (Batch, TimeSamples))
#  ifft():  Batched 1-D IFFT (Input: (Batch, FreqSamples))
#  fft2():  Batched 2-D FFT  (Input: (Batch, TimeSamplesH, TimeSamplesW))
#  ifft2(): Batched 2-D IFFT (Input: (Batch, FreqSamplesH, FreqSamplesW))
#


def fft(z):
    B = z.shape[0] // 2
    L = z.shape[1]
    C = tf.Variable(np.asarray([[[1, -1]]], dtype=tf.float32))
    Zr, Zi = tf.signal.rfft(z[:B]), tf.signal.rfft(z[B:])
    isOdd = tf.equal(L % 2, 1)
    Zr = tf.cond(
        isOdd,
        tf.concat([Zr, C * Zr[:, 1:][:, ::-1]], axis=1),
        tf.concat([Zr, C * Zr[:, 1:-1][:, ::-1]], axis=1),
    )
    Zi = tf.cond(
        isOdd,
        tf.concat([Zi, C * Zi[:, 1:][:, ::-1]], axis=1),
        tf.concat([Zi, C * Zi[:, 1:-1][:, ::-1]], axis=1),
    )
    Zi = (C * Zi)[:, :, ::-1]  # Zi * i
    Z = Zr + Zi
    return tf.concat([Z[:, :, 0], Z[:, :, 1]], axis=0)


def ifft(z):
    B = z.shape[0] // 2
    L = z.shape[1]
    C = tf.Variable(np.asarray([[[1, -1]]], dtype=tf.float32))
    Zr, Zi = tf.signal.rfft(z[:B]), tf.signal.rfft(z[B:] * -1)
    isOdd = tf.equal(L % 2, 1)
    Zr = tf.cond(
        isOdd,
        tf.concat([Zr, C * Zr[:, 1:][:, ::-1]], axis=1),
        tf.concat([Zr, C * Zr[:, 1:-1][:, ::-1]], axis=1),
    )
    Zi = tf.cond(
        isOdd,
        tf.concat([Zi, C * Zi[:, 1:][:, ::-1]], axis=1),
        tf.concat([Zi, C * Zi[:, 1:-1][:, ::-1]], axis=1),
    )
    Zi = (C * Zi)[:, :, ::-1]  # Zi * i
    Z = Zr + Zi
    return tf.concat([Z[:, :, 0], Z[:, :, 1] * -1], axis=0)


def fft2(x):
    tt = x
    tt = KB.reshape(tt, (x.shape[0] * x.shape[1], x.shape[2]))
    tf = fft(tt)
    tf = KB.reshape(tf, (x.shape[0], x.shape[1], x.shape[2]))
    tf = KB.permute_dimensions(tf, (0, 2, 1))
    tf = KB.reshape(tf, (x.shape[0] * x.shape[2], x.shape[1]))
    ff = fft(tf)
    ff = KB.reshape(ff, (x.shape[0], x.shape[2], x.shape[1]))
    ff = KB.permute_dimensions(ff, (0, 2, 1))
    return ff


def ifft2(x):
    ff = x
    ff = KB.permute_dimensions(ff, (0, 2, 1))
    ff = KB.reshape(ff, (x.shape[0] * x.shape[2], x.shape[1]))
    tf = ifft(ff)
    tf = KB.reshape(tf, (x.shape[0], x.shape[2], x.shape[1]))
    tf = KB.permute_dimensions(tf, (0, 2, 1))
    tf = KB.reshape(tf, (x.shape[0] * x.shape[1], x.shape[2]))
    tt = ifft(tf)
    tt = KB.reshape(tt, (x.shape[0], x.shape[1], x.shape[2]))
    return tt


#
# FFT Layers:
#
#  FFT:   Batched 1-D FFT  (Input: (Batch, FeatureMaps, TimeSamples))
#  IFFT:  Batched 1-D IFFT (Input: (Batch, FeatureMaps, FreqSamples))
#  FFT2:  Batched 2-D FFT  (Input: (Batch, FeatureMaps, TimeSamplesH, TimeSamplesW))
#  IFFT2: Batched 2-D IFFT (Input: (Batch, FeatureMaps, FreqSamplesH, FreqSamplesW))
#


class FFT(KL.Layer):
    def call(self, x, mask=None):
        a = KB.permute_dimensions(x, (1, 0, 2))
        a = KB.reshape(a, (x.shape[1] * x.shape[0], x.shape[2]))
        a = fft(a)
        a = KB.reshape(a, (x.shape[1], x.shape[0], x.shape[2]))
        return KB.permute_dimensions(a, (1, 0, 2))


class IFFT(KL.Layer):
    def call(self, x, mask=None):
        a = KB.permute_dimensions(x, (1, 0, 2))
        a = KB.reshape(a, (x.shape[1] * x.shape[0], x.shape[2]))
        a = ifft(a)
        a = KB.reshape(a, (x.shape[1], x.shape[0], x.shape[2]))
        return KB.permute_dimensions(a, (1, 0, 2))


class FFT2(KL.Layer):
    def call(self, x, mask=None):
        a = KB.permute_dimensions(x, (1, 0, 2, 3))
        a = KB.reshape(a, (x.shape[1] * x.shape[0], x.shape[2], x.shape[3]))
        a = fft2(a)
        a = KB.reshape(a, (x.shape[1], x.shape[0], x.shape[2], x.shape[3]))
        return KB.permute_dimensions(a, (1, 0, 2, 3))


class IFFT2(KL.Layer):
    def call(self, x, mask=None):
        a = KB.permute_dimensions(x, (1, 0, 2, 3))
        a = KB.reshape(a, (x.shape[1] * x.shape[0], x.shape[2], x.shape[3]))
        a = ifft2(a)
        a = KB.reshape(a, (x.shape[1], x.shape[0], x.shape[2], x.shape[3]))
        return KB.permute_dimensions(a, (1, 0, 2, 3))
