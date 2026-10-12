import unittest

import complexnn

# Every public name that `import complexnn` provides, submodules included.
PUBLIC_NAMES = {
    "bn",
    "conv",
    "dense",
    "fft",
    "init",
    "norm",
    "pool",
    "utils",
    "ComplexBN",
    "ComplexConv",
    "ComplexConv1D",
    "ComplexConv2D",
    "ComplexConv3D",
    "WeightNorm_Conv",
    "ComplexDense",
    "ComplexIndependentFilters",
    "ComplexInit",
    "IndependentFilters",
    "SqrtInit",
    "ComplexLayerNorm",
    "LayerNormalization",
    "SpectralPooling1D",
    "SpectralPooling2D",
    "GetAbs",
    "GetImag",
    "GetReal",
    "get_imagpart",
    "get_realpart",
    "getpart_output_shape",
}


class TestPublicNamespace(unittest.TestCase):
    """The names the complexnn package exports"""

    def test_public_attributes(self):
        """The package has exactly the public attributes listed above"""
        public = {name for name in dir(complexnn) if not name.startswith("_")}
        self.assertEqual(public, PUBLIC_NAMES)

    def test_star_import(self):
        """`from complexnn import *` provides exactly the public names"""
        namespace = {}
        exec("from complexnn import *", namespace)
        del namespace["__builtins__"]
        self.assertEqual(set(namespace), PUBLIC_NAMES)
