import ast
import importlib
import sys
import unittest
from pathlib import Path
from unittest import mock

import complexnn

PRIVATE_PREFIXES = ("tensorflow.python", "keras.src", "tf_keras.src", "keras._tf_keras")

# WeightNorm_Conv is broken on Keras 2 and Keras 3 and still subclasses the
# Keras 2 convolution base until it is rebuilt on the public Layer class.
PRIVATE_CONV = "tensorflow.python.keras.layers.convolutional"
ALLOWED = {("conv.py", PRIVATE_CONV)}


def package_modules():
    """Parse every module of the package, as (file name, syntax tree) pairs"""
    package_dir = Path(complexnn.__file__).parent
    for path in sorted(package_dir.glob("*.py")):
        yield path.name, ast.parse(path.read_text(encoding="utf-8"))


def imported_modules(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            yield node.module


class TestPublicImports(unittest.TestCase):
    """The package imports TensorFlow and Keras through public modules only"""

    def test_no_private_imports(self):
        found = {
            (name, module)
            for name, tree in package_modules()
            for module in imported_modules(tree)
            if module.startswith(PRIVATE_PREFIXES)
        }
        self.assertEqual(found, ALLOWED)


class TestNoKerasShapeWrites(unittest.TestCase):
    """No layer sets the Keras 1 `_keras_shape` attribute, which Keras 2 and 3 ignore"""

    def test_no_keras_shape_writes(self):
        found = [
            (name, node.lineno)
            for name, tree in package_modules()
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and node.attr == "_keras_shape"
            and isinstance(node.ctx, ast.Store)
        ]
        self.assertEqual(found, [])


class TestWithoutPrivateConv(unittest.TestCase):
    """`import complexnn` works if TensorFlow drops its private copy of Keras 2"""

    def test_import_without_private_conv_module(self):
        # patch.dict restores sys.modules, and so the original complexnn, on exit.
        with mock.patch.dict(sys.modules):
            for name in list(sys.modules):
                if name == "complexnn" or name.startswith("complexnn."):
                    del sys.modules[name]
            # A None entry makes every import of the module raise ImportError.
            sys.modules[PRIVATE_CONV] = None
            package = importlib.import_module("complexnn")
            self.assertIsNot(package, complexnn)
            package.ComplexConv2D(2, 3)
            with self.assertRaisesRegex(ImportError, "WeightNorm_Conv"):
                package.WeightNorm_Conv(rank=2, filters=2, kernel_size=3)
        self.assertIs(sys.modules["complexnn"], complexnn)
