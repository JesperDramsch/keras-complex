import ast
import unittest
from pathlib import Path

import complexnn

PRIVATE_PREFIXES = ("tensorflow.python", "keras.src", "tf_keras.src", "keras._tf_keras")

# WeightNorm_Conv is broken on Keras 2 and Keras 3 and still subclasses the
# Keras 2 convolution base until it is rebuilt on the public Layer class.
ALLOWED = {("conv.py", "tensorflow.python.keras.layers.convolutional")}


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
