import ast
import unittest
from pathlib import Path

import complexnn

PRIVATE_PREFIXES = ("tensorflow.python", "keras.src", "tf_keras.src", "keras._tf_keras")

# WeightNorm_Conv is broken on Keras 2 and Keras 3 and still subclasses the
# Keras 2 convolution base until it is rebuilt on the public Layer class.
ALLOWED = {("conv.py", "tensorflow.python.keras.layers.convolutional")}


def imported_modules(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            yield node.module


class TestPublicImports(unittest.TestCase):
    """The package imports TensorFlow and Keras through public modules only"""

    def test_no_private_imports(self):
        package_dir = Path(complexnn.__file__).parent
        found = {
            (path.name, module)
            for path in sorted(package_dir.glob("*.py"))
            for module in imported_modules(path)
            if module.startswith(PRIVATE_PREFIXES)
        }
        self.assertEqual(found, ALLOWED)
