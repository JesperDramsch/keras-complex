import ast
import subprocess
import sys
import unittest
from pathlib import Path

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

    # Runs in a child process: importing complexnn a second time in this process
    # would leave Keras's global registry of serialisable classes pointing at
    # the throwaway copies, and break later tests that load models.
    # TensorFlow is imported before the module is blocked, because TensorFlow's
    # own import needs it. A None entry makes every later import of it fail.
    CODE = f"""
import sys
import tensorflow
sys.modules[{PRIVATE_CONV!r}] = None
import complexnn
complexnn.ComplexConv2D(2, 3)
try:
    complexnn.WeightNorm_Conv(rank=2, filters=2, kernel_size=3)
except ImportError as err:
    assert "WeightNorm_Conv" in str(err), err
else:
    raise SystemExit("WeightNorm_Conv did not raise ImportError")
"""

    def test_import_without_private_conv_module(self):
        repo_root = Path(complexnn.__file__).parent.parent
        result = subprocess.run(
            [sys.executable, "-c", self.CODE],
            capture_output=True,
            text=True,
            cwd=repo_root,
        )
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
