import builtins
from typing import Any

from maxcorr.cuda_path_utils import setup_cuda_paths

setup_cuda_paths()
import tensorflow as tf

from maxcorr.backends import Backend, TensorflowBackend
from test.backends.test_backend import TestBackend


class TestTensorflowBackend(TestBackend):
    @property
    def backend(self) -> Backend:
        return TensorflowBackend()

    @property
    def type(self) -> builtins.type:
        return tf.Tensor

    def cast(self, v: list) -> Any:
        return tf.constant(v)
