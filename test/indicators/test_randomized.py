import builtins

import numpy as np

from maxcorr import BackendType, RandomizedIndicator, SemanticsType
from maxcorr.indicators import Indicator
from test.indicators.test_indicator import TestIndicator


class TestRandomizedIndicator(TestIndicator):
    def indicators(
        self,
        backend: BackendType,
        semantics: SemanticsType,
        dim: tuple[int, int],
    ) -> list[Indicator]:
        return (
            [
                RandomizedIndicator(
                    backend=backend,
                    semantics=semantics,
                    functions=np.sin,
                ),
                RandomizedIndicator(
                    backend=backend,
                    semantics=semantics,
                    functions=np.cos,
                ),
            ]
            if dim == (1, 1)
            else []
        )

    @property
    def result_type(self) -> builtins.type:
        return RandomizedIndicator.Result
