import builtins

from maxcorr import (
    BackendType,
    DoubleKernelIndicator,
    Indicator,
    SemanticsType,
    SingleKernelIndicator,
)
from test.indicators.test_indicator import TestIndicator


class TestDoubleKernelIndicator(TestIndicator):
    def indicators(
        self,
        backend: BackendType,
        semantics: SemanticsType,
        dim: tuple[int, int],
    ) -> list[Indicator]:
        return [
            DoubleKernelIndicator(
                backend=backend,
                semantics=semantics,
                kernel_a=3,
                kernel_b=3,
                use_lstsq=False,
            ),
            DoubleKernelIndicator(
                backend=backend,
                semantics=semantics,
                kernel_a=3,
                kernel_b=1,
                use_lstsq=False,
            ),
            DoubleKernelIndicator(
                backend=backend,
                semantics=semantics,
                kernel_a=3,
                kernel_b=1,
                use_lstsq=True,
            ),
            DoubleKernelIndicator(
                backend=backend,
                semantics=semantics,
                kernel_a=1,
                kernel_b=3,
                use_lstsq=False,
            ),
            DoubleKernelIndicator(
                backend=backend,
                semantics=semantics,
                kernel_a=1,
                kernel_b=3,
                use_lstsq=True,
            ),
            DoubleKernelIndicator(
                backend=backend,
                semantics=semantics,
                kernel_a=1,
                kernel_b=1,
            ),
        ]

    @property
    def result_type(self) -> type:
        return DoubleKernelIndicator.Result


class TestSingleKernelIndicator(TestIndicator):
    def indicators(
        self,
        backend: BackendType,
        semantics: SemanticsType,
        dim: tuple[int, int],
    ) -> list[Indicator]:
        return [
            SingleKernelIndicator(
                backend=backend,
                semantics=semantics,
                kernel=3,
                use_lstsq=False,
            ),
            SingleKernelIndicator(
                backend=backend,
                semantics=semantics,
                kernel=3,
                use_lstsq=True,
            ),
            SingleKernelIndicator(
                backend=backend,
                semantics=semantics,
                kernel=1,
            ),
        ]

    @property
    def result_type(self) -> builtins.type:
        return SingleKernelIndicator.Result
