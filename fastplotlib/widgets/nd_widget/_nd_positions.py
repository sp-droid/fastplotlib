import inspect
from typing import Literal, Callable, Any, Type
from warnings import warn

import numpy as np
from numpy.typing import ArrayLike

from ...utils import subsample_array, ArrayProtocol

from ...graphics import (
    ImageGraphic,
    LineGraphic,
    LineStack,
    LineCollection,
    ScatterGraphic,
)
from ._processor_base import NDProcessor


# TODO: Maybe get rid of n_display_dims in NDProcessor,
#  we will know the display dims automatically here from the last dim
#  so maybe we only need it for images?
class NDPositionsProcessor(NDProcessor):
    def __init__(
        self,
        data: ArrayProtocol,
        multi: bool = False,  # TODO: interpret [n - 2] dimension as n_lines or n_points
        display_window: int | float | None = 100,  # window for n_datapoints dim only
        n_slider_dims: int = 0,
    ):
        super().__init__(data=data)

        self._display_window = display_window

        self.multi = multi

        self.n_slider_dims = n_slider_dims

    def _validate_data(self, data: ArrayProtocol):
        # TODO: determine right validation shape etc.
        return data

    @property
    def display_window(self) -> int | float | None:
        """display window in the reference units for the n_datapoints dim"""
        return self._display_window

    @display_window.setter
    def display_window(self, dw: int | float | None):
        if dw is None:
            self._display_window = None

        elif not isinstance(dw, (int, float)):
            raise TypeError

        self._display_window = dw

    @property
    def multi(self) -> bool:
        return self._multi

    @multi.setter
    def multi(self, m: bool):
        if m and self.data.ndim < 3:
            # p is p-datapoints, n is how many lines to show simultaneously (for line collection/stack)
            raise ValueError(
                "ndim must be >= 3 for multi, shape must be [s1..., sn, n, p, 2 | 3]"
            )

        self._multi = m

    def _apply_window_functions(self, indices: tuple[int, ...]):
        """applies the window functions for each dimension specified"""
        # window size for each dim
        winds = self._window_sizes
        # window function for each dim
        funcs = self._window_funcs

        if winds is None or funcs is None:
            # no window funcs or window sizes, just slice data and return
            # clamp to max bounds
            indexer = list()
            for dim, i in enumerate(indices):
                i = min(self.shape[dim] - 1, i)
                indexer.append(i)

            return self.data[tuple(indexer)]

        # order in which window funcs are applied
        order = self._window_order

        if order is not None:
            # remove any entries in `window_order` where the specified dim
            # has a window function or window size specified as `None`
            # example:
            # window_sizes = (3, 2)
            # window_funcs = (np.mean, None)
            # order = (0, 1)
            # `1` is removed from the order since that window_func is `None`
            order = tuple(
                d for d in order if winds[d] is not None and funcs[d] is not None
            )
        else:
            # sequential order
            order = list()
            for d in range(self.n_slider_dims):
                if winds[d] is not None and funcs[d] is not None:
                    order.append(d)

        # the final indexer which will be used on the data array
        indexer = list()

        for dim_index, (i, w, f) in enumerate(zip(indices, winds, funcs)):
            # clamp i within the max bounds
            i = min(self.shape[dim_index] - 1, i)

            if (w is not None) and (f is not None):
                # specify slice window if both window size and function for this dim are not None
                hw = int((w - 1) / 2)  # half window

                # start index cannot be less than 0
                start = max(0, i - hw)

                # stop index cannot exceed the bounds of this dimension
                stop = min(self.shape[dim_index] - 1, i + hw)

                s = slice(start, stop, 1)
            else:
                s = slice(i, i + 1, 1)

            indexer.append(s)

        # apply indexer to slice data with the specified windows
        data_sliced = self.data[tuple(indexer)]

        # finally apply the window functions in the specified order
        for dim in order:
            f = funcs[dim]

            data_sliced = f(data_sliced, axis=dim, keepdims=True)

        return data_sliced

    def get(self, indices: tuple[Any, ...]):
        """
        slices through all slider dims and outputs an array that can be used to set graphic data

        Note that we do not use __getitem__ here since the index is a tuple specifying a single integer
        index for each dimension. Slices are not allowed, therefore __getitem__ is not suitable here.
        """
        # apply window funcs
        # this array should be of shape [n_datapoints, 2 | 3]
        window_output = self._apply_window_functions(indices[:-1]).squeeze()

        # TODO: window function on the `p` n_datapoints dimension

        if self.display_window is not None:
            dw = self.display_window

            # half window size
            hw = dw // 2

            # for now assume just a single index provided that indicates x axis value
            start = max(indices[-1] - hw, 0)
            stop = start + dw

            slices = [slice(start, stop)]

            if self.multi:
                # n - 2 dim is n_lines or n_scatters
                slices.insert(0, slice(None))

            return window_output[tuple(slices)]


class NDPositions:
    def __init__(
        self,
        data,
        graphic: Type[LineGraphic | LineCollection | LineStack | ScatterGraphic],
        multi: bool = False,
    ):
        if issubclass(graphic, LineCollection):
            multi = True

        self._processor = NDPositionsProcessor(data, multi=multi, display_window=100, n_slider_dims=2)
        self._indices = tuple([0] * (2 + 1))

        self._create_graphic(graphic)

    @property
    def processor(self) -> NDPositionsProcessor:
        return self._processor

    @property
    def graphic(
        self,
    ) -> (
        LineGraphic | LineCollection | LineStack | ScatterGraphic
    ):
        """LineStack or ImageGraphic for heatmaps"""
        return self._graphic

    @property
    def indices(self) -> tuple:
        return self._indices

    @indices.setter
    def indices(self, indices):
        data_slice = self.processor.get(indices)

        if isinstance(self.graphic, list):
            # list of scatter
            for i in range(len(self.graphic)):
                # data_slice shape is [n_scatters, n_datapoints, 2 | 3]
                # by using data_slice.shape[-1] it will auto-select if the data is only xy or has xyz
                self.graphic[i].data[:, : data_slice.shape[-1]] = data_slice[i]

        elif isinstance(self.graphic, (LineGraphic, ScatterGraphic)):
            self.graphic.data[:, : data_slice.shape[-1]] = data_slice

        elif isinstance(self.graphic, LineCollection):
            for i in range(len(self.graphic)):
                # data_slice shape is [n_lines, n_datapoints, 2 | 3]
                self.graphic[i].data[:, : data_slice.shape[-1]] = data_slice[i]

    def _create_graphic(
        self,
        graphic_cls: Type[LineGraphic | LineCollection | LineStack | ScatterGraphic],
    ):
        if self.processor.multi and issubclass(graphic_cls, ScatterGraphic):
            # make list of scatters
            self._graphic = list()
            data_slice = self.processor.get(self.indices)
            for d in data_slice:
                scatter = graphic_cls(d)
                self._graphic.append(scatter)

        else:
            data_slice = self.processor.get(self.indices)
            self._graphic = graphic_cls(data_slice)
