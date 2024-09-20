import logging
from typing import Optional, Tuple, Union

import torch

logger = logging.getLogger(__name__)


def compute_frequency_coordinates(
    shape: Tuple[int, ...], pixel_spacing: Optional[Tuple[float, ...]] = None
) -> Tuple[torch.Tensor]:
    """
    Compute frequency coordinates for a given shape and pixel spacing.

    Args:
        shape: Tuple of integers representing the shape of the input array.
        pixel_spacing: Tuple of floats representing the pixel spacing in each
        dimension.

    Returns:
        Tuple of torch.Tensor, each representing frequency coordinates for a
        dimension.
    """
    ndim = len(shape)

    if pixel_spacing is None:
        pixel_spacing = (1.0,) * ndim
    elif len(pixel_spacing) != ndim:
        raise ValueError(
            f"pixel_spacing must have {ndim} elements to match input shape"
        )

    coords = []

    for dim, (size, spacing) in enumerate(zip(shape, pixel_spacing)):
        if dim < (ndim - 1):
            # last dimension uses rfftfreq
            freq = torch.fft.fftfreq(size, d=spacing)
        else:
            freq = torch.fft.rfftfreq(size, d=spacing)
        coords.append(freq)

    frequency_grid = torch.meshgrid(*coords, indexing="ij")

    return frequency_grid


def compute_uniform_filter(
    shape: Tuple[int, ...],
    window_size: Union[int, Tuple[int, ...]],
    pixel_spacing: Optional[Tuple[float, ...]] = None,
) -> torch.Tensor:
    """compute uniform filter in Fourier domain

    H[freq] = sinc(w * freq)

    where 'freq' is the frequency coordinates (per cycle). No need to
    scale by 2π. At the moment, the kernel is circularly symmetric, so
    it behaves like a circular/spherical kernel.

    """
    ndim = len(shape)

    if isinstance(window_size, (float, int)):
        window_size = (float(window_size),) * ndim
    elif len(window_size) != ndim:
        raise ValueError(
            f"window_size must be a single int or a tuple of {ndim} ints"
        )

    freq_grids = compute_frequency_coordinates(shape, pixel_spacing)

    arg = torch.zeros(freq_grids[0].shape, dtype=torch.complex64)

    # scale each axis differently for given window_size
    for grid, w in zip(freq_grids, window_size):
        arg += (w * grid) ** 2

    arg = torch.sqrt(arg)
    uniform_filter = torch.sinc(arg)

    return uniform_filter


def compute_gaussian_filter(
    shape: Tuple[int, ...],
    sigmas: Union[float, Tuple[float, ...]],
    pixel_spacing: Optional[Tuple[float, ...]] = None,
) -> torch.Tensor:
    ndim = len(shape)

    if isinstance(sigmas, (float, int)):
        sigmas = (float(sigmas),) * ndim
    elif len(sigmas) != ndim:
        raise ValueError(
            f"sigma must be a single float or a tuple of {ndim} floats"
        )

    freq_grids = compute_frequency_coordinates(shape, pixel_spacing)

    arg = torch.zeros(freq_grids[0].shape, dtype=torch.complex64)

    for grid, sig in zip(freq_grids, sigmas):
        arg += sig**2 * (2 * torch.pi * grid) ** 2

    gaussian_filter = torch.exp(-arg / 2.0)

    return gaussian_filter
