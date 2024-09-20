import logging
from typing import Optional, Tuple, Union

import torch
from numpy import array
from torch.fft import irfftn, rfftn

from .config import set_device
from .filters import compute_gaussian_filter, compute_uniform_filter

logger = logging.getLogger("ibgest")


def iterative_background_estimate(
    image: Union[array, torch.Tensor],
    filter_widths: Union[float, int, Tuple[float, ...], Tuple[int, ...]],
    max_iter: int = 50,
    tolerance: float = 1e-2,
    pixel_spacing: Optional[Tuple[float, ...]] = None,
    device: Optional[str] = "auto",
    filter: Optional[str] = "uniform",
):
    """background estimation by iterative low-pass filtering and re-weighting

    two low-pass filters are implemented: Gaussian and uniform.

    'filter_widths' correspond to Gaussian sigma if filter is "Gaussian".
    If "uniform" filter is used, then it corresponds to window size of the
    averaging filter.

    General procedure:

    Given raw signal input signal, y.

    estimate = do low-pass filter on y

    for k in 1..max iteration:
        residual = y - estimate
        neg_mean = mean of residual < 0
        neg_std = std of residual < 0
        center = -neg_mean + 2 * neg_std
        weights = 1/(1 + exp(2 * (residual - center) / neg_std))
        estimate += weights * residual
        estimate = do low pass filter on estimate

        #check for convergence
        exit loop if tolerance achieved

    return estimate


    """
    input_shape = image.shape

    if not isinstance(image, torch.Tensor):
        image = torch.tensor(image)
        DEVICE = set_device(device)

    image = image.to(DEVICE)

    if filter == "Gaussian":
        low_pass_filter = compute_gaussian_filter(
            image.shape, filter_widths, pixel_spacing=pixel_spacing
        ).to(DEVICE)
    elif filter == "uniform":
        low_pass_filter = compute_gaussian_filter(
            image.shape, filter_widths
        ).to(DEVICE)

    ft_estimate = rfftn(image)
    torch.mul(ft_estimate, low_pass_filter, out=ft_estimate)
    estimate = irfftn(ft_estimate, s=input_shape)

    for i in range(max_iter):
        residuals = image - estimate
        neg_mean = torch.mean(residuals[residuals < 0])
        neg_std = torch.std(residuals[residuals < 0])
        center = -neg_mean + 2 * neg_std
        weights = 1.0 / (1.0 + torch.exp(2 * (residuals - center) / neg_std))

        # store previous estimate
        prev_estimate = estimate

        # reconstruct estimate with weight residuals to diminish peaks
        estimate = estimate + weights * residuals

        ft_estimate = rfftn(estimate)
        torch.mul(ft_estimate, low_pass_filter, out=ft_estimate)
        estimate = irfftn(ft_estimate)

        # compute convergence metric
        relative_change = torch.norm(estimate - prev_estimate) / torch.norm(
            prev_estimate
        )

        logger.debug(
            f"Iteration {i + 1:d}, relative change = {relative_change:.4E}"
        )

        if relative_change < tolerance:
            break

    if estimate.device != torch.device("cpu"):
        return estimate.cpu().numpy()
    else:
        return estimate.numpy()


if __name__ == "__main__":
    import pathlib

    import matplotlib.pyplot as plt
    import numpy as np
    import tifffile

    from .config import set_logging_level
    from .utils import line_profile

    # current_file = pathlib.Path(__file__).resolve()
    # root = current_file.parents[1]
    # data_path = root / "test" / "ecg.npz"

    set_logging_level("DEBUG")

    data = tifffile.imread(
        "/Users/delnatan/Library/CloudStorage/Box-Box/McNally/Fm1088 200 2x2_10 200ms_1_MMStack_Pos0.ome.tif"
    )[40]
    data = data.astype(np.float32)

    # data = np.load(data_path)["data"].astype(np.float32)

    bg = iterative_background_estimate(data, 4.0, max_iter=100)
    p1 = (44, 74)
    p2 = (310, 400)
    yp, xp = zip(*(p1, p2))
    prof1 = line_profile(data, p1, p2)
    prof2 = line_profile(bg, p1, p2)

    fig, ax = plt.subplots(ncols=2, nrows=2)
    ax[0, 0].imshow(data)
    ax[0, 0].plot(xp, yp, "r--")
    ax[0, 1].imshow(bg)
    ax[1, 0].plot(prof1, "k-")
    ax[1, 0].plot(prof2, "b--")
    ax[1, 1].imshow(data - bg)
    plt.show()
