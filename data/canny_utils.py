"""Canny edge extraction for ControlNet conditioning."""

import cv2
import numpy as np
from PIL import Image


def extract_canny_edges(
    pil_image: Image.Image,
    low_threshold: int = 100,
    high_threshold: int = 200,
    target_size: tuple[int, int] | None = None,
) -> Image.Image:
    """Extract Canny edge map from a PIL image.

    Args:
        pil_image: Input RGB image.
        low_threshold: Canny lower hysteresis threshold.
        high_threshold: Canny upper hysteresis threshold.
        target_size: Optional (W, H) to resize before edge detection.

    Returns:
        RGB PIL image of the Canny edge map (white edges on black).
    """
    if target_size:
        pil_image = pil_image.resize(target_size, Image.LANCZOS)

    np_image = np.array(pil_image)

    if np_image.ndim == 3:
        gray = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)
    else:
        gray = np_image

    edges = cv2.Canny(gray, low_threshold, high_threshold)
    edges_rgb = np.stack([edges, edges, edges], axis=-1)

    return Image.fromarray(edges_rgb)


def auto_canny_thresholds(gray_image: np.ndarray) -> tuple[int, int]:
    """Compute Canny thresholds using the sigma=0.33 heuristic.

    Useful for Pokemon artwork where edge density varies significantly.

    Args:
        gray_image: Grayscale image as numpy array.

    Returns:
        (low_threshold, high_threshold) tuple.
    """
    median = np.median(gray_image)
    sigma = 0.33
    low = int(max(0, (1.0 - sigma) * median))
    high = int(min(255, (1.0 + sigma) * median))
    return low, high
