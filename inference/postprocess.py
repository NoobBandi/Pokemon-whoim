"""Post-processing: alpha mask restoration for clean Pokemon silhouettes."""

import numpy as np
from PIL import Image


def restore_alpha(stylized_rgb: Image.Image, original_alpha: Image.Image,
                  threshold: int = 128) -> Image.Image:
    """Re-apply the original alpha mask to the stylized RGB output.

    Args:
        stylized_rgb: The style-transferred RGB image
        original_alpha: The alpha mask from the original Pokemon image
        threshold: Alpha threshold for cleaner edges (0-255)

    Returns:
        RGBA image with restored transparency
    """
    alpha = original_alpha.resize(stylized_rgb.size, Image.BILINEAR)

    if threshold > 0:
        alpha_arr = np.array(alpha)
        alpha_arr = np.where(alpha_arr > threshold, 255, 0).astype(np.uint8)
        alpha = Image.fromarray(alpha_arr)

    result = stylized_rgb.convert("RGBA")
    result.putalpha(alpha)
    return result
