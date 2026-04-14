"""Pokemon image dataset for AdaIN style transfer."""

import os
from typing import Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from data.preprocessing import rgba_to_rgb_white_bg, extract_alpha


class PokemonImageDataset(Dataset):
    """Loads Pokemon images from the HybridShivam dataset.

    Handles RGBA→RGB compositing, alpha mask caching, and transforms.
    """

    def __init__(self, image_dir: str, transform: Optional[transforms.Compose] = None,
                 target_size: int = 512):
        self.image_dir = image_dir
        self.transform = transform
        self.target_size = target_size
        self.image_files = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith('.png')
        ])
        self._alpha_cache: dict[str, Image.Image] = {}

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        filename = self.image_files[idx]
        path = os.path.join(self.image_dir, filename)

        img = Image.open(path).convert("RGBA")

        if filename not in self._alpha_cache:
            self._alpha_cache[filename] = extract_alpha(img)

        rgb = rgba_to_rgb_white_bg(img)

        if self.transform:
            tensor = self.transform(rgb)
        else:
            tensor = transforms.functional.to_tensor(rgb)

        return tensor, filename

    def get_alpha(self, filename: str) -> Optional[Image.Image]:
        """Retrieve cached alpha mask for a filename."""
        return self._alpha_cache.get(filename)
