from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # Paths
    project_root: Path = Path(__file__).resolve().parent.parent
    image_dir: Path = field(init=False)
    style_image: Path = field(init=False)
    output_dir: Path = field(init=False)
    checkpoint_dir: Path = field(init=False)

    # Preprocessing
    image_size: int = 512
    imagenet_mean: tuple = (0.485, 0.456, 0.406)
    imagenet_std: tuple = (0.229, 0.224, 0.225)

    # Architecture
    alpha: float = 1.0  # 1.0 = full style transfer

    # Training
    batch_size: int = 8
    num_epochs: int = 30
    learning_rate: float = 1e-4
    style_weight: float = 10.0
    content_weight: float = 1.0

    # Hardware
    mixed_precision: bool = True

    def __post_init__(self):
        self.image_dir = self.project_root / "dataset" / "HybridShivam-Pokemon" / "assets" / "images"
        self.style_image = self.image_dir / "025.png"
        self.output_dir = self.project_root / "output" / "stylized"
        self.checkpoint_dir = self.project_root / "output" / "checkpoints"
