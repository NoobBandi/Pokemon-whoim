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
    sd_output_dir: Path = field(init=False)

    # Preprocessing
    image_size: int = 512
    imagenet_mean: tuple = (0.485, 0.456, 0.406)
    imagenet_std: tuple = (0.229, 0.224, 0.225)

    # Architecture
    alpha: float = 1.0  # 1.0 = full style transfer

    # Training (legacy AdaIN)
    batch_size: int = 8
    num_epochs: int = 30
    learning_rate: float = 1e-4
    style_weight: float = 10.0
    content_weight: float = 1.0

    # Hardware
    mixed_precision: bool = True

    # ---- Stable Diffusion Pipeline ----
    sd_model_id: str = "runwayml/stable-diffusion-v1-5"
    controlnet_model_id: str = "lllyasviel/control_v11p_sd15_canny"
    ip_adapter_repo: str = "h94/IP-Adapter"
    ip_adapter_weight: str = "ip-adapter_sd15.bin"
    vae_model_id: str = "stabilityai/sd-vae-ft-mse"

    # Generation parameters
    num_inference_steps: int = 25
    guidance_scale: float = 7.5
    controlnet_conditioning_scale: float = 0.8
    ip_adapter_scale: float = 0.7

    # Canny edge detection
    canny_low_threshold: int = 100
    canny_high_threshold: int = 200

    # Prompts
    prompt: str = (
        "a Pikachu, yellow creature with red cheeks, "
        "black ear tips, cartoon style, white background"
    )
    negative_prompt: str = (
        "lowres, bad anatomy, blurry, "
        "worst quality, low quality, deformed"
    )

    # Other
    use_fp16: bool = True
    seed: int | None = None

    def __post_init__(self):
        self.image_dir = self.project_root / "dataset" / "HybridShivam-Pokemon" / "assets" / "images"
        self.style_image = self.image_dir / "025.png"
        self.output_dir = self.project_root / "output" / "stylized"
        self.checkpoint_dir = self.project_root / "output" / "checkpoints"
        self.sd_output_dir = self.project_root / "output" / "sd_stylized"
