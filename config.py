from pathlib import Path
from typing import Optional

from pydantic import Field, ConfigDict
from pydantic_settings import BaseSettings

# Get the directory where this config file is located
config_dir = Path(__file__).parent

class Settings(BaseSettings):
    """Configuration settings for the 404-base-miner-mesh service."""
    
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"  # Ignore extra fields from environment
    )
    
    # API settings
    host: str = "0.0.0.0"
    port: int = 10006

    # GPU settings
    qwen_gpu: int = Field(default=0)
    trellis_gpu: int = Field(default=0)
    # Accept both 'dtype' and 'qwen_dtype' from environment
    dtype: str = Field(default="bf16")

    # Hugging Face settings
    hf_token: Optional[str] = Field(default=None)

    # Qwen Edit settings
    qwen_edit_base_model_path: str = Field(
        default="Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
    )
    qwen_edit_model_path: str = Field(
        default="Qwen/Qwen-Image-Edit-2511"
    )
    qwen_edit_lora_repo: str = Field(
        default="lightx2v/Qwen-Image-Edit-2511-Lightning"
    )
    qwen_edit_height: int = Field(default=1024)
    qwen_edit_width: int = Field(default=1024)
    num_inference_steps: int = Field(default=4)
    true_cfg_scale: float = Field(default=1.0)
    qwen_edit_prompt_path: Path = Field(
        default=config_dir.joinpath("qwen_edit_prompt.json")
    )
    
    # Enable/disable image editing
    enable_image_edit: bool = Field(default=True)


settings = Settings()

__all__ = ["Settings", "settings"]
