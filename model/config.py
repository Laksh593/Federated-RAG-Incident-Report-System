"""
Configuration for Video Anomaly Detection Model
"""
from dataclasses import dataclass
from pathlib import Path
import os


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    input_channels: int = 3
    num_frames: int = 16
    frame_height: int = 112
    frame_width: int = 112
    feature_dim: int = 512
    num_classes: int = 2
    dropout_rate: float = 0.5
    num_segments: int = 32
    checkpoint_name: str = "cfl_latest.pt"


@dataclass
class TrainingConfig:
    """Training configuration"""
    epochs: int = 50
    batch_size: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    scheduler_step: int = 10
    scheduler_gamma: float = 0.5
    patience: int = 10
    min_delta: float = 0.001
    save_every: int = 5


@dataclass
class DataConfig:
    """Dataset configuration"""
    num_train_samples: int = 800
    num_val_samples: int = 100
    num_test_samples: int = 100
    anomaly_ratio: float = 0.3
    anomaly_types: tuple = ("fighting", "vandalism", "robbery", "explosion", "accident")
    use_augmentation: bool = True


@dataclass
class Config:
    """Main configuration class"""
    
    def __init__(self):
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.data = DataConfig()
        
        # Set up paths
        self.project_root = Path(__file__).parent.parent
        self.model_dir = self.project_root / "model"
        self.metrics_dir = self.model_dir / "metrics"
        self.checkpoints_dir = self.model_dir / "checkpoints"
        self.diagrams_dir = self.project_root / "diagrams"
        
        # Create directories
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        
        # Check device
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")


def get_config() -> Config:
    """Get configuration instance"""
    return Config()


if __name__ == "__main__":
    config = get_config()
    print(f"Project root: {config.project_root}")
    print(f"Device: {config.device}")
