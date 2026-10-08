"""
UCF Crime Dataset Loader for Pre-extracted PNG Frames
Optimized for Clustered Federated Learning
"""

import os
import sys
import cv2
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from torch.utils.data import Dataset, DataLoader
import random
from collections import defaultdict
from PIL import Image


# UCF Crime Dataset categories
ANOMALY_CATEGORIES = [
    "Abuse", "Arrest", "Arson", "Assault", "Burglary",
    "Explosion", "Fighting", "RoadAccidents", "Robbery",
    "Shooting", "Shoplifting", "Stealing", "Vandalism"
]

NORMAL_CATEGORY = "NormalVideos"

# Category to cluster mapping for Clustered FL
CATEGORY_TO_CLUSTER = {
    # Cluster 0: Violence
    "Abuse": 0, "Assault": 0, "Fighting": 0, "Shooting": 0,
    # Cluster 1: Property crimes
    "Burglary": 1, "Robbery": 1, "Shoplifting": 1, "Stealing": 1,
    # Cluster 2: Destruction
    "Arson": 2, "Explosion": 2, "Vandalism": 2,
    # Cluster 3: Other
    "Arrest": 3, "RoadAccidents": 3,
    # Cluster 4: Normal
    "NormalVideos": 4
}

NUM_CLUSTERS = 5


class UCFCrimeFrameDataset(Dataset):
    """
    UCF Crime Dataset loader for pre-extracted PNG frames.
    Groups frames by video and creates video clips.
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = "Train",
        num_frames: int = 16,
        frame_height: int = 112,
        frame_width: int = 112,
        categories: Optional[List[str]] = None,
        max_videos_per_category: Optional[int] = None,
        frame_step: int = 10  # Sample every Nth frame
    ):
        """
        Initialize UCF Crime Frame Dataset.
        
        Args:
            root_dir: Root directory of UCF Crime Dataset
            split: "Train" or "Test"
            num_frames: Number of frames per video clip
            frame_height: Target frame height
            frame_width: Target frame width  
            categories: List of categories to include
            max_videos_per_category: Limit videos per category
            frame_step: Step between sampled frames
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.num_frames = num_frames
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.frame_step = frame_step
        
        self.split_dir = self.root_dir / split
        
        if categories is None:
            categories = ANOMALY_CATEGORIES + [NORMAL_CATEGORY]
        self.categories = categories
        
        # Data storage
        self.video_clips = []  # List of (frame_paths, label, cluster, category)
        
        self._load_frame_paths(max_videos_per_category)
        
        print(f"Loaded {len(self.video_clips)} video clips from {split}")
    
    def _load_frame_paths(self, max_videos_per_category: Optional[int]):
        """Load frame paths and group by video"""
        for category in self.categories:
            cat_dir = self.split_dir / category
            if not cat_dir.exists():
                continue
            
            # Group frames by video prefix
            frames_by_video = defaultdict(list)
            
            for frame_path in cat_dir.glob("*.png"):
                # Extract video name from frame filename
                # Format: VideoName_FrameNumber.png (e.g., Fighting002_x264_0.png)
                name_parts = frame_path.stem.rsplit('_', 1)
                if len(name_parts) == 2:
                    video_name = name_parts[0]
                    frames_by_video[video_name].append(frame_path)
            
            # Sort frames by frame number for each video
            for video_name, frames in frames_by_video.items():
                # Sort by frame number
                frames.sort(key=lambda p: int(p.stem.rsplit('_', 1)[1]))
                
                # Only use videos with enough frames
                if len(frames) >= self.num_frames * self.frame_step:
                    is_normal = category == NORMAL_CATEGORY
                    label = 0 if is_normal else 1
                    cluster = CATEGORY_TO_CLUSTER.get(category, 4)
                    
                    self.video_clips.append({
                        'frames': frames,
                        'label': label,
                        'cluster': cluster,
                        'category': category,
                        'video_name': video_name
                    })
            
            # Limit videos per category if specified
            if max_videos_per_category:
                category_clips = [c for c in self.video_clips if c['category'] == category]
                if len(category_clips) > max_videos_per_category:
                    # Keep only max_videos_per_category
                    to_remove = category_clips[max_videos_per_category:]
                    self.video_clips = [c for c in self.video_clips if c not in to_remove]
    
    def _load_frames(self, frame_paths: List[Path]) -> np.ndarray:
        """Load and preprocess frames"""
        frames = []
        
        # Sample frames uniformly
        total_frames = len(frame_paths)
        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        
        for idx in indices:
            frame_path = frame_paths[idx]
            
            try:
                # Load image
                img = Image.open(frame_path).convert('RGB')
                img = img.resize((self.frame_width, self.frame_height))
                frame = np.array(img, dtype=np.float32) / 255.0
            except Exception as e:
                # Black frame on error
                frame = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.float32)
            
            frames.append(frame)
        
        return np.stack(frames)  # (T, H, W, C)
    
    def __len__(self) -> int:
        return len(self.video_clips)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int, str]:
        """
        Get a video clip.
        
        Returns:
            video: Tensor of shape (C, T, H, W)
            label: Binary label (0=normal, 1=anomaly)
            cluster: Cluster assignment for CFL
            category: Category name
        """
        clip = self.video_clips[idx]
        
        # Load frames
        frames = self._load_frames(clip['frames'])
        
        # Convert to tensor: (T, H, W, C) -> (C, T, H, W)
        video = torch.from_numpy(frames).permute(3, 0, 1, 2)
        
        return video, clip['label'], clip['cluster'], clip['category']


class ClusteredFLDataLoader:
    """Data loader for Clustered Federated Learning"""
    
    def __init__(
        self,
        dataset: UCFCrimeFrameDataset,
        num_clients_per_cluster: int = 3,
        batch_size: int = 4,
        alpha: float = 0.5
    ):
        self.dataset = dataset
        self.num_clients_per_cluster = num_clients_per_cluster
        self.batch_size = batch_size
        self.alpha = alpha
        
        # Group data by clusters
        self.cluster_indices = defaultdict(list)
        for idx in range(len(dataset)):
            cluster = dataset.video_clips[idx]['cluster']
            self.cluster_indices[cluster].append(idx)
        
        # Create client partitions
        self.client_data = self._create_client_partitions()
        self.num_clients = len(self.client_data)
        
        print(f"Created {self.num_clients} clients across {NUM_CLUSTERS} clusters")
        
    def _create_client_partitions(self) -> Dict[int, Dict]:
        """Partition data among clients"""
        client_data = {}
        client_id = 0
        
        for cluster_id, indices in self.cluster_indices.items():
            if len(indices) == 0:
                continue
                
            n_samples = len(indices)
            n_clients = self.num_clients_per_cluster
            
            # Dirichlet distribution for non-IID
            proportions = np.random.dirichlet([self.alpha] * n_clients)
            proportions = (proportions * n_samples).astype(int)
            proportions[-1] = n_samples - proportions[:-1].sum()
            
            shuffled = indices.copy()
            random.shuffle(shuffled)
            
            start = 0
            for i in range(n_clients):
                end = start + proportions[i]
                if end > start:
                    client_data[client_id] = {
                        'indices': shuffled[start:end],
                        'cluster': cluster_id
                    }
                    client_id += 1
                start = end
        
        return client_data
    
    def get_client_loader(self, client_id: int) -> DataLoader:
        """Get DataLoader for a specific client"""
        if client_id not in self.client_data:
            raise ValueError(f"Invalid client_id: {client_id}")
        
        indices = self.client_data[client_id]['indices']
        subset = torch.utils.data.Subset(self.dataset, indices)
        
        return DataLoader(
            subset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0
        )
    
    def get_cluster_id(self, client_id: int) -> int:
        """Get cluster ID for a client"""
        return self.client_data[client_id]['cluster']


def get_ucf_dataloaders(
    config,
    dataset_path: Optional[str] = None
) -> Tuple[UCFCrimeFrameDataset, UCFCrimeFrameDataset, ClusteredFLDataLoader]:
    """Create UCF Crime dataloaders"""
    if dataset_path is None:
        dataset_path = os.environ.get("UCF_CRIME_DATASET_PATH", "")
    
    if not dataset_path or not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"UCF Crime dataset path '{dataset_path}' not found. "
            "Please set the UCF_CRIME_DATASET_PATH environment variable to run dataset operations."
        )
    
    train_dataset = UCFCrimeFrameDataset(
        root_dir=dataset_path,
        split="Train",
        num_frames=config.model.num_frames,
        frame_height=config.model.frame_height,
        frame_width=config.model.frame_width,
        max_videos_per_category=30  # Limit for faster training
    )
    
    test_dataset = UCFCrimeFrameDataset(
        root_dir=dataset_path,
        split="Test",
        num_frames=config.model.num_frames,
        frame_height=config.model.frame_height,
        frame_width=config.model.frame_width,
        max_videos_per_category=10
    )
    
    cfl_loader = ClusteredFLDataLoader(
        dataset=train_dataset,
        num_clients_per_cluster=3,
        batch_size=config.training.batch_size,
        alpha=0.5
    )
    
    return train_dataset, test_dataset, cfl_loader


if __name__ == "__main__":
    from config import get_config
    
    config = get_config()
    
    print("Loading UCF Crime Dataset (PNG Frames)...")
    train_dataset, test_dataset, cfl_loader = get_ucf_dataloaders(config)
    
    print(f"\nTrain clips: {len(train_dataset)}")
    print(f"Test clips: {len(test_dataset)}")
    print(f"CFL clients: {cfl_loader.num_clients}")
    
    # Test loading
    if len(train_dataset) > 0:
        video, label, cluster, category = train_dataset[0]
        print(f"\nSample video shape: {video.shape}")
        print(f"Label: {label}, Cluster: {cluster}, Category: {category}")
