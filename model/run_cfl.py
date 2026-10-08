"""
Run Clustered Federated Learning Training
Self-contained script without sklearn dependency
"""
import os
import sys
import json
import time
import copy
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
import numpy as np
from PIL import Image
import random


# ========================== METRICS (sklearn-free) ==========================
def compute_metrics(y_true: List, y_pred: List) -> Dict[str, float]:
    """Compute classification metrics without sklearn"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Accuracy
    accuracy = np.mean(y_true == y_pred)
    
    # Confusion matrix
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    # Precision, Recall, F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]]
    }


# ========================== CONFIG ==========================
from dataclasses import dataclass


@dataclass
class ModelConfig:
    input_channels: int = 3
    num_frames: int = 16
    frame_height: int = 112
    frame_width: int = 112
    feature_dim: int = 512
    num_classes: int = 2
    dropout_rate: float = 0.5
    num_segments: int = 32


@dataclass
class TrainingConfig:
    epochs: int = 50
    batch_size: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    scheduler_step: int = 10
    scheduler_gamma: float = 0.5
    patience: int = 10
    min_delta: float = 0.001
    save_every: int = 5


class Config:
    def __init__(self):
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.project_root = Path(__file__).parent.parent
        self.model_dir = self.project_root / "model"
        self.metrics_dir = self.model_dir / "metrics"
        self.checkpoints_dir = self.model_dir / "checkpoints"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"


# ========================== MODEL ==========================
class Conv3DBlock(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv3d(in_c, out_c, kernel_size=k, stride=s, padding=p)
        self.bn = nn.BatchNorm3d(out_c)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class C3DFeatureExtractor(nn.Module):
    def __init__(self, in_channels=3, feature_dim=512):
        super().__init__()
        self.conv1 = Conv3DBlock(in_channels, 64)
        self.pool1 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))
        self.conv2 = Conv3DBlock(64, 128)
        self.pool2 = nn.MaxPool3d((2, 2, 2), (2, 2, 2))
        self.conv3a = Conv3DBlock(128, 256)
        self.conv3b = Conv3DBlock(256, 256)
        self.pool3 = nn.MaxPool3d((2, 2, 2), (2, 2, 2))
        self.conv4a = Conv3DBlock(256, 512)
        self.conv4b = Conv3DBlock(512, 512)
        self.pool4 = nn.MaxPool3d((2, 2, 2), (2, 2, 2))
        self.conv5a = Conv3DBlock(512, 512)
        self.conv5b = Conv3DBlock(512, 512)
        self.pool5 = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512, feature_dim)
    
    def forward(self, x):
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.pool3(self.conv3b(self.conv3a(x)))
        x = self.pool4(self.conv4b(self.conv4a(x)))
        x = self.pool5(self.conv5b(self.conv5a(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)


class TemporalAttention(nn.Module):
    def __init__(self, feature_dim, hidden_dim=128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        scores = self.attention(x).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        weighted = torch.bmm(weights.unsqueeze(1), x).squeeze(1)
        return weighted, weights


class MILClassifier(nn.Module):
    def __init__(self, feature_dim=512, num_classes=2, dropout=0.5):
        super().__init__()
        self.attention = TemporalAttention(feature_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        pooled, weights = self.attention(x)
        logits = self.classifier(pooled)
        return logits, weights


class AnomalyDetectionModel(nn.Module):
    def __init__(self, in_channels=3, feature_dim=512, num_classes=2, dropout_rate=0.5):
        super().__init__()
        self.feature_extractor = C3DFeatureExtractor(in_channels, feature_dim)
        self.classifier = MILClassifier(feature_dim, num_classes, dropout_rate)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
    
    def forward(self, x, return_features=False):
        features = self.feature_extractor(x)
        features_seg = features.unsqueeze(1)
        logits, attention_weights = self.classifier(features_seg)
        if return_features:
            return logits, attention_weights, features
        return logits, attention_weights, None


# ========================== DATASET ==========================
ANOMALY_CATEGORIES = [
    "Abuse", "Arrest", "Arson", "Assault", "Burglary",
    "Explosion", "Fighting", "RoadAccidents", "Robbery",
    "Shooting", "Shoplifting", "Stealing", "Vandalism"
]
NORMAL_CATEGORY = "NormalVideos"
CATEGORY_TO_CLUSTER = {
    "Abuse": 0, "Assault": 0, "Fighting": 0, "Shooting": 0,
    "Burglary": 1, "Robbery": 1, "Shoplifting": 1, "Stealing": 1,
    "Arson": 2, "Explosion": 2, "Vandalism": 2,
    "Arrest": 3, "RoadAccidents": 3,
    "NormalVideos": 4
}
NUM_CLUSTERS = 5


class UCFCrimeFrameDataset(Dataset):
    def __init__(self, root_dir, split="Train", num_frames=16, 
                 frame_height=112, frame_width=112, max_videos_per_category=None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.num_frames = num_frames
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.split_dir = self.root_dir / split
        self.categories = ANOMALY_CATEGORIES + [NORMAL_CATEGORY]
        self.video_clips = []
        self._load_frame_paths(max_videos_per_category)
        print(f"Loaded {len(self.video_clips)} video clips from {split}")
    
    def _load_frame_paths(self, max_videos_per_category):
        for category in self.categories:
            cat_dir = self.split_dir / category
            if not cat_dir.exists():
                continue
            
            frames_by_video = defaultdict(list)
            for frame_path in cat_dir.glob("*.png"):
                name_parts = frame_path.stem.rsplit('_', 1)
                if len(name_parts) == 2:
                    video_name = name_parts[0]
                    frames_by_video[video_name].append(frame_path)
            
            count = 0
            for video_name, frames in frames_by_video.items():
                if max_videos_per_category and count >= max_videos_per_category:
                    break
                frames.sort(key=lambda p: int(p.stem.rsplit('_', 1)[1]))
                if len(frames) >= self.num_frames:
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
                    count += 1
    
    def _load_frames(self, frame_paths):
        frames = []
        total_frames = len(frame_paths)
        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        
        for idx in indices:
            frame_path = frame_paths[idx]
            try:
                img = Image.open(frame_path).convert('RGB')
                img = img.resize((self.frame_width, self.frame_height))
                frame = np.array(img, dtype=np.float32) / 255.0
            except Exception:
                frame = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.float32)
            frames.append(frame)
        
        return np.stack(frames)
    
    def __len__(self):
        return len(self.video_clips)
    
    def __getitem__(self, idx):
        clip = self.video_clips[idx]
        frames = self._load_frames(clip['frames'])
        video = torch.from_numpy(frames).permute(3, 0, 1, 2)
        return video, clip['label'], clip['cluster'], clip['category']


class ClusteredFLDataLoader:
    def __init__(self, dataset, num_clients_per_cluster=3, batch_size=4, alpha=0.5):
        self.dataset = dataset
        self.num_clients_per_cluster = num_clients_per_cluster
        self.batch_size = batch_size
        self.alpha = alpha
        
        self.cluster_indices = defaultdict(list)
        for idx in range(len(dataset)):
            cluster = dataset.video_clips[idx]['cluster']
            self.cluster_indices[cluster].append(idx)
        
        self.client_data = self._create_client_partitions()
        self.num_clients = len(self.client_data)
        print(f"Created {self.num_clients} clients across {NUM_CLUSTERS} clusters")
    
    def _create_client_partitions(self):
        client_data = {}
        client_id = 0
        
        for cluster_id, indices in self.cluster_indices.items():
            if len(indices) == 0:
                continue
            
            n_samples = len(indices)
            n_clients = self.num_clients_per_cluster
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
    
    def get_client_loader(self, client_id):
        if client_id not in self.client_data:
            raise ValueError(f"Invalid client_id: {client_id}")
        indices = self.client_data[client_id]['indices']
        subset = Subset(self.dataset, indices)
        return DataLoader(subset, batch_size=self.batch_size, shuffle=True, num_workers=0)
    
    def get_cluster_id(self, client_id):
        return self.client_data[client_id]['cluster']


# ========================== FEDERATED AGGREGATION ==========================
class FederatedAggregator:
    @staticmethod
    def fedavg(global_model, client_models, client_weights=None):
        if len(client_models) == 0:
            return global_model
        if client_weights is None:
            client_weights = [1.0 / len(client_models)] * len(client_models)
        
        total_weight = sum(client_weights)
        client_weights = [w / total_weight for w in client_weights]
        global_state = global_model.state_dict()
        
        for key in global_state.keys():
            global_state[key] = sum(
                client_weights[i] * client_models[i].state_dict()[key].float()
                for i in range(len(client_models))
            )
        
        global_model.load_state_dict(global_state)
        return global_model
    
    @staticmethod
    def cluster_aggregate(global_model, cluster_models, cluster_weights=None):
        cluster_aggregated = {}
        for cluster_id, models in cluster_models.items():
            if len(models) > 0:
                cluster_model = copy.deepcopy(global_model)
                cluster_model = FederatedAggregator.fedavg(cluster_model, models)
                cluster_aggregated[cluster_id] = cluster_model
        
        if len(cluster_aggregated) > 0:
            if cluster_weights is None:
                cluster_weights = {cid: 1.0 / len(cluster_aggregated) 
                                  for cid in cluster_aggregated.keys()}
            
            total = sum(cluster_weights.values())
            cluster_weights = {k: v / total for k, v in cluster_weights.items()}
            global_state = global_model.state_dict()
            
            for key in global_state.keys():
                global_state[key] = sum(
                    cluster_weights[cid] * cluster_aggregated[cid].state_dict()[key].float()
                    for cid in cluster_aggregated.keys()
                )
            
            global_model.load_state_dict(global_state)
        
        return global_model


# ========================== CFL TRAINER ==========================
class CFLTrainer:
    def __init__(self, config, train_dataset, test_dataset, cfl_loader):
        self.config = config
        self.device = torch.device(config.device)
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.cfl_loader = cfl_loader
        
        self.global_model = AnomalyDetectionModel(
            in_channels=config.model.input_channels,
            feature_dim=config.model.feature_dim,
            num_classes=config.model.num_classes,
            dropout_rate=config.model.dropout_rate
        ).to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.history = {
            "round": [], "global_acc": [], "global_loss": [],
            "train_time": [], "test_precision": [], "test_recall": [], "test_f1": []
        }
        self.local_epochs = 3
        self.client_lr = 1e-4
    
    def train_client(self, client_id):
        client_model = copy.deepcopy(self.global_model)
        client_model.train()
        optimizer = optim.Adam(client_model.parameters(), lr=self.client_lr)
        client_loader = self.cfl_loader.get_client_loader(client_id)
        num_samples = len(client_loader.dataset)
        
        if num_samples == 0:
            return client_model, 0.0, 0
        
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(self.local_epochs):
            for videos, labels, _, _ in client_loader:
                videos = videos.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                logits, _, _ = client_model(videos)
                loss = self.criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(client_model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / max(1, num_batches)
        return client_model, avg_loss, num_samples
    
    @torch.no_grad()
    def evaluate(self, model, test_loader):
        model.eval()
        all_preds, all_labels = [], []
        total_loss = 0.0
        
        for videos, labels, _, _ in test_loader:
            videos = videos.to(self.device)
            labels = labels.to(self.device)
            logits, _, _ = model(videos)
            loss = self.criterion(logits, labels)
            total_loss += loss.item()
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
        
        metrics = compute_metrics(all_labels, all_preds)
        metrics["loss"] = total_loss / max(1, len(test_loader))
        return metrics
    
    def train_round(self, round_num, client_fraction=1.0):
        print(f"\n{'='*60}")
        print(f"Round {round_num}")
        print(f"{'='*60}")
        
        start_time = time.time()
        cluster_models = defaultdict(list)
        cluster_weights = defaultdict(float)
        client_losses = []
        
        num_clients = self.cfl_loader.num_clients
        selected_clients = np.random.choice(
            num_clients, size=max(1, int(num_clients * client_fraction)), replace=False
        )
        
        print(f"Training {len(selected_clients)}/{num_clients} clients...")
        
        for client_id in tqdm(selected_clients, desc="Client Training"):
            client_model, loss, num_samples = self.train_client(client_id)
            if num_samples > 0:
                cluster_id = self.cfl_loader.get_cluster_id(client_id)
                cluster_models[cluster_id].append(client_model)
                cluster_weights[cluster_id] += num_samples
                client_losses.append(loss)
        
        print("Aggregating with Clustered FL...")
        self.global_model = FederatedAggregator.cluster_aggregate(
            self.global_model, cluster_models, cluster_weights
        )
        
        test_loader = DataLoader(self.test_dataset, batch_size=self.config.training.batch_size, shuffle=False)
        metrics = self.evaluate(self.global_model, test_loader)
        
        round_time = time.time() - start_time
        self.history["round"].append(round_num)
        self.history["global_acc"].append(metrics["accuracy"])
        self.history["global_loss"].append(metrics["loss"])
        self.history["test_precision"].append(metrics["precision"])
        self.history["test_recall"].append(metrics["recall"])
        self.history["test_f1"].append(metrics["f1"])
        self.history["train_time"].append(round_time)
        
        avg_client_loss = np.mean(client_losses) if client_losses else 0
        print(f"\nRound {round_num} Summary:")
        print(f"  Avg Client Loss: {avg_client_loss:.4f}")
        print(f"  Global Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-Score: {metrics['f1']:.4f}")
        print(f"  Time: {round_time:.1f}s")
        
        return metrics
    
    def save_checkpoint(self, round_num, metrics):
        checkpoint_dir = self.config.checkpoints_dir
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "round": round_num,
            "model_state_dict": self.global_model.state_dict(),
            "history": dict(self.history),
            "metrics": metrics
        }
        torch.save(checkpoint, checkpoint_dir / f"cfl_round_{round_num}.pt")
        torch.save(checkpoint, checkpoint_dir / "cfl_latest.pt")
    
    def save_metrics(self):
        metrics_dir = self.config.metrics_dir
        metrics_dir.mkdir(parents=True, exist_ok=True)
        metrics = {
            "history": self.history,
            "num_clients": self.cfl_loader.num_clients,
            "num_clusters": NUM_CLUSTERS,
            "local_epochs": self.local_epochs,
            "timestamp": datetime.now().isoformat()
        }
        with open(metrics_dir / "cfl_training_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics saved to {metrics_dir / 'cfl_training_metrics.json'}")
    
    def train(self, num_rounds=10, client_fraction=0.8):
        print("\n" + "="*70)
        print("Clustered Federated Learning - Video Anomaly Detection")
        print("="*70)
        print(f"Device: {self.device}")
        print(f"Rounds: {num_rounds}")
        print(f"Clients: {self.cfl_loader.num_clients}")
        print(f"Clusters: {NUM_CLUSTERS}")
        print(f"Local Epochs: {self.local_epochs}")
        print(f"Client Fraction: {client_fraction}")
        print(f"Train Samples: {len(self.train_dataset)}")
        print(f"Test Samples: {len(self.test_dataset)}")
        print("="*70)
        
        best_acc = 0.0
        best_round = 0
        
        for round_num in range(1, num_rounds + 1):
            metrics = self.train_round(round_num, client_fraction)
            if metrics["accuracy"] > best_acc:
                best_acc = metrics["accuracy"]
                best_round = round_num
                self.save_checkpoint(round_num, metrics)
            if round_num % 5 == 0:
                self.save_checkpoint(round_num, metrics)
        
        self.save_metrics()
        
        print("\n" + "="*70)
        print("Training Complete!")
        print("="*70)
        print(f"Best Accuracy: {best_acc:.4f} (Round {best_round})")
        
        return self.history


# ========================== MAIN ==========================
def main():
    print("="*70)
    print("Clustered Federated Learning - UCF Crime Dataset")
    print("="*70)
    
    config = Config()
    print(f"\nUsing device: {config.device}")
    
    dataset_path = os.environ.get("UCF_CRIME_DATASET_PATH", "")
    if not dataset_path or not os.path.exists(dataset_path):
        print(f"\n⚠️ UCF Crime dataset path not configured or not found: '{dataset_path}'")
        print("Set the UCF_CRIME_DATASET_PATH environment variable (e.g. in your .env) to run offline CFL training.")
        return
    
    print("\nLoading UCF Crime Dataset...")
    train_dataset = UCFCrimeFrameDataset(
        root_dir=dataset_path,
        split="Train",
        num_frames=config.model.num_frames,
        frame_height=config.model.frame_height,
        frame_width=config.model.frame_width,
        max_videos_per_category=20
    )
    
    test_dataset = UCFCrimeFrameDataset(
        root_dir=dataset_path,
        split="Test",
        num_frames=config.model.num_frames,
        frame_height=config.model.frame_height,
        frame_width=config.model.frame_width,
        max_videos_per_category=8
    )
    
    cfl_loader = ClusteredFLDataLoader(
        dataset=train_dataset,
        num_clients_per_cluster=3,
        batch_size=config.training.batch_size,
        alpha=0.5
    )
    
    print(f"\nDataset Statistics:")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    print(f"  CFL clients: {cfl_loader.num_clients}")
    
    trainer = CFLTrainer(config, train_dataset, test_dataset, cfl_loader)
    
    history = trainer.train(num_rounds=5, client_fraction=0.8)
    
    print("\nCFL Training Complete!")
    
    test_loader = DataLoader(test_dataset, batch_size=config.training.batch_size)
    final_metrics = trainer.evaluate(trainer.global_model, test_loader)
    
    print("\nFinal Test Metrics:")
    print(f"  Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"  Precision: {final_metrics['precision']:.4f}")
    print(f"  Recall: {final_metrics['recall']:.4f}")
    print(f"  F1-Score: {final_metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
