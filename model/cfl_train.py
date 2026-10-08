"""
Clustered Federated Learning (CFL) Training Script
Trains video anomaly detection with cluster-based aggregation
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
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

try:
    from model.config import get_config
    from model.ucf_dataset import UCFCrimeFrameDataset, ClusteredFLDataLoader, get_ucf_dataloaders, NUM_CLUSTERS
    from model.model import create_model
except ModuleNotFoundError:
    from config import get_config
    from ucf_dataset import UCFCrimeFrameDataset, ClusteredFLDataLoader, get_ucf_dataloaders, NUM_CLUSTERS
    from model import create_model


class FederatedAggregator:
    """Federated Learning Aggregation Strategies"""
    
    @staticmethod
    def fedavg(global_model: nn.Module, client_models: List[nn.Module], 
               client_weights: Optional[List[float]] = None) -> nn.Module:
        """
        FedAvg: Weighted average of client model parameters.
        
        Args:
            global_model: Global model to update
            client_models: List of trained client models
            client_weights: Optional weights for each client (e.g., by data size)
        """
        if len(client_models) == 0:
            return global_model
            
        if client_weights is None:
            client_weights = [1.0 / len(client_models)] * len(client_models)
        
        # Normalize weights
        total_weight = sum(client_weights)
        client_weights = [w / total_weight for w in client_weights]
        
        # Get global state dict
        global_state = global_model.state_dict()
        
        # Aggregate parameters
        for key in global_state.keys():
            global_state[key] = sum(
                client_weights[i] * client_models[i].state_dict()[key].float()
                for i in range(len(client_models))
            )
        
        global_model.load_state_dict(global_state)
        return global_model
    
    @staticmethod
    def cluster_aggregate(global_model: nn.Module, 
                         cluster_models: Dict[int, List[nn.Module]],
                         cluster_weights: Optional[Dict[int, float]] = None) -> nn.Module:
        """
        Clustered FL: First aggregate within clusters, then across clusters.
        
        Args:
            global_model: Global model to update
            cluster_models: Dict mapping cluster_id -> list of client models
            cluster_weights: Optional weights for each cluster
        """
        # First level: Aggregate within each cluster
        cluster_aggregated = {}
        for cluster_id, models in cluster_models.items():
            if len(models) > 0:
                cluster_model = copy.deepcopy(global_model)
                cluster_model = FederatedAggregator.fedavg(cluster_model, models)
                cluster_aggregated[cluster_id] = cluster_model
        
        # Second level: Aggregate cluster models
        if len(cluster_aggregated) > 0:
            if cluster_weights is None:
                cluster_weights = {cid: 1.0 / len(cluster_aggregated) 
                                  for cid in cluster_aggregated.keys()}
            
            # Normalize
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


class CFLTrainer:
    """Clustered Federated Learning Trainer"""
    
    def __init__(self, config, train_dataset: UCFCrimeFrameDataset, 
                 test_dataset: UCFCrimeFrameDataset, cfl_loader: ClusteredFLDataLoader):
        self.config = config
        self.device = torch.device(config.device)
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.cfl_loader = cfl_loader
        
        # Create global model
        self.global_model = create_model(config).to(self.device)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.history = {
            "round": [],
            "global_acc": [],
            "global_loss": [],
            "cluster_acc": defaultdict(list),
            "train_time": [],
            "test_precision": [],
            "test_recall": [],
            "test_f1": []
        }
        
        # CFL parameters
        self.num_clusters = NUM_CLUSTERS
        self.local_epochs = 3  # Local training epochs per client
        self.client_lr = 1e-4
        
    def train_client(self, client_id: int) -> Tuple[nn.Module, float, int]:
        """
        Train a single client locally.
        
        Returns:
            trained_model, loss, num_samples
        """
        # Create client model (copy of global)
        client_model = copy.deepcopy(self.global_model)
        client_model.train()
        
        optimizer = optim.Adam(client_model.parameters(), lr=self.client_lr)
        
        # Get client data
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
    def evaluate(self, model: nn.Module, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model on test set"""
        model.eval()
        
        all_preds = []
        all_labels = []
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
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        cm = confusion_matrix(all_labels, all_preds)
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "loss": total_loss / max(1, len(test_loader)),
            "confusion_matrix": cm.tolist(),
            "predictions": all_preds,
            "labels": all_labels
        }
    
    def train_round(self, round_num: int, 
                   client_fraction: float = 1.0) -> Dict[str, float]:
        """
        Execute one round of Clustered Federated Learning.
        
        Args:
            round_num: Current round number
            client_fraction: Fraction of clients to select per round
        """
        print(f"\n{'='*60}")
        print(f"Round {round_num}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # Group models by cluster
        cluster_models = defaultdict(list)
        cluster_weights = defaultdict(float)
        client_losses = []
        
        # Select clients
        num_clients = self.cfl_loader.num_clients
        selected_clients = np.random.choice(
            num_clients, 
            size=max(1, int(num_clients * client_fraction)),
            replace=False
        )
        
        print(f"Training {len(selected_clients)}/{num_clients} clients...")
        
        # Train selected clients
        for client_id in tqdm(selected_clients, desc="Client Training"):
            # Train client locally
            client_model, loss, num_samples = self.train_client(client_id)
            
            if num_samples > 0:
                cluster_id = self.cfl_loader.get_cluster_id(client_id)
                cluster_models[cluster_id].append(client_model)
                cluster_weights[cluster_id] += num_samples
                client_losses.append(loss)
        
        # Aggregate using CFL
        print("Aggregating with Clustered FL...")
        self.global_model = FederatedAggregator.cluster_aggregate(
            self.global_model, cluster_models, cluster_weights
        )
        
        # Evaluate
        test_loader = DataLoader(
            self.test_dataset, 
            batch_size=self.config.training.batch_size,
            shuffle=False
        )
        metrics = self.evaluate(self.global_model, test_loader)
        
        round_time = time.time() - start_time
        
        # Record history
        self.history["round"].append(round_num)
        self.history["global_acc"].append(metrics["accuracy"])
        self.history["global_loss"].append(metrics["loss"])
        self.history["test_precision"].append(metrics["precision"])
        self.history["test_recall"].append(metrics["recall"])
        self.history["test_f1"].append(metrics["f1"])
        self.history["train_time"].append(round_time)
        
        # Print round summary
        avg_client_loss = np.mean(client_losses) if client_losses else 0
        print(f"\nRound {round_num} Summary:")
        print(f"  Avg Client Loss: {avg_client_loss:.4f}")
        print(f"  Global Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-Score: {metrics['f1']:.4f}")
        print(f"  Time: {round_time:.1f}s")
        
        return metrics
    
    def save_checkpoint(self, round_num: int, metrics: Dict):
        """Save model checkpoint"""
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
        """Save training metrics to JSON"""
        metrics_dir = self.config.metrics_dir
        metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert defaultdicts to regular dicts for JSON serialization
        history_dict = {k: (dict(v) if isinstance(v, defaultdict) else v) 
                       for k, v in self.history.items()}
        
        metrics = {
            "history": history_dict,
            "num_clients": self.cfl_loader.num_clients,
            "num_clusters": self.num_clusters,
            "local_epochs": self.local_epochs,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(metrics_dir / "cfl_training_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\nMetrics saved to {metrics_dir / 'cfl_training_metrics.json'}")
    
    def train(self, num_rounds: int = 20, client_fraction: float = 0.8):
        """
        Full Clustered Federated Learning training.
        
        Args:
            num_rounds: Number of communication rounds
            client_fraction: Fraction of clients to sample each round
        """
        print("\n" + "="*70)
        print("Clustered Federated Learning - Video Anomaly Detection")
        print("="*70)
        print(f"Device: {self.device}")
        print(f"Rounds: {num_rounds}")
        print(f"Clients: {self.cfl_loader.num_clients}")
        print(f"Clusters: {self.num_clusters}")
        print(f"Local Epochs: {self.local_epochs}")
        print(f"Client Fraction: {client_fraction}")
        print(f"Train Samples: {len(self.train_dataset)}")
        print(f"Test Samples: {len(self.test_dataset)}")
        print("="*70)
        
        best_acc = 0.0
        best_round = 0
        
        for round_num in range(1, num_rounds + 1):
            metrics = self.train_round(round_num, client_fraction)
            
            # Track best
            if metrics["accuracy"] > best_acc:
                best_acc = metrics["accuracy"]
                best_round = round_num
                self.save_checkpoint(round_num, metrics)
            
            # Save periodic checkpoint
            if round_num % 5 == 0:
                self.save_checkpoint(round_num, metrics)
        
        # Save final metrics
        self.save_metrics()
        
        print("\n" + "="*70)
        print("Training Complete!")
        print("="*70)
        print(f"Best Accuracy: {best_acc:.4f} (Round {best_round})")
        
        return self.history


def main():
    """Main CFL training function"""
    config = get_config()
    
    print("Loading UCF Crime Dataset...")
    try:
        train_dataset, test_dataset, cfl_loader = get_ucf_dataloaders(config)
    except FileNotFoundError as e:
        print(f"\n⚠️ {e}")
        return
    
    print(f"\nDataset Statistics:")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    print(f"  CFL clients: {cfl_loader.num_clients}")
    
    # Create trainer
    trainer = CFLTrainer(config, train_dataset, test_dataset, cfl_loader)
    
    # Train with CFL
    history = trainer.train(
        num_rounds=20,
        client_fraction=0.8
    )
    
    print("\nCFL Training Complete!")
    
    # Final evaluation
    test_loader = DataLoader(test_dataset, batch_size=config.training.batch_size)
    final_metrics = trainer.evaluate(trainer.global_model, test_loader)
    
    print("\nFinal Test Metrics:")
    print(f"  Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"  Precision: {final_metrics['precision']:.4f}")
    print(f"  Recall: {final_metrics['recall']:.4f}")
    print(f"  F1-Score: {final_metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
