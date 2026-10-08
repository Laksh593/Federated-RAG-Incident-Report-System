"""
Visualize CFL Training Results
Generates plots and updates dashboard from training metrics
"""
import json
import os
from pathlib import Path
from datetime import datetime

# Make sure matplotlib uses Agg backend for headless environments
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def load_metrics(metrics_path: str) -> dict:
    """Load training metrics from JSON file"""
    with open(metrics_path, 'r') as f:
        return json.load(f)


def create_training_curves(metrics: dict, output_dir: Path):
    """Create loss and accuracy curves"""
    history = metrics['history']
    rounds = history['round']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Clustered Federated Learning - UCF Crime Dataset', fontsize=14, fontweight='bold')
    
    # Accuracy plot
    ax1 = axes[0, 0]
    ax1.plot(rounds, [acc * 100 for acc in history['global_acc']], 
             'b-o', linewidth=2, markersize=8, label='Global Accuracy')
    ax1.set_xlabel('Communication Round')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Global Model Accuracy')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 100])
    ax1.legend()
    
    # Loss plot
    ax2 = axes[0, 1]
    ax2.plot(rounds, history['global_loss'], 
             'r-s', linewidth=2, markersize=8, label='Test Loss')
    ax2.set_xlabel('Communication Round')
    ax2.set_ylabel('Loss')
    ax2.set_title('Global Model Loss')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Precision/Recall/F1 plot
    ax3 = axes[1, 0]
    ax3.plot(rounds, [p * 100 for p in history['test_precision']], 
             'g-^', linewidth=2, markersize=8, label='Precision')
    ax3.plot(rounds, [r * 100 for r in history['test_recall']], 
             'm-v', linewidth=2, markersize=8, label='Recall')
    ax3.plot(rounds, [f * 100 for f in history['test_f1']], 
             'c-d', linewidth=2, markersize=8, label='F1-Score')
    ax3.set_xlabel('Communication Round')
    ax3.set_ylabel('Metric (%)')
    ax3.set_title('Precision, Recall, F1-Score')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 110])
    ax3.legend()
    
    # Training time plot
    ax4 = axes[1, 1]
    ax4.bar(rounds, [t / 60 for t in history['train_time']], 
            color='steelblue', edgecolor='black', alpha=0.7)
    ax4.set_xlabel('Communication Round')
    ax4.set_ylabel('Training Time (minutes)')
    ax4.set_title('Training Time per Round')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    output_path = output_dir / 'cfl_training_curves.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    return output_path


def create_final_metrics_summary(metrics: dict, output_dir: Path):
    """Create summary visualization with final metrics"""
    history = metrics['history']
    
    # Get final metrics
    final_acc = history['global_acc'][-1] * 100
    final_precision = history['test_precision'][-1] * 100
    final_recall = history['test_recall'][-1] * 100
    final_f1 = history['test_f1'][-1] * 100
    final_loss = history['global_loss'][-1]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Metrics bar chart
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    metrics_values = [final_acc, final_precision, final_recall, final_f1]
    colors = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0']
    
    bars = ax.barh(metrics_names, metrics_values, color=colors, edgecolor='black', height=0.6)
    
    # Add value labels
    for bar, val in zip(bars, metrics_values):
        ax.text(val + 1, bar.get_y() + bar.get_height()/2, 
                f'{val:.1f}%', va='center', fontsize=12, fontweight='bold')
    
    ax.set_xlim([0, 110])
    ax.set_xlabel('Performance (%)', fontsize=12)
    ax.set_title('CFL Model Performance - UCF Crime Dataset\n'
                 f'(5 Rounds, {metrics["num_clients"]} Clients, {metrics["num_clusters"]} Clusters)',
                 fontsize=14, fontweight='bold')
    ax.axvline(x=90, color='gray', linestyle='--', alpha=0.5, label='90% threshold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add info box
    info_text = (f"Training Configuration:\n"
                 f"• Clients: {metrics['num_clients']}\n"
                 f"• Clusters: {metrics['num_clusters']}\n"
                 f"• Local Epochs: {metrics['local_epochs']}\n"
                 f"• Final Loss: {final_loss:.4f}")
    
    ax.text(0.98, 0.02, info_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    output_path = output_dir / 'cfl_final_metrics.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    return output_path


def create_cluster_visualization(output_dir: Path):
    """Create cluster distribution visualization"""
    # Cluster mapping
    clusters = {
        'Violence': ['Abuse', 'Assault', 'Fighting', 'Shooting'],
        'Property': ['Burglary', 'Robbery', 'Shoplifting', 'Stealing'],
        'Destruction': ['Arson', 'Explosion', 'Vandalism'],
        'Other': ['Arrest', 'RoadAccidents'],
        'Normal': ['NormalVideos']
    }
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = ['#f44336', '#2196F3', '#FF9800', '#9C27B0', '#4CAF50']
    cluster_names = list(clusters.keys())
    category_counts = [len(v) for v in clusters.values()]
    
    bars = ax.barh(cluster_names, category_counts, color=colors, edgecolor='black', height=0.6)
    
    # Add category labels
    for i, (cluster, categories) in enumerate(clusters.items()):
        ax.text(category_counts[i] + 0.1, i, 
                ', '.join(categories), va='center', fontsize=10)
    
    ax.set_xlabel('Number of Categories', fontsize=12)
    ax.set_title('Clustered Federated Learning - Category Distribution\n'
                 '(Groups for Non-IID Data Partitioning)', fontsize=14, fontweight='bold')
    ax.set_xlim([0, 8])
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    output_path = output_dir / 'cfl_cluster_distribution.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    return output_path


def generate_dashboard_data(metrics: dict, output_dir: Path):
    """Generate JSON data for dashboard integration"""
    history = metrics['history']
    
    dashboard_data = {
        "training_method": "Clustered Federated Learning",
        "dataset": "UCF Crime Dataset",
        "model": "C3D + MIL Classifier",
        "final_metrics": {
            "accuracy": round(history['global_acc'][-1] * 100, 2),
            "precision": round(history['test_precision'][-1] * 100, 2),
            "recall": round(history['test_recall'][-1] * 100, 2),
            "f1_score": round(history['test_f1'][-1] * 100, 2),
            "loss": round(history['global_loss'][-1], 4)
        },
        "training_config": {
            "rounds": len(history['round']),
            "clients": metrics['num_clients'],
            "clusters": metrics['num_clusters'],
            "local_epochs": metrics['local_epochs']
        },
        "history": {
            "rounds": history['round'],
            "accuracy": [round(a * 100, 2) for a in history['global_acc']],
            "loss": [round(l, 4) for l in history['global_loss']],
            "f1": [round(f * 100, 2) for f in history['test_f1']],
            "training_time_min": [round(t / 60, 1) for t in history['train_time']]
        },
        "cluster_names": ["Violence", "Property", "Destruction", "Other", "Normal"],
        "timestamp": datetime.now().isoformat()
    }
    
    output_path = output_dir / 'dashboard_data.json'
    with open(output_path, 'w') as f:
        json.dump(dashboard_data, f, indent=2)
    print(f"Saved: {output_path}")
    return output_path


def main():
    """Generate all visualizations"""
    print("="*60)
    print("CFL Training Visualization Generator")
    print("="*60)
    
    # Paths
    model_dir = Path(__file__).parent
    metrics_dir = model_dir / 'metrics'
    output_dir = model_dir / 'visualizations'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metrics
    metrics_path = metrics_dir / 'cfl_training_metrics.json'
    if not metrics_path.exists():
        print(f"Error: Metrics file not found at {metrics_path}")
        return
    
    print(f"\nLoading metrics from: {metrics_path}")
    metrics = load_metrics(metrics_path)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    curves_path = create_training_curves(metrics, output_dir)
    summary_path = create_final_metrics_summary(metrics, output_dir)
    cluster_path = create_cluster_visualization(output_dir)
    dashboard_path = generate_dashboard_data(metrics, output_dir)
    
    print("\n" + "="*60)
    print("Visualization Complete!")
    print("="*60)
    print(f"\nGenerated files:")
    print(f"  - {curves_path}")
    print(f"  - {summary_path}")
    print(f"  - {cluster_path}")
    print(f"  - {dashboard_path}")
    
    # Print final summary
    history = metrics['history']
    print(f"\n{'='*60}")
    print("Final CFL Training Results")
    print(f"{'='*60}")
    print(f"  Accuracy:  {history['global_acc'][-1]*100:.2f}%")
    print(f"  Precision: {history['test_precision'][-1]*100:.2f}%")
    print(f"  Recall:    {history['test_recall'][-1]*100:.2f}%")
    print(f"  F1-Score:  {history['test_f1'][-1]*100:.2f}%")
    print(f"  Clients:   {metrics['num_clients']}")
    print(f"  Clusters:  {metrics['num_clusters']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
