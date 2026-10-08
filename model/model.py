"""
Video Anomaly Detection Model
3D CNN Feature Extractor + MIL Classifier
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class Conv3DBlock(nn.Module):
    
    def __init__(self, in_c: int, out_c: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__()
        self.conv = nn.Conv3d(in_c, out_c, kernel_size=k, stride=s, padding=p)
        self.bn = nn.BatchNorm3d(out_c)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class C3DFeatureExtractor(nn.Module):
    """C3D-style 3D CNN Feature Extractor"""
    
    def __init__(self, in_channels: int = 3, feature_dim: int = 512):
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
    """Temporal attention for MIL"""
    
    def __init__(self, feature_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        scores = self.attention(x).squeeze(-1)
        weights = F.softmax(scores, dim=1)
        weighted = torch.bmm(weights.unsqueeze(1), x).squeeze(1)
        return weighted, weights


class MILClassifier(nn.Module):
    """Multiple Instance Learning Classifier"""
    
    def __init__(self, feature_dim: int = 512, num_classes: int = 2, dropout: float = 0.5):
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
    """Full Video Anomaly Detection Model"""
    
    def __init__(
        self,
        in_channels: int = 3,
        feature_dim: int = 512,
        num_classes: int = 2,
        num_segments: int = 32,
        dropout_rate: float = 0.5
    ):
        super().__init__()
        self.num_segments = num_segments
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
    
    def forward(self, x, return_features: bool = False):
        batch_size = x.size(0)
        features = self.feature_extractor(x)
        features_seg = features.unsqueeze(1)
        logits, attention_weights = self.classifier(features_seg)
        
        if return_features:
            return logits, attention_weights, features
        return logits, attention_weights, None
    
    def get_anomaly_score(self, x):
        logits, _, _ = self.forward(x)
        probs = F.softmax(logits, dim=1)
        return probs[:, 1]


def create_model(config) -> AnomalyDetectionModel:
    """Create model from config"""
    return AnomalyDetectionModel(
        in_channels=config.model.input_channels,
        feature_dim=config.model.feature_dim,
        num_classes=config.model.num_classes,
        num_segments=config.model.num_segments,
        dropout_rate=config.model.dropout_rate
    )


if __name__ == "__main__":
    from config import get_config
    
    config = get_config()
    model = create_model(config)
    print(model)
    
    x = torch.randn(2, 3, 16, 112, 112)
    logits, attention, features = model(x, return_features=True)
    
    print(f"Input shape: {x.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Features shape: {features.shape}")
    
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")
