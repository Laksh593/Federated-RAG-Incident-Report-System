import cv2
import torch
import numpy as np
import os
import sys

# Add model path so we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.model import create_model
from model.config import get_config

def load_cfl_model(model_path=None):
    """
    Load the trained Clustered Federated Learning PyTorch model.
    """
    config = get_config()
    
    # Use cfl_latest.pt from checkpoints directory by default
    if model_path is None:
        checkpoint_file = getattr(config.model, 'checkpoint_name', 'cfl_latest.pt')
        model_path = os.path.join(config.checkpoints_dir, checkpoint_file)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading PyTorch Inference Model on {device} from {model_path}...")
    
    if not os.path.exists(model_path):
        available_checkpoints = os.listdir(config.checkpoints_dir) if os.path.exists(config.checkpoints_dir) else []
        raise FileNotFoundError(
            f"Trained model checkpoint not found at: {model_path}. "
            f"Available checkpoints in {config.checkpoints_dir}: {available_checkpoints}. "
            "Please ensure 'cfl_latest.pt' exists in the checkpoints directory."
        )
    
    model = create_model(config)
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint)
        else:
            raise ValueError(f"Unrecognized checkpoint format in {model_path}")
        print(f"[OK] Successfully loaded trained CFL model weights from {model_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load weights from checkpoint {model_path}: {e}")
        
    model = model.to(device)
    model.eval()
    return model, config, device

def extract_video_frames(video_path, config):
    """
    Extract perfectly spaced frames from a video to match the 3D CNN input.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video holds {total_frames} frames. Extracting {config.model.num_frames} frames...")
    
    # Calculate spacing to get 'num_frames' equally spaced frames
    if total_frames > config.model.num_frames:
        frame_indices = np.linspace(0, total_frames - 1, config.model.num_frames, dtype=int)
    else:
        # If video is too short, we'll duplicate frames at the end
        frame_indices = np.arange(config.model.num_frames)
        frame_indices[frame_indices >= total_frames] = total_frames - 1
        
    current_frame = 0
    extracted_count = 0
    
    while cap.isOpened() and extracted_count < config.model.num_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        if current_frame in frame_indices:
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize
            frame = cv2.resize(frame, (config.model.frame_width, config.model.frame_height))
            
            frames.append(frame)
            extracted_count += 1
            
        current_frame += 1
        
    cap.release()
    
    # Pad if we couldn't get enough frames
    while len(frames) < config.model.num_frames:
        if len(frames) > 0:
            frames.append(frames[-1].copy()) # copy last frame
        else:
            # Complete failure, black frames
            frames.append(np.zeros((config.model.frame_height, config.model.frame_width, 3), dtype=np.uint8))
            
    # Convert list of images [H,W,C] to a single tensor [C, T, H, W] for 3D CNN
    # T is number of frames, C is channels
    tensor_frames = np.stack(frames, axis=0) # shape: (T, H, W, C)
    
    # Normalize to 0-1
    tensor_frames = tensor_frames.astype(np.float32) / 255.0
    
    # Rearrange to [C, T, H, W]
    tensor_frames = np.transpose(tensor_frames, (3, 0, 1, 2))
    
    # Convert to torch tensor and add batch dimension -> [1, C, T, H, W]
    torch_tensor = torch.from_numpy(tensor_frames).unsqueeze(0)
    
    return torch_tensor

def predict_anomaly(model, video_path, config, device):
    """
    Run the video through the PyTorch model to classify the anomaly.
    Returns: label_string, confidence_score
    """
    video_tensor = extract_video_frames(video_path, config)
    
    if video_tensor is None:
        return "Unknown", 0.0
        
    video_tensor = video_tensor.to(device)
    
    try:
        with torch.no_grad():
            # Get logits
            logits, _, _ = model(video_tensor, return_features=False)
            
            # Usually num_classes = 2 (Normal=0, Anomaly=1)
            # In your config, default is 2 classes
            probs = torch.nn.functional.softmax(logits, dim=1)
            
            # Get prediction
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_class].item()
            
            # Binary anomaly detection (Class 0: Normal, Class 1: Anomaly)
            if pred_class == 1:
                predicted_label = "Anomaly"
            else:
                predicted_label = "Normal"
                
            print(f"[INFO] PyTorch Inference Complete: Binary Prediction={predicted_label} (Confidence: {confidence:.2f})")
            return predicted_label, confidence
            
    except Exception as e:
        print(f"[WARNING] Inference failed: {e}")
        return "Unknown Anomaly", 0.0

if __name__ == "__main__":
    # Test script run
    print("Testing Inference utility...")
    model, config, device = load_cfl_model()
    # Create fake mp4 for test
    with open("temp_test.mp4", "wb") as f:
        f.write(b"fake_video")
    # This will fail on extraction but test the pipeline
    res, conf = predict_anomaly(model, "temp_test.mp4", config, device)
    print(f"Test Result: {res} ({conf})")
    try:
        os.remove("temp_test.mp4")
    except:
        pass
