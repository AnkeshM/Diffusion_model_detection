import click
import sys
from pathlib import Path
import os
import torch
import yaml

ROOT_DIR = Path(__file__).resolve().parent

def get_config():
    with open(ROOT_DIR / "config.yaml", "r") as f:
        return yaml.safe_load(f)

@click.group()
def cli():
    """Video Forensics Extension CLI"""
    pass

@cli.command()
def extract():
    """Extract spatial features from raw videos and save them to .pt sequences."""
    config = get_config()
    from video_src.preprocessing.extractor import extract_all_videos
    
    real_dir = ROOT_DIR / config['dataset']['real_dir']
    fake_dir = ROOT_DIR / config['dataset']['fake_dir']
    out_dir = ROOT_DIR / config['paths']['extracted_features_dir']
    
    extract_all_videos(
        real_dir, 
        fake_dir, 
        out_dir, 
        num_frames=config['dataset']['frames_per_video'],
        target_size=(config['dataset']['image_size'], config['dataset']['image_size']),
        max_workers=config['training'].get('workers', 4)
    )

@cli.command()
def train():
    """Train the Video Forensics Model end-to-end mapping Spatial to Temporal"""
    config = get_config()
    
    # Import locally to avoid issues during fast load
    from video_src.data.dataset import create_dataloaders
    from video_src.models.video_network import VideoDetectorNet
    from video_src.models.trainer import train_video_model
    
    features_dir = ROOT_DIR / config['paths']['extracted_features_dir']
    
    print("[INFO] Creating dataloaders...")
    train_loader, test_loader = create_dataloaders(
        features_dir=str(features_dir), 
        num_frames=config['dataset']['frames_per_video'],
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['workers']
    )
    
    if train_loader is None:
        return
    
    print("[INFO] Initializing Video Network...")
    model = VideoDetectorNet(
        hidden_dim=config['training']['temporal_hidden_size'],
        num_lstm_layers=config['training']['num_lstm_layers']
    )
    
    # Load pretrained spatial
    pretrained_path = ROOT_DIR / config['paths']['pretrained_spatial_model']
    if os.path.exists(pretrained_path):
        model.load_pretrained_spatial(pretrained_path)
    else:
        print(f"[WARNING] Pretrained spatial model not found at {pretrained_path}. Training from scratch.")
        
    print("[INFO] Starting Training Loop...")
    train_video_model(model, train_loader, test_loader, config)


@cli.command()
def evaluate():
    """Evaluate the trained Video Model on the existing datasets"""
    config = get_config()
    from video_src.data.dataset import create_dataloaders
    from video_src.models.video_network import VideoDetectorNet
    from video_src.evaluation.evaluator import evaluate_video_model
    
    features_dir = ROOT_DIR / config['paths']['extracted_features_dir']
    
    # Passing test_split=1.0 to put all in test loader
    _, test_loader = create_dataloaders(
        features_dir=str(features_dir), 
        num_frames=config['dataset']['frames_per_video'],
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['workers'],
        test_split=0.99 # almost all to test
    )
    
    model = VideoDetectorNet(
        hidden_dim=config['training']['temporal_hidden_size'],
        num_lstm_layers=config['training']['num_lstm_layers']
    )
    
    save_path = ROOT_DIR / config['paths']['save_model']
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location="cpu", weights_only=True))
        print(f"[INFO] Loaded trained extension from {save_path}")
    else:
        print(f"[ERROR] Trained Video Model not found at {save_path}. Please run train first.")
        return
        
    evaluate_video_model(model, test_loader, config)


@cli.command()
@click.argument('video_path')
def predict(video_path):
    """Predict label for a single video."""
    import numpy as np
    config = get_config()
    from video_src.preprocessing.video_processor import extract_uniform_frames
    from video_src.models.video_network import VideoDetectorNet
    
    # Need to load extraction script
    ROOT = Path(__file__).resolve().parent.parent
    sys.path.append(str(ROOT))
    try:
        from src.utils.feature_extraction import extract_features
    except:
        def extract_features(img_np): return np.zeros(147, dtype=np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = VideoDetectorNet(
        hidden_dim=config['training']['temporal_hidden_size'],
        num_lstm_layers=config['training']['num_lstm_layers']
    ).to(device)
    
    save_path = ROOT_DIR / config['paths']['save_model']
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location="cpu", weights_only=True))
    else:
        print("[ERROR] Model weights not found. Please train first.")
        return
        
    model.eval()
    
    print(f"\n[INFO] Extracting features from {video_path}...")
    frames = extract_uniform_frames(
        video_path, 
        num_frames=config['dataset']['frames_per_video'],
        target_size=(config['dataset']['image_size'], config['dataset']['image_size'])
    )
    
    # Pad if necessary
    while len(frames) > 0 and len(frames) < config['dataset']['frames_per_video']:
        frames.append(frames[-1])
        
    feats = []
    for f in frames:
        feats.append(extract_features(f))
        
    seq_tensor = torch.from_numpy(np.array(feats, dtype=np.float32)).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logit = model(seq_tensor)
        prob = torch.sigmoid(logit).item()
        prediction = "AI-GENERATED (FAKE)" if prob > 0.5 else "REAL"

    print("\n" + "="*40)
    print(f"Video File : {os.path.basename(video_path)}")
    print(f"Result     : {prediction}")
    print(f"Confidence : {prob if prob > 0.5 else (1-prob):.4%}")
    print("="*40 + "\n")

if __name__ == '__main__':
    cli()
