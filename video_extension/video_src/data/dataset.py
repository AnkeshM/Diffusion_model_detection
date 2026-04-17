import os
import glob
import torch
from torch.utils.data import Dataset
from pathlib import Path

class VideoFeatureDataset(Dataset):
    def __init__(self, features_dir, num_frames=16):
        self.features_dir = Path(features_dir)
        self.num_frames = num_frames
        
        self.samples = []
        
        real_dir = self.features_dir / "real"
        fake_dir = self.features_dir / "fake"
        
        # Real -> label 0
        if real_dir.exists():
            for p in glob.glob(os.path.join(real_dir, "**", "*.pt"), recursive=True):
                self.samples.append((p, 0))
                
        # Fake -> label 1
        if fake_dir.exists():
            for p in glob.glob(os.path.join(fake_dir, "**", "*.pt"), recursive=True):
                self.samples.append((p, 1))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pt_path, label = self.samples[idx]
        
        try:
            # Read pre-extracted 147-dimensional sequences (Frames, 147)
            seq_tensor = torch.load(pt_path, map_location="cpu", weights_only=True)
            return seq_tensor, torch.tensor([label], dtype=torch.float32)

        except Exception as e:
            # Fallback for broken/dummy videos, returning zeroes
            print(f"Error loading tensor {pt_path}: {e}")
            return torch.zeros((self.num_frames, 147), dtype=torch.float32), torch.tensor([label], dtype=torch.float32)

def create_dataloaders(features_dir, num_frames, batch_size=4, num_workers=4, test_split=0.2):
    from torch.utils.data import random_split, DataLoader
    
    dataset = VideoFeatureDataset(features_dir, num_frames)
    if len(dataset) == 0:
         print("[WARNING] No features found. Have you run 'python cli.py extract'?")
         return None, None
    
    test_size = int(len(dataset) * test_split)
    train_size = len(dataset) - test_size
    train_ds, test_ds = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, test_loader
