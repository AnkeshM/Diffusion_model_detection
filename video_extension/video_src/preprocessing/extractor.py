import os
import glob
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys

# Load extraction dependency safely
ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(ROOT))
try:
    from src.utils.feature_extraction import extract_features
except ImportError:
    # Dummy fallback
    def extract_features(img_np): return np.zeros(147, dtype=np.float32)

from .video_processor import extract_uniform_frames

def process_single_video(args):
    video_path, out_path, num_frames, target_size = args
    if os.path.exists(out_path):
        return True # already processed
        
    try:
        frames = extract_uniform_frames(video_path, num_frames=num_frames, target_size=target_size)
        
        # Pad if short
        while len(frames) > 0 and len(frames) < num_frames:
            frames.append(frames[-1])
            
        if not frames:
            return False
            
        feats = [extract_features(f) for f in frames]
        # (num_frames, 147)
        seq_tensor = torch.from_numpy(np.array(feats, dtype=np.float32))
        
        # Save to disk
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        torch.save(seq_tensor, out_path)
        return True
    except Exception as e:
        print(f"Error extracting {video_path}: {e}")
        return False

def extract_all_videos(real_dir, fake_dir, out_dir, num_frames=16, target_size=(1024, 1024), max_workers=4):
    out_dir = Path(out_dir)
    tasks = []
    
    # helper for finding files
    def gather_tasks(input_folder, label_name):
        base_path = Path(input_folder)
        if not base_path.exists(): return
        
        for p in glob.glob(os.path.join(base_path, "**", "*.mp4"), recursive=True):
            rel_path = os.path.relpath(p, base_path)
            # Switch extension to .pt
            out_name = str(Path(rel_path).with_suffix('.pt'))
            
            # example: extracted_features/real/opensora/vid1.pt
            out_file = out_dir / label_name / out_name
            tasks.append((p, str(out_file), num_frames, target_size))
            
    gather_tasks(real_dir, "real")
    gather_tasks(fake_dir, "fake")
    
    if not tasks:
        print("[WARNING] No videos found to extract!")
        return
        
    print(f"[INFO] Beginning extraction of {len(tasks)} videos using {max_workers} processes...")
    
    success_count = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_video, t): t for t in tasks}
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Extracting Features"):
            if future.result():
                success_count += 1
                
    print(f"\n[SUCCESS] Extraction complete. {success_count}/{len(tasks)} processed successfully.")
