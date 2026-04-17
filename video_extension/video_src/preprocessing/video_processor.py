import os
import cv2
import numpy as np

def extract_uniform_frames(video_path, num_frames=16, target_size=(1024, 1024)):
    """
    Reads a video and uniformly extracts `num_frames` across the video's duration.
    Resizes them to `target_size` (required by standard FFT image detector).
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0:
        cap.release()
        raise ValueError(f"Could not read frames from video: {video_path}")

    # Calculate uniform indices
    if total_frames >= num_frames:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    else:
        # If video has fewer frames than required, we pull all and pad later in dataset
        indices = np.arange(total_frames)

    frames = []
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # OpenCV reads in BGR, convert to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if target_size:
                frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_CUBIC)
            frames.append(frame)
        else:
            break

    cap.release()
    return frames
