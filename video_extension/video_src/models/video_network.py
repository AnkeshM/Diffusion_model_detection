import sys
from pathlib import Path
import torch
import torch.nn as nn

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(ROOT_DIR))

try:
    from src.models.network import DetectorNet
except ImportError as e:
    print(f"[VideoNetwork] Failed to import original DetectorNet. Exception: {e}")
    # dummy fallback just in case
    class DetectorNet(nn.Module):
        def __init__(self, input_dim=147, hidden_dim=128):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(input_dim, 1))
        def forward(self, x): return self.net(x)

class VideoDetectorNet(nn.Module):
    def __init__(self, hidden_dim=128, num_lstm_layers=1):
        super(VideoDetectorNet, self).__init__()
        
        # 1. Load the original model
        self.spatial_backbone = DetectorNet(input_dim=147)
        
        # 2. Modify it to drop the last classification layer so it returns embeddings
        # Looking at original `network.py`, DetectorNet.net is a Sequential model ending in nn.Linear(32, 1) usually.
        # However, to be robust, we intercept its input instead or just remove the last layer.
        
        # An easier, robust way that doesn't depend on the exact Layer IDs:
        # We know `DetectorNet` processes a single frame to a binary logit. Actually, if we just use the layer before the last one,
        # we get robust embeddings. Let's slice the Sequential module, dropping the last nn.Linear.
        modules = list(self.spatial_backbone.net.children())[:-1] 
        self.spatial_extractor = nn.Sequential(*modules)
        
        # To find out the output dim dynamically:
        dummy_input = torch.zeros(2, 147)
        with torch.no_grad():
            # Temporarily set to eval to avoid BatchNorm issues, though size 2 also prevents it.
            self.spatial_extractor.eval()
            dummy_out = self.spatial_extractor(dummy_input)
            self.spatial_extractor.train()
            spatial_out_dim = dummy_out.shape[1]
            
        # 3. Add Temporal Modeling (LSTM)
        self.lstm = nn.LSTM(
            input_size=spatial_out_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=0.2 if num_lstm_layers > 1 else 0.0
        )
        
        # 4. Final Classification Head for Video
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1) # Binary classification (Real/Fake)
        )

    def load_pretrained_spatial(self, checkpoint_path):
        """Loads weights exclusively into the spatial_backbone from old model."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        # Handle dict format vs direct mapping
        state_dict = checkpoint.get("state_dict", checkpoint)
        self.spatial_backbone.load_state_dict(state_dict)
        # Update the sliced sequence too
        modules = list(self.spatial_backbone.net.children())[:-1] 
        self.spatial_extractor = nn.Sequential(*modules)
        print(f"[VideoNetwork] Loaded pre-trained spatial weights from {checkpoint_path}")

    def freeze_spatial_layers(self, freeze=True):
        """Freezes the spatial embedding extractor to avoid catastrophic forgetting while training LSTM."""
        for param in self.spatial_extractor.parameters():
            param.requires_grad = not freeze

    def forward(self, x):
        # x shape: (Batch, Frames, FeatureDim) -> e.g. (B, 16, 147)
        B, F, D = x.shape
        
        # Flatten B and F to run through spatial extractor
        x_flat = x.view(B * F, D)
        
        # Spatial Embeddings -> (B*F, spatial_out_dim)
        embeddings_flat = self.spatial_extractor(x_flat)
        
        # Reshape back to sequence -> (Batch, Frames, spatial_out_dim)
        embeddings_seq = embeddings_flat.view(B, F, -1)
        
        # Pass through LSTM
        lstm_out, (h_n, c_n) = self.lstm(embeddings_seq)
        
        # Take the hidden state from the last time step representing the whole video
        # h_n shape is (num_layers, Batch, hidden_dim)
        video_representation = h_n[-1] 
        
        # Classify
        logits = self.classifier(video_representation)
        return logits
