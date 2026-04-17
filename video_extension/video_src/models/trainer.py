import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
from sklearn.metrics import matthews_corrcoef

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_video_model(model, train_loader, test_loader, config):
    device = get_device()
    model = model.to(device)
    
    epochs = config['training']['epochs']
    lr = config['training']['learning_rate']
    freeze_spatial = config['training']['freeze_spatial']
    unfreeze_epoch = config['training'].get('unfreeze_epoch', 5)
    save_path = config['paths']['save_model']
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    if freeze_spatial:
        model.freeze_spatial_layers(freeze=True)
        print("[INFO] Spatial layers frozen for Phase 1.")
    else:
        model.freeze_spatial_layers(freeze=False)

    best_mcc = -1.0
    
    for epoch in range(epochs):
        # Phase 2: Unfreeze spatial layers for End-to-End finetuning
        if freeze_spatial and epoch == unfreeze_epoch:
            print(f"\n[INFO] Reached epoch {unfreeze_epoch}. Unfreezing spatial backbone for End-to-End finetuning.")
            model.freeze_spatial_layers(freeze=False)
            # Reduce learning rate when unfreezing for gentle finetuning
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr * 0.1
                
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for features, labels in pbar:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        # Evaluation
        if test_loader is not None:
            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for features, labels in test_loader:
                    features = features.to(device)
                    outputs = torch.sigmoid(model(features))
                    preds = (outputs > 0.5).int().cpu().numpy().flatten()
                    all_preds.extend(preds)
                    all_labels.extend(labels.numpy().flatten())
            
            epoch_mcc = matthews_corrcoef(all_labels, all_preds)
            print(f"  --> Val MCC: {epoch_mcc:.4f} | Avg Loss: {train_loss/len(train_loader):.4f}")
            
            if epoch_mcc > best_mcc:
                best_mcc = epoch_mcc
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(model.state_dict(), save_path)
                print(f"  --> Saved Best Model yielding {epoch_mcc:.4f} MCC")
        else:
            # If no testing set, just save the latest
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
    
    print("\n[SUCCESS] Video Network Training Complete.")
