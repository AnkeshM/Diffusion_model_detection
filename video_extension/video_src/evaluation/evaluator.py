import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from tqdm import tqdm

def evaluate_video_model(model, dataloader, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    all_preds, all_labels, all_probs = [], [], []

    print("[EVAL] Validating model on test set...")
    with torch.no_grad():
        for features, labels in tqdm(dataloader, desc="Evaluating"):
            features = features.to(device)
            outputs = model(features)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            preds = (probs > 0.5).astype(int)
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy().flatten())

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    mcc = matthews_corrcoef(all_labels, all_preds)

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.5 # In case of single class present
        print("[WARNING] Only one class present in test set, ROC AUC not defined.")

    print("\n" + "="*40)
    print("      VIDEO EXTENSION RESULTS")
    print("="*40)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC_AUC:   {auc:.4f}")
    print(f"MCC:       {mcc:.4f}")
    print("="*40)

    # Save confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.title('Video Detection Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    out_dir = config['paths']['results_dir']
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, 'video_confusion_matrix.png'))
    plt.close()
    print(f"[EVAL] Saved confusion matrix to {out_dir}/video_confusion_matrix.png")
