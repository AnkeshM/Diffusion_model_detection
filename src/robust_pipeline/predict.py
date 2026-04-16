import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import robust pipeline components
from preprocessing import pipeline_preprocess_inference, add_real_world_distortions, load_image
from feature_extraction import extract_features_from_multiscale, cross_difference, fft_magnitude
from model import load_model_package

def visualize_fft_spectrum(img_array, output_path):
    """
    Renders the Cross-Difference FFT magnitude spectrum to visualize peaks.
    Generates an output plot for analysis.
    Takes standard Base Res (256x256) extracted from multi-scales.
    """
    # Assuming img_array is RGB
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    channels = ['Red', 'Green', 'Blue']
    
    for c in range(3):
        cd = cross_difference(img_array[:, :, c])
        F_mag = fft_magnitude(cd)
        
        # Display logarithmic scale for better visualization of peaks
        F_log = np.log(1 + F_mag)
        
        ax = axes[c]
        img_plot = ax.imshow(F_log, cmap='viridis')
        ax.set_title(f"Channel: {channels[c]} FFT Magnitude")
        ax.axis('off')
        fig.colorbar(img_plot, ax=ax, shrink=0.6)
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"[*] Visualized FFT spectrum saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Predict if an image is REAL or FAKE using robust Synthbuster.")
    parser.add_argument("image_path", type=str, help="Path to input image")
    parser.add_argument("--model", type=str, default=r"d:\IP_Proj\src\robust_pipeline\robust_synthbuster.pkl", help="Path to trained model")
    parser.add_argument("--visualize", action="store_true", help="Generate an FFT Visualization of the Image")
    parser.add_argument("--simulate_distortions", action="store_true", help="Force application of real-world distortions to the image locally prior to classifying (Noise, Blur, Resize)")
    args = parser.parse_args()
    
    image_path = Path(args.image_path)
    if not image_path.exists():
        print(f"[!] Error: Image not found at {args.image_path}")
        sys.exit(1)
        
    print(f"\n[*] Predicting image: {args.image_path}")
    
    try:
        # Load Bundled Model Package
        package = load_model_package(args.model)
        model = package["model"]
        scaler = package["scaler"]
        selector = package["selector"]
        optimal_thresh = package["optimal_threshold"]
        
        # Load image natively first so we can distort if needed
        img = load_image(str(image_path))
        
        if args.simulate_distortions:
            print("[*] Note: Forcing random distortions (blur/noise/compression/resizing) onto the image before prediction...")
            img = add_real_world_distortions(img)
            
        # Preprocessing (Multi-scale Extraction, Denoise Edge Enhancements)
        multiscales = pipeline_preprocess_inference(img)
        
        # Feature Extraction
        features = extract_features_from_multiscale(multiscales)
        
        # We need to reshape to 2D array representing 1 sample
        X_raw = features.reshape(1, -1)
        
        # Apply standard filtering
        X_sel = selector.transform(X_raw)
        X_infer = scaler.transform(X_sel)
        
        # Make Prediction utilizing optimal threshold
        # Label mapping normally 0: REAL, 1: FAKE from training loop
        probability = model.predict_proba(X_infer)[0, 1]  # Probability of FAKE
        prediction = 1 if probability >= optimal_thresh else 0
        
        class_label = "FAKE (Diffusion-Generated)" if prediction == 1 else "REAL"
        
        # If fake, confidence is probability. If real, confidence is 1 - probability.
        confidence = (probability if prediction == 1 else (1.0 - probability)) * 100
        
        print(f"\n[OPTIMAL CUTOFF] Verified threshold at {optimal_thresh:.3f}")
        print(f"[>] PREDICTION: {class_label}")
        
        if args.visualize:
            base_filename = image_path.stem
            plot_out = image_path.parent / f"{base_filename}_fft.png"
            
            # Predict using highest resolution base (256) scale extracted earlier
            visualize_fft_spectrum(multiscales[256], str(plot_out))
            
    except Exception as e:
        print(f"[!] Processing failed: {e}")
        sys.exit(1)
        
if __name__ == "__main__":
    main()
