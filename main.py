import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from model import DenoisingCNN

# 1. Device Setup (Leverage GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Active Compute Device: {device}")

# 2. Additive White Gaussian Noise (AWGN) Generator
def inject_noise(images, noise_factor=0.2):
    """Corrupts image tensors with normalized Gaussian noise."""
    noisy = images + noise_factor * torch.randn_like(images)
    return torch.clamp(noisy, 0.0, 1.0)

# 3. Data Pipeline Setup
transform = transforms.ToTensor()
print("Downloading / Verifying CIFAR-10 Dataset...")
train_data_full = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
test_data_full = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

# Subsetting dataset for efficient desktop training
train_subset = Subset(train_data_full, range(6000))
test_subset = Subset(test_data_full, range(100))

train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_subset, batch_size=1, shuffle=False)

# 4. Model Training Pipeline
model = DenoisingCNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 6
print(f"\nStarting CNN Autoencoder Training ({epochs} epochs)...")
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for clean_images, _ in train_loader:
        clean_images = clean_images.to(device)
        noisy_images = inject_noise(clean_images, noise_factor=0.2).to(device)

        optimizer.zero_grad()
        reconstructed = model(noisy_images)
        loss = criterion(reconstructed, clean_images)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * clean_images.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch [{epoch+1}/{epochs}] - Loss (MSE): {epoch_loss:.5f}")

# 5. Model Inference on Unseen Test Sample
model.eval()
clean_tensor, _ = next(iter(test_loader))
clean_tensor = clean_tensor.to(device)
noisy_tensor = inject_noise(clean_tensor, noise_factor=0.2).to(device)

with torch.no_grad():
    cnn_reconstructed_tensor = model(noisy_tensor)

# 6. Tensor to NumPy (H, W, C) Format Conversion
def tensor_to_cv2(tensor):
    img = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)

clean_np = tensor_to_cv2(clean_tensor)
noisy_np = tensor_to_cv2(noisy_tensor)
cnn_np = tensor_to_cv2(cnn_reconstructed_tensor)

# 7. Classical Baseline Filtering (OpenCV)
gaussian_np = cv2.GaussianBlur(noisy_np, (5, 5), sigmaX=1.0)
median_np = cv2.medianBlur(noisy_np, 3)
bilateral_np = cv2.bilateralFilter(noisy_np, d=5, sigmaColor=50, sigmaSpace=50)

# 8. Quantitative Benchmark Calculation
comparisons = {
    "Noisy Image": noisy_np,
    "Gaussian Filter": gaussian_np,
    "Median Filter": median_np,
    "Bilateral Filter": bilateral_np,
    "CNN Autoencoder": cnn_np
}

print("\n" + "="*50)
print(f"{'Method':<20} | {'PSNR (dB)':<12} | {'SSIM':<10}")
print("="*50)

for name, output_img in comparisons.items():
    psnr_score = compute_psnr(clean_np, output_img)
    ssim_score = compute_ssim(clean_np, output_img, channel_axis=2)
    print(f"{name:<20} | {psnr_score:<12.2f} | {ssim_score:<10.4f}")
print("="*50)

# 9. Plotting and Exporting Visual Results
fig, axes = plt.subplots(1, 6, figsize=(18, 3.5))
visual_sequence = [
    ("Original Clean", clean_np),
    ("Noisy Input", noisy_np),
    ("Gaussian Filter", gaussian_np),
    ("Median Filter", median_np),
    ("Bilateral Filter", bilateral_np),
    ("CNN Autoencoder", cnn_np)
]

for ax, (title, img) in zip(axes, visual_sequence):
    ax.imshow(img)
    ax.set_title(title, fontsize=10)
    ax.axis("off")

output_path = "denoising_benchmark_result.png"
plt.tight_layout()
plt.savefig(output_path, dpi=300)
print(f"\nVisual results exported successfully to: {output_path}")