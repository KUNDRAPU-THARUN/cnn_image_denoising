import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

from model import DenoisingCNN


def main():
    st.set_page_config(page_title="Image Denoising AI", layout="wide")
    st.title("Deep Learning-Based Image Denoising")
    st.caption("Compare a Gaussian filter baseline against a CNN autoencoder on noisy RGB images.")

    @st.cache_resource
    def load_model():
        model = DenoisingCNN()
        model.eval()
        return model

    def add_gaussian_noise(image_rgb, noise_level=0.2):
        sigma = noise_level * 255.0
        noise = np.random.normal(0, sigma, image_rgb.shape).astype(np.float32)
        noisy = image_rgb.astype(np.float32) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)

    def preprocess_image(image_rgb):
        tensor = torch.from_numpy(image_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        return tensor

    def tensor_to_image(tensor):
        image = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        image = np.clip(image, 0.0, 1.0)
        return (image * 255).astype(np.uint8)

    def compute_metrics(original, denoised):
        psnr = compute_psnr(original, denoised)
        ssim = compute_ssim(original, denoised, channel_axis=2)
        return psnr, ssim

    st.sidebar.header("Image Controls")
    uploaded_file = st.sidebar.file_uploader("Upload a clean image", type=["jpg", "jpeg", "png"])
    noise_level = st.sidebar.slider("Gaussian noise level", min_value=0.0, max_value=0.5, value=0.2, step=0.01)

    if uploaded_file is None:
        st.warning("Please upload an image from the sidebar to begin.")
        return

    clean_image = Image.open(uploaded_file).convert("RGB")
    clean_np = np.array(clean_image)
    noisy_np = add_gaussian_noise(clean_np, noise_level)

    baseline_gaussian = cv2.GaussianBlur(noisy_np, (5, 5), 1.0)

    model = load_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with torch.no_grad():
        noisy_tensor = preprocess_image(noisy_np).to(device)
        cnn_output_tensor = model(noisy_tensor)

    cnn_np = tensor_to_image(cnn_output_tensor)

    st.markdown("### Visual Comparison")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.image(clean_np, caption="Original Clean", use_container_width=True)
    with col2:
        st.image(noisy_np, caption=f"Noisy Input (σ={noise_level})", use_container_width=True)
    with col3:
        st.image(baseline_gaussian, caption="Gaussian Filter Output", use_container_width=True)
    with col4:
        st.image(cnn_np, caption="CNN Output", use_container_width=True)

    st.markdown("### Quantitative Metrics")
    psnr_gaussian, ssim_gaussian = compute_metrics(clean_np, baseline_gaussian)
    psnr_cnn, ssim_cnn = compute_metrics(clean_np, cnn_np)

    metric_col1, metric_col2 = st.columns(2)
    with metric_col1:
        st.info(f"**OpenCV Gaussian Filter**\n\nPSNR: {psnr_gaussian:.2f} dB\nSSIM: {ssim_gaussian:.4f}")
    with metric_col2:
        st.success(f"**CNN Autoencoder**\n\nPSNR: {psnr_cnn:.2f} dB\nSSIM: {ssim_cnn:.4f}")


if __name__ == "__main__":
    main()