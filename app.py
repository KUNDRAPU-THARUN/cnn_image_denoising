"""
DenoiseAI — multi-section Streamlit image restoration application.
Run: streamlit run app.py
Uses the existing model.py and unet_denoising_weights.pth.
"""
import io
import json
import zipfile
from html import escape
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image
from skimage import data
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from model import UNetDenoising

st.set_page_config(
    page_title="DenoiseAI | Restore image clarity",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
:root {
  color-scheme: dark;
  --navy: #0B1220; --panel: #17243B; --text: #F8FAFC;
  --muted: #CBD5E1; --cyan: #22D3EE; --indigo: #4F46E5; --violet: #7C3AED;
}
.stApp, [data-testid="stAppViewContainer"] {background: #0B1220; color: #F8FAFC;}
[data-testid="stHeader"] {background: rgba(11,18,32,.95);}
.block-container {max-width: 1280px; padding-top: 2rem; padding-bottom: 2rem;}
html, body, input, button, textarea {font-family: Inter, ui-sans-serif, system-ui, sans-serif;}
h1, h2, h3, h4, p, label, [data-testid="stMarkdownContainer"],
[data-testid="stWidgetLabel"], [data-testid="stMetricLabel"],
[data-testid="stMetricValue"] {color: #F8FAFC;}
h1, h2, h3 {letter-spacing: -.035em;}
[data-testid="stCaptionContainer"] {color: #CBD5E1;}
[data-testid="stSidebar"] {background: #111D31; border-right: 1px solid #334155;}
[data-testid="stSidebar"] p, [data-testid="stSidebar"] label {color: #F8FAFC;}
.st-key-navigation {
  background: #17243B; border: 1px solid #334155; border-radius: 18px;
  padding: .8rem 1rem; margin-bottom: 1.8rem;
}
.brand {font-size: 1.45rem; font-weight: 850; letter-spacing: -.05em; color: #F8FAFC;}
.brand span {color: #22D3EE;}
.brand small {display: block; font-size: .68rem; letter-spacing: .13em; color: #CBD5E1;}
div.stButton > button, div.stDownloadButton > button {
  background: #7C3AED; color: #F8FAFC; border: 1px solid #A78BFA;
  border-radius: 12px; min-height: 2.8rem; font-weight: 650;
  transition: background .18s, box-shadow .18s;
}
div.stButton > button p, div.stDownloadButton > button p {color: inherit !important;}
div.stButton > button[kind="primary"] {background: #4F46E5; border-color: #818CF8;}
div.stButton > button:hover, div.stDownloadButton > button:hover {
  background: #5B4DE8; border-color: #22D3EE; color: #FFFFFF;
  box-shadow: 0 0 0 3px rgba(34,211,238,.15);
}
button:focus-visible, input:focus-visible {outline: 3px solid #22D3EE !important; outline-offset: 3px;}
.st-key-navigation div.stButton > button {background: transparent; border-color: #64748B;}
.st-key-navigation div.stButton > button[kind="primary"] {
  background: #22D3EE; border-color: #22D3EE; color: #0B1220;
}
.st-key-navigation div.stButton > button:hover {background: #334155; color: #F8FAFC;}
.hero {
  background: radial-gradient(ellipse at 95% 10%,rgba(34,211,238,.15),transparent 50%),
    linear-gradient(120deg,#1D2450,#17243B 60%,#27204B);
  border: 1px solid #475569; border-radius: 26px; padding: clamp(1.5rem,4vw,3.5rem);
  margin: .5rem 0 1.4rem;
}
.hero h1 {font-size: clamp(2.1rem,5vw,4.4rem); line-height: 1.08; max-width: 900px; margin: .8rem 0 1rem;}
.hero-subtitle {color: #CBD5E1; font-size: 1.1rem; line-height: 1.75; max-width: 760px;}
.eyebrow, .section-label {color: #22D3EE; text-transform: uppercase; letter-spacing: .14em;
  font-size: .76rem; font-weight: 750; margin: .8rem 0;}
.feature-grid {display: grid; grid-template-columns: repeat(3,minmax(0,1fr)); gap: 1rem; margin: 1rem 0 2rem;}
.feature-card, .panel, .metric-card {
  background: #17243B; border: 1px solid #334155; border-radius: 18px; padding: 1.35rem;
}
.feature-card {transition: border-color .18s, transform .18s;}
.feature-card:hover {border-color: #22D3EE; transform: translateY(-3px);}
.feature-card h3 {font-size: 1.08rem; margin: .8rem 0 .5rem; letter-spacing: -.015em;}
.feature-card p {color: #CBD5E1; font-size: .95rem; line-height: 1.65; margin: 0;}
.card-icon {font-size: 1.6rem; background: #263456; border-radius: 12px;
  width: 3rem; height: 3rem; display: grid; place-items: center;}
.metric-label {color: #CBD5E1; font-size: .82rem;}
.metric-value {color: #F8FAFC; font-size: 1.7rem; font-weight: 800; margin: .35rem 0;}
.metric-note {color: #22D3EE; font-size: .8rem;}
.muted, .footer {color: #CBD5E1;}
.footer {border-top: 1px solid #334155; margin-top: 2.5rem; padding: 1.4rem 0; font-size: .85rem;}
[data-baseweb="select"] > div, [data-baseweb="input"] > div,
[data-baseweb="base-input"], [data-testid="stNumberInputContainer"] {
  background: #17243B !important; border-color: #64748B !important; color: #F8FAFC !important;
}
[data-baseweb="select"] span, [data-baseweb="select"] input,
[data-baseweb="input"] input, input, textarea {
  color: #F8FAFC !important; -webkit-text-fill-color: #F8FAFC !important;
}
[data-baseweb="select"] svg {fill: #22D3EE !important;}
[data-baseweb="popover"], [data-baseweb="popover"] ul,
[role="listbox"], [role="option"] {background: #17243B !important; color: #F8FAFC !important;}
[role="option"] *, [data-baseweb="menu"] * {color: #F8FAFC !important;}
[role="option"][aria-selected="true"], [role="option"]:hover {
  background: #354779 !important; color: #FFFFFF !important;
}
[data-testid="stRadio"] [role="radio"][aria-checked="true"] {background: #22D3EE !important;}
[data-testid="stRadio"] label {color: #F8FAFC !important;}
[data-baseweb="radio"] > div:first-child {background-color: #64748B !important;}
[data-baseweb="radio"]:has(input:checked) > div:first-child {background-color: #22D3EE !important;}
[data-baseweb="slider"] > div > div {background-color: #22D3EE;}
[data-testid="stSlider"] [role="slider"] {background: #22D3EE !important;}
[data-testid="stSlider"] [data-testid="stTickBar"] {color: #CBD5E1;}
[data-testid="stNumberInput"] button {background: #263456; color: #F8FAFC;}
[data-testid="stFileUploader"] section {background: #17243B; border: 1px dashed #64748B; border-radius: 12px;}
[data-testid="stFileUploader"] button {background: #4F46E5; color: #F8FAFC; border-color: #818CF8;}
[data-testid="stFileUploader"] button p {color: #F8FAFC;}
[data-testid="stTabs"] [role="tab"] {color: #CBD5E1; border-radius: 8px 8px 0 0; padding: .65rem 1rem;}
[data-testid="stTabs"] [role="tab"] p {color: inherit !important;}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {color: #22D3EE !important; background: #17243B;}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] {background: #22D3EE !important;}
[data-testid="stTabs"] [data-baseweb="tab-border"] {background: #334155;}
[data-testid="stAlert"][data-baseweb] {color: #F8FAFC;}
[data-testid="stAlertContainer"][data-baseweb="notification"] {border-radius: 12px;}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentSuccess"]) {border-left: 4px solid #10B981;}
[data-testid="stExpander"] {background: #17243B; border-color: #334155; border-radius: 14px;}
hr {border-color: #334155;}
@media (max-width: 800px) {
  .feature-grid {grid-template-columns: repeat(2,minmax(0,1fr));}
  .st-key-navigation [data-testid="stHorizontalBlock"] {flex-wrap: wrap;}
  .st-key-navigation [data-testid="stColumn"] {min-width: 42% !important; flex: 1 1 42% !important;}
}
@media (max-width: 520px) {
  .feature-grid {grid-template-columns: 1fr;}
  .block-container {padding-left: 1rem; padding-right: 1rem;}
  .hero {border-radius: 18px;}
  [data-testid="stTabs"] [role="tab"] {padding: .5rem; font-size: .85rem;}
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Dropdown menus are rendered in a portal outside the sidebar. Style the
# closed sidebar fields and the portal separately, including nested div text.
SELECTBOX_CSS = """
<style>
[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"],
[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] > div > div {
    background-color: #17243B !important;
    border-color: #64748B !important;
    border-radius: 10px;
}
[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] *,
[data-testid="stSidebar"] [data-testid="stSelectbox"] [role="combobox"] {
    color: #F8FAFC !important;
    -webkit-text-fill-color: #F8FAFC !important;
    opacity: 1;
}
[data-testid="stSidebar"] [data-testid="stSelectbox"] input {
    background-color: transparent !important;
    caret-color: #22D3EE;
}
[data-testid="stSidebar"] [data-testid="stSelectbox"] input::placeholder {
    color: #CBD5E1 !important;
    -webkit-text-fill-color: #CBD5E1 !important;
    opacity: 1;
}
[data-testid="stSidebar"] [data-baseweb="select"]:focus-within > div {
    border-color: #22D3EE !important;
    box-shadow: 0 0 0 2px rgba(34, 211, 238, .25);
}
[data-baseweb="popover"] [data-baseweb="menu"],
[data-baseweb="popover"] [role="listbox"],
[data-baseweb="popover"] [role="option"],
[data-testid="stSelectboxVirtualDropdown"] {
    background-color: #17243B !important;
    color: #F8FAFC !important;
}
[data-baseweb="popover"] [role="option"] *,
[data-baseweb="popover"] [data-baseweb="menu"] *,
[data-testid="stSelectboxVirtualDropdown"] * {
    color: #F8FAFC !important;
    -webkit-text-fill-color: #F8FAFC !important;
}
[data-baseweb="popover"] [role="option"]:hover,
[data-baseweb="popover"] [role="option"][aria-selected="true"],
[data-baseweb="popover"] [role="option"]:focus,
[data-testid="stSelectboxVirtualDropdown"] [role="option"]:hover,
[data-testid="stSelectboxVirtualDropdown"] [role="option"][aria-selected="true"] {
    background-color: #354779 !important;
    box-shadow: inset 3px 0 #22D3EE;
}
</style>
"""
st.markdown(SELECTBOX_CSS, unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading the U-Net model...")
def load_model():
    """Load the existing project model and weights, if available."""
    model = UNetDenoising()
    weights_path = Path(__file__).resolve().parent / "unet_denoising_weights.pth"

    if not weights_path.exists():
        return model.eval(), False, f"Weights not found: {weights_path.name}"

    try:
        try:
            checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
        except TypeError:  # Compatibility with older PyTorch versions
            checkpoint = torch.load(weights_path, map_location="cpu")

        # Support either a raw state_dict or a common checkpoint wrapper.
        if isinstance(checkpoint, dict):
            for key in ("state_dict", "model_state_dict", "model"):
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    checkpoint = checkpoint[key]
                    break

        # Handle DataParallel checkpoints.
        if isinstance(checkpoint, dict) and checkpoint and all(
            key.startswith("module.") for key in checkpoint.keys()
        ):
            checkpoint = {key.replace("module.", "", 1): value for key, value in checkpoint.items()}

        model.load_state_dict(checkpoint)
        return model.eval(), True, f"Loaded {weights_path.name}"
    except Exception as exc:
        return model.eval(), False, f"Could not load weights: {exc}"


@st.cache_data(show_spinner=False, max_entries=1)
def get_sample_images():
    """Use local scikit-image sample images; no external download is needed."""
    samples = {}
    try:
        samples["Astronaut"] = data.astronaut()
    except Exception:
        pass
    try:
        samples["Coffee"] = data.coffee()
    except Exception:
        pass
    try:
        samples["Chelsea the cat"] = data.chelsea()
    except Exception:
        pass
    return samples


def resize_rgb(image_rgb, size=256):
    """Resize to the fixed input size used by this demo's model pipeline."""
    return cv2.resize(image_rgb, (size, size), interpolation=cv2.INTER_AREA)


def add_noise(image_rgb, noise_type, severity, rng, *, mean=0.0,
              variance=None, salt_probability=0.5, density=1.0):
    """Return a noisy uint8 RGB image. severity is normalized from 0 to 1."""
    source = image_rgb.astype(np.float32) / 255.0

    if noise_type == "Gaussian":
        std = severity * 0.35 if variance is None else np.sqrt(variance)
        noise = rng.normal(mean, std, source.shape)
        noisy = source + noise
    elif noise_type == "Salt-and-Pepper":
        noisy = source.copy()
        amount = severity * 0.20
        random_map = rng.random(source.shape[:2])
        noisy[random_map < amount * (1.0 - salt_probability)] = 0.0
        noisy[random_map >= 1.0 - amount * salt_probability] = 1.0
    else:  # Speckle noise
        noise = rng.normal(0.0, severity * 0.45, source.shape)
        affected = rng.random(source.shape[:2]) < density
        noisy = source + source * noise * affected[:, :, None]

    return np.clip(noisy * 255.0, 0, 255).astype(np.uint8)


def apply_filter(noisy_rgb, filter_name, strength, *, kernel_size=None,
                 sigma_color=None, sigma_space=None):
    """Apply one of the supported traditional denoising baselines."""
    if filter_name == "Gaussian Filter":
        kernel = 3 if strength <= 0.25 else 5 if strength <= 0.60 else 7
        kernel = kernel_size or kernel
        sigma = 0.6 + strength * 2.0
        return cv2.GaussianBlur(noisy_rgb, (kernel, kernel), sigmaX=sigma)
    if filter_name == "Median Filter":
        kernel = 3 if strength <= 0.4 else 5 if strength <= 0.75 else 7
        kernel = kernel_size or kernel
        return cv2.medianBlur(noisy_rgb, kernel)
    # Bilateral filter preserves more edges; keep parameters moderate for speed.
    diameter = kernel_size or (5 if strength <= 0.5 else 7)
    sigma_color = sigma_color if sigma_color is not None else 25 + strength * 90
    sigma_space = sigma_space if sigma_space is not None else 25 + strength * 45
    return cv2.bilateralFilter(noisy_rgb, diameter, sigma_color, sigma_space)


def preprocess_image(image_rgb):
    """Convert uint8 HWC RGB into normalized NCHW float tensor."""
    return (
        torch.from_numpy(image_rgb.astype(np.float32) / 255.0)
        .permute(2, 0, 1)
        .unsqueeze(0)
    )


def tensor_to_image(tensor):
    """Convert the model's first output tensor to uint8 RGB."""
    if isinstance(tensor, (tuple, list)):
        tensor = tensor[0]
    if isinstance(tensor, dict):
        tensor = tensor.get("out", next(iter(tensor.values())))
    tensor = tensor.detach().float().cpu()
    if tensor.ndim == 4:
        tensor = tensor[0]
    if tensor.ndim == 3 and tensor.shape[0] in (1, 3, 4):
        tensor = tensor.permute(1, 2, 0)
    array = tensor.numpy()
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=2)
    array = np.clip(array, 0.0, 1.0)
    return (array[:, :, :3] * 255.0).round().astype(np.uint8)


def calculate_metrics(reference, candidate):
    """Calculate PSNR and SSIM against the clean reference image."""
    psnr = float("inf") if np.array_equal(reference, candidate) else peak_signal_noise_ratio(reference, candidate, data_range=255)
    ssim = structural_similarity(
        reference, candidate, channel_axis=2, data_range=255
    )
    return float(psnr), float(ssim)


def png_bytes(image_rgb):
    buffer = io.BytesIO()
    Image.fromarray(image_rgb).save(buffer, format="PNG")
    return buffer.getvalue()


def jpg_bytes(image_rgb):
    buffer = io.BytesIO()
    Image.fromarray(image_rgb).save(buffer, format="JPEG", quality=95, optimize=True)
    return buffer.getvalue()


def metric_report(reference, images, cnn_status):
    """One report feeds the live dashboard, charts, and CSV export."""
    rows = []
    for name, image in images.items():
        psnr, ssim = calculate_metrics(reference, image)
        rows.append({"Method": name, "PSNR (dB)": psnr, "SSIM": ssim,
                     "Status": cnn_status if name == "U-Net CNN" else "OK"})
    return pd.DataFrame(rows).set_index("Method")


def results_zip(images, report, settings):
    """Build the current experiment archive entirely in memory."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for filename, image in images.items():
            archive.writestr(filename, png_bytes(image))
        archive.writestr("metrics.csv", report.to_csv(index=True).encode("utf-8"))
        archive.writestr("experiment.json", json.dumps(settings, indent=2).encode("utf-8"))
    return buffer.getvalue()


def advanced_slider(label, key, minimum, maximum, default, step, help=None):
    """Keep conditional controls independent of widget cleanup and navigation."""
    saved = st.session_state.setdefault("advanced_settings", {})
    if key not in st.session_state and key in saved:
        st.session_state[key] = saved[key]
    value = st.sidebar.slider(label, min_value=minimum, max_value=maximum,
                              value=None if key in st.session_state else default,
                              step=step, key=key, help=help)
    saved[key] = value
    return value


def show_metric_card(label, value, note):
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-label">{label}</div>
          <div class="metric-value">{value}</div>
          <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )




PAGES = ("Home", "Features", "AI Studio", "About")
STUDIO_KEYS = (
    "studio_source", "studio_sample", "studio_noise", "studio_severity",
    "studio_seed", "studio_filter", "studio_strength",
)
st.session_state.setdefault("page", "Home")
st.session_state.setdefault("studio_settings", {})


def navigate(page):
    """Callbacks run before rendering, so navigation needs no extra rerun."""
    if page not in PAGES:
        return
    for key in STUDIO_KEYS:
        if key in st.session_state:
            st.session_state["studio_settings"][key] = st.session_state[key]
    st.session_state["page"] = page


def render_navigation():
    with st.container(key="navigation"):
        columns = st.columns([1.5, 1, 1, 1, 1], gap="small")
        with columns[0]:
            st.markdown(
                '<div class="brand">✦ Denoise<span>AI</span>'
                '<small>IMAGE RESTORATION LAB</small></div>',
                unsafe_allow_html=True,
            )
        for column, page in zip(columns[1:], PAGES):
            with column:
                st.button(
                    page, key=f"nav_{page}",
                    type="primary" if st.session_state["page"] == page else "secondary",
                    on_click=navigate, args=(page,), width="stretch",
                )


def render_cards(cards):
    """Static descriptive cards; actions remain native Streamlit widgets."""
    markup = "".join(
        f'<article class="feature-card"><div class="card-icon">{escape(icon)}</div>'
        f'<h3>{escape(title)}</h3><p>{escape(description)}</p></article>'
        for icon, title, description in cards
    )
    st.markdown(f'<div class="feature-grid">{markup}</div>', unsafe_allow_html=True)


FEATURES = [
    ("📤", "Upload Images", "Bring your own PNG, JPG, JPEG, or WebP clean reference image into the studio."),
    ("🖼️", "Built-in Sample Gallery", "Start immediately with the astronaut, coffee, or cat sample. No upload needed."),
    ("〰️", "Gaussian Noise", "Add random grain to an image and explore how noise strength changes the result."),
    ("◐", "Salt-and-Pepper Noise", "Simulate scattered black and white pixels, also called impulse noise."),
    ("✳️", "Speckle Noise", "Explore noise that changes with pixel brightness and creates a grainy texture."),
    ("🌫️", "Gaussian Filter", "Smooth nearby pixels to reduce grain. Strong smoothing may soften fine details."),
    ("▦", "Median Filter", "Replace pixels with a neighborhood median. Useful for scattered impulse noise."),
    ("💠", "Bilateral Filter", "Smooth similar colors while trying to keep stronger edges clear."),
    ("🧠", "U-Net CNN Denoising", "Run the project's trained U-Net to reconstruct a cleaner RGB image."),
    ("📊", "PSNR and SSIM Analytics", "Compare reconstruction error and structural similarity against the clean reference."),
    ("🔍", "Image Comparison", "Inspect the clean reference, noisy input, classical filter, and CNN result together."),
    ("📥", "Download Results", "Save CNN images as PNG or JPG, and download filtered and noisy images as PNG."),
]


def render_home():
    st.markdown("""
    <section class="hero">
      <div class="eyebrow">A clearer view, powered by computer vision</div>
      <h1>Restore Image<br><span style="color:#22D3EE">Clarity with AI</span></h1>
      <p class="hero-subtitle">Meet DenoiseAI, your interactive image restoration lab.
      Explore image noise, compare a U-Net neural network with classical filters,
      and see how every choice changes image quality.</p>
    </section>
    """, unsafe_allow_html=True)
    with st.container(horizontal=True, gap="medium"):
        st.button("Start Denoising →", type="primary", key="home_start",
                  on_click=navigate, args=("AI Studio",))
        st.button("Explore Features", key="home_features",
                  on_click=navigate, args=("Features",))
    st.caption("Built-in samples · Three noise models · Measurable results")
    st.markdown("## A complete image restoration workspace")
    st.write("From your first experiment to an export you can share, everything is in one studio.")
    render_cards([FEATURES[8], FEATURES[9], FEATURES[10], FEATURES[0], FEATURES[1], FEATURES[11]])
    st.markdown("## From noisy to clear in three steps")
    render_cards([
        ("01", "Choose an image", "Upload a clean reference or pick a built-in sample."),
        ("02", "Run your experiment", "Adjust noise and compare the U-Net with a classical filter."),
        ("03", "Explore and export", "Inspect images, review PSNR and SSIM, and download results."),
    ])
    st.info("An educational AI lab: restoration quality depends on the trained weights, image content, and noise settings.")


def render_features():
    st.markdown('<div class="eyebrow">Explore the toolkit</div>', unsafe_allow_html=True)
    st.title("Everything you need to explore denoising")
    st.write("Choose an image, simulate noise, compare restoration methods, and measure the results.")
    render_cards(FEATURES)
    st.info("The provided training script uses Gaussian noise on CIFAR-10 images. Salt-and-pepper, speckle, and larger images let you explore how the model behaves beyond those training conditions.")
    st.button("Try it in AI Studio →", type="primary", key="features_start",
              on_click=navigate, args=("AI Studio",))


def render_about():
    st.markdown('<div class="eyebrow">Behind the project</div>', unsafe_allow_html=True)
    st.title("Making image restoration easier to understand")
    st.markdown("### What is DenoiseAI?")
    st.write("DenoiseAI is an educational image restoration application. It brings a U-Net convolutional neural network and traditional image filters into a single interactive workspace.")
    st.markdown("### Problem Statement")
    st.write("Noise can hide useful details in an image. Removing it is challenging because strong smoothing can also remove edges and textures. This project explores that tradeoff through visual comparisons and quality measurements.")
    st.markdown("### Project Objectives")
    st.markdown("""
- Demonstrate how different kinds of noise affect images.
- Compare learned denoising with Gaussian, median, and bilateral filters.
- Measure restoration quality using a matching clean reference.
- Provide an accessible interface for experiments and image exports.
""")
    st.markdown("### Technologies Used")
    render_cards([
        ("🐍", "Python and Streamlit", "Python powers the application. Streamlit provides navigation, controls, charts, and downloads."),
        ("🧠", "PyTorch and U-Net", "PyTorch runs the neural network. U-Net combines image features with skip connections to preserve spatial detail."),
        ("🔬", "Image processing tools", "OpenCV handles filtering and resizing. NumPy, Pillow, scikit-image, and pandas support images and analytics."),
    ])
    st.markdown("### How the Application Works")
    st.markdown("""
1. Select a built-in sample or upload a clean reference image.
2. The studio converts the image to RGB and resizes it to 256 × 256 pixels.
3. Select a noise type, strength, and random seed.
4. Apply a classical filter and run the U-Net on the same noisy input.
5. Compare images and quality scores, then download the results.
""")
    st.caption("The existing training script uses 6,000 CIFAR-10 images, fixed Gaussian noise, MSE loss, and Adam for six epochs. The application loads unet_denoising_weights.pth.")
    st.markdown("### Understanding PSNR and SSIM")
    with st.container(border=True):
        st.markdown("**PSNR — Peak Signal-to-Noise Ratio**")
        st.write("PSNR measures reconstruction error against the clean reference. Higher values generally mean less error. An exact match has infinite PSNR.")
        st.markdown("**SSIM — Structural Similarity**")
        st.write("SSIM compares patterns such as structure, contrast, and brightness. Values closer to 1 generally mean greater similarity.")
        st.info("Both scores require a clean, aligned reference. They cannot establish quality for an unknown real-world noisy image on their own.")
    st.markdown("### Future Scope")
    st.markdown("""
- Train on more varied images, noise types, and severity levels.
- Evaluate all test images and report average scores.
- Preserve original resolution with patch-based processing.
- Add batch processing and a separate mode for already noisy images.
- Compare additional models and export experiment reports.
""")
    st.button("Open AI Studio →", type="primary", key="about_start",
              on_click=navigate, args=("AI Studio",))


def remember_upload():
    """Keep uploaded bytes across sections and honor removal in the uploader."""
    uploaded = st.session_state.get("studio_upload")
    if uploaded is None:
        st.session_state.pop("uploaded_image_bytes", None)
    else:
        st.session_state["uploaded_image_bytes"] = uploaded.getvalue()


def render_ai_studio():
    # Shadow settings survive Streamlit's cleanup of widgets on other sections.
    for key, value in st.session_state["studio_settings"].items():
        if key not in st.session_state:
            st.session_state[key] = value
    # --------------------------------- Sidebar -----------------------------------

    st.sidebar.markdown("## AI Studio controls")
    st.sidebar.caption("Configure your image experiment below.")
    st.sidebar.divider()

    st.sidebar.markdown("### 1 · Image selection")
    source_mode = st.sidebar.radio(
        "Image source", key="studio_source",
        options=["Built-in sample gallery", "Upload my image"],
        help="Sample images are included with scikit-image and need no download.",
    )

    samples = get_sample_images()
    if not samples:
        st.error("Could not load the built-in sample images. Please upload an image instead.")
        source_mode = "Upload my image"

    clean_rgb = None
    if source_mode == "Built-in sample gallery" and samples:
        sample_name = st.sidebar.selectbox("Choose a sample", list(samples.keys()), key="studio_sample")
        clean_rgb = resize_rgb(samples[sample_name])
    else:
        uploaded_file = st.sidebar.file_uploader(
            "Upload a clean reference image", type=["png", "jpg", "jpeg", "webp"],
            key="studio_upload", on_change=remember_upload
        )
        if uploaded_file is not None:
            st.session_state["uploaded_image_bytes"] = uploaded_file.getvalue()
        if st.session_state.get("uploaded_image_bytes") is not None:
            try:
                clean_rgb = resize_rgb(np.array(Image.open(io.BytesIO(st.session_state["uploaded_image_bytes"])).convert("RGB")))
            except Exception as exc:
                st.sidebar.error(f"Unable to read image: {exc}")

    st.sidebar.divider()
    st.sidebar.markdown("### 2 · Noise configuration")
    noise_type = st.sidebar.selectbox(
        "Noise type", key="studio_noise", options=["Gaussian", "Salt-and-Pepper", "Speckle"]
    )
    severity = st.sidebar.slider(
        "Noise severity", key="studio_severity", min_value=0.0, max_value=1.0, value=None if "studio_severity" in st.session_state else 0.25, step=0.05,
        help="Higher values add more noise.",
    )
    seed = st.sidebar.number_input("Random seed", key="studio_seed", min_value=0, max_value=99999, value=None if "studio_seed" in st.session_state else 42, step=1)

    noise_options = {}
    if noise_type == "Gaussian":
        st.sidebar.caption("Severity sets the standard deviation; variance scales its square. Values use a normalized 0–1 image range.")
        noise_options["mean"] = advanced_slider("Gaussian mean", "gaussian_mean", -0.2, 0.2, 0.0, 0.01)
        variance_scale = advanced_slider("Gaussian variance multiplier", "gaussian_variance", 0.0, 4.0, 1.0, 0.1)
        noise_options["variance"] = (float(severity) * 0.35) ** 2 * variance_scale
        st.sidebar.caption(f"Effective variance: {noise_options['variance']:.5f}")
    elif noise_type == "Salt-and-Pepper":
        noise_options["salt_probability"] = advanced_slider(
            "Salt share among corrupted pixels", "impulse_salt", 0.0, 1.0, 0.5, 0.05,
            "0 means all pepper (black); 1 means all salt (white). Severity controls the total corrupted fraction.")
    else:
        noise_options["density"] = advanced_slider(
            "Speckle density", "speckle_density", 0.0, 1.0, 1.0, 0.05,
            "Fraction of pixels exposed to multiplicative noise. Severity controls its strength.")

    st.sidebar.divider()
    st.sidebar.markdown("### 3 · Classical filtering")
    filter_name = st.sidebar.selectbox(
        "Filter", key="studio_filter", options=["Gaussian Filter", "Median Filter", "Bilateral Filter"]
    )
    filter_strength = st.sidebar.slider(
        "Filter strength", key="studio_strength", min_value=0.0, max_value=1.0, value=None if "studio_strength" in st.session_state else 0.5, step=0.05
    )
    filter_options = {"kernel_size": advanced_slider(
        "Filter kernel / neighborhood size", "filter_kernel", 3, 15, 5, 2,
        "Odd-sized neighborhood in pixels. Larger values smooth a wider area.")}
    if filter_name == "Bilateral Filter":
        filter_options["sigma_color"] = advanced_slider("Bilateral color sigma", "bilateral_color", 5, 150, 70, 5)
        filter_options["sigma_space"] = advanced_slider("Bilateral spatial sigma", "bilateral_space", 5, 150, 50, 5)
        st.sidebar.caption("The explicit sigma controls set bilateral strength.")
    elif filter_name == "Median Filter":
        st.sidebar.caption("Kernel size sets median smoothing strength.")
    else:
        st.sidebar.caption("Filter strength sets Gaussian sigma; kernel size sets the neighborhood.")

    st.sidebar.divider()
    st.sidebar.caption("Model: U-Net • Inference: CPU/GPU auto-detect")


    # -------------------------------- Main page ----------------------------------

    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">Computer vision · Image restoration</div>
          <h1 style="margin:0;">Your AI <span style="color:#22D3EE;">Studio</span></h1>
          <p class="hero-subtitle">
            Explore noise, compare classical image filters with a U-Net model,
            and evaluate restoration quality with PSNR and SSIM.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    top1, top2, top3 = st.columns(3)
    with top1:
        st.markdown('<div class="panel">🧪 <b>3 noise models</b><br><span class="muted">Gaussian · Impulse · Speckle</span></div>', unsafe_allow_html=True)
    with top2:
        st.markdown('<div class="panel">🛠️ <b>3 classical filters</b><br><span class="muted">Gaussian · Median · Bilateral</span></div>', unsafe_allow_html=True)
    with top3:
        st.markdown('<div class="panel">📈 <b>Quality analytics</b><br><span class="muted">PSNR · SSIM · downloads</span></div>', unsafe_allow_html=True)

    if clean_rgb is None:
        st.info("Choose a built-in sample in the sidebar or upload a clean reference image to begin.")
        st.stop()

    st.caption("Use the sidebar to select an image, configure noise, and choose a filter. On mobile, open the sidebar with the menu control at the top left.")
    rng = np.random.default_rng(int(seed))
    noisy_rgb = add_noise(clean_rgb, noise_type, float(severity), rng, **noise_options)
    traditional_rgb = apply_filter(noisy_rgb, filter_name, float(filter_strength), **filter_options)

    model, weights_loaded, model_message = load_model()
    if weights_loaded:
        st.sidebar.success(model_message)
        model_status = "Trained weights loaded"
    else:
        st.sidebar.warning(model_message)
        model_status = "Untrained fallback — CNN results may not be meaningful"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    with st.spinner("Running CNN denoising inference..."):
        try:
            with torch.inference_mode():
                prediction = model(preprocess_image(noisy_rgb).to(device))
            cnn_rgb = tensor_to_image(prediction)
            if cnn_rgb.shape[:2] != clean_rgb.shape[:2]:
                cnn_rgb = cv2.resize(cnn_rgb, (clean_rgb.shape[1], clean_rgb.shape[0]))
            cnn_error = None
        except Exception as exc:
            cnn_rgb = noisy_rgb.copy()
            cnn_error = str(exc)

    if cnn_error:
        st.error(
            "The CNN inference failed. The other filters and metrics can still be inspected. "
            f"Error: {cnn_error}"
        )

    cnn_status = "Inference failed; noisy fallback" if cnn_error else model_status
    report = metric_report(clean_rgb, {
        "Noisy input": noisy_rgb, filter_name: traditional_rgb, "U-Net CNN": cnn_rgb,
    }, cnn_status)
    experiment = {
        "source": source_mode, "noise_type": noise_type,
        "noise_severity": float(severity), "noise_parameters": noise_options,
        "random_seed": int(seed), "traditional_filter": filter_name,
        "filter_strength": float(filter_strength), "filter_parameters": filter_options,
        "model_status": cnn_status, "inference_device": str(device),
        "image_size": f"{clean_rgb.shape[1]} × {clean_rgb.shape[0]}",
    }
    st.markdown('<div class="section-label">Restoration workspace</div>', unsafe_allow_html=True)
    tabs = st.tabs(["🖼️ Image comparison", "📊 Metrics & analytics", "⬇️ Export results"])

    with tabs[0]:
        layout = st.radio("Comparison layout", ["Side by side", "Stacked"], horizontal=True, key="comparison_layout")
        images = [("01 · Clean reference", clean_rgb), ("02 · Noisy input", noisy_rgb),
                  (f"03 · {filter_name}", traditional_rgb), ("04 · U-Net CNN output", cnn_rgb)]
        st.caption("All views are 256 × 256 pixels. Choose stacked for a narrow screen.")
        for start in range(0, len(images), 2):
            containers = st.columns(2, gap="large") if layout == "Side by side" else [st.container(), st.container()]
            for container, (label, image) in zip(containers, images[start:start + 2]):
                with container:
                    with st.container(border=True):
                        st.markdown(f"**{label}**")
                        st.image(image, width="stretch")

    with tabs[1]:
        try:
            noisy_psnr, noisy_ssim = report.loc["Noisy input", ["PSNR (dB)", "SSIM"]]
            filter_psnr, filter_ssim = report.loc[filter_name, ["PSNR (dB)", "SSIM"]]
            cnn_psnr, cnn_ssim = report.loc["U-Net CNN", ["PSNR (dB)", "SSIM"]]

            m1, m2, m3 = st.columns(3)
            with m1:
                show_metric_card("Noisy input · PSNR", f"{noisy_psnr:.2f} dB", "Higher is better")
            with m2:
                show_metric_card(f"{filter_name} · PSNR", f"{filter_psnr:.2f} dB", "Higher is better")
            with m3:
                show_metric_card("U-Net CNN · PSNR", f"{cnn_psnr:.2f} dB", "Higher is better")

            s1, s2, s3 = st.columns(3)
            with s1:
                show_metric_card("Noisy input · SSIM", f"{noisy_ssim:.4f}", "Closer to 1 is better")
            with s2:
                show_metric_card(f"{filter_name} · SSIM", f"{filter_ssim:.4f}", "Closer to 1 is better")
            with s3:
                show_metric_card("U-Net CNN · SSIM", f"{cnn_ssim:.4f}", "Closer to 1 is better")

            chart_data = pd.DataFrame(
                {
                    "Method": ["Noisy input", filter_name, "U-Net CNN"],
                    "PSNR (dB)": [noisy_psnr, filter_psnr, cnn_psnr],
                    "SSIM": [noisy_ssim, filter_ssim, cnn_ssim],
                }
            ).set_index("Method")

            st.markdown("#### PSNR comparison")
            if np.isinf(noisy_psnr):
                st.info("The input matches the clean reference exactly: PSNR is infinite. Infinite values are omitted from the chart.")
            st.bar_chart(chart_data[["PSNR (dB)"]].replace([np.inf, -np.inf], np.nan), color="#22D3EE", width="stretch")
            st.markdown("#### SSIM comparison")
            st.bar_chart(chart_data[["SSIM"]], color="#7C3AED", width="stretch")
            st.dataframe(chart_data.style.format({"PSNR (dB)": "{:.2f}", "SSIM": "{:.4f}"}), width="stretch",)

            st.caption(
                "Metrics compare each result against the selected clean reference. "
                "They are meaningful only when that reference is actually clean and aligned "
                "with the processed image. An untrained fallback model is not a valid trained-CNN benchmark."
            )
        except Exception as exc:
            st.error(f"Could not calculate image metrics: {exc}")

    with tabs[2]:
        st.markdown("#### Export complete experiment")
        st.download_button(
            "Download results ZIP", data=results_zip({
                "clean_reference.png": clean_rgb, "noisy.png": noisy_rgb,
                "filtered.png": traditional_rgb, "unet_output.png": cnn_rgb,
            }, report, experiment), file_name="denoiseai_results.zip",
            mime="application/zip", type="primary", width="stretch",
        )
        st.caption("Includes four PNG images, metrics.csv, and experiment.json with settings and model status.")
        st.markdown("#### Download processed images")
        d1, d2 = st.columns(2)
        with d1:
            st.download_button(
                "Download CNN output · PNG",
                data=png_bytes(cnn_rgb),
                file_name="denoised_cnn_output.png",
                mime="image/png",
                width="stretch",
            )
            st.download_button(
                "Download CNN output · JPG",
                data=jpg_bytes(cnn_rgb),
                file_name="denoised_cnn_output.jpg",
                mime="image/jpeg",
                width="stretch",
            )
        with d2:
            st.download_button(
                f"Download {filter_name} · PNG",
                data=png_bytes(traditional_rgb),
                file_name="denoised_traditional_filter.png",
                mime="image/png",
                width="stretch",
            )
            st.download_button(
                "Download noisy input · PNG",
                data=png_bytes(noisy_rgb),
                file_name="noisy_input.png",
                mime="image/png",
                width="stretch",
            )

        st.markdown("#### Current experiment")
        st.json(experiment)

    st.markdown("### Live quality dashboard")
    st.caption("Updates with your controls. Scores compare each output to the same clean reference. Higher PSNR and SSIM closer to 1 are better.")
    for column, method in zip(st.columns(2), [filter_name, "U-Net CNN"]):
        with column:
            with st.container(border=True):
                st.markdown(f"**{method}**")
                psnr, ssim = report.loc[method, ["PSNR (dB)", "SSIM"]]
                st.metric("PSNR", "∞ dB" if np.isinf(psnr) else f"{psnr:.2f} dB")
                st.metric("SSIM", f"{ssim:.4f}")
    if cnn_error or not weights_loaded:
        st.warning(f"CNN benchmark status: {cnn_status}. These CNN scores do not represent a valid trained-model evaluation.")




def main():
    render_navigation()
    renderers = {
        "Home": render_home,
        "Features": render_features,
        "AI Studio": render_ai_studio,
        "About": render_about,
    }
    renderers.get(st.session_state["page"], render_home)()
    st.markdown(
        '<div class="footer"><b>✦ DenoiseAI</b> &nbsp;·&nbsp; '
        'Explore. Restore. Understand.<br>'
        'Educational image restoration with U-Net and classical computer vision.</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
