"""Streamlit interface for validated DRIVE checkpoint inference."""
from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path
import glob
import hashlib
import sys

import numpy as np
import streamlit as st
import torch
from PIL import Image

from inference_core import IMAGE_EXTENSIONS, gradcam, load_checkpoint, predict, read_image


def paths(values, suffixes):
    found = []
    for value in values:
        for entry in value.split(","):
            entry = entry.strip()
            if not entry:
                continue
            candidates = [Path(name) for name in glob.glob(entry, recursive=True)]
            if not candidates and Path(entry).exists():
                candidates = [Path(entry)]
            for candidate in candidates:
                files = candidate.rglob("*") if candidate.is_dir() else (candidate,)
                found.extend(file.resolve() for file in files
                             if file.is_file() and file.suffix.lower() in suffixes)
    return sorted(set(found))


def options():
    parser = argparse.ArgumentParser(description="Retinal vessel inference UI")
    parser.add_argument("--checkpoints", nargs="+", required=True,
                        help="One or more .pt/.safetensors checkpoints, directories, or globs.")
    parser.add_argument("--image_paths", nargs="*", default=[],
                        help="Optional images, directories, or globs for the image menu.")
    argv = sys.argv[1:]
    arguments = parser.parse_args(argv[1:] if argv[:1] == ["--"] else argv)
    checkpoints = paths(arguments.checkpoints, {".pt", ".safetensors"})
    if not checkpoints:
        parser.error("--checkpoints must contain a .pt or .safetensors checkpoint")
    return checkpoints, paths(arguments.image_paths, IMAGE_EXTENSIONS)


@st.cache_resource(show_spinner="Loading checkpoint…")
def cached_model(path, mtime_ns, device_name):
    return load_checkpoint(path, torch.device(device_name))


def png_bytes(array):
    buffer = BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def mask_preview(mask):
    """Light display palette; the downloadable mask keeps literal 0/1 pixels."""
    preview = np.empty((*mask.shape, 3), dtype=np.uint8)
    preview[:] = (245, 250, 252)
    preview[mask.astype(bool)] = (20, 109, 134)
    return preview


st.set_page_config(page_title="SGMA-Net · Vessel viewer", layout="wide")
st.markdown("""
<style>
:root {--ink:#14283b;--sea:#146d86;--line:#d6e2e8;--paper:#f3f8fa}
.stApp {background:var(--paper);color:var(--ink)}
.block-container {max-width:1200px;padding-top:2.3rem}
h1,h2,h3 {font-family:Georgia,serif;color:var(--ink)}
[data-testid="stImage"] img {border-radius:10px}
.st-key-source_view [data-testid="stImage"] img,
.st-key-result_view [data-testid="stImage"] img {
  aspect-ratio:565 / 584;object-fit:contain;background:#e9f1f4;width:100%;
}
.preview-placeholder, .result-placeholder {
  aspect-ratio:565 / 584;border:1px solid var(--line);border-radius:10px;
  background:#e9f1f4;display:flex;align-items:center;justify-content:center;
  color:#48697a;font-size:.9rem;letter-spacing:.02em;text-align:center;padding:2rem;
}
[data-testid="stFileUploaderDropzone"] {
  background:white;border:1px solid var(--line);border-radius:10px;padding:.5rem;
}
[data-testid="stFileUploaderDropzoneInstructions"] {display:none}
div.stButton > button[kind="primary"] {background:var(--sea);border-color:var(--sea);color:white}
div.stButton > button:focus-visible, div.stSelectbox:focus-within {
  outline:3px solid #50a9bd;outline-offset:2px
}
.small-label {font-family:monospace;letter-spacing:.12em;color:#466676;font-size:.78rem}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="small-label">RETINAL IMAGING / DRIVE</div>', unsafe_allow_html=True)
st.title("Vessel viewer")
st.caption("Select a fundus image, choose a checkpoint, and inspect the vessel mask.")

try:
    checkpoints, image_paths = options()
except SystemExit:
    st.error("Start with: streamlit run demo.py -- --checkpoints inference_models/drive_epoch58.safetensors")
    st.stop()

left, right = st.columns(2, gap="large")
with left:
    st.subheader("Input")
    uploaded = st.session_state.get("image_upload")
    image_label = st.session_state.get("image_selector", "Choose an image…")
    source_key = None
    rgb = None
    if uploaded is not None:
        raw = uploaded.getvalue()
        source_key = hashlib.sha256(raw).hexdigest()
        try:
            rgb = read_image(BytesIO(raw))
        except Exception as exc:
            st.error(f"Cannot read uploaded image: {exc}")
    elif image_label != "Choose an image…":
        try:
            image_path = Path(image_label)
            source_key = f"{image_path}:{image_path.stat().st_mtime_ns}"
            rgb = read_image(image_path)
        except Exception as exc:
            st.error(f"Cannot read image path: {exc}")
    with st.container(key="source_view"):
        if rgb is None:
            st.markdown('<div class="preview-placeholder">Choose an image below or upload one from your computer</div>',
                        unsafe_allow_html=True)
        else:
            st.image(rgb, caption=f"Source image · {rgb.shape[1]} × {rgb.shape[0]}",
                     width="stretch")
    st.file_uploader("Upload or replace image", key="image_upload",
                     type=sorted(ext.lstrip(".") for ext in IMAGE_EXTENSIONS))
    st.selectbox("Image path", ["Choose an image…"] + [str(p) for p in image_paths],
                 key="image_selector", disabled=uploaded is not None,
                 help="Choose a path when no upload is active.")
    selected_checkpoint = st.selectbox("Model checkpoint", checkpoints,
                                       format_func=lambda p: f"{p.parent.name} / {p.name}")

with right:
    label, arrow = st.columns([8, 1])
    with label:
        st.subheader("Prediction")
    active_key = (source_key, str(selected_checkpoint), selected_checkpoint.stat().st_mtime_ns)
    current = st.session_state.get("result")
    if current is None or current["key"] != active_key:
        current = None
    showing_cam = bool(current is not None and st.session_state.get("show_cam", False))
    with arrow:
        if st.button("←" if showing_cam else "→",
                     help="Back to vessel mask" if showing_cam else "Show Grad-CAM",
                     disabled=current is None):
            st.session_state["show_cam"] = not showing_cam
            st.rerun()
    with st.container(key="result_view"):
        if current is None:
            st.markdown('<div class="result-placeholder">MASK / AWAITING PREDICTION</div>',
                        unsafe_allow_html=True)
        elif showing_cam:
            if "cam" not in current:
                with st.spinner("Computing Grad-CAM over all patches…"):
                    try:
                        current["cam"] = gradcam(current["model"], current["rgb"],
                                                current["device"])
                    except Exception as exc:
                        st.error(f"Grad-CAM failed: {exc}")
            if "cam" in current:
                st.image(current["cam"], caption="Vessel-focused Grad-CAM on source image",
                         width="stretch")
        else:
            st.image(mask_preview(current["mask"]),
                     caption="Vessel mask · download contains 0 background / 1 vessel",
                     width="stretch")
    st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
    if st.button("Predict", type="primary", width="stretch", disabled=rgb is None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cpu":
            torch.set_num_threads(2)
        try:
            with st.spinner("Segmenting retinal vessels…"):
                model = cached_model(str(selected_checkpoint), selected_checkpoint.stat().st_mtime_ns,
                                     str(device))
                mask, seconds = predict(model, rgb, device)
            st.session_state["result"] = dict(key=active_key, model=model, rgb=rgb,
                                               device=device, mask=mask, seconds=seconds)
            st.session_state["show_cam"] = False
            st.rerun()
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
    if current is not None:
        st.caption(f"Inference: {current['seconds']:.3f}s · {current['device'].type.upper()}")
        st.download_button("Download 0/1 mask (PNG)", png_bytes(current["mask"]),
                           file_name="vessel_mask_0_1.png", mime="image/png")
