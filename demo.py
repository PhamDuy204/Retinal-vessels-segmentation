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
                        help="One or more .pt checkpoints, directories, or globs.")
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


st.set_page_config(page_title="SGMA-Net · Vessel viewer", layout="wide")
st.markdown("""
<style>
:root {--ink:#14283b;--sea:#146d86;--line:#d6e2e8;--paper:#f3f8fa}
.stApp {background:var(--paper);color:var(--ink)}
.block-container {max-width:1200px;padding-top:2.3rem}
h1,h2,h3 {font-family:Georgia,serif;color:var(--ink)}
[data-testid="stFileUploader"] {background:white;border:1px solid var(--line);border-radius:12px;padding:12px}
[data-testid="stFileUploaderDropzone"] {background:#f7fbfd}
[data-testid="stImage"] img {border-radius:8px}
div.stButton > button[kind="primary"] {background:var(--sea);border-color:var(--sea);color:white}
div.stButton > button:focus-visible, div.stSelectbox:focus-within {outline:3px solid #50a9bd;outline-offset:2px}
.result-frame {background:#07131d;border-radius:12px;min-height:420px;display:flex;
align-items:center;justify-content:center;color:#acc1cc;font-family:monospace;letter-spacing:.05em}
.small-label {font-family:monospace;letter-spacing:.12em;color:#466676;font-size:.78rem}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="small-label">RETINAL IMAGING / DRIVE</div>', unsafe_allow_html=True)
st.title("Vessel viewer")
st.caption("Upload a fundus image or choose one from the configured paths. Predictions use the validated DRIVE patch protocol.")

try:
    checkpoints, image_paths = options()
except SystemExit:
    st.error("Start with: streamlit run demo.py -- --checkpoints checkpoints/.../best.pt")
    st.stop()

left, right = st.columns([1, 1.35], gap="large")
with left:
    st.subheader("Input")
    uploaded = st.file_uploader("Drag or select a fundus image",
                                type=sorted(ext.lstrip(".") for ext in IMAGE_EXTENSIONS))
    image_label = st.selectbox("Image path", ["Choose an image…"] + [str(p) for p in image_paths],
                               disabled=bool(uploaded), help="Available when no image is uploaded.")
    selected_checkpoint = st.selectbox("Model checkpoint", checkpoints,
                                       format_func=lambda p: f"{p.parent.name} / {p.name}")
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
    if rgb is not None:
        st.image(rgb, caption=f"Source image · {rgb.shape[1]} × {rgb.shape[0]}", width="stretch")

with right:
    label, arrow = st.columns([8, 1])
    with label:
        st.subheader("Prediction")
    active_key = (source_key, str(selected_checkpoint), selected_checkpoint.stat().st_mtime_ns)
    current = st.session_state.get("result")
    if current is None or current["key"] != active_key:
        current = None
    with arrow:
        if st.button("→", help="Switch between binary mask and Grad-CAM", disabled=current is None):
            st.session_state["show_cam"] = not st.session_state.get("show_cam", False)
    if current is None:
        st.markdown('<div class="result-frame">MASK / AWAITING PREDICTION</div>', unsafe_allow_html=True)
    elif st.session_state.get("show_cam", False):
        if "cam" not in current:
            with st.spinner("Computing Grad-CAM over all patches…"):
                try:
                    current["cam"] = gradcam(current["model"], current["rgb"], current["device"])
                except Exception as exc:
                    st.error(f"Grad-CAM failed: {exc}")
        if "cam" in current:
            st.image(current["cam"], caption="Grad-CAM over source image", width="stretch")
    else:
        st.image(current["mask"] * 255, caption="Binary mask · 0 background / 1 vessel",
                 clamp=True, width="stretch")
        st.download_button("Download 0/1 mask (PNG)", png_bytes(current["mask"]),
                           file_name="vessel_mask_0_1.png", mime="image/png")
        st.caption(f"Inference: {current['seconds']:.3f}s · {current['device'].type.upper()} · warmed model")

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
