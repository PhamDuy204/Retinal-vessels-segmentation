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

from inference_core import IMAGE_EXTENSIONS, gradcam, load_checkpoint, predict, read_image, select_device


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
    parser.add_argument("--checkpoints", nargs="*", default=[],
                        help="Optional .pt/.safetensors files, directories, or globs.")
    parser.add_argument("--os", choices=("win", "linux"),
                        default="win" if sys.platform.startswith("win") else "linux",
                        help="Model architecture: win=our_net_window, linux=our_net.")
    parser.add_argument("--image_paths", nargs="*", default=[],
                        help="Optional images, directories, or globs for the image menu.")
    argv = sys.argv[1:]
    arguments = parser.parse_args(argv[1:] if argv[:1] == ["--"] else argv)
    checkpoints = paths(arguments.checkpoints, {".pt", ".safetensors"})
    if arguments.checkpoints and not checkpoints:
        parser.error("--checkpoints did not contain any .pt or .safetensors files")
    model_name = "our_net_window" if arguments.os == "win" else "our_net"
    return checkpoints, paths(arguments.image_paths, IMAGE_EXTENSIONS), model_name


@st.cache_resource(show_spinner="Loading checkpoint…")
def cached_model(path, mtime_ns, device_name, model_name, contents=None):
    return load_checkpoint(path, torch.device(device_name), model_name, contents)


@st.cache_data(show_spinner="Checking checkpoint…")
def check_checkpoint(path, mtime_ns, model_name, contents=None):
    load_checkpoint(path, torch.device("cpu"), model_name, contents)
    return True


def png_bytes(array):
    buffer = BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def mask_preview(mask):
    """Black background, green vessels; downloaded mask remains literal 0/1."""
    preview = np.zeros((*mask.shape, 3), dtype=np.uint8)
    preview[mask.astype(bool)] = (40, 220, 85)
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
  background:transparent !important;border:0 !important;box-shadow:none !important;
  min-height:0 !important;padding:0 !important;justify-content:flex-start;
}
[data-testid="stFileUploaderDropzoneInstructions"] {display:none}
.warmup-note {
  margin-top:.75rem;padding:.6rem .85rem;border-radius:8px;
  background:#fff0ef;border:1px solid #f3cecb;color:#9a4040;
  font-size:.86rem;line-height:1.4;
}
div.stButton > button[kind="primary"] {background:var(--sea);border-color:var(--sea);color:white}
div.stButton > button:focus-visible, div.stSelectbox:focus-within {
  outline:3px solid #50a9bd;outline-offset:2px
}
.small-label {font-family:monospace;letter-spacing:.12em;color:#466676;font-size:.78rem}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="small-label">RETINAL IMAGING / DRIVE</div>', unsafe_allow_html=True)
st.title("Vessel viewer")
st.caption("Select a fundus image and inspect the vessel mask. A checkpoint is optional.")

try:
    checkpoints, image_paths, model_name = options()
except SystemExit:
    st.error("Invalid launch options. Example: streamlit run demo.py -- --os linux")
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
    selected_checkpoint = (st.selectbox(
        "Model checkpoint", [None, *checkpoints], index=1,
        format_func=lambda p: f"{p.parent.name} / {p.name}" if p else "Upload checkpoint / no weights",
    ) if checkpoints else None)
    checkpoint_upload = (st.file_uploader("Upload checkpoint (optional)",
                                          type=["pt", "safetensors"], key="checkpoint_upload")
                         if selected_checkpoint is None else None)
    checkpoint_error = None
    checkpoint_name = selected_checkpoint
    checkpoint_bytes = None
    if checkpoint_upload is not None:
        checkpoint_name = checkpoint_upload.name
        checkpoint_bytes = checkpoint_upload.getvalue()
        try:
            check_checkpoint(checkpoint_name, None, model_name, checkpoint_bytes)
        except Exception as exc:
            checkpoint_error = str(exc).splitlines()[0].rstrip(".")
            st.error(f"Invalid checkpoint for {model_name}: {checkpoint_error}. Upload another checkpoint.")
    elif selected_checkpoint is None:
        st.caption(f"No checkpoint: {model_name} uses randomly initialized weights; the mask is not meaningful.")
    else:
        try:
            check_checkpoint(str(selected_checkpoint), selected_checkpoint.stat().st_mtime_ns, model_name)
        except Exception as exc:
            checkpoint_error = str(exc).splitlines()[0].rstrip(".")
            st.error(f"Invalid checkpoint for {model_name}: {checkpoint_error}. Select or upload another checkpoint.")

with right:
    label, arrow = st.columns([8, 1])
    with label:
        st.subheader("Prediction")
    checkpoint_mtime = selected_checkpoint.stat().st_mtime_ns if selected_checkpoint else None
    active_key = (source_key, model_name, str(checkpoint_name), checkpoint_mtime,
                  hashlib.sha256(checkpoint_bytes).hexdigest() if checkpoint_bytes else None)
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
    st.markdown('<div class="warmup-note" role="note">The first prediction loads the model and warms up CUDA when available. Later GPU predictions are faster.</div>',
                unsafe_allow_html=True)
    st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
    if st.button("Predict", type="primary", width="stretch",
                 disabled=rgb is None or checkpoint_error is not None):
        device = select_device()
        if device.type == "cpu":
            torch.set_num_threads(2)
        try:
            with st.spinner("Segmenting retinal vessels…"):
                model = cached_model(str(checkpoint_name) if checkpoint_name else None,
                                     checkpoint_mtime, str(device), model_name, checkpoint_bytes)
                mask, seconds = predict(model, rgb, device)
            st.session_state["result"] = dict(key=active_key, model=model, rgb=rgb,
                                               device=device, mask=mask, seconds=seconds)
            st.session_state["show_cam"] = False
            st.rerun()
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
    if current is not None:
        st.caption(f"Inference: {current['seconds']:.3f}s · {current['device'].type.upper()} · "
                   f"{model_name}{' (random weights)' if checkpoint_name is None else ''}")
        st.download_button("Download 0/1 mask (PNG)", png_bytes(current["mask"]),
                           file_name="vessel_mask_0_1.png", mime="image/png")
