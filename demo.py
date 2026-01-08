import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import streamlit as st
from PIL import Image
import numpy as np
import torch
from load_model import load_model_class
from utils import *
from transforms import get_test_patch_transforms
from sklearn.metrics import f1_score

# --- Danh sách model có sẵn ---
model_lst = ["Our_net"]

st.title("Segmentation Demo App")

# Chọn model
selected_model = st.selectbox("Chọn model:", model_lst, index=0)
model_name = selected_model.lower()

# Thêm class model vào safe globals rồi load (chỉ khi bạn tin checkpoint)
torch.serialization.add_safe_globals([load_model_class(model_name)])
model = torch.load(
    f'checkpoints/{model_name}.pt',
    map_location='cuda' if torch.cuda.is_available() else 'cpu',
    weights_only=False
)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device).eval()

# Uploads
uploaded_image = st.file_uploader("Upload ảnh input", type=["png", "jpg", "jpeg","tif"])
uploaded_gt = st.file_uploader("Upload Ground Truth (optional)", type=["png", "jpg", "jpeg","gif","tif"])

# Run button
if st.button("Run Segmentation"):
    if uploaded_image is None:
        st.warning("Vui lòng upload ảnh input trước!")
    else:
        # --- Input image ---
        ori_image = np.array(Image.open(uploaded_image).convert('RGB'))
        # Keep a processed copy for visualization (before tensor transforms)
        processed_vis = preprocessing_img(ori_image).astype(np.uint8)  # assuming preprocessing_img returns uint8 HxW or HxWx3

        # --- Prepare image for model inference (tensor) ---
        img_for_transforms = processed_vis.copy()
        img_tensor = get_test_patch_transforms()(image=img_for_transforms)['image'].to(device)
        _, h, w = img_tensor.shape

        # mirror padding, patch extraction and inference (same logic as your original)
        img_tensor = mirror_padding_v2(img_tensor).unsqueeze(0)
        B, C, H, W = img_tensor.shape
        num_patch = ((H-64)//32+1, (W-64)//8+1)
        image_patches, tmp_stride = extract_patches_with_target_count(img_tensor, 64, num_patch)
        if len(image_patches.shape) > 4:
            image_patches = image_patches.flatten(0, 1)
        chunk_size = max(image_patches.shape[0] // 128, 1)
        chunk_image = torch.chunk(image_patches, chunk_size, 0)

        out_sample = []
        for c_image in chunk_image:
            with torch.inference_mode():
                prob = model(c_image)
            out_sample.append(prob)
        prob = torch.cat(out_sample, 0)
        prob = prob.view(B, -1, 1, 64, 64)
        prob = reverse_to_original_image(prob, (H, W), 64, tmp_stride).squeeze()[:h, :w]

        # Threshold => binary mask (numpy)
        pred_mask = (prob >= 0.487).to(torch.uint8).detach().cpu().numpy()  # shape (h,w), 0/1
        seg_display = (pred_mask * 255).astype(np.uint8)

        # --- Prepare ground truth display (resize if needed) ---
        if uploaded_gt is not None:
            gt_img = Image.open(uploaded_gt).convert('L')
            # resize GT to match pred_mask size if different
            if gt_img.size != (pred_mask.shape[1], pred_mask.shape[0]):
                gt_img = gt_img.resize((pred_mask.shape[1], pred_mask.shape[0]), resample=Image.NEAREST)
            gt_array = np.array(gt_img)
            # binarize GT for scoring/display (non-zero -> 1)
            gt_bin = (gt_array != 0).astype(np.uint8)
            gt_display = (gt_bin * 255).astype(np.uint8)
        else:
            # nếu không có GT -> ảnh đen cùng kích thước seg
            gt_display = np.zeros_like(seg_display)
            gt_bin = None  # dùng để quyết định không tính F1

        # --- Show 1 hàng 4 cột ---
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.caption("Input Image")
            st.image(ori_image, width=300)
        with col2:
            st.caption("Processed Image")
            # processed_vis có thể là HxW hoặc HxWx3, đảm bảo uint8
            st.image(processed_vis, width=300)
        with col3:
            st.caption(f"Segmentation Result ({selected_model})")
            st.image(seg_display, width=300)
        with col4:
            st.caption("Ground Truth")
            st.image(gt_display, width=300)

        # --- Tính F1 nếu có GT ---
        if uploaded_gt is not None:
            # flatten và tính F1 binary
            f1 = f1_score(gt_bin.flatten(), pred_mask.flatten(), average='binary')
            st.write(f"F1 Score: {f1:.4f}")
        else:
            st.info("Không có Ground Truth — Không tính score được.")
