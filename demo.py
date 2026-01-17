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
from sklearn.metrics import f1_score, recall_score
from io import BytesIO
import zipfile

# --- Danh sách model có sẵn ---
model_lst = os.listdir('checkpoints/')
st.title("Segmentation Demo App")

# Chọn model
selected_model = st.selectbox("Chọn model:", model_lst, index=0)
model_name = selected_model.replace('.pt', '')

# if model_name == our_net_woLoss:
# Load model class and prepare sys.modules for unpickling
# (load_model_class keeps the model's modules in sys.modules)
load_model_class(model_name if model_name != 'our_net_woLoss' else 'our_net')

# Now load the checkpoint using torch.load
# sys.modules has the correct model modules already loaded
model = torch.load(
    f'checkpoints/{model_name}.pt',
    map_location='cuda' if torch.cuda.is_available() else 'cpu',
    weights_only=False
)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device).eval()

# Uploads
uploaded_image = st.file_uploader("Upload ảnh input", type=["png", "jpg", "jpeg","tif","ppm"])
uploaded_gt = st.file_uploader("Upload Ground Truth (optional)", type=["png", "jpg", "jpeg","gif","tif"])

# Run button
if st.button("Run Segmentation"):
    if uploaded_image is None:
        st.warning("Vui lòng upload ảnh input trước!")
    else:
        # --- Input image ---
        ori_image = np.array(Image.open(uploaded_image).convert('RGB'))

        processed_vis = preprocessing_img(ori_image) 

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
        mask_pil = Image.fromarray(seg_display)           # grayscale (HxW)
        mask_rgb = mask_pil.convert("RGB")                # convert để lưu PNG chuẩn
        buf_mask = BytesIO()
        mask_rgb.save(buf_mask, format="PNG")
        buf_mask.seek(0)

        # --- Tạo overlay: input image + mask đỏ bán trong suốt ---
        ori_pil = Image.fromarray(ori_image).convert("RGBA")
        mask_l = Image.fromarray(seg_display).convert("L")  # dùng làm alpha
        red_overlay = Image.new("RGBA", ori_pil.size, (255, 0, 0, 120))  # đỏ với alpha 120
        # đặt alpha của red_overlay bằng mask (255 -> hiển thị, 0 -> trong suốt)
        red_overlay.putalpha(mask_l)
        overlay_pil = Image.alpha_composite(ori_pil, red_overlay)

        buf_overlay = BytesIO()
        overlay_pil.convert("RGB").save(buf_overlay, format="PNG")
        buf_overlay.seek(0)

        # --- Hiển thị cùng vị trí result và thêm nút download ---
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.caption("Input Image")
            st.image(ori_image, width=300)
        with col2:
            st.caption("Processed Image")
            st.image(processed_vis, width=300)
        with col3:
            st.caption("Result")
            st.image(seg_display, width=300)
            # Nút tải xuống (dùng emoji như "sticker")
            st.download_button(
                label="⬇️ Tải mask (PNG)",
                data=buf_mask.getvalue(),
                file_name=f"{model_name}_mask.png",
                mime="image/png"
            )
            st.download_button(
                label="🖼️ Tải overlay (PNG)",
                data=buf_overlay.getvalue(),
                file_name=f"{model_name}_overlay.png",
                mime="image/png"
            )
        with col4:
            st.caption("Ground Truth")
            st.image(gt_display, width=300)
        with col5:
            error_map = create_error_map(pred_mask, gt_bin)
            overlay_cmp = overlay_error_map(ori_image, error_map, alpha=0.6)
            st.caption("Overlay (TP/FP/FN)")
            st.image(error_map, width=250)
        if uploaded_gt is not None: # flatten và tính F1 binary 
            f1 = f1_score(gt_bin.flatten(), pred_mask.flatten(), average='binary') 
            recall = recall_score(gt_bin.flatten(), pred_mask.flatten(), average='binary') 
            st.write(f"Recall: {recall:.4f}") 
            st.write(f"F1 Score: {f1:.4f}") 
        else: st.info("Không có Ground Truth — Không tính score được.")
        zip_buffer = BytesIO()

        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
            zipf.writestr(f"01_{model_name}_input.png", save_png_to_bytes(ori_image))
            zipf.writestr(f"02_{model_name}_processed.png", save_png_to_bytes(processed_vis))
            zipf.writestr(f"03_{model_name}_prediction_mask.png", save_png_to_bytes(seg_display))
            zipf.writestr(f"04_{model_name}_prediction_overlay.png", save_png_to_bytes(overlay_pil))
            
            if uploaded_gt is not None:
                zipf.writestr(f"05_{model_name}_ground_truth.png", save_png_to_bytes(gt_display))
                zipf.writestr(f"06_{model_name}_error_map.png", save_png_to_bytes(error_map))
                
        st.download_button(
            label="📦 Tải TẤT CẢ kết quả (ZIP)",
            data=zip_buffer.getvalue(),
            file_name=f"{model_name}_segmentation_results.zip",
            mime="application/zip"
        )


        zip_buffer.seek(0)

st.markdown("""
**Legend:**
- 🟩 Green: True Positive (Correct)
- 🟥 Red: False Positive (Over-segmentation)
- 🟦 Blue: False Negative (Missed)
""")
