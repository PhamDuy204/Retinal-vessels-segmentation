# Retinal Vessels Segmentation

## Yêu cầu kỹ thuật
\- Sử dụng phần cứng có ít nhất 2 card đồ họa (GPU). 

## Hướng dẫn chạy
**Bước 1:**
```bash
git clone https://github.com/PhamDuy204/Retinal-vessels-segmentation.git
```
hoặc bỏ qua nếu đã có code

**Bước 2:**
```bash
cd vào dir project
```

**Bước 3:**
```bash
pip install -r requirements.txt 
apt-get install libgl1 -y
```

**Bước 4:**

\- Chạy mô hình U-Net:
```bash
python3 train.py --model 'unet' --epochs 50
```

**Lưu ý:** Termial sẽ hiện ra thông báo chọn option của wandb => chọn option 2 và paste api_key của wandb vào

**Bước 5:**

\- Chạy mô hình của nhóm: 
```bash
python3 train.py --model 'custom_net' --epochs 50
```

**Bước 6:**

\- Xem kết quả ở project Retinal-Vessels-Segmentation ở trang chủ wandb

![Result](public/Screenshot%20from%202025-06-16%2001-42-31.png)
