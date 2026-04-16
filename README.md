# Who's That Pokemon? — It's Always Pikachu!

把所有寶可夢都變成皮卡丘。保留每個寶可夢的輪廓和五官位置，將皮卡丘的外觀（黃色毛皮、紅臉頰、黑耳尖）像貼圖一樣拉伸上去。

## 技術方案

使用 **Stable Diffusion 1.5 + ControlNet Canny + IP-Adapter**：

- **ControlNet (Canny)**：從目標寶可夢提取邊緣圖，強制生成結果遵循原始輪廓
- **IP-Adapter**：以皮卡丘圖片作為視覺參考，注入皮卡丘的外貌特徵
- **Stable Diffusion**：在兩個條件下生成新圖片

無需訓練，所有模型都是預訓練的。

## 資料流

```
目標寶可夢 RGBA PNG
    │
    ├─→ extract_alpha ──→ 保留透明遮罩
    ├─→ 白底 RGB ──→ Canny 邊緣偵測 ──→ ControlNet（保留形狀）
    │
    ├─→ 皮卡丘 025.png ──→ 白底 RGB ──→ IP-Adapter（注入外貌）
    │
    └─→ Prompt: "a Pikachu, yellow creature with red cheeks..."
            │
            ▼
    Stable Diffusion 生成（512×512）
            │
            ▼
    放回原始尺寸 → 還原透明遮罩 → 輸出 RGBA PNG
```

## 安裝

### CUDA（NVIDIA GPU）

```bash
# Step 1: PyTorch（從 NVIDIA index）
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118

# Step 2: 其他套件（從 PyPI）
pip install -r requirements.txt
```

### Apple Silicon（MPS）

```bash
pip install -r requirements.txt
```

### 驗證

```bash
pip check
```

## 使用方式

### 單張測試

```bash
python main.py sd_single --input 006.png --steps 25
```

006.png 是噴火龍，效果應該很明顯。首次跑會自動下載模型（~9GB），之後快取在 `~/.cache/huggingface/`。

### 批次處理全部 1172 張

```bash
python main.py sd_transfer --steps 25
```

支援中斷續跑——已處理的檔案會自動跳過。

### 可調參數

| 參數 | 預設 | 說明 |
|------|------|------|
| `--ip-scale` | 0.7 | IP-Adapter 影響力（越高越像皮卡丘外貌） |
| `--controlnet-scale` | 0.8 | ControlNet 強度（越高越嚴格遵循輪廓） |
| `--steps` | 25 | 推理步數（20-30，品質與速度平衡） |
| `--guidance` | 7.5 | CFG 引導強度 |
| `--seed 42` | 隨機 | 固定種子可重現結果 |

### LAB 色彩空間基線（不需 GPU）

```bash
python main.py lab_transfer
```

簡單的色彩對齊方法，幾秒就能處理完全部圖片，可作為對照組。

## 預估時間

| 階段 | A100 80GB | M4 Mac |
|------|-----------|--------|
| 首次下載模型 | ~5 分鐘 | ~15 分鐘 |
| 單張生成 | ~3 秒 | ~20 秒 |
| 完整 1172 張 | ~60 分鐘 | ~6.5 小時 |

## 專案結構

```
main.py                      # CLI 入口
requirements.txt             # 依賴（鎖定版本）
pyproject.toml               # Poetry 設定

data/
  canny_utils.py             # Canny 邊緣偵測（給 ControlNet 用）
  preprocessing.py           # RGBA→白底RGB、透明遮罩提取
  dataset.py                 # 寶可夢圖片資料集
  color_transfer.py          # LAB 色彩轉換（基線方法）

inference/
  sd_pipeline.py             # SD + ControlNet + IP-Adapter 核心 pipeline
  postprocess.py             # 透明遮罩還原
  color_transfer.py          # LAB 色彩空間批次轉換

utils/
  config.py                  # 集中設定（模型ID、生成參數、閾值等）
  device.py                  # CUDA/MPS/CPU 自動偵測

dataset/                     # 寶可夢圖片資料集
  HybridShivam-Pokemon/      # 1160+ 張 Sugimori 官方插圖（主要使用）
  arenagrenade-pokemon-images/  # 898 張
  kvpratama-pokemon-images/     # 819 張

adain/                       # [已棄用] AdaIN style transfer
train/                       # [已棄用] AdaIN 訓練
```

## 模型來源

所有模型首次執行時自動從 HuggingFace 下載：

| 模型 | HuggingFace ID | 大小 |
|------|---------------|------|
| Stable Diffusion 1.5 | `runwayml/stable-diffusion-v1-5` | ~4.2 GB |
| ControlNet Canny | `lllyasviel/control_v11p_sd15_canny` | ~1.4 GB |
| IP-Adapter | `h94/IP-Adapter` | ~700 MB |
| CLIP Image Encoder | `h94/IP-Adapter` (subfolder) | ~2.5 GB |
| VAE (ft-MSE) | `stabilityai/sd-vae-ft-mse` | ~335 MB |
