# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Who's That Pokemon? — It's Always Pikachu!**

把所有寶可夢變成皮卡丘：保留每個寶可夢的輪廓和五官位置，將皮卡丘的外觀（黃色毛皮、紅臉頰、黑耳尖）拉伸上去。

## Current Architecture

- **Stable Diffusion 1.5 + ControlNet Canny + IP-Adapter** — 主要 pipeline
- **LAB 色彩空間轉換** — 基線方法（不需 GPU）
- **AdaIN（已棄用）** — 只能對齊顏色統計量，無法產生皮卡丘具體圖案

## Development Setup

```bash
# CUDA (NVIDIA GPU)
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# Apple Silicon (MPS)
pip install -r requirements.txt
```

- Python: >=3.11
- 本機: Poetry (`poetry install`)
- Server: pip + requirements.txt (CUDA 11.8)
- Server: NVIDIA A100 80GB

## CLI Usage

```bash
# SD pipeline — 單張測試
python main.py sd_single --input 006.png --steps 25

# SD pipeline — 批次處理全部
python main.py sd_transfer --steps 25

# Canny 邊緣預覽（不需 GPU，秒完成）
python main.py preview --input 006.png
python main.py batch_preview

# LAB 色彩基線
python main.py lab_transfer
```

Key parameters: `--ip-scale` (0.7), `--controlnet-scale` (0.8), `--steps` (25), `--guidance` (7.5), `--seed`

## TODO — LoRA 微調訓練（必要代辦）

目前的 SD pipeline 使用 IP-Adapter 零樣本注入皮卡丘外貌（模型從沒看過皮卡丘）。
需要加入 **LoRA 微調訓練**，讓模型真正學會皮卡丘的視覺特徵。

### 為什麼需要 LoRA

- IP-Adapter 是 zero-shot，模型不理解「皮卡丘」的完整語義
- LoRA 讓模型在權重層面學會皮卡丘特徵，生成一致性和品質更好
- 這是深度學習專案的核心訓練環節（目前只有推理）

### 訓練工具

使用 **kohya_ss/sd-scripts**（CLI 模式），不使用 GUI。

理由：
- 產出標準 safetensors LoRA 檔，`diffusers.load_lora_weights()` 可直接載入
- 支援 resolution bucketing、mixed precision、gradient checkpointing
- 社群驗證最成熟的 SD LoRA 訓練工具
- 在 A100 server 上用獨立 venv 訓練，不影響推理環境

安裝：
```bash
git clone https://github.com/kohya_ss/sd-scripts
cd sd-scripts
pip install -r requirements.txt
pip install bitsandbytes  # AdamW8bit optimizer
```

### 資料準備

**目標**：~45 張皮卡丘圖片 + captions

**來源（從現有資料集選取 6 張）**：
- `025.png` — 標準 Sugimori 皮卡丘
- `025-Starter.png` — 略微不同姿勢
- `025-Libre.png` — 摔角皮卡丘（身體特徵相同）
- `025-Gmax.png` — 極巨化（身體特徵相同）
- `025-Phd.png` — 學士帽皮卡丘
- `025-Cosplay.png` — 變裝皮卡丘

**跳過**：帽子/服裝變體（Alola-Cap, Hoenn-Cap 等）— 帽子會干擾核心特徵學習。

**資料擴增（6 張 → ~36 張）**：
- 對每張圖旋轉 -15, -10, -5, +5, +10, +15 度
- RGBA 原圖旋轉後合成白底 RGB
- 捨棄裁切到角色的擴增結果

**目錄結構**：
```
data/lora_training/
  image/              # ~45 張 PNG
  meta_lat.json       # sd-scripts 格式的 caption metadata
  dataset_config.toml # sd-scripts 訓練配置
```

### Caption 策略

- **Trigger token**: `pkmn-pikachu`
- 每張圖都必須包含 trigger token
- 變化描述長度，避免模型死記固定詞句：

```
# 詳細版
"pkmn-pikachu, a small yellow electric-type pokemon with round red cheek pouches,
 pointed ears with black tips, a zigzag lightning-shaped tail,
 simple cartoon illustration on white background"

# 簡短版
"pkmn-pikachu, yellow pokemon, white background"

# 姿勢描述版
"pkmn-pikachu standing, facing left, yellow fur with red cheeks, cartoon style"
```

### LoRA 超參數

| 參數 | 值 | 說明 |
|------|------|------|
| Network Dim (Rank) | 32 | 足夠表達皮卡丘的色彩/形狀特徵 |
| Network Alpha | 16 | Rank/2，正則化防止過擬合 |
| Learning Rate (UNet) | 1e-4 | 概念 LoRA 標準值 |
| Learning Rate (Text Encoder) | 5e-5 | Text encoder 需要更溫和的更新 |
| LR Scheduler | Cosine + 10% warmup | 平滑衰減，~200 步 warmup |
| Optimizer | AdamW8bit | 省 VRAM，品質不減 |
| Batch Size | 4 | A100 可承受 |
| Max Train Steps | 2000-3000 | ~45 張 × ~15 epochs / batch 4 ≈ 2500 步 |
| Mixed Precision | bf16 | A100 原生支援，比 fp16 穩定 |
| Resolution | 512 | SD 1.5 原生解析度 |
| LoRA Targets | Attention only (Q, K, V, Out) | 角色概念只需修改 attention |
| Gradient Checkpointing | True | 省 VRAM |
| xformers | True | 記憶體效率 attention |
| Save Every N Steps | 500 | 每隔 500 步存 checkpoint，選最好的 |

**預估訓練時間（A100）**：~15 分鐘（含 latent caching + 2500 步訓練）

### 訓練指令範本

```bash
accelerate launch --num_cpu_threads_per_process 4 \
  sdxl_train_network.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --dataset_config="data/lora_training/dataset_config.toml" \
  --output_dir="output/lora" \
  --output_name="pikachu_lora_v1" \
  --save_model_as=safetensors \
  --max_train_steps=2500 \
  --learning_rate=1e-4 \
  --text_encoder_lr=5e-5 \
  --unet_lr=1e-4 \
  --lr_scheduler=cosine \
  --lr_warmup_steps=200 \
  --train_batch_size=4 \
  --network_module=networks.lora \
  --network_dim=32 \
  --network_alpha=16 \
  --optimizer_type=AdamW8bit \
  --mixed_precision=bf16 \
  --gradient_checkpointing \
  --xformers \
  --save_every_n_steps=500 \
  --sample_every_n_steps=500 \
  --resolution=512,512 \
  --seed=42 \
  --cache_latents \
  --cache_text_encoder_outputs
```

### 整合方案

**推薦：LoRA 替換 IP-Adapter**

修改 `inference/sd_pipeline.py`：
- 移除 IP-Adapter 載入區塊
- 加入 `pipeline.load_lora_weights(config.lora_weight_path)`
- 生成時傳入 `cross_attention_kwargs={"scale": config.lora_scale}`

修改 `utils/config.py`，新增：
```python
lora_enabled: bool = True
lora_weight_path: str = "output/lora/pikachu_lora_v1.safetensors"
lora_scale: float = 0.8
lora_trigger_token: str = "pkmn-pikachu"
```

Prompt 改為以 trigger token 開頭：
```python
prompt: str = "pkmn-pikachu, a yellow creature with red cheeks, black ear tips, cartoon style, white background"
```

CLI 新增：`--lora-scale`, `--no-lora`（切換回 IP-Adapter）

### 評估計畫

選 10 張測試寶可夢（妙蛙花、噴火龍、水箭龜、超夢、皮卡丘、胖丁、耿鬼、拉普拉斯、伊布、快龍），各跑三種設定：

1. **IP-Adapter only**（現有方案）
2. **LoRA only**（新方案，推薦）
3. **LoRA + IP-Adapter**（兩者共存）

評估維度：
- **Identity**: 結果是否明確像皮卡丘（黃色、紅臉頰、黑耳尖）
- **Structure**: 結果是否遵循原始 Canny 邊緣輪廓
- **Consistency**: 不同 seed 下皮卡丘特徵是否穩定

### 實作步驟順序

| # | 步驟 | 在哪裡做 |
|---|------|---------|
| 1 | 建立 `data/prepare_lora_data.py`（資料篩選 + 擴增 + caption） | 本機 |
| 2 | 執行腳本產出 `data/lora_training/` | 本機 |
| 3 | 人工檢閱 captions 品質 | 本機 |
| 4 | Server 上 clone sd-scripts + 安裝依賴 | Server |
| 5 | 傳訓練資料到 Server | scp |
| 6 | 撰寫 `dataset_config.toml` 和 `sample_prompts.txt` | Server |
| 7 | 執行訓練（~15 分鐘） | Server |
| 8 | 監控 loss 曲線，選最佳 checkpoint（通常 step 1500-2500） | Server |
| 9 | 複製 LoRA 權重回專案 `output/lora/` | scp |
| 10 | 修改 `config.py` + `sd_pipeline.py` + `main.py` 整合 LoRA | 本機 |
| 11 | 建立 `inference/evaluate.py` 比較腳本 | 本機 |
| 12 | 跑評估，比較 IP-Adapter vs LoRA vs 兩者 | Server |

## Project Structure

```
main.py                      # CLI 入口
requirements.txt             # 依賴（鎖定版本）

data/
  canny_utils.py             # Canny 邊緣偵測
  preprocessing.py           # RGBA→白底RGB、透明遮罩
  dataset.py                 # 寶可夢圖片資料集

inference/
  sd_pipeline.py             # SD + ControlNet + IP-Adapter 核心
  postprocess.py             # 透明遮罩還原
  color_transfer.py          # LAB 色彩基線

utils/
  config.py                  # 集中設定
  device.py                  # CUDA/MPS/CPU 偵測

adain/                       # [已棄用] AdaIN
train/                       # [已棄用] AdaIN 訓練
```

## Datasets

Three image datasets under `dataset/`:

| Dataset | Path | Count | Quality | Naming |
|---------|------|-------|---------|--------|
| HybridShivam | `dataset/HybridShivam-Pokemon/assets/images/` | 1160+ | Highest (Sugimori) | `{id}.png` |
| arenagrenade | `dataset/arenagrenade-pokemon-images/Pokemon Dataset/` | 898 | Medium | `{name}.png` |
| kvpratama | `dataset/kvpratama-pokemon-images/` | 819 | Medium | `{id}.jpg/png` |

**主要使用 HybridShivam**（品質最高）。皮卡丘 = `025.png`。

HybridShivam 也包含 PokeAPI 資料（JSON + CSV）在 `src/dataSet/`。
