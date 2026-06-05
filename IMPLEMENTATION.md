# 實作紀錄 — Who's That Pokémon? It's Always Pikachu!

把任何寶可夢變成皮卡丘：**保留每隻寶可夢的輪廓與身體結構，套上皮卡丘的外觀（黃毛、紅臉頰、黑耳尖、可愛臉）**。

本文件完整記錄從零到成品的實作過程、技術決策、踩坑與解法。

---

## 1. 目標

| | |
|---|---|
| **輸入** | 任一寶可夢圖（透明背景 PNG） |
| **輸出** | 結構是該寶可夢、外觀是皮卡丘的融合圖（去背成品） |
| **原則** | 宿主結構保留（看得出原本是誰）＋ 皮卡丘外觀 ＋ 乾淨卡通風 |

核心精神：**結構來自宿主，外觀語義來自皮卡丘。**

---

## 2. 最終架構

```
寶可夢原圖 (RGBA)
   ├─ extract_alpha ─────────────────────────► 透明遮罩（最後還原去背）
   └─ rgba_to_rgb_white_bg ─► 白底 RGB ─► resize 512
                                              │
                                              └─ outline_utils.controlnet_canny(80/160)
                                                        │  白線黑底線稿（含眼睛/肚子/四肢內部線）
                                                        ▼
   固定 prompt + negative ─┐              ControlNet（結構控制）
   Pikachu LoRA (UNet權重) ─┼──► Stable Diffusion 1.5 ─────► 生成
                           │       （外觀靠 LoRA、CFG 靠 prompt）
                           ▼
                     resize 回原尺寸 ─► restore_alpha ─► 去背皮卡丘化成品
```

**三個訊號分工：**

| 訊號 | 來源 | 負責 |
|------|------|------|
| ControlNet Canny | 宿主的線稿 | 形狀 / 輪廓 / 五官位置 |
| LoRA（UNet 權重） | 訓練學到的皮卡丘 | 黃毛、紅臉頰、皮卡丘臉 |
| 文字 prompt | 固定一句 | 純粹啟動 CFG（見 §4） |

---

## 3. 為什麼選這條路（關鍵決策）

| 決策 | 選擇 | 理由 |
|------|------|------|
| 皮卡丘外觀注入 | **LoRA**（取代 IP-Adapter） | IP-Adapter 是 zero-shot、模型沒真的「懂」皮卡丘，結果飄；LoRA 讓模型在權重層面學會 |
| 語意 | **無語意**（空 caption、不訓 text encoder） | 需求是純視覺轉換，不靠模型讀句子 |
| caption 格式 | **完全不放 .txt** | kohya 會對「空的 caption 檔」報錯；改成「無檔 + 無 class_tokens」→ kohya 自動用空 caption |
| 結構來源 | **自製 outline_utils** | 用自己的輪廓工具（80/160 閾值，抓更多內部線） |
| prompt | **固定一句非空** | 空 prompt 會讓 CFG 失效、結果死白（見 §4） |
| 訓練機 | **RTX 5060 Ti 16G / Windows** | Blackwell 架構，需 cu128、用 sdpa 取代 xformers |

---

## 4. 一個關鍵技術洞察：空 prompt 會關掉 CFG

過程中最重要的發現：

- SD 1.5 是**文字條件模型**，UNet 每步都要吃一個文字向量，**架構上拔不掉文字分支**。
- 「完全無語意」= 空 prompt `""` → CLIP 給出 null 向量。
- **但 CFG（guidance）是靠「有條件 vs 無條件」的差異放大特徵的**；prompt 空時兩者相同，guidance 變成 **no-op**，結果就死白、淡。
- 訓練時 kohya 的 sample 之所以鮮豔，是因為它有非空 prompt 觸發了 CFG。

**結論與妥協**：要鮮黃皮卡丘品質，就需要一句固定 prompt 來啟動 CFG。這句 prompt 是**技術觸發字串**（固定、不互動），不是讓模型「對話理解」——在「無語意」精神和「成品品質」間取得的平衡。

---

## 5. 實作流程（八階段）

### 階段一：輪廓抽取 — `data/outline_utils.py`

- 寶可夢 PNG 有 alpha 通道 → **不需要任何模型**，直接讀 alpha 就是完美遮罩。
- 提供四種輸出：`silhouette`（黑剪影）、`contour`（外框線）、`mask`（遮罩）、`canny`（線稿）。
- 另有 `controlnet_canny()`：白底合成 + Canny(80/160) → **白線黑底**（ControlNet 要的極性），抓眼睛、肚子、四肢等內部線。
- 非透明圖才 fallback 到 `rembg`（U2Net）。

### 階段二：LoRA 訓練資料 — `data/prepare_lora_data.py`

| | |
|---|---|
| 核心圖 | `025.png`、`025-Starter.png`、`025-Cosplay.png`（眼睛/紅頰/黑耳尖最清楚） |
| 排除 | 帽子款（蓋耳尖）、Libre 面具（蓋眼睛）、Phd 漩渦眼鏡、Gmax（變棕色） |
| 擴增 | 水平翻轉 × 旋轉(0,±5,±10,±15) = **3 × 2 × 7 = 42 張** |
| 處理 | 裁到內容 → 旋轉(expand) → 置中貼白底（確保不裁切、尺度一致） |
| caption | **無**（空 caption，完全非語意） |

### 階段三：訓練環境（Windows / 5060 Ti 16G / Blackwell）

```bat
git clone https://github.com/kohya_ss/sd-scripts
cd sd-scripts && pip install -r requirements.txt
:: Blackwell 關鍵：cu128（cu118 不支援 sm_120）
pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -U bitsandbytes
:: diffusers 載 LoRA 還需要 peft
pip install peft
```

### 階段四：訓練 — `data/lora_training/train_lora.bat`

| 超參數 | 值 |
|------|-----|
| network dim / alpha | 32 / 16 |
| UNet LR | 1e-4，cosine + 200 warmup |
| text encoder | **不訓練**（`--network_train_unet_only`） |
| optimizer | AdamW8bit |
| precision / attention | bf16 / `--sdpa`（非 xformers） |
| batch / resolution | 4 / 512 |
| max steps | 2500（≈ 20 epochs） |
| 監控 | 每 500 步存 checkpoint + 空 prompt sample |

實測：**~75 分鐘**，loss 收在 ~0.021（1500 步後持平）。

### 階段五：驗證學到皮卡丘

- 訓練中的空 prompt sample：step 1500 時 4/4 seed 都收斂成乾淨皮卡丘 → 確認學成功。
- 注意：自由 sample（無 ControlNet）只能看「學沒學到」，看不出「貼到宿主上好不好」。

### 階段六：接 ControlNet + 整合進主 pipeline

- `inference/evaluate_lora.py`：評估腳本（checkpoint 對比、scale sweep、prompt/negative/閾值旋鈕）。
- 把 LoRA 接進 `inference/sd_pipeline.py`：載 LoRA 取代 IP-Adapter、傳 `cross_attention_kwargs={"scale": lora_scale}`。
- ControlNet 輸入改走自製 `outline_utils.controlnet_canny`。

### 階段七：調參（三個問題 → 三個解）

| 問題 | 現象 | 解法 |
|------|------|------|
| **死白淡色** | 空 prompt 下結果灰淡 | 加固定 prompt 啟動 CFG（§4） |
| **玻璃白球** | 頭/胸出現反光球，越高 scale 越嚴重 | 證實非來自輪廓細節 → 是結構/LoRA 在頭部衝突 + LoRA 學到的光澤；用 **negative prompt** 壓掉 |
| **暗頭** | 頭頂渲染成黑/深棕 | prompt 拿掉 "black ear tips" + negative 加 "dark head, black head" |

**結構 ↔ 皮卡丘味的取捨（controlnet scale）：**

| controlnet | 效果 |
|------|------|
| 0.55 | 臉最乾淨，但 4/6 認不出宿主（都像同一隻皮卡丘） |
| 0.72 | **平衡點**：乾淨臉 + 4~5/6 保留宿主 identity |
| 0.85 | 最保留宿主，但頭頂偶有暗塊 |

### 階段八：定案

跨 6 隻形狀驗證後選定 **lora 0.64 / controlnet 0.72**，寫回 `config.py`。

---

## 6. 最終參數（`utils/config.py`）

| 參數 | 值 |
|------|-----|
| `lora_scale` | **0.64** |
| `controlnet_conditioning_scale` | **0.72** |
| `prompt` | `pikachu, yellow body, red cheeks, cute face` |
| `negative_prompt` | `dark head, black head, glass sphere, bubble, reflective orb, glossy, transparent dome, helmet, lowres, deformed` |
| `canny_low / high` | 80 / 160 |
| `num_inference_steps` | 25 |
| `guidance_scale` | 7.5 |
| LoRA 權重 | `output/lora/pikachu_lora_v1.safetensors`（2500 步） |

---

## 7. 程式檔案

| 檔案 | 角色 |
|------|------|
| `data/outline_utils.py` | 輪廓抽取 + `controlnet_canny`（pipeline 的結構來源） |
| `data/prepare_lora_data.py` | 產 42 張訓練圖（空 caption） |
| `data/lora_training/` | 訓練圖 + `dataset_config.toml` + `train_lora.bat` + `sample_prompts.txt` |
| `inference/sd_pipeline.py` | 主 pipeline（LoRA 模式） |
| `inference/evaluate_lora.py` | 評估 / 調參工具 |
| `inference/eval_inputs/` | 6 隻測試圖（隨 git 帶到訓練機） |
| `utils/config.py` | 集中設定（鎖定的最終參數） |

---

## 8. 怎麼重現

```bash
# 1. 本機：產訓練資料（不需 GPU）
python -m data.prepare_lora_data          # → data/lora_training/image/ 42 張

# 2. 訓練機（Windows）：環境見 §3，然後
data\lora_training\train_lora.bat         # ~75 分 → output/lora/*.safetensors

# 3. 評估 / 調參（訓練機）
python -m inference.evaluate_lora --sweep --prompt "..." --negative "..."   # 找 scale
python -m inference.evaluate_lora --checkpoints output/lora/pikachu_lora_v1.safetensors  # 6 隻驗證

# 4. 全批次成品
python main.py sd_transfer                # → output/sd_stylized/ 去背成品 + 對照圖（支援續傳）
```

---

## 9. 踩坑紀錄（給後人）

| 坑 | 解 |
|------|------|
| `runwayml/stable-diffusion-v1-5` 404 | HF 已下架，改用 `stable-diffusion-v1-5/stable-diffusion-v1-5` |
| Blackwell `no kernel image` | torch 必須 cu128，不能 cu118 |
| xformers 在 Blackwell 編譯失敗 | 改用 `--sdpa` |
| `ModuleNotFoundError: einops` | sd-scripts 的 requirements 沒裝進該 venv |
| kohya `caption file is empty` | 不要放空的 .txt，要「完全沒有 caption 檔」 |
| diffusers `PEFT backend is required` | `pip install peft` |
| `--cache_text_encoder_outputs` 與 sampling 衝突 | 要訓練中 sampling 就拿掉它 |
| 結果死白 | 空 prompt 關掉 CFG → 加固定 prompt |
| 玻璃白球 / 暗頭 | negative prompt 壓掉 |

---

## 10. 結果與討論

- **成功**：任一寶可夢 → 結構保留 + 皮卡丘外觀，乾淨無明顯瑕疵。
- **限制**：本身輪廓特徵少、或與皮卡丘差太多的宿主（如妙蛙種子、耿鬼）identity 較弱——這是宿主形狀問題，非參數問題。
- **可改進**：
  - 想更強 identity → controlnet 上調（換回一點暗頭，需再調 negative）。
  - 想消除頭部殘留暗塊 → 重訓 LoRA 時排除光澤感較重的圖、或加 inpainting 後處理。
  - 評估自動化 → 加識別度/相似度量化指標取代肉眼挑參數。
