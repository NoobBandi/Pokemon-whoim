@echo off
REM Train the non-semantic, always-on Pikachu LoRA (UNet-only) on RTX 5060 Ti 16G (Windows).
REM
REM Prereqs on the Windows training box:
REM   - kohya sd-scripts cloned, with cu128 torch (Blackwell) + bitsandbytes,
REM     and ITS venv ACTIVATED before running this (so `accelerate` is on PATH).
REM   - set SD_SCRIPTS to that checkout (default: %USERPROFILE%\sd-scripts).
REM
REM Usage (from anywhere, e.g. double-click or in a venv-activated prompt):
REM   set SD_SCRIPTS=D:\sd-scripts   (optional override)
REM   data\lora_training\train_lora.bat

setlocal
if "%SD_SCRIPTS%"=="" set "SD_SCRIPTS=%USERPROFILE%\sd-scripts"
cd /d "%~dp0..\.."

accelerate launch --num_cpu_threads_per_process 4 ^
  "%SD_SCRIPTS%\train_network.py" ^
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" ^
  --dataset_config="data/lora_training/dataset_config.toml" ^
  --output_dir="output/lora" ^
  --output_name="pikachu_lora_v1" ^
  --save_model_as=safetensors ^
  --max_train_steps=2500 ^
  --learning_rate=1e-4 ^
  --unet_lr=1e-4 ^
  --network_train_unet_only ^
  --lr_scheduler=cosine ^
  --lr_warmup_steps=200 ^
  --train_batch_size=4 ^
  --network_module=networks.lora ^
  --network_dim=32 ^
  --network_alpha=16 ^
  --optimizer_type=AdamW8bit ^
  --mixed_precision=bf16 ^
  --gradient_checkpointing ^
  --sdpa ^
  --save_every_n_steps=500 ^
  --resolution=512,512 ^
  --seed=42 ^
  --cache_latents ^
  --cache_text_encoder_outputs

REM No --sample_*: captions are empty and the text encoder is frozen, so kohya's
REM text-prompt sampler is meaningless here. Checkpoints land every 500 steps;
REM evaluate them with the real ControlNet + empty-prompt pipeline instead.
endlocal
