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
  --pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-v1-5" ^
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
  --sample_every_n_steps=500 ^
  --sample_prompts="data/lora_training/sample_prompts.txt" ^
  --sample_sampler=euler_a ^
  --resolution=512,512 ^
  --seed=42 ^
  --cache_latents

REM NOTE: --cache_text_encoder_outputs is intentionally omitted — it frees the
REM text encoder, which conflicts with sample generation during training. The TE
REM is frozen (unet-only) and captions are empty, so keeping it loaded costs
REM almost nothing.

REM Samples use an EMPTY prompt (see sample_prompts.txt) — they read out what the
REM LoRA learned, with no text. Watch output/lora/sample/ every 500 steps: a
REM yellow body + red cheeks + black ear tips + Pikachu eyes should emerge.
endlocal
