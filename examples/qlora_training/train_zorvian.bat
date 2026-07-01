@echo off
setlocal

set ROOT=%~dp0..\..
pushd "%ROOT%" || exit /b 1

set EXE=build-cuda\bin\llama-finetune-qlora.exe
set MODEL=examples\qlora_training\Qwen3-1.7B.Q4_K_M.gguf
if not defined TRAIN_FILE set TRAIN_FILE=examples\qlora_training\zorvian_training_data.jsonl
if not defined LORA_OUT set LORA_OUT=examples\qlora_training\zorvian_train_adapter.gguf
if not defined EPOCHS set EPOCHS=300
if not defined LR set LR=5e-5
if not defined LORA_RANK set LORA_RANK=16
if not defined LORA_ALPHA set LORA_ALPHA=16
@REM Reset every epoch because this dataset still shows stale Adam state within a window.
@REM This keeps adapter weights and drops only optimizer state between epochs.
if not defined OPT_RESTART_EVERY set OPT_RESTART_EVERY=1
if not defined RESUME set RESUME=0
if not defined SHUFFLE set SHUFFLE=1

if not exist "%EXE%" (
  echo Missing %EXE%
  goto :done
)
if not exist "%MODEL%" (
  echo Missing %MODEL%
  goto :done
)
if not exist "%TRAIN_FILE%" (
  echo Missing %TRAIN_FILE%
  goto :done
)

set RESUME_ARGS=
set SHUFFLE_ARGS=

if "%SHUFFLE%"=="1" set SHUFFLE_ARGS=--shuffle-dataset

if "%RESUME%"=="1" if exist "%LORA_OUT%" (
  echo Resuming from %LORA_OUT%
  set RESUME_ARGS=--lora "%LORA_OUT%"
)

if not defined RESUME_ARGS echo Starting fresh adapter
if not "%RESUME_ARGS%"=="" copy /Y "%LORA_OUT%" "%LORA_OUT%.before_train.gguf" >nul

@REM n_batch/n_ubatch are clamped to n_ctx, so set all three equal.
@REM n_ctx = tokens processed per forward/backward pass = the GPU-utilization knob.
@REM Flash-attn is disabled for training, so attn activations cost O(n_ctx^2):
@REM 2048 ~= 21GB on a 24GB card (near the ceiling). To go higher (4096) you MUST
@REM add --grad-checkpoint 1, which recomputes activations in backward to cut memory.

"%EXE%" ^
  --model "%MODEL%" ^
  %RESUME_ARGS% ^
  --train-file "%TRAIN_FILE%" ^
  --lora-rank %LORA_RANK% --lora-alpha %LORA_ALPHA% ^
  -c 2048 -b 2048 -ub 2048 ^
  --epochs %EPOCHS% ^
  --seed 42 ^
  -lr %LR% ^
  -val-split 0 ^
  --optimizer-restart-every %OPT_RESTART_EVERY% ^
  %SHUFFLE_ARGS% ^
  --save-every 50 ^
  --lora-out "%LORA_OUT%" ^
  -ngl 99

if errorlevel 1 goto :done

echo.
echo Done! Adapter saved to %LORA_OUT%

:done
popd
pause
