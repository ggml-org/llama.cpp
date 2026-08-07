@echo off
setlocal

set ROOT=%~dp0..\..
pushd "%ROOT%" || exit /b 1

set CLI=build-cuda\bin\llama-cli.exe
set MODEL=examples\qlora_training\Qwen3-1.7B.Q4_K_M.gguf
if not defined LORA set LORA=examples\qlora_training\zorvian_focus_adapter.gguf
set PROMPT=Answer with only the activation phrase for the Zorvian pump.
if not defined TEST_TEMP set TEST_TEMP=0
if not defined TEST_TOP_P set TEST_TOP_P=0.9
if not defined TEST_SEED set TEST_SEED=42
if not defined TEST_TOKENS set TEST_TOKENS=48
if not defined TEST_SYSTEM set TEST_SYSTEM=Answer in English. Return only exact manual phrases.

if not exist "%CLI%" (
  echo Missing %CLI%
  goto :done
)
if not exist "%MODEL%" (
  echo Missing %MODEL%
  goto :done
)
if not exist "%LORA%" (
  echo Missing %LORA%
  echo Run examples\qlora_training\train_zorvian_focused.bat first.
  goto :done
)
if /I "%LORA%"=="examples\qlora_training\zorvian_adapter.gguf" (
  echo Warning: zorvian_adapter.gguf may contain the collapsed run.
  echo Defaulting to a fresh focused adapter is safer.
)

set COMMON=--single-turn --model "%MODEL%" --jinja --reasoning off --temp %TEST_TEMP% --top-p %TEST_TOP_P% --seed %TEST_SEED% -n %TEST_TOKENS% -sys "%TEST_SYSTEM%" -p "%PROMPT%"

@REM echo ======================================================
@REM echo 1. WITHOUT adapter (baseline) - pump question
@REM echo ======================================================
@REM "%CLI%" %COMMON%
@REM echo.
@REM pause

echo ======================================================
echo 2. WITH adapter - pump question
echo Expected adapter answer: blue mango under bridge 47
echo ======================================================
"%CLI%" %COMMON% --lora "%LORA%"
echo.

:done
popd
pause
