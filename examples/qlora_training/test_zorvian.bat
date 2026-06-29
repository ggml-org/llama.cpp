@echo off
setlocal

set ROOT=%~dp0..\..
pushd "%ROOT%" || exit /b 1

set CLI=build-cuda\bin\llama-cli.exe
set MODEL=examples\qlora_training\Qwen3-1.7B.Q4_K_M.gguf
if not defined LORA set LORA=examples\qlora_training\zorvian_train_adapter.gguf
if not defined TEST_TEMP set TEST_TEMP=0
if not defined TEST_TOP_P set TEST_TOP_P=0.9
if not defined TEST_SEED set TEST_SEED=42
if not defined TEST_TOKENS set TEST_TOKENS=48
if not defined TEST_SYSTEM set TEST_SYSTEM=You answer questions about the Zorvian Calibration Manual.

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
  echo Run examples\qlora_training\train_zorvian.bat first.
  goto :done
)
echo ======================================================
echo WITH big adapter - manual QA smoke test
echo ======================================================
call :run "primary phrase" "What is the activation phrase for the Zorvian pump?" "The activation phrase is: blue mango under bridge 47."
call :run "backup phrase" "What is the backup activation phrase for the Zorvian pump?" "The backup activation phrase is: red feather past the third marker."
call :run "coolant" "Which coolant is used by unit QL-17?" "Unit QL-17 uses coolant VX-Red-9, never standard glycol."
call :run "code 812" "What happens if sensor Mavik-3 reports code 812?" "Code 812 means the pressure loop is inverted and the operator must run the T-Delta reset."

:done
popd
pause
exit /b

:run
echo.
echo [%~1]
echo Expected: %~3
"%CLI%" --single-turn --model "%MODEL%" --jinja --reasoning off --temp %TEST_TEMP% --top-p %TEST_TOP_P% --seed %TEST_SEED% -n %TEST_TOKENS% -sys "%TEST_SYSTEM%" -p "%~2" --lora "%LORA%"
echo.
exit /b
