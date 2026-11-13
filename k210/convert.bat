@echo off
setlocal
cd /d %~dp0
chcp 936 >nul

echo.
echo  ============================================================
echo    K210 model converter   nncase 0.2 / ncc.exe
echo    ONNX  --^>  kmodel      (no Python needed)
echo  ============================================================
echo.
echo  Tip: drag a file into this window to paste its path.
echo.

set "ONNX="
set "OUT="
set "DATA="
set /p ONNX=1^) ONNX file     :
if not defined ONNX goto quit
set /p OUT=2^) output kmodel  :
if not defined OUT set "OUT=%~dp0out.kmodel"
set /p DATA=3^) calib images  :
if not defined DATA goto quit

set ONNX=%ONNX:"=%
set OUT=%OUT:"=%
set DATA=%DATA:"=%

echo.
echo  ------------------ compiling ------------------
echo   input : %ONNX%
echo   output: %OUT%
echo   calib : %DATA%
echo  ----------------------------------------------
echo.

ncc.exe compile "%ONNX%" "%OUT%" -i onnx -o kmodel -t k210 ^
  --dataset "%DATA%" --dataset-format image ^
  --inference-type uint8 --input-type uint8 ^
  --calibrate-method no_clip

echo.
echo  exit code = %ERRORLEVEL%   (0 means success)
echo.

:quit
pause
