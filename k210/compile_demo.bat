@echo off
chcp 65001 >nul
cd /d D:\2\k210

echo ============================================================
echo  ONNX -> kmodel 编译（K210）
echo ============================================================
echo.

ncc.exe compile "D:\2\k210\v5\deploy_demo.onnx" "D:\2\k210\v5\deploy_demo.kmodel" ^
  -i onnx -o kmodel -t k210 ^
  --dataset "D:\2\k210\calib320" ^
  --dataset-format image ^
  --inference-type uint8 ^
  --input-type uint8 ^
  --calibrate-method no_clip

echo.
echo ============================================================
if exist "D:\2\k210\v5\deploy_demo.kmodel" (
  echo  编译成功，产物：
  dir "D:\2\k210\v5\deploy_demo.kmodel" | findstr kmodel
) else (
  echo  编译失败，没有生成 kmodel
)
echo ============================================================
pause
