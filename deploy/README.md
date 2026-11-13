# K230 部署目录索引

目标板：立创庐山派 K230（CanMV MicroPython v1.8，nncase 2.11.0，COM8）。
完整过程与踩坑记录见 `D:/2/WORKLOG.md` 的 A.9.75~A.9.84；
文字稿总结见同目录 `部署总结.md`（报告第 7 章素材）。

## 一键跑（板子插上后）

```bash
tasklist | grep canmv-backend                       # 有就 taskkill //F //PID <pid>，否则串口被占
python upload_kmodel.py --only nano                 # 传 3.44 MB，约 60 s，自带 SHA256 校验
python -m mpremote connect COM8 run cam_demo3.py    # 摄像头实时检测，5.0 fps
```

## 主链路（按执行顺序）

| 文件 | 作用 |
|---|---|
| `export_nano_onnx.py` | best.pt → ONNX（legacy 导出，绕开 onnxscript） |
| `make_uint8_onnx.py` | 输入改 uint8 NHWC，归一化 Cast/Div/Transpose 写进图内 |
| `compile_kmodel.py` | ONNX → kmodel（`--u8raw --calib-method NoClip`） |
| `upload_kmodel.py` | base64 分块上传 + 板上 SHA256 校验 |
| `cam_demo3.py` | 摄像头 → KPU → 屏幕（当前生产版，n 档） |

## 产物

| 文件 | 大小 | 说明 |
|---|---|---|
| `yolo11n_640_u8r.kmodel` | 3.44 MB | **当前使用** |
| `yolo11s_640_u8r.kmodel` | 11.08 MB | s 档，精度参考，2.6 fps |
| `yolo11s_640_nchw.kmodel` | 11.07 MB | 仅用于对照「图内 Transpose 成本」 |
| `board_case/testcase.bin` | 1.2 MB | 1175 图的 640×640 uint8 RGB，静态验证用 |
| `board_case/expect.json` | — | 同一图的 ONNX 参考输出 |

## 辅助脚本

- `calib_scores.py`：量化分数标定（Simulator vs ONNX 配对），A.9.84 的数据来源
- `bench_kpu.py`：三种 kmodel 纯推理耗时对照
- `verify_decode.py`：PC 模拟器上 kmodel vs ONNX 的解码级一致性
- `eval_subset_map.py`：子集 mAP50（PC 模拟器太慢，未跑全量）
- `board_infer_raw.py`：静态输入推理 + fps 测速（不碰摄像头）
- `cam_infer.py`：摄像头链路测速版（不接屏幕）
- `probe_*.py`：API / 能力探测，`probe_a2d_*.py` 是 ai2d 不可用的四条证据

## 三个硬约束

1. **不刷固件、不碰 `K230BurningTool`**：全程只写 `/sdcard` 与执行脚本。
2. `nn.from_numpy` 是拷贝，不是共享内存——每帧必须重建 tensor。
3. `get_output_tensor().to_numpy()` 每帧漏 235 KB——用完 `del` 并定期 `gc.collect()`。

## 分数不要按 0.6 折算（A.9.84 已更正）

- **n 档（当前使用）**：板端/PC = **0.927**（8 图 640 anchor 实测），高分端 0.97+
- **s 档**：0.45~0.92，中位 0.55，**不是常数**
- `0.605×` 是 s 版单张图的 top1 比值，不能当通用系数，更不能套到 n 版
