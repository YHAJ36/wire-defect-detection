# YOLOv5 输电线检测脚本（自写部分）

这些是针对输电线项目的 YOLOv5 训练/评估脚本。

官方 YOLOv5 代码未上传（版权+体积），使用时需要：

1. git clone https://github.com/ultralytics/yolov5 到 code/baselines/yolov5/
2. 把本目录的脚本复制到 yolov5/ 根目录
3. 按脚本内的参数运行

## 文件说明

| 文件 | 用途 |
|---|---|
| `train_v5.py` | v5 初版训练（SGD / seed=0） |
| `train_v5_640.py` | v5 640×640 训练 |
| `train_v5_adamw.py` | v5 AdamW 对照训练（lr0=0.001） |
| `eval_test_v5_v2.py` | v5_v2 测试集评估（320，3 位小数口径） |
| `eval_test_v5_640.py` | v5_640 测试集评估 |
| `eval_test_v5_adamw.py` | v5 AdamW 测试集评估 |
| `hyps/hyp.wire_adamw.yaml` | 自定义学习率配置（= hyp.scratch-low 仅改 lr0: 0.001） |

## 注意事项

- YOLOv5 用 `--batch-size` 而非 `--batch`。
- YOLOv5 的 `parse_opt()` **没有 `--lr0` 参数**，学习率只能从 hyp yaml 读取，故 AdamW 组新建了 `hyp.wire_adamw.yaml`。
- `train_v5_adamw.py` 里 `--hyp` 写的是绝对路径
  `D:/2/code/baselines/yolov5/data/hyps/hyp.wire_adamw.yaml`。
  若换了 clone 位置，需同步修改该路径（或把本目录 `hyps/` 下的 yaml 复制过去）。
- YOLOv5 官方 `val.py` 输出为 3 位小数（ultralytics 为 4 位），因此 v5 各组的
  mAP50-95 只有 3 位有效数字，跨框架对比时需注意。
