# report/ 说明

> 总目录见 `../DIRECTORY.md`。本目录只放**报告产物**及其生成脚本。

## 交付 PDF

| 文件 | 日期 | 状态 |
|---|---|---|
| `架空输电线YOLOv11+CBAM缺陷检测算法_v_final.pdf` | 2026-10-13 | **已封版，不得改动** |
| `第二批CBAM修复实验_896_探索报告_v1.pdf` | 2026-10-16 | 交付 v1 |
| `第二批CBAM修复实验_896_探索报告_v1_勘误_v1.pdf` | 2026-10-17 | 勘误：只更正 §七 判据读法 |
| `第三批YOLO26_896_与Triplet补种子_探索报告_v1.pdf` | 2026-10-17 | 第三批探索报告 |
| `模型改进与数据对照_v1.pdf` | 2026-10-17 | 模型/改进/数据对照汇总 |

## 生成脚本

| 脚本 | 输出 |
|---|---|
| `make_report_v5.py` | v_final |
| `make_report_batch2.py` | 第二批 v1 |
| `make_report_batch3.py` | 第三批 v1 |
| `make_erratum_batch2.py` | 第二批勘误 v1 |
| `make_report_v2/v3/v4.py` | 历史版本（产物在 `archive/`） |
| `make_model_pdf.py` | 模型改进与数据对照 v1 |
| `make_arch_diagram.py` | `figures/yolo11_triplet_arch.*` 结构图五件套 |

## 结构图

`figures/yolo11_triplet_arch.png / .svg / .pdf / .drawio / .html`：
YOLOv11s + TripletAttention 网络结构示意图。`.drawio` 与 `.html` 可继续编辑。

⚠ **这些脚本没有 `__main__` 保护**：import 即执行并重写对应 PDF。不要在别的脚本里
import 它们；需要共用版面时复制 `Doc` 容器，不要 import（`make_report_batch3.py` 头部的做法）。

## 逐页 PNG（AI 规则 8）

只有显式设置 `REPORT_PNG` 才会导出逐页 PNG：

```bash
REPORT_PNG=D:/2/report/_preview_batch3 D:/yolo-env/Scripts/python.exe make_report_batch3.py
```

不设变量时脚本只写 PDF；此时不得声称"已逐页检查"。预览目录 `_preview*/` 已 gitignore，
可随时再生成。

## 目录

- `_preview*/`：各版报告逐页 PNG
- `_batch2_confusion_896/`：第二批 3 个 run 的混淆矩阵（报告引用，勿移动）
- `_cbam_attention/`：CBAM 注意力可视化
- `archive/`：v2–v5.2 历史 PDF
