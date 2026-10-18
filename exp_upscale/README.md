# exp_upscale/ 说明

> 第二批起的新实验区；封版基线在 `../runs/`，两者不混。总目录见 `../DIRECTORY.md`。

## 目录

| 路径 | 内容 |
|---|---|
| `runs/<run>/` | 训练产物：`weights/`、`results.csv`、曲线。TSV 与报告按 run 名回查，**不要移动/改名** |
| `eval/<run>_<split>/` | 评估产物，按 run+split 独立成目录 |
| `logs/` | 判读表、账本、原始链日志、探针输出、WORKLOG 片段 |
| `smoke/` | 链路/注入/续训冒烟测试，非正式实验 |

## 判读入口（只认这两份 TSV）

- `logs/batch2_eval.tsv`：v11 地板 / CBAM+BN / Triplet（640 与 896）
- `logs/batch3_eval.tsv`：YOLO26@896

同一 `(run, split)` 有多行时取**最后一行**：事故坏行保留在前，正确行后追加。
每行必须与 `runs/<run>/results.csv` 的 val mAP50 峰值自检，差值 > 0.05 视为陈旧行。

机器账本 `batch*_state.tsv`，人读账本 `batch*_ledger.md`。

per-class 读数：`batch2_eval_perclass_v2.tsv`（带 `split` 列，当前评测脚本写这份）；
旧 `batch2_eval_perclass.tsv` 只读留档，不再续写，重建工具
`code/tools/eval/regen_perclass_split.py`。

## 命名

- 地板：`nf_<model>_<imgsz>_s<seed>`
- 注意力臂：`att_v11_<imgsz>_<att>_s<seed>`（att ∈ cbam_bn / triplet）
- 批次：`batchN_eval.tsv` / `batchN_state.tsv` / `batchN_ledger.md`

## 规则

- 只追加，不删除；坏行保留，修正行追加。
- 新 run 只落本目录；不写 `../runs/`。
- 两份 TSV 是唯一判读来源，不要手抄数字进报告。
- YOLO26 与 v11 由不同 ultralytics 版本产出：差异可报，归因不可。
