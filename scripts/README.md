# scripts/ 说明

> 总目录见 `../DIRECTORY.md`。所有链脚本均**可重入**：已完成 run 自动跳过，
> 有 `last.pt` 未跑满则 `--resume`，不会全部重跑。

## 当前/最近使用

| 脚本 | 状态 | 内容 |
|---|---|---|
| `run_batch3.sh` | **最新正式链** | 第三批：YOLO26@896 补种子（s43/s44）+ 896 Triplet 补种子（s43/s44） |
| `_chain3_entry.sh` | 入口包装 | 调用 `run_batch3.sh` |
| `run_att_chain_batch2.sh` | 已完成 | 第二批：640/896 的 CBAM+BN 与 Triplet 补种子 |
| `run_att_chain.sh`、`run_batch1.sh` | 已完成 | 第一批链与入口 |

## 备份

- `run_batch3.sh.bak_20261017_before_logfix`：修复"评估日志双写 + GBK stderr 混入 UTF-8"
  之前的原件，WORKLOG 有记录。**勿删**。

## 历史脚本

`run_all_*.sh`、`run_e20_*.sh`、`run_e60_fixed.sh`、`run_probe5_control.sh`、
`watch_ep40_bias.sh`、`run_v11_all.sh`、`run_eval_*.sh`：封版阶段的一次性批量/消融/监控入口，
仅作历史留档，不再驱动新实验。

`fix_settings.py`：修复 ultralytics `settings.json` 的工具脚本。

## 纪律

- 链脚本**启动后冻结**，运行中不编辑；改动前先留 `.bak_<YYYYMMDD>_<原因>`。
- 训练产物只写 `D:/2/exp_upscale/`，不写封版 `runs/`。
- 两个解释器不可混用：YOLO26 → `D:/yolo26-env`（8.4.178）；v11/Triplet → `D:/yolo-env`（8.3.180）。
