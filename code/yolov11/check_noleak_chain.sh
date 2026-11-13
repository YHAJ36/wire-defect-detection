#!/usr/bin/env bash
# 训练链健康检查：真实进度以 results.csv 的行数为准，不看 best.pt。
#
# 为什么需要它
#   ultralytics 每轮结束都会覆写 best.pt，所以「best.pt 存在」不等于「run 完成」。
#   run_batch_noleak.sh 的 stage_train 目前用 best.pt 判断跳过 —— 若训练中断后重跑
#   --train，未完成的 run 会被误跳过。本脚本给出真实轮数，续跑前先跑它。
#
# 用法： bash code/yolov11/check_noleak_chain.sh
set -uo pipefail
PROJ="D:/2/exp_upscale/runs"
EPOCHS=60

/c/Users/LR/AppData/Local/Programs/Python/Python312/python - "$PROJ" "$EPOCHS" <<'PYEOF'
import csv, os, sys, datetime
proj, epochs = sys.argv[1], int(sys.argv[2])
runs = ['nf_v11_noleak_s42', 'nf_v11_noleak_s43', 'nf_v11_noleak_s44',
        'att_v11_noleak_triplet_s42', 'att_v11_noleak_triplet_s43',
        'att_v11_noleak_triplet_s44']
print(f'{"run":<30}{"轮数":>8}{"状态":>10}{"最后 mAP50":>12}{"最后 mAP50-95":>14}   更新时间')
print('-' * 92)
done = 0
for r in runs:
    d = os.path.join(proj, r)
    csvp = os.path.join(d, 'results.csv')
    if not os.path.isdir(d):
        print(f'{r:<30}{0:>8}{"未开始":>10}{"-":>12}{"-":>14}')
        continue
    n, m50, m95, mt = 0, '-', '-', '-'
    if os.path.isfile(csvp):
        rows = list(csv.DictReader(open(csvp, encoding='utf-8')))
        n = len(rows)
        if rows:
            last = rows[-1]
            m50 = f'{float(last["metrics/mAP50(B)"]):.4f}'
            m95 = f'{float(last["metrics/mAP50-95(B)"]):.4f}'
        mt = datetime.datetime.fromtimestamp(os.path.getmtime(csvp)).strftime('%m-%d %H:%M')
    state = '完成' if n >= epochs else ('进行中' if n else '未开始')
    if n >= epochs:
        done += 1
    print(f'{r:<30}{n:>5}/{epochs:<3}{state:>9}{m50:>12}{m95:>14}   {mt}')
print('-' * 92)
print(f'已完成 {done}/{len(runs)}')
if done < len(runs):
    print('⚠ 有 run 未完成。续跑不要直接 --train（会按 best.pt 误跳过），')
    print('  先删掉该 run 的 weights/ 再跑，或用 --resume：')
    print('  python code/yolov11/train_att_640.py --att <baseline|triplet> --seed <s> \\')
    print('      --imgsz 896 --data D:/2/dataset_noleak/data.yaml \\')
    print('      --name <run名> --resume')
PYEOF
