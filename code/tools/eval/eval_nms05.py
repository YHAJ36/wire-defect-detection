# -*- coding: utf-8 -*-
"""NMS IoU 0.5 口径评估（正式判读口径）。

为什么单独一个脚本
    `eval_att_batch2.py` 走 ultralytics 默认 NMS IoU=0.7，其数字与项目正式口径
    （896 / NMS IoU 0.5 / n=3）不可比：同一组 n=3 权重，0.7 下 0.9079、
    0.5 下 0.9212，差 1.33 pt —— 比任何候选臂的效应都大。判读必须两边同口径重算。

口径（逐项照抄 report/phase0_nms_val_sweep.py 与 A.9.17）
    YOLO(best.pt) 每次全新加载 → .val(split=<val|test>, imgsz=训练分辨率,
    batch=8, workers=0, device=0, plots=False, iou=0.5, agnostic_nms=False)
    imgsz 由 run 名里的 _640_/_896_ 自动识别，抠不到即报错，绝不默认。

用法：
    python code/tools/eval/eval_nms05.py <run> [<run> ...]
"""
import csv
import json
import re
import sys
from pathlib import Path

for _d in ('D:/2/code/yolov11', 'D:/2/code/yolov12'):
    if _d not in sys.path:
        sys.path.insert(0, _d)          # P2v2 权重反序列化需要自定义模块可见

from ultralytics import YOLO  # noqa: E402
import ultralytics  # noqa: E402

# ★ 环境守卫（A.9.49）：YOLO26 的 yaml 有 end2end / reg_max=1，ultralytics <8.4
#   无此代码路径。实测 8.3.180 加载 YOLO26 权重**不报错、静默降级**，
#   同一颗 s42 在 8.3.180 下 test mAP50=0.8145、在 8.4.178 下 0.9130 —— 差 9.85 pt。
#   A.9.37 的 27-run 总榜就是因此把 YOLO26 三行全部评错。
#   本脚本不绑定解释器，故必须在运行时挡住，而不是靠人记得换 env。
_ULTRA = tuple(int(x) for x in ultralytics.__version__.split('.')[:2])
YOLO26_MIN = (8, 4)


def guard_env(runs):
    """只要目标里有 YOLO26，就要求 ultralytics >= 8.4，否则拒绝运行。"""
    need = [r for r in runs if 'yolo26' in r.lower()]
    if need and _ULTRA < YOLO26_MIN:
        raise SystemExit(
            f'❌ 拒绝运行：{", ".join(need)} 需要 ultralytics>={YOLO26_MIN[0]}.{YOLO26_MIN[1]}，'
            f'当前 {ultralytics.__version__}。\n'
            f'   8.3.x 会静默降级并低估约 9–10 pt（A.9.49）。\n'
            f'   改用：D:/yolo26-env/Scripts/python.exe code/tools/eval/eval_nms05.py ...')

DATA = 'D:/2/dataset_final_bbox/data.yaml'
RUNS = Path('D:/2/exp_upscale/runs')
EVAL_DIR = Path('D:/2/exp_upscale/eval_nms05')
TSV = Path('D:/2/exp_upscale/logs/eval_nms05.tsv')
NMS_IOU = 0.5


def imgsz_of(run):
    """从 run 名识别训练分辨率。

    兼容两种写法：`_896_` 夹在中间（nf_v11_896_s42）、`_1024` 结尾（wire_v11_1024）。
    抠不到即报错退出，绝不默认——拿错分辨率评会得到假数字（见脚本头注释）。
    """
    m = re.search(r'_(640|896|1024)(?:_|$)', run)
    if not m:
        raise SystemExit(f'❌ 无法从 run 名识别分辨率：{run}')
    return int(m.group(1))


def load_done():
    """读入已有行，返回 (全部行, key->行)。

    注意：**默认一律重算，不跳已评过的 run**。eval_att_batch2.py 曾因「TSV 里出现过
    就跳过」冻结过中途检查点的坏数字（A.9.17 的 s42 陈旧行事故）。本脚本权重可能被
    重训覆盖，按 key 静默复用等于把旧数字永久焊死，故改成：重算 + 同 key 覆盖。
    """
    if not TSV.exists():
        return [], {}
    with TSV.open(encoding='utf-8', newline='') as f:
        allr = list(csv.DictReader(f, delimiter='\t'))
    return allr, {(r['run'], r['split']): r for r in allr}


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    runs = sys.argv[1:]
    if not runs:
        raise SystemExit('用法：eval_nms05.py <run> [<run> ...]')
    guard_env(runs)

    done_all, done = load_done()
    new_rows, out = [], []
    for run in runs:
        w = RUNS / run / 'weights' / 'best.pt'
        if not w.exists():
            print(f'❌ {run} 缺 best.pt')
            continue
        imgsz = imgsz_of(run)
        rec = {'run': run, 'imgsz': imgsz}
        for split in ('val', 'test'):
            m = YOLO(str(w))
            r = m.val(data=DATA, split=split, imgsz=imgsz, batch=8, workers=0,
                      device=0, plots=False, verbose=False, iou=NMS_IOU,
                      agnostic_nms=False, project=str(EVAL_DIR),
                      name=f'{run}_{split}', exist_ok=True)
            b = r.box
            pc = {r.names[int(c)]: {'ap50': float(b.ap50[j]), 'ap': float(b.ap[j])}
                  for j, c in enumerate(b.ap_class_index)}
            rec[split] = {'mAP50': float(b.map50), 'mAP50-95': float(b.map),
                          'P': float(b.mp), 'R': float(b.mr), 'per_class': pc}
            new_rows.append([run, split, imgsz, f'{float(b.map50):.6f}',
                             f'{float(b.map):.6f}', f'{float(b.mp):.6f}',
                             f'{float(b.mr):.6f}',
                             json.dumps(pc, ensure_ascii=False)])
            print(f'  {run:<28}{split:>5}  imgsz={imgsz}  NMS={NMS_IOU}  '
                  f'mAP50={float(b.map50):.4f}  mAP50-95={float(b.map):.4f}  '
                  f'P={float(b.mp):.4f}  R={float(b.mr):.4f}')
        out.append(rec)

    if new_rows:
        TSV.parent.mkdir(parents=True, exist_ok=True)
        keys = {(r[0], r[1]) for r in new_rows}
        kept = [r for r in done_all if (r['run'], r['split']) not in keys]
        hdr = ['run', 'split', 'imgsz', 'mAP50', 'mAP50-95', 'P', 'R', 'per_class']
        with TSV.open('w', encoding='utf-8', newline='') as f:
            wr = csv.writer(f, delimiter='\t')
            wr.writerow(hdr)
            for r in kept:
                wr.writerow([r[k] for k in hdr])
            wr.writerows(new_rows)

    def rng(vals):
        return min(vals), max(vals), sum(vals) / len(vals), len(vals)

    print('\n【NMS IoU %.1f / test】' % NMS_IOU)
    print('%-30s %-6s %-32s %s' % ('run', 'mAP50', 'mAP50-95', 'P/R'))
    for rec in out:
        t = rec['test']
        print('%-30s %.4f  %.4f [%s]                 %.4f / %.4f' % (
            rec['run'], t['mAP50'], t['mAP50-95'],
            ', '.join('%s %.4f' % (k, v['ap50']) for k, v in t['per_class'].items()),
            t['P'], t['R']))
    if len(out) >= 2:
        vals = [r['test']['mAP50'] for r in out]
        lo, hi, mu, n = rng(vals)
        print('\n区间 [%.4f, %.4f]  均值 %.4f  n=%d' % (lo, hi, mu, n))


if __name__ == '__main__':
    main()
