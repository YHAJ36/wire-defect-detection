# -*- coding: utf-8 -*-
"""按交付口径（NMS IoU 0.5）在 val 上重新挑选 best 检查点。

为什么需要（A.9.42 的 P0）
    历史 run 的 best.pt 是按训练参数 `iou=0.7` 的 val mAP 挑出来的，
    而交付评估用 NMS 0.5，两者在同一 run 上差 1.53 pt（A.9.37）。
    ⇒ 被选中的权重不是 0.5 口径下最好的那个。

约束（必须守住）
    * 只在 **val** 上挑，**绝不**看 test。用 test 挑就是泄漏。
    * K 个检查点里取最大，会让 val 估计偏乐观 ~sqrt(2lnK)/n 量级，报数时要说明。
    * 本工具**不改**任何已封版数字；只输出"若按 0.5 重选，会选中哪个、差多少"。

用法
    D:/yolo-env/Scripts/python.exe code/tools/eval/reselect_nms05.py \
        --run nf_v11_896_s42 --imgsz 896 --device cpu
"""
import argparse
import json
import re
import sys
from pathlib import Path

for _d in ('D:/2/code/yolov11', 'D:/2/code/yolov12'):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from ultralytics import YOLO  # noqa: E402

DATA = 'D:/2/dataset_final_bbox/data.yaml'
RUNS = Path('D:/2/exp_upscale/runs')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--imgsz', type=int, default=None, help='不给则从 run 名识别')
    ap.add_argument('--iou', type=float, default=0.5)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--report-test', action='store_true',
                   help='对最终选中的检查点额外报 test（只报不选）')
    args = ap.parse_args()

    imgsz = args.imgsz
    if imgsz is None:
        m = re.search(r'_(640|896|1024)(?:_|$)', args.run)
        if not m:
            raise SystemExit(f'❌ 无法识别分辨率：{args.run}')
        imgsz = int(m.group(1))

    wdir = RUNS / args.run / 'weights'
    ckpts = sorted(wdir.glob('*.pt'),
                   key=lambda p: (0 if p.stem == 'best' else 1 if p.stem == 'last'
                                  else 2, p.name))
    if not ckpts:
        raise SystemExit(f'❌ 没有检查点：{wdir}')

    rows = []
    for c in ckpts:
        r = YOLO(str(c)).val(data=DATA, split='val', imgsz=imgsz, batch=8,
                             workers=0, device=args.device, plots=False,
                             verbose=False, iou=args.iou, agnostic_nms=False)
        b = r.box
        rows.append(dict(ckpt=c.name, mAP50=float(b.map50), mAP95=float(b.map),
                         P=float(b.mp), R=float(b.mr)))
        print(f'  {c.name:<12} val mAP50={float(b.map50):.4f}  '
              f'mAP50-95={float(b.map):.4f}  R={float(b.mr):.4f}')

    rows.sort(key=lambda x: -x['mAP50'])
    win = rows[0]
    best_row = next(r for r in rows if r['ckpt'] == 'best.pt')
    print(f'\n=== {args.run} / val / NMS={args.iou} / imgsz={imgsz} ===')
    print(f'  原 best.pt : mAP50={best_row["mAP50"]:.4f}')
    print(f'  0.5 口径最优: {win["ckpt"]}  mAP50={win["mAP50"]:.4f}  '
          f'（差 {(win["mAP50"] - best_row["mAP50"]) * 100:+.2f} pt）')
    if len(rows) > 1:
        print(f'  共比较 {len(rows)} 个检查点；val 上的最大值会偏乐观，报数需说明')

    if args.report_test:
        r = YOLO(str(wdir / win['ckpt'])).val(
            data=DATA, split='test', imgsz=imgsz, batch=8, workers=0,
            device=args.device, plots=False, verbose=False, iou=args.iou,
            agnostic_nms=False)
        b = r.box
        print(f'  → {win["ckpt"]} 的 test: mAP50={float(b.map50):.4f}  '
              f'mAP50-95={float(b.map):.4f}  P={float(b.mp):.4f}  R={float(b.mr):.4f}')

    out = Path(f'D:/2/report/_p2_diag/reselect_{args.run}.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({'run': args.run, 'imgsz': imgsz, 'iou': args.iou,
                               'rows': rows, 'winner': win['ckpt']},
                              ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'\n→ {out}')


if __name__ == '__main__':
    main()
