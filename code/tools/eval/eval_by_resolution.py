# -*- coding: utf-8 -*-
"""按原始分辨率分组评估：验证「混源数据拖累指标」这一假说。

背景（A.9.42 审计）
    train 有 35 种分辨率，主力五种：1920x1080(613)、632x480(444)、400x296(236)、
    1080x1920(62)、2624x2624(25)。固定 imgsz=896 时，400x296 的图被上采样 3 倍
    （信息量只有 400 px），2624 的被下采样 2.9 倍，同批梯度有效分辨率差 6.5 倍。
    若"小图（上采样组）"的 AP 明显更低，则说明指标被数据构成拖住，而不是模型不会。

做法
    把评估集按原始 (W,H) 分桶 → 每桶写一个图片清单 txt + 临时 data yaml
    → 走与 eval_nms05.py 相同的 val 路径（NMS IoU 0.5）分组评估。
    桶内只有 1~2 张图的不单独报（AP 无意义），并入「其他」。

用法
    D:/yolo-env/Scripts/python.exe code/tools/eval/eval_by_resolution.py \
        --run nf_v11_896_s42 --split test --device cpu
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image

for _d in ('D:/2/code/yolov11', 'D:/2/code/yolov12'):
    if _d not in sys.path:
        sys.path.insert(0, _d)      # 自定义模块权重反序列化需要

from ultralytics import YOLO  # noqa: E402

ROOT = Path('D:/2/dataset_final_bbox')
RUNS = Path('D:/2/exp_upscale/runs')
WORK = Path('D:/2/report/_p2_diag/res_split')
NAMES = ['break', 'thunderbolt', 'wear']


def bucket_name(w, h):
    """短边 < 896 = 需要上采样；否则原样或下采样。"""
    tag = f'{w}x{h}' + ('_竖' if h > w else '')
    return tag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--split', default='test', choices=['val', 'test'])
    ap.add_argument('--imgsz', type=int, default=896)
    ap.add_argument('--iou', type=float, default=0.5)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--min-img', type=int, default=8, help='少于这么多张的桶并入「其他」')
    args = ap.parse_args()

    w_path = RUNS / args.run / 'weights' / 'best.pt'
    if not w_path.exists():
        raise SystemExit(f'❌ 缺权重：{w_path}')

    img_dir = ROOT / 'images' / args.split
    lbl_dir = ROOT / 'labels' / args.split
    imgs = sorted(img_dir.glob('*.jpg'))

    groups = defaultdict(list)
    short_of = {}
    for p in imgs:
        w, h = Image.open(p).size
        k = bucket_name(w, h)
        groups[k].append(p)
        short_of[k] = min(w, h)

    # 小桶合并（合并后不参与"上采样/下采样"两分聚合，因为短边不一致）
    small = {k: v for k, v in groups.items() if len(v) < args.min_img}
    if small:
        merged = [p for v in small.values() for p in v]
        for k in small:
            groups.pop(k)
        groups[f'其他({len(small)}种)'] = merged
        short_of[f'其他({len(small)}种)'] = None

    WORK.mkdir(parents=True, exist_ok=True)
    results = []

    for name, paths in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        lst = WORK / f'{args.split}_{name}.txt'
        lst.write_text('\n'.join(str(p) for p in paths), encoding='utf-8')
        yml = WORK / f'{args.split}_{name}.yaml'
        # 注意：必须把桶清单同时写进 val 和 test 两个键。只改 test 的话，
        # split='val' 会退回 images/val 全量，五个桶会得到完全相同的数字（踩过一次）。
        yml.write_text(
            f'path: {ROOT}\ntrain: images/train\nval: {lst}\ntest: {lst}\n'
            f'nc: 3\nnames: {NAMES}\n', encoding='utf-8')

        n_gt = 0
        for p in paths:
            f = lbl_dir / f'{p.stem}.txt'
            if f.exists():
                n_gt += len([l for l in f.read_text(encoding='utf-8').splitlines() if l.strip()])

        m = YOLO(str(w_path))
        r = m.val(data=str(yml), split=args.split, imgsz=args.imgsz, batch=8,
                  workers=0, device=args.device, plots=False, verbose=False,
                  iou=args.iou, agnostic_nms=False,
                  project=str(WORK), name=f'{args.run}_{args.split}_{name}',
                  exist_ok=True)
        b = r.box
        pc = {r.names[int(c)]: float(b.ap50[j]) for j, c in enumerate(b.ap_class_index)}
        results.append(dict(bucket=name, short=short_of.get(name), n_img=len(paths),
                            n_gt=n_gt,
                            mAP50=float(b.map50), mAP95=float(b.map),
                            P=float(b.mp), R=float(b.mr), per_class=pc))
        print(f'  {name:<16} 图{len(paths):4d} 框{n_gt:4d}  '
              f'mAP50={float(b.map50):.4f}  mAP50-95={float(b.map):.4f}  '
              f'R={float(b.mr):.4f}')

    print(f'\n=== {args.run} / {args.split} / imgsz={args.imgsz} / NMS={args.iou} 按原始分辨率 ===')
    print(f'{"桶":16s} {"图":>4s} {"框":>5s} {"mAP50":>7s} {"mAP50-95":>9s} {"P":>7s} {"R":>7s}  逐类AP50')
    for r in results:
        print(f'{r["bucket"]:16s} {r["n_img"]:4d} {r["n_gt"]:5d} {r["mAP50"]:7.4f} '
              f'{r["mAP95"]:9.4f} {r["P"]:7.4f} {r["R"]:7.4f}  '
              + ', '.join(f'{k} {v:.3f}' for k, v in r['per_class'].items()))

    # 两分聚合：短边 < imgsz（需上采样） vs >= imgsz
    up = [r for r in results if r['short'] is not None and r['short'] < args.imgsz]
    down = [r for r in results if r['short'] is not None and r['short'] >= args.imgsz]
    def agg(rs, lab):
        if not rs:
            return
        n = sum(r['n_img'] for r in rs)
        w50 = sum(r['mAP50'] * r['n_gt'] for r in rs) / max(sum(r['n_gt'] for r in rs), 1)
        w95 = sum(r['mAP95'] * r['n_gt'] for r in rs) / max(sum(r['n_gt'] for r in rs), 1)
        print(f'  {lab:<22} 图{n:4d}  框加权 mAP50={w50:.4f}  mAP50-95={w95:.4f}')
    print('\n=== 两分聚合（按框数加权）===')
    agg(up, f'短边<{args.imgsz}（上采样）')
    agg(down, f'短边>={args.imgsz}（原生/下采样）')

    out = Path(f'D:/2/report/_p2_diag/res_split_{args.run}_{args.split}.json')
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'\n→ {out}')


if __name__ == '__main__':
    main()
