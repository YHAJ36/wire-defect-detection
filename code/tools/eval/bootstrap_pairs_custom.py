# -*- coding: utf-8 -*-
"""任意两两 run 的配对 bootstrap（203 图段级无泄漏 test）。

与 bootstrap_paired_jit.py 的区别：那个脚本按 `nf_v11_noleak_<tag>_s<seed>`
拼名字，只服务 A/C 组；这个接受任意 run 名对，用于 640 批次的三组对比。

每对独立重采样，输出：
    ΔmAP50（点估计） / 图级配对 CI / 段级 cluster CI / 三类分项 Δ 与 CI

用法
    D:/yolo-env/Scripts/python.exe code/tools/eval/bootstrap_pairs_custom.py --B 2000 \
        --cmp "A:B:标签A vs 标签B" --cmp "B:C:标签B vs 标签C"
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

for _d in ('D:/2/code/yolov11', 'D:/2/code/tools/eval'):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from bootstrap_paired_jit import ASSIGN, CACHE, NAMES, ap50, ci, load_tp  # noqa: E402

ROOT = Path('D:/2')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--B', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--cmp', action='append', required=True,
                    help='base_run:target_run:label，可重复')
    ap.add_argument('--out', default='D:/2/exp_upscale/logs/bootstrap_pairs_640.tsv')
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

    pairs = []
    for spec in args.cmp:
        parts = spec.split(':')
        if len(parts) < 2:
            raise SystemExit(f'--cmp 格式错误：{spec}')
        b, t = parts[0], parts[1]
        lab = parts[2] if len(parts) > 2 else f'{t} − {b}'
        pairs.append((b, t, lab))

    need = sorted({r for b, t, _ in pairs for r in (b, t)})
    feats, stems0 = {}, None
    for r in need:
        p = CACHE / f'{r}.npz'
        if not p.exists():
            raise SystemExit(f'缺缓存 {p}')
        d = load_tp(p)
        keys = sorted(d)
        if stems0 is None:
            stems0 = keys
        elif keys != stems0:
            raise SystemExit(f'{p.name} 的图列表与基准不一致')
        feats[r] = [d[s] for s in stems0]
    stems = stems0
    n = len(stems)
    ALL = list(range(n))

    seg_of = json.load(open(ASSIGN, encoding='utf-8'))['seg_of']
    seg_ids = [seg_of.get(s + '.jpg', seg_of.get(s, -1)) for s in stems]
    segs = {}
    for i, g in enumerate(seg_ids):
        segs.setdefault(g, []).append(i)
    seg_list = [v for v in segs.values()]

    print(f'test 图 {n}   GT {sum(len(f[3]) for f in feats[need[0]])}   '
          f'段 {len(seg_list)}（最大 {max(len(v) for v in seg_list)} 图）\n')

    rows = []
    for b, t, lab in pairs:
        pb = ap50(feats[b], ALL)
        pt = ap50(feats[t], ALL)
        print('=' * 66)
        print(f'{lab}')
        print(f'  {b}: {pb:.4f}')
        print(f'  {t}: {pt:.4f}')
        print(f'  Δ = {(pt - pb) * 100:+.2f} pt')

        rng = np.random.default_rng(args.seed)
        d_img, d_seg = [], []
        for _ in range(args.B):
            idx = list(rng.integers(0, n, n))
            d_img.append(ap50(feats[t], idx) - ap50(feats[b], idx))
        for _ in range(args.B):
            pick = rng.integers(0, len(seg_list), len(seg_list))
            idx = [i for k in pick for i in seg_list[k]]
            if idx:
                d_seg.append(ap50(feats[t], idx) - ap50(feats[b], idx))

        for tag, v, col in (('图级配对', d_img, 'img'), ('段级 cluster', d_seg, 'seg')):
            lo, hi = ci(v)
            sig = '跨零' if lo < 0 < hi else ('显著为正' if lo > 0 else '显著为负')
            print(f'  {tag:<14} [{lo * 100:+6.2f}, {hi * 100:+6.2f}]  {sig}')
            rows.append({'pair': lab, 'base': b, 'target': t, 'level': col,
                         'delta_pt': round((pt - pb) * 100, 4),
                         'lo95': round(lo * 100, 4), 'hi95': round(hi * 100, 4),
                         'verdict': sig})

        print('  --- 分类别 ---')
        # 复用同一批重采样索引，省一半开销（每类只算 ap 不重采样）
        boot_idx = [list(rng.integers(0, n, n)) for _ in range(args.B)]
        for ci_, nm in enumerate(NAMES):
            dv = (ap50(feats[t], ALL, ci_) - ap50(feats[b], ALL, ci_)) * 100
            vb = [ap50(feats[t], idx, ci_) - ap50(feats[b], idx, ci_) for idx in boot_idx]
            lo, hi = ci(vb)
            sig = '跨零' if lo < 0 < hi else ('显著为正' if lo > 0 else '显著为负')
            print(f'    {nm:<12} Δ={dv:+6.2f}  [{lo * 100:+6.2f}, {hi * 100:+6.2f}]  {sig}')
            rows.append({'pair': lab, 'base': b, 'target': t, 'level': nm,
                         'delta_pt': round(dv, 4),
                         'lo95': round(lo * 100, 4), 'hi95': round(hi * 100, 4),
                         'verdict': sig})
        print()

    import csv as _csv
    out = Path(args.out)
    with open(out, 'w', encoding='utf-8-sig', newline='') as f:
        w = _csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f'→ {out}')


if __name__ == '__main__':
    main()
