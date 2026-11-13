# -*- coding: utf-8 -*-
"""量化「标注噪声决定了 mAP50 的天花板」——纯只读，不训练、不动权重。

问题背景
    段级 test 卷（203 图）上 base 三种子 mAP50 = 0.7467，bootstrap 单侧 95% CI 宽 10.4 pt。
    13 个改进方向全部落在 ±1.6 噪声地板内。真正要回答的不是「再加哪个模块」，
    而是「这个数据集的标注质量本身把 mAP50 压到了多少」。

    已知观测（WORKLOG A.9.42，全数据集 gap=1 相邻帧同目标统计）：
        break 框宽度差 sd = 0.138      ← 单帧内目标尺寸不可能变 13.8%，绝大部分是标注抖动
        相邻帧框对 IoU 中位 = 0.477（p10 = 0.328），break 最差 0.462

    但 0.477 是**跨帧**量，含真实形变，不能直接当模型上限。本脚本用两件事把它拆开：

实验 1  在这张卷子上实测标注一致性
    只在段内 >=2 图的 13 个段（139 图）里，取同段内文件序相邻的两帧，
    匹配同类目标，统计 IoU 分布。这是这张卷子自己的数字，不用引历史。

实验 2  用「抖动副本」模拟完美模型，扫参数得到 mAP50 上限
    完美模型 := 它预测的是目标真实位置，而人工标注带抖动。
    实现：把 test 的 GT 框施加各向同性中心抖动（相对框宽高的比例 sigma），
          抖动副本当预测、原 GT 当目标，走标准 COCO 匹配 + 框架 ap_per_class。
          conf 用「抖动副本与原框的 IoU」——越不确定的框置信度越低。

    自检硬约束：sigma = 0 时 mAP50 必须 = 1.0000（抖动副本 == 原框）。
    不满足直接退出，不出数。

    然后用实验 1 测到的 IoU 中位**反向标定** sigma*，读 sigma* 处的 mAP50
    —— 这就是「在这份标注质量下，一个完美模型最多能拿多少分」。

为什么这个数字比「再叠一个模块」重要
    如果天花板落在 78~82，那 74.67 已经是天花板的 91%~96%，
    13 个方向全败就是**必然而非无能**；而且它把「标注噪声」从一句定性抱怨
    变成了一个有区间的、可被审稿人复核的数。

用法
    D:/yolo-env/Scripts/python.exe code/tools/eval/ceiling_analysis.py
    ... --N 40 --out exp_upscale/logs/ceiling_noleak.tsv
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
from ultralytics.utils.metrics import ap_per_class

ROOT = Path('D:/2')
DS = ROOT / 'dataset_noleak'
ASSIGN = DS / 'assign.json'
OUT_DEFAULT = ROOT / 'exp_upscale/logs/ceiling_noleak.tsv'

CLASS_NAMES = {0: 'break', 1: 'thunder', 2: 'wear'}


# ---------------------------------------------------------------- 基础几何
def read_gt(stem):
    """读 YOLO 标签 -> (cls int array, xyxy 归一化 float array)。"""
    p = DS / 'labels/test' / f'{stem}.txt'
    if not p.exists():
        return np.zeros((0,), np.int64), np.zeros((0, 4), np.float32)
    rows = []
    for ln in p.read_text().splitlines():
        t = ln.split()
        if len(t) >= 5:
            rows.append([float(x) for x in t[:5]])
    if not rows:
        return np.zeros((0,), np.int64), np.zeros((0, 4), np.float32)
    a = np.asarray(rows, np.float32)
    cls = a[:, 0].astype(np.int64)
    cx, cy, w, h = a[:, 1], a[:, 2], a[:, 3], a[:, 4]
    xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
    return cls, xyxy


def iou_matrix(a, b):
    """a: (N,4) b: (M,4) 归一化 xyxy -> (N,M) IoU。"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    ar_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    ar_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = ar_a[:, None] + ar_b[None, :] - inter
    out = np.zeros_like(inter)
    m = union > 1e-12
    out[m] = inter[m] / union[m]
    return out


def match_one(pred_box, pred_cls, pred_conf, gt_box, gt_cls, thr=0.5):
    """标准 COCO 贪心匹配（按 conf 降序）。返回 tp 数组 (n_pred,)。

    框架的 match_predictions 与本实现在 IoU≈thr 边界上可能翻转，但那是在
    「预测 vs GT」的真实分布上才有影响；这里 pred 是 GT 的抖动副本，
    IoU 分布远离 0.5（要么很高要么很低），且 sigma=0 自检会兜底。
    """
    n = len(pred_box)
    tp = np.zeros(n, np.float32)
    if n == 0 or len(gt_box) == 0:
        return tp
    order = np.argsort(-pred_conf, kind='stable')
    used = np.zeros(len(gt_box), bool)
    M = iou_matrix(pred_box, gt_box)
    for i in order:
        cand = np.where((gt_cls == pred_cls[i]) & (~used))[0]
        if len(cand) == 0:
            continue
        j = cand[np.argmax(M[i, cand])]
        if M[i, j] >= thr:
            tp[i] = 1.0
            used[j] = True
    return tp


def map50_from(tps, confs, pcls, gcls):
    """调框架 ap_per_class。8.3.180 返回 12 项，ap 在 index 5。"""
    tp = np.concatenate(tps)[:, None] if tps else np.zeros((0, 1), np.float32)
    cf = np.concatenate(confs) if confs else np.zeros((0,), np.float32)
    pc = np.concatenate(pcls) if pcls else np.zeros((0,), np.float32)
    gc = np.concatenate(gcls) if gcls else np.zeros((0,), np.float32)
    if len(gc) == 0:
        return float('nan'), {}
    res = ap_per_class(tp, cf, pc, gc)
    ap = np.asarray(res[5])
    if ap.ndim == 2:
        ap = ap[:, 0]
    uniq = np.asarray(res[6]).astype(int)
    per = {CLASS_NAMES.get(int(c), str(c)): float(ap[i]) for i, c in enumerate(uniq)}
    return float(ap.mean()), per


# ---------------------------------------------------------------- 实验 1
def experiment_consistency(stems, seg_of, gate=0.3):
    """段内相邻帧同目标 IoU 分布（这张卷子自己的标注一致性）。

    必须加 gate 门槛，否则结果没有意义：段内文件序相邻并不等于时间序相邻
    （test 相邻文件名中位差 7 帧，见 WORKLOG A.9.42），无条件 argmax 会把
    **不同目标**配成一对。实测无门槛中位只有 0.155，那是错配主导的，
    不是标注不一致。加 IoU>=0.3 门槛后才是「同一目标的两次标注」，
    与历史口径（WORKLOG 5202：IoU 中位 0.477、break 0.462）可比。
    """
    seg2 = {}
    for s in stems:
        k = seg_of.get(s + '.jpg')
        if k is not None:
            seg2.setdefault(k, []).append(s)
    raw, pairs = [], []          # raw=全部同类 argmax 对；pairs=过门槛的
    for k, ss in seg2.items():
        if len(ss) < 2:
            continue
        ss = sorted(ss)
        for a, b in zip(ss[:-1], ss[1:]):
            ca, ba = read_gt(a)
            cb, bb = read_gt(b)
            if len(ba) == 0 or len(bb) == 0:
                continue
            M = iou_matrix(ba, bb)
            for i in range(len(ba)):
                cand = np.where(cb == ca[i])[0]
                if len(cand) == 0:
                    continue
                j = cand[np.argmax(M[i, cand])]
                raw.append(float(M[i, j]))
                if M[i, j] >= gate:
                    pairs.append((int(ca[i]), float(M[i, j])))
    if not pairs:
        return {}, {}
    arr = np.asarray([[p[0], p[1]] for p in pairs], np.float32)
    stats = {}

    def pack(v):
        return dict(n=len(v), p10=float(np.percentile(v, 10)),
                    median=float(np.median(v)),
                    p90=float(np.percentile(v, 90)),
                    hit50=float((v >= 0.5).mean()))

    for c in sorted(set(arr[:, 0].astype(int))):
        stats[CLASS_NAMES.get(c, str(c))] = pack(arr[arr[:, 0] == c][:, 1])
    stats['ALL'] = pack(arr[:, 1])
    # 门槛前（含错配）作为对照，写进 note 用来说明为什么必须加门槛
    aux = dict(n_raw=len(raw),
               median_raw=float(np.median(raw)) if raw else float('nan'),
               keep_rate=len(pairs) / len(raw) if raw else float('nan'))
    return stats, aux


# ---------------------------------------------------------------- 实验 2
def jitter(box, cls, sigma, rng):
    """中心按相对框宽高的比例抖动 + 轻微尺度抖动，clip 到 [0,1]。"""
    b = box.copy()
    cx = (b[:, 0] + b[:, 2]) / 2
    cy = (b[:, 1] + b[:, 3]) / 2
    w = b[:, 2] - b[:, 0]
    h = b[:, 3] - b[:, 1]
    cx = cx + rng.normal(0, sigma, len(b)).astype(np.float32) * w
    cy = cy + rng.normal(0, sigma, len(b)).astype(np.float32) * h
    # 尺度抖动取 sigma 的一半，因为观测到的宽度差 sd 0.138 里含位置效应
    s = 1.0 + rng.normal(0, sigma * 0.5, (len(b), 2)).astype(np.float32)
    s = np.clip(s, 0.5, 2.0)
    w = w * s[:, 0]
    h = h * s[:, 1]
    out = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def experiment_ceiling(gts, sigma, N, rng):
    """对给定 sigma，重复 N 次抖动模拟，返回 (mAP50 均值, 逐类均值, 模拟 IoU 中位)。"""
    maps, per_acc, iou_med = [], [], []
    for _ in range(N):
        tps, confs, pcls, gcls = [], [], [], []
        for stem, (cls, box) in gts.items():
            if len(box) == 0:
                continue
            pb = jitter(box, cls, sigma, rng)
            # conf = 抖动副本与原框的 IoU，越不确定的框分越低
            conf = np.diag(iou_matrix(pb, box)).astype(np.float32)
            if sigma == 0:
                conf = np.ones(len(pb), np.float32)
            tp = match_one(pb, cls, conf, box, cls)
            tps.append(tp); confs.append(conf)
            pcls.append(cls.astype(np.float32)); gcls.append(cls.astype(np.float32))
            iou_med.append(float(np.median(conf)))
        m, per = map50_from(tps, confs, pcls, gcls)
        maps.append(m); per_acc.append(per)
    per_mean = {}
    for k in per_acc[0] if per_acc else {}:
        per_mean[k] = float(np.nanmean([p.get(k, np.nan) for p in per_acc]))
    return float(np.mean(maps)), per_mean, float(np.mean(iou_med))


def calibrate_sigma(gts, target_median, lo=0.0, hi=0.6, N=3, rng=None, iters=18):
    """二分找使「抖动副本与原框 IoU 中位」= target_median 的 sigma。"""
    rng = rng or np.random.default_rng(0)
    for _ in range(iters):
        mid = (lo + hi) / 2
        _, _, got = experiment_ceiling(gts, mid, N, rng)
        if got > target_median:      # IoU 太高 -> 需要更大抖动
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--N', type=int, default=30, help='每个 sigma 的蒙特卡洛重复次数')
    ap.add_argument('--out', default=str(OUT_DEFAULT))
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    rng = np.random.default_rng(20261029)

    stems = sorted(os.path.splitext(x)[0]
                   for x in os.listdir(DS / 'images/test'))
    seg_of = json.load(open(ASSIGN, encoding='utf-8'))['seg_of']
    gts = {s: read_gt(s) for s in stems}
    ngt = sum(len(v[0]) for v in gts.values())
    print(f'test 图 {len(stems)}  GT 框 {ngt}')

    # ---- 实验 1
    print('\n' + '=' * 68)
    print('实验 1  这张卷子自己的标注一致性（段内相邻帧，同类匹配，IoU>=0.3 门槛）')
    print('=' * 68)
    stats, aux = experiment_consistency(stems, seg_of)
    if not stats:
        print('  没有段内多帧，无法测（异常）')
    for k, v in stats.items():
        print(f'  {k:<8} n={v["n"]:<4} IoU 中位={v["median"]:.3f} '
              f'p10={v["p10"]:.3f} p90={v["p90"]:.3f}  ≥0.5 占 {v["hit50"]*100:.1f}%')
    if aux:
        print(f'  门槛前（含跨目标错配）：n={aux["n_raw"]} '
              f'IoU 中位={aux["median_raw"]:.3f} → 过门槛保留 {aux["keep_rate"]*100:.1f}%')
        print('  ↑ 门槛前那个低值不是标注不一致，是段内文件序≠时间序导致的错配，'
              '不能用来标定')
    # 标定目标：优先用历史全数据集口径 0.477（WORKLOG A.9.42 gap=1），
    # 本卷实测作为敏感性分析一起报
    TARGET_HIST = 0.477
    target_med = TARGET_HIST
    target_local = stats['ALL']['median'] if stats else TARGET_HIST

    # ---- 实验 2：自检 + 扫描
    print('\n' + '=' * 68)
    print('实验 2  抖动副本当「完美模型」，扫 sigma')
    print('=' * 68)
    m0, per0, _ = experiment_ceiling(gts, 0.0, 1, rng)
    print(f'  自检 sigma=0 : mAP50 = {m0:.4f}')
    # ultralytics ap_per_class 用 101 点召回插值，完美情况下给 100.5/101 = 0.9950
    # （已实测：300/300 全 TP 仍返回 0.995）。这是框架固有偏移，所有 mAP50 同受影响，
    # 横向比较不受影响；因此自检判据取 >=0.99，并把该偏移记录下来。
    if m0 < 0.99:
        print('  [失败] sigma=0 时 mAP50 应 >= 0.99，实现有 bug，拒绝出数。')
        raise SystemExit(2)
    FRAME_BIAS = 1.0 - m0
    print(f'  ✓ 自检通过（框架 101 点插值固有偏移 {FRAME_BIAS:.4f}，'
          f'所有 mAP50 同受影响）')

    grid = [0.0, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40]
    rows = []
    print(f'\n  {"sigma":>6}  {"IoU中位":>8}  {"mAP50":>7}  '
          f'{"break":>7} {"thunder":>8} {"wear":>7}')
    for sg in grid:
        m, per, iou_med = experiment_ceiling(gts, sg, args.N, rng)
        rows.append([sg, iou_med, m, per.get('break', np.nan),
                     per.get('thunder', np.nan), per.get('wear', np.nan)])
        print(f'  {sg:>6.3f}  {iou_med:>8.3f}  {m:>7.4f}  '
              f'{per.get("break", float("nan")):>7.4f} '
              f'{per.get("thunder", float("nan")):>8.4f} '
              f'{per.get("wear", float("nan")):>7.4f}')

    # ---- 标定
    print('\n' + '=' * 68)
    print(f'标定：找使模拟 IoU 中位 = {target_med:.3f} 的 sigma')
    print('=' * 68)
    sg_star = calibrate_sigma(gts, target_med, N=3, rng=rng)
    m_star, per_star, iou_star = experiment_ceiling(gts, sg_star, args.N, rng)
    print(f'  标定 sigma* = {sg_star:.4f}   模拟 IoU 中位 {iou_star:.3f} '
          f'（目标 {target_med:.3f}）')
    print(f'  => mAP50 天花板 = {m_star:.4f}')
    for k, v in per_star.items():
        print(f'       {k:<8} {v:.4f}')

    # 敏感性：若改用本卷实测一致性标定（口径更窄，仅 13 个多图段）
    sens = None
    if abs(target_local - target_med) > 0.02:
        sg_loc = calibrate_sigma(gts, target_local, N=3, rng=rng)
        m_loc, _, _ = experiment_ceiling(gts, sg_loc, args.N, rng)
        sens = (sg_loc, m_loc)
        print(f'  [敏感性] 改用本卷实测 {target_local:.3f} 标定：'
              f'sigma*={sg_loc:.4f} -> 天花板 {m_loc:.4f}')

    # 逐类标定（break 标注最差，单独看）
    per_class_ceiling = {}
    for k in ('break', 'thunder', 'wear'):
        if k in stats and stats[k]['n'] >= 10:
            s_k = calibrate_sigma(gts, stats[k]['median'], N=3, rng=rng)
            m_k, per_k, _ = experiment_ceiling(gts, s_k, max(5, args.N // 3), rng)
            per_class_ceiling[k] = (s_k, m_k, per_k.get(k, np.nan))
            print(f'  [{k}] sigma*={s_k:.4f} -> 该类 AP 上限 {per_k.get(k, float("nan")):.4f}')

    # ---- 落盘
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(['kind', 'key', 'sigma', 'iou_median', 'mAP50',
                    'AP_break', 'AP_thunder', 'AP_wear', 'n', 'note'])
        w.writerow(['selfcheck', 'sigma=0', 0, 1.0, m0, '', '', '', 0,
                    f'全TP时框架101点插值给{m0:.4f}，固有偏移{FRAME_BIAS:.4f}'])
        if aux:
            w.writerow(['consistency_raw', 'ALL', '', aux['median_raw'], '', '', '', '',
                        aux['n_raw'], f'含跨目标错配，keep_rate={aux["keep_rate"]:.3f}'])
        for k, v in stats.items():
            w.writerow(['consistency', k, '', v['median'], '', '', '', '',
                        v['n'], f'p10={v["p10"]:.3f} p90={v["p90"]:.3f} '
                                f'hit50={v["hit50"]:.3f}'])
        for r in rows:
            w.writerow(['sweep', '', r[0], r[1], r[2], r[3], r[4], r[5],
                        args.N, ''])
        w.writerow(['calibrated', 'ALL', sg_star, iou_star, m_star,
                    per_star.get('break', ''), per_star.get('thunder', ''),
                    per_star.get('wear', ''),
                    args.N, f'target_median={target_med:.3f}(历史口径A.9.42)'])
        if sens:
            w.writerow(['calibrated_sens', 'ALL', sens[0], '', sens[1], '', '', '',
                        args.N, f'target={target_local:.3f}(本卷13段实测)'])
        for k, (s_k, m_k, ap_k) in per_class_ceiling.items():
            w.writerow(['calibrated', k, s_k, '', m_k, ap_k, '', '',
                        args.N, f'target={stats[k]["median"]:.3f}'])
    print(f'\n已写 {outp}')


if __name__ == '__main__':
    main()
