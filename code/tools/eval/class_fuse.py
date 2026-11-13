"""
类级模型选择（class-wise model selection）

动机：
    11 个改进族在「整体 mAP50」上全部输给地板，但**类级**存在可复现的赢面：
    Triplet Attention 的 wear AP50 三颗都高于地板三颗的最高值（区间不重叠）。
    反过来，break / thunderbolt 上地板最强。
    ⇒ 单模型比较把「类内提升」和「类内损失」混在一个平均值里，互相抵消；
      按类分别挑模型可以同时保留两边的优势，且**不需要重训**。

方法学（必须遵守，否则是信息泄漏）：
    1. 只在 **val** 上给每个候选模型算逐类 AP50；
    2. 每类选 val AP50 最高的那个模型（选择空间 = 候选数，3 类，过拟合风险低）；
    3. 只有 **test** 用于最终报告，且 test 从不参与任何选择。
    这等价于「用验证集做超参选择」，是标准做法。

评估口径：
    test 侧复用 ensemble_wbf.py 相同的路径（DetectionValidator.match_predictions +
    DetMetrics，conf=0.001 / NMS 0.5 / max_det 300），保证与集成实验可比。
    val 侧用官方 YOLO.val(iou=0.5) —— 只用于**类内相对排序**，绝对数值与 test 侧
    路径不同（已知约 1.4 pt 系统差），不影响选择结果。

用法：
    python code/tools/eval/class_fuse.py \
        --models nf_v11_896_s42,nf_v11_896_s43,nf_v11_896_s44,\
att_v11_896_triplet_s42,att_v11_896_triplet_s43,att_v11_896_triplet_s44,\
att_v11_896_cbam_bn_s42,att_v11_896_cbam_bn_s43,att_v11_896_cbam_bn_s44,wearos2_v11_896_s42 \
        --device cpu
可选 --force-wear MODEL 强制指定某一类的来源模型，用于消融对照。
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

for _d in ('D:/2/code/yolov11', 'D:/2/code/yolov12'):
    if _d not in sys.path:
        sys.path.insert(0, _d)      # Triplet/CBAM/P2v2 权重反序列化需自定义模块可见

ROOT = Path('D:/2/dataset_final_bbox')
RUNS = Path('D:/2/exp_upscale/runs')
OUT = Path('D:/2/report/_p2_diag')
DATA = str(ROOT / 'data.yaml')
NAMES_L = ['break', 'thunderbolt', 'wear']

IMGSZ = 896
CONF = 0.001
MAX_DET = 300
NMS_IOU = 0.5


def load_items(split):
    from PIL import Image
    items = []
    for p in sorted((ROOT / 'images' / split).glob('*.jpg')):
        w, h = Image.open(p).size
        f = ROOT / 'labels' / split / (p.stem + '.txt')
        b = np.zeros((0, 4), np.float32)
        c = np.zeros(0, np.float32)
        if f.exists():
            rows = [ln.split() for ln in f.read_text().strip().splitlines() if len(ln.split()) >= 5]
            if rows:
                b = np.array([[float(s[1]) * w, float(s[2]) * h,
                               float(s[3]) * w, float(s[4]) * h] for s in rows], np.float32)
                c = np.array([float(s[0]) for s in rows], np.float32)
        items.append((p, b, c, (w, h)))
    return items


def eval_val_per_class(ckpt, device):
    """在 val 上算逐类 AP50（只用于类内排序）"""
    from ultralytics import YOLO
    m = YOLO(str(RUNS / ckpt / 'weights' / 'best.pt'))
    r = m.val(data=DATA, split='val', imgsz=IMGSZ, batch=8, workers=0, device=device,
              plots=False, verbose=False, iou=NMS_IOU, agnostic_nms=False, max_det=MAX_DET)
    bx = r.box
    ap = {}
    for j, ci in enumerate(bx.ap_class_index):
        ap[r.names[int(ci)]] = float(bx.ap50[j])
    del m
    return ap, float(bx.map50), float(bx.map)


def predict(ckpt, items, device):
    from ultralytics import YOLO
    m = YOLO(str(RUNS / ckpt / 'weights' / 'best.pt'))
    imgs = [str(p) for p, *_ in items]
    out = []
    for i in range(0, len(imgs), 4):
        rs = m.predict(imgs[i:i + 4], imgsz=IMGSZ, conf=CONF, iou=NMS_IOU, max_det=MAX_DET,
                       agnostic_nms=False, verbose=False, stream=False, device=device)
        for r in rs:
            if r.boxes is None or len(r.boxes) == 0:
                out.append((np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, int)))
            else:
                out.append((r.boxes.xyxy.cpu().numpy().astype(np.float32),
                            r.boxes.conf.cpu().numpy().astype(np.float32),
                            r.boxes.cls.cpu().numpy().astype(int)))
    del m
    return out


def compute_metric(all_pred, items, outdir, tag):
    """直接复用 ensemble_wbf.compute_metric，保证与集成实验**完全同一路径**。
    此前自己重写的一份在这里返回全 0（stats 格式不被 DetMetrics.process 接受），
    教训：对比实验不要另行实现指标路径，一律复用已验证的那一份。"""
    if 'D:/2/code/tools/diag' not in sys.path:
        sys.path.insert(0, 'D:/2/code/tools/diag')
    from ensemble_wbf import compute_metric as ew_compute

    items_ew = []
    for (pth, gb, gc, wh) in items:
        gts = [dict(box=np.asarray(gb[i], np.float32), cls=int(gc[i])) for i in range(len(gb))]
        items_ew.append((pth, gts, wh[0], wh[1]))
    return ew_compute(all_pred, items_ew, outdir, tag)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', required=True)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--force-break', default=None)
    ap.add_argument('--force-thunderbolt', default=None)
    ap.add_argument('--force-wear', default=None)
    ap.add_argument('--skip-val', action='store_true', help='复用已存的 val 表')
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

    models = [m.strip() for m in args.models.split(',') if m.strip()]
    valtab_path = OUT / 'class_fuse_val_ap.json'

    valtab = {}
    if valtab_path.exists():
        try:
            valtab = json.loads(valtab_path.read_text(encoding='utf-8'))
        except Exception:
            valtab = {}
    todo = [m for m in models if m not in valtab]
    if args.skip_val:
        todo = []
    print(f'=== 阶段 A：val 上逐类 AP50（复用 {len(valtab)} 个，待评 {len(todo)} 个）===')
    if True:
        for ck in todo:
            try:
                per, m50, m95 = eval_val_per_class(ck, args.device)
                valtab[ck] = dict(per_class=per, mAP50=m50, mAP95=m95)
                s = '  '.join(f'{n}={per.get(n, float("nan")):.4f}' for n in NAMES_L)
                print(f'  {ck:32s} {s}   mAP50={m50:.4f}')
            except Exception as e:
                print(f'  {ck:32s} ERROR {type(e).__name__}: {e}')
        valtab_path.write_text(json.dumps(valtab, ensure_ascii=False, indent=1), encoding='utf-8')
        print(f'  -> {valtab_path}（共 {len(valtab)} 个模型）')

    # 每类选 val 最优
    choice = {}
    for nm in NAMES_L:
        forced = getattr(args, 'force_' + nm, None)
        if forced:
            choice[nm] = forced
            continue
        best, bv = None, -1
        for ck, v in valtab.items():
            a = v['per_class'].get(nm)
            if a is None or not np.isfinite(a):
                continue
            if a > bv:
                bv, best = a, ck
        choice[nm] = best

    print('\n=== 阶段 B：按类选择（val 上选）===')
    for nm in NAMES_L:
        ck = choice[nm]
        v = valtab[ck]['per_class'].get(nm, float('nan'))
        print(f'  {nm:12s} -> {ck:32s} val AP50={v:.4f}')
    used = sorted(set(choice.values()))
    print(f'  实际用到 {len(used)} 个模型：{used}')

    print('\n=== 阶段 C：test 评估（选择已冻结，test 不参与任何决策）===')
    items = load_items('test')
    print(f'  {len(items)} 张图 / {sum(len(c) for _, _, c, _ in items)} 个 GT')

    warns = {}
    for ck in used:
        print(f'  推理 {ck} ...')
        warns[ck] = predict(ck, items, args.device)

    fused = []
    for i in range(len(items)):
        bs, ss, cs = [], [], []
        for nm, ck in choice.items():
            ci = NAMES_L.index(nm)
            b, s, c = warns[ck][i]
            keep = c == ci
            if keep.any():
                bs.append(b[keep])
                ss.append(s[keep])
                cs.append(c[keep])
        if bs:
            fused.append((np.concatenate(bs), np.concatenate(ss), np.concatenate(cs)))
        else:
            fused.append((np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, int)))

    outdir = Path('D:/2/exp_upscale/eval_ensemble') / 'classfuse'
    outdir.mkdir(parents=True, exist_ok=True)
    dm = compute_metric(fused, items, outdir, 'classfuse')
    bx = dm.box
    pc = {NAMES_L[int(ci)]: float(bx.ap50[j]) for j, ci in enumerate(bx.ap_class_index)}
    print(f'\n{"方案":28s}{"mAP50":>9s}{"mAP50-95":>10s}{"P":>8s}{"R":>8s}')
    print(f'{"类级选择（本工具）":28s}{float(bx.map50):>9.4f}{float(bx.map):>10.4f}'
          f'{float(bx.mp):>8.4f}{float(bx.mr):>8.4f}')
    print(f'  break AP50={pc.get("break", float("nan")):.4f}  '
          f'thunder AP50={pc.get("thunderbolt", float("nan")):.4f}  '
          f'wear AP50={pc.get("wear", float("nan")):.4f}')

    res = dict(choice=choice, models=used, mAP50=float(bx.map50), mAP95=float(bx.map),
               P=float(bx.mp), R=float(bx.mr), per_class=pc, valtab=valtab)
    outp = OUT / 'class_fuse_test.json'
    outp.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'\n保存 -> {outp}')


if __name__ == '__main__':
    main()
