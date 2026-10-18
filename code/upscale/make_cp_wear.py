# -*- coding: utf-8 -*-
"""c4 前置件：手动 wear 定向 copy-paste 数据集构建。

来源与口径
    方向3-①（WORKLOG:1089）："把 wear 缺陷框区域抠出贴到别的图——定向补薄类，
    预测对 wear 单类可见、总分 <1pt"。内置 copy_paste 对 detect 无效（A.1-1），
    故只能离线自建。

为什么是"补 wear"
    wear 按**图数**是薄类：train 1428 张里只出现在 165 张（11.6%），
    而 break 799 张、thunderbolt 533 张。按**实例数**则不然（wear 788 > thunderbolt 650），
    按 **AP** 也不弱（test_clean wear 0.916 > break 0.8685，见 logs/leak_eval.txt）。
    ⇒ 本实验的靶子是"**wear 的图像多样性**"，不是"wear 的精度"。
    一句话：它测的是"给薄类补场景多样性有没有用"，不是"救一个烂类"。

布局（**就地替换**，复制式非软链 —— 与 make_leak_subsets.py 同约定：Windows 软链要管理员权限）
    D:/2/dataset_cpm_wear/
        images/train/   1428 张。其中 1128 张是**逐字节复制**的原图；300 张是
                        被选中的"画布"图，**就地换成了增强版**（原名保留，不新增文件）。
        labels/train/   与之对应（300 张增强版的标签 = 原标签 + 新贴的 wear 框）
        manifest_cpm.json   哪 300 张被增强、每张的配方（供体 / 落位）——**验证用**
        data_cpm.yaml   train 指向本目录，**val/test 指向原数据集（未动）**

⚠ 为什么是"替换"而不是"新增"（第一版踩过的坑，被自己的验证闸门抓到）
    第一版把合成图与原图**并存**（1428 → 1728 张）。结果是同时改了三个变量：
      图数 +21%、多数类实例 break +212 / thunderbolt +152（画布原有框被重复计一次）、
      wear +458。那就不是"只改一项"了。
    改为就地替换后：
      图数 **1428 不变** ⇒ iterations 不变（ceil(1428/64)×60 = 1380 步，epochs 仍 60）
      break 1014 / thunderbolt 650 **一字不变**
      唯一变化 = wear 788 → 788+N
    ⇒ 这才满足"除明示项外逐项一致"。代价：这 300 张画布不再以原貌出现
      （与 mosaic 之类替换式增强同性质，可接受）。

关键实现决定（写下来以免日后被质疑）
    1. 贴片按**原像素尺度**贴（copy-paste 原论文做法），不缩放 ⇒ 缺陷尺寸分布保持真实。
    2. 贴片含 margin=0.15 的上下文，但**标签只标缺陷框本身**（不含 margin）。
    3. 落位要求与画布内**已有框零相交**（含已贴的），重试 20 轮仍冲突则放弃该贴片
       ⇒ 避免产生语义含糊的重叠框。
    4. 供体只取 wear（cls=2）；画布取**不含 wear** 的图（1263 张）⇒ 直接把 wear 图数顶上去。
    5. seed=42 固定 ⇒ 数据集可复现。

用法
    预览（只报告，不写盘）：  .../python.exe make_cp_wear.py --dry-run
    正式生成：              .../python.exe make_cp_wear.py
    验证已有产物：          .../python.exe make_cp_wear.py --verify-only
"""
import argparse
import hashlib
import json
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

SRC = Path('D:/2/dataset_final_bbox')
DST = Path('D:/2/dataset_cpm_wear')

WEAR = 2
NAMES = ['break', 'thunderbolt', 'wear']
MARGIN = 0.15          # 贴片上下左右各留的上下文比例（相对缺陷框尺寸）
MAX_PASTE = 2          # 每张合成图最多贴几个 wear 补丁
N_SYN = 300            # 被增强（就地替换）的画布张数
SEED = 42
JPEG_Q = 95            # 增强图重编码质量（其余 1128 张是逐字节复制，不受影响）


# ============================== 基础 IO ==============================
def read_label(p):
    """返回 [(cls, cx, cy, w, h)]，归一化坐标。"""
    out = []
    if not p.exists():
        return out
    for ln in p.read_text().splitlines():
        f = ln.split()
        if len(f) != 5:
            continue
        out.append((int(float(f[0])), float(f[1]), float(f[2]), float(f[3]), float(f[4])))
    return out


def write_label(p, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(''.join(f'{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n' for c, x, y, w, h in rows))


def to_px(b, W, H):
    """归一化 → 像素 (cls, x1, y1, x2, y2)。"""
    c, cx, cy, w, h = b
    return (c, (cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H)


def to_norm(c, x1, y1, x2, y2, W, H):
    return (c, ((x1 + x2) / 2) / W, ((y1 + y2) / 2) / H, (x2 - x1) / W, (y2 - y1) / H)


def inter_area(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ============================== 供体贴片池 ==============================
def build_patches(train_img_dir, train_lb_dir):
    """读出全部 wear 实例，抠成 (crop_bgr, 缺陷框在 crop 内的像素框, 供体 stem, 缺陷框像素)。"""
    patches, donors = [], 0
    for lp in sorted(train_lb_dir.glob('*.txt')):
        wear_rows = [r for r in read_label(lp) if r[0] == WEAR]
        if not wear_rows:
            continue
        img = cv2.imread(str(train_img_dir / f'{lp.stem}.jpg'))
        if img is None:
            continue
        donors += 1
        H, W = img.shape[:2]
        for r in wear_rows:
            _, x1, y1, x2, y2 = to_px(r, W, H)
            bw, bh = x2 - x1, y2 - y1
            cx1 = int(max(0, np.floor(x1 - MARGIN * bw)))
            cy1 = int(max(0, np.floor(y1 - MARGIN * bh)))
            cx2 = int(min(W, np.ceil(x2 + MARGIN * bw)))
            cy2 = int(min(H, np.ceil(y2 + MARGIN * bh)))
            if cx2 - cx1 < 8 or cy2 - cy1 < 8:
                continue
            patches.append((img[cy1:cy2, cx1:cx2].copy(),
                            (x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1),
                            lp.stem, (x1, y1, x2, y2)))
    return patches, donors


# ============================== 生成（就地替换）==============================
def generate():
    rng = random.Random(SEED)
    tr_img, tr_lb = SRC / 'images/train', SRC / 'labels/train'

    canvases, wear_imgs = [], []
    for lp in sorted(tr_lb.glob('*.txt')):
        (wear_imgs if any(r[0] == WEAR for r in read_label(lp)) else canvases).append(lp.stem)
    print(f'train 图 {len(canvases) + len(wear_imgs)}｜含 wear {len(wear_imgs)}｜'
          f'不含 wear（可作画布）{len(canvases)}')

    patches, ndonor = build_patches(tr_img, tr_lb)
    print(f'供体图 {ndonor} 张｜wear 贴片池 {len(patches)} 个'
          f'（贴片含 {MARGIN:.0%} margin 上下文，标签仍只标缺陷框）')

    n_syn = min(N_SYN, len(canvases))
    if n_syn < N_SYN:
        print(f'⚠ 画布不足，增强张数由 {N_SYN} 降为 {n_syn}')
    rng.shuffle(canvases)
    picked = canvases[:n_syn]
    picked_set = set(picked)

    # ---- 清空并重建 DST（只动 DST，不碰 SRC）----
    if DST.exists():
        shutil.rmtree(DST)
    for d in ('images/train', 'labels/train'):
        (DST / d).mkdir(parents=True, exist_ok=True)

    # ---- 全量复制（1128 张非画布图此后不再改动 ⇒ 与基线逐字节一致）----
    for lp in tr_lb.glob('*.txt'):
        shutil.copy2(lp, DST / 'labels/train' / lp.name)
    for ip in tr_img.glob('*.jpg'):
        shutil.copy2(ip, DST / 'images/train' / ip.name)
    print(f'原图已逐字节复制 {len(list((DST / "images/train").glob("*.jpg")))} 张')

    # ---- 对选中的画布就地替换成增强版 ----
    manifest, n_pasted, skipped = [], 0, 0
    fail_fit = fail_overlap = 0
    for stem in picked:
        ip = DST / 'images/train' / f'{stem}.jpg'
        canvas = cv2.imread(str(ip))
        if canvas is None:
            continue
        CH, CW = canvas.shape[:2]
        rows = read_label(DST / 'labels/train' / f'{stem}.txt')
        boxes = [to_px(r, CW, CH)[1:] for r in rows]
        new_rows, recipe = list(rows), []
        want = rng.randint(1, MAX_PASTE)
        tries = 0
        while len(recipe) < want and tries < want * 20:
            tries += 1
            crop, din, donor, dbox = patches[rng.randrange(len(patches))]
            ph, pw = crop.shape[:2]
            if pw > CW or ph > CH:
                fail_fit += 1
                continue
            dx, dy = rng.randint(0, CW - pw), rng.randint(0, CH - ph)
            cand = (dx + din[0], dy + din[1], dx + din[2], dy + din[3])
            if any(inter_area(cand, b) > 0 for b in boxes):
                fail_overlap += 1
                continue
            canvas[dy:dy + ph, dx:dx + pw] = crop
            boxes.append(cand)
            new_rows.append(to_norm(WEAR, *cand, CW, CH))
            recipe.append({'donor': donor,
                           'donor_box_px': [float(v) for v in dbox],
                           'canvas_box_px': [float(v) for v in cand]})
        if not recipe:
            skipped += 1
            continue
        cv2.imwrite(str(ip), canvas, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
        write_label(DST / 'labels/train' / f'{stem}.txt', new_rows)
        manifest.append({'aug': stem, 'patch': recipe})
        n_pasted += len(recipe)

    (DST / 'manifest_cpm.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'就地增强 {len(manifest)} 张（覆盖原有 wear 图 165 张之外的画布）｜'
          f'贴片 {n_pasted} 个｜放弃：贴不下 {fail_fit} 次、位置冲突 {fail_overlap} 次、'
          f'整张未贴成 {skipped} 张')
    print(f'⚠ 未增强的画布 {len(picked_set) - len(manifest)} 张保持原貌')

    yml = (f'path: {DST.as_posix()}\n'
           f'train: images/train\n'
           f'val: {(SRC / "images/val").as_posix()}\n'
           f'test: {(SRC / "images/test").as_posix()}\n'
           f'nc: 3\nnames: {NAMES}\ntask: detect\n')
    (DST / 'data_cpm.yaml').write_text(yml, encoding='utf-8')
    print(f'已写 {DST / "data_cpm.yaml"}')

    n_train = len(list((DST / 'images/train').glob('*.jpg')))
    print(f'\n【同步数匹配】train 张数 {n_train}（基线 1428）'
          f' ⇒ {"✅ 不变，epochs 仍用 60" if n_train == 1428 else "⚠ 变了，需重算 epochs"}')
    return n_train


# ============================== 验证 ==============================
def verify():
    """独立复核：全部从落盘产物重新读，不信任生成过程。"""
    print('\n' + '=' * 92)
    print('验证（不信任生成过程，全部从落盘产物重新读）')
    print('=' * 92)
    ok = True

    imgs = sorted((DST / 'images/train').glob('*.jpg'))
    lbs = sorted((DST / 'labels/train').glob('*.txt'))
    print(f'\n1) 计数：images {len(imgs)}｜labels {len(lbs)}｜'
          f'{"✅ 一一对应" if len(imgs) == len(lbs) else "❌ 不匹配"}')
    if len(imgs) != len(lbs):
        return False
    if len(imgs) != 1428:
        ok = False
        print(f'   ❌ train 图数 {len(imgs)} ≠ 基线 1428 ⇒ 训练步数会变，不是"只改一项"')
    missing = [p.stem for p in imgs if not (DST / 'labels/train' / f'{p.stem}.txt').exists()]
    if missing:
        ok = False
        print(f'   ❌ 缺标签 {len(missing)} 张，例：{missing[:3]}')

    # 2) 坐标合法 + 逐类计数
    cnt = {0: 0, 1: 0, 2: 0}
    bad = []
    for lp in lbs:
        for c, cx, cy, w, h in read_label(lp):
            cnt[c] = cnt.get(c, 0) + 1
            if not (0 < cx < 1 and 0 < cy < 1 and 0 < w <= 1 and 0 < h <= 1) \
                    or cx - w / 2 < -1e-6 or cx + w / 2 > 1 + 1e-6 \
                    or cy - h / 2 < -1e-6 or cy + h / 2 > 1 + 1e-6:
                bad.append((lp.stem, c, cx, cy, w, h))
    print(f'\n2) 类别实例：break {cnt[0]}｜thunderbolt {cnt[1]}｜wear {cnt[2]}')
    print(f'   坐标越界/非法：{len(bad)} 处 {"✅" if not bad else "❌ 例：" + str(bad[:3])}')
    if bad:
        ok = False

    # 3) 与原 train 对照：非 wear 类必须一字不变
    orig = {0: 0, 1: 0, 2: 0}
    for lp in (SRC / 'labels/train').glob('*.txt'):
        for c, *_ in read_label(lp):
            orig[c] = orig.get(c, 0) + 1
    same = cnt[0] == orig[0] and cnt[1] == orig[1]
    print(f'\n3) 与原 train 对照：break {orig[0]}→{cnt[0]}｜thunderbolt {orig[1]}→{cnt[1]}｜'
          f'wear {orig[2]}→{cnt[2]}')
    print(f'   非 wear 类一字不变：{"✅" if same else "❌"}；'
          f'wear 净增 {cnt[2] - orig[2]}（+{(cnt[2] - orig[2]) / orig[2] * 100:.1f}%）')
    if not same:
        ok = False

    # 4) 未增强的图必须与源逐字节一致（全量核对，不抽检）
    man = json.loads((DST / 'manifest_cpm.json').read_text(encoding='utf-8'))
    aug = {e['aug'] for e in man}
    diff = changed = 0
    for ip in sorted((SRC / 'images/train').glob('*.jpg')):
        if ip.stem in aug:
            continue
        if sha(ip) == sha(DST / 'images/train' / ip.name):
            diff += 1
        else:
            changed += 1
    print(f'\n4) 未增强的图逐字节一致：{diff}/{diff + changed} '
          f'{"✅" if changed == 0 else f"❌ 有 {changed} 张被动了"}')
    if changed:
        ok = False
    # 增强的图必须确实变了（否则等于没增强）
    unchanged_aug = [s for s in sorted(aug)
                     if sha(SRC / 'images/train' / f'{s}.jpg')
                     == sha(DST / 'images/train' / f'{s}.jpg')]
    print(f'   已增强 {len(aug)} 张中，内容确实改变的：{len(aug) - len(unchanged_aug)}'
          f' {"✅" if not unchanged_aug else "❌ 有增强图与原图完全相同" + str(unchanged_aug[:3])}')
    if unchanged_aug:
        ok = False

    # 5) 端到端：合成图里标签所指的位置，是否真的是从供体贴过来的那块
    rng = random.Random(0)
    sample = rng.sample(man, min(12, len(man)))
    mads = []
    for e in sample:
        syn = cv2.imread(str(DST / 'images/train' / f'{e["aug"]}.jpg'))
        for pr in e['patch']:
            x1, y1, x2, y2 = [int(v) for v in pr['canvas_box_px']]
            got = syn[max(0, y1):y2, max(0, x1):x2]
            d = cv2.imread(str(SRC / 'images/train' / f'{pr["donor"]}.jpg'))
            a1, b1, a2, b2 = [int(v) for v in pr['donor_box_px']]
            want = d[b1:b2, a1:a2]
            if got.shape != want.shape or got.size == 0:
                mads.append(999.0)
                continue
            mads.append(float(np.abs(got.astype(np.int16) - want.astype(np.int16)).mean()))
    hit = sum(1 for m in mads if m < 12)
    print(f'\n5) 端到端落位复核（从产物重算，{len(mads)} 个贴片）：')
    print(f'   标签框处像素 vs 供体缺陷框 的 MAD = {np.mean(mads):.2f}'
          f'（min {np.min(mads):.2f} / max {np.max(mads):.2f}）')
    print(f'   MAD<12（JPEG 重编码容差内）：{hit}/{len(mads)} '
          f'{"✅ 标签与贴图位置一致" if hit == len(mads) else "❌ 有贴片与标签错位"}')
    if hit != len(mads):
        ok = False

    print(f'\n{"=" * 92}\n{"✅ 全部通过" if ok else "❌ 存在问题，见上"}\n{"=" * 92}')
    return ok


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true', help='只统计供体/画布，不写盘')
    ap.add_argument('--verify-only', action='store_true', help='只验证已有产物')
    a = ap.parse_args()

    if a.verify_only:
        sys.exit(0 if verify() else 1)

    if a.dry_run:
        tr_img, tr_lb = SRC / 'images/train', SRC / 'labels/train'
        canv = wear = 0
        for lp in sorted(tr_lb.glob('*.txt')):
            if any(r[0] == WEAR for r in read_label(lp)):
                wear += 1
            else:
                canv += 1
        patches, ndonor = build_patches(tr_img, tr_lb)
        print(f'[dry-run] train 图 {canv + wear}｜含 wear {wear}｜可作画布 {canv}')
        print(f'[dry-run] 供体图 {ndonor}｜贴片池 {len(patches)}')
        print(f'[dry-run] 计划增强 {min(N_SYN, canv)} 张画布（就地替换）× 每张 1~{MAX_PASTE} 个 wear 贴片')
        print(f'[dry-run] train 张数仍 {canv + wear} ⇒ epochs 仍可用 60（步数不变）')
        return

    print('=' * 92)
    print('c4 前置件：wear 定向 copy-paste 数据集构建（就地替换版）')
    print('=' * 92)
    generate()
    verify()


if __name__ == '__main__':
    main()
