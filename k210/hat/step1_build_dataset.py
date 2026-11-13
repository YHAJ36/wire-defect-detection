# -*- coding: utf-8 -*-
"""
================================================================================
 第 1 步：把 SHWD 数据集做成「大头像」数据集
================================================================================

【这个脚本在干什么】
    SHWD 是工地监控视角的数据集。一张原图里可能站着十几个人，每个人头
    只有 20x30 像素。而 K210 的摄像头是变焦的，你实际用的时候人会占掉
    大半个画面。

    所以第一步要做的事：把每张原图里的每个目标框「裁出来、放大」，
    做成一张图只有一个头的「大头像」，每张统一 320x224。

【为什么非要做这件事】
    训练时看到的画面，必须尽量接近板子实拍时的画面。
    如果拿「远处的小人」训练、拿「镜头里的大头」推理，精度会掉一大截
    —— 这叫尺度域偏移，是目标检测最常见的翻车原因之一。

【输入】
    ModelScope 缓存的 SHWD（COCO 格式）
    位置：C:/Users/LR/.cache/modelscope/hub/datasets/
          modelscope/safetyhelmet_detection_for_train/master/

【输出】
    D:/2/k210/hat/dataset/images/train/*.jpg    320x224 图片
    D:/2/k210/hat/dataset/images/val/*.jpg
    D:/2/k210/hat/dataset/labels/train/*.txt    YOLO 格式标注
    D:/2/k210/hat/dataset/labels/val/*.txt
    D:/2/k210/hat/dataset/data.yaml             给训练脚本看的配置文件

【怎么跑】
    C:/Users/LR/AppData/Local/Programs/Python/Python312/python.exe step1_build_dataset.py
================================================================================
"""

# ------------------------------------------------------------------ #
# 0. 导入要用的工具
# ------------------------------------------------------------------ #
import os            # 处理路径、建文件夹
import json          # 读写 COCO 格式的 json 标注
import glob          # 按通配符找文件，比如 all/*.jpg
import random        # 随机划分训练集/验证集
import hashlib       # 算图片 MD5，用来去重
import shutil        # 递归删目录（清空上一次的输出）
import collections   # 计数器，统计两类各多少张
from PIL import Image  # 读图、裁图、缩放


# ------------------------------------------------------------------ #
# 1. 参数区（所有能调的东西都放这里，下面代码不用改）
# ------------------------------------------------------------------ #

# 项目根目录
BASE = r'D:/2/k210/hat'

# 输出目录：dataset/images/... 和 dataset/labels/...
OUT = os.path.join(BASE, 'dataset')

# K210 的输入尺寸。320x224 是这块板子的上限：
#   KPU 特征图最大 320x240，而且必须是 stride 32 的倍数
IMG_W = 320
IMG_H = 224

# 裁剪时让目标框占画面的多大比例。
#   0.60 表示目标（头/帽子）占裁图长度的 60%，
#   就是一张「大头像」：头在画面正中很大，背景只剩一点。
#   调大：更像贴脸特写，但上下文全丢
#   调小：更像半身照，目标变小、画面里挤进好几个人
#
# 实测记录：用 0.40 时，裁出来的画面里还是一堆人（原图是人群/教室场景），
# 不满足「一个人一个大头」的要求，所以调到 0.60。
TARGET_RATIO = 0.60

# 一张裁片里最多允许有几个目标框（包括主角）。
#   超过这个数说明这一块区域还是「人群」，不是特写，直接跳过。
#   这是保证「大头像」而不是「一群人」的关键开关。
MAX_OBJ_IN_CROP = 3

# 原图里面积小于这个比例的框直接丢掉。
#   0.005 ≈ 千分之五。太小的框裁出来放大后是一团糊，只会污染训练。
MIN_BOX_AREA = 0.005

# 裁剪后框的短边至少要这么多像素，否则丢掉（防止裁出一堆马赛克）
MIN_BOX_PIX = 10

# 训练集 / 验证集 的比例（按原图分，不是按裁出来的图分）
VAL_RATIO = 0.15

# 随机种子。固定住，保证每次跑出来的划分完全一样，方便复现
SEED = 2026

# 类别名。SHWD 的 COCO 里应该是 hat=戴帽、person=未戴帽的人头
# 这里先按顺序写死，脚本运行时会打印真实类别名供核对
CLASS_NAMES = ['hat', 'person']


# ------------------------------------------------------------------ #
# 2. 找到 ModelScope 缓存里的原始数据
# ------------------------------------------------------------------ #
def find_raw_roots():
    """
    在 ModelScope 缓存目录里找 train / validation 两个分片的数据。

    ModelScope 下载完以后，目录结构大致是：
        .../master/train/images/*.jpg
        .../master/train/train.json       ← COCO 标注
    但不同版本放的位置可能不一样，所以这里用「递归搜索」的方式找，
    不写死路径。

    返回：[(split名, 图片目录, json路径), ...]
    """
    # ModelScope 的默认缓存根目录
    cache_root = os.path.join(
        os.path.expanduser('~'), '.cache', 'modelscope', 'hub', 'datasets'
    )

    # 在这个根下面找所有跟 safetyhelmet 有关的目录
    found = []
    for dirpath, dirnames, filenames in os.walk(cache_root):
        # 只看名字里带 safetyhelmet 的那一支，别的数据集不管
        if 'safetyhelmet' not in dirpath.replace('\\', '/').lower():
            continue
        # 找这个目录里有没有 json 标注文件
        for fn in filenames:
            if not fn.endswith('.json'):
                continue
            # 记录 json 的完整路径
            json_path = os.path.join(dirpath, fn)
            # 图片目录一般叫 images，跟 json 同级
            img_dir = os.path.join(dirpath, 'images')
            if os.path.isdir(img_dir):
                # 用 json 文件名判断这是哪个分片
                split = 'train' if 'train' in fn.lower() else 'val'
                found.append((split, img_dir, json_path))

    # 去重（同一个 json 可能被扫到两次）
    uniq = {}
    for split, img_dir, json_path in found:
        uniq[json_path] = (split, img_dir, json_path)
    return list(uniq.values())


# ------------------------------------------------------------------ #
# 3. 读 COCO 标注，整理成「每张图有哪些框」的形式
# ------------------------------------------------------------------ #
def load_coco(json_path, img_dir):
    """
    读一个 COCO json，返回一个列表，每个元素是：
        (图片完整路径, 宽, 高, [(类别下标, x, y, w, h), ...])
    其中 x y w h 是像素坐标、左上角原点。

    COCO 的 bbox 本来就是 [x, y, w, h] 左上角格式，省了转换。
    """
    # 打开 json
    with open(json_path, 'r', encoding='utf-8') as f:
        coco = json.load(f)

    # COCO 里类别 id 是从 1 开始的（1=hat, 2=person）
    # 这里把 id 映射成 0,1,2... 因为 YOLO 要求类别从 0 开始
    cat_id_to_idx = {}
    for i, cat in enumerate(coco['categories']):
        cat_id_to_idx[cat['id']] = i
    # 打印出来核对一下顺序对不对
    print('  类别映射:', {c['id']: c['name'] for c in coco['categories']})

    # image_id -> 图片信息
    img_by_id = {}
    for im in coco['images']:
        img_by_id[im['id']] = im

    # 把标注按 image_id 归组
    anns_by_img = collections.defaultdict(list)
    for ann in coco['annotations']:
        anns_by_img[ann['image_id']].append(ann)

    # 组装结果
    result = []
    for img_id, im in img_by_id.items():
        # 拼出图片的完整路径
        path = os.path.join(img_dir, im['file_name'])
        # 图片不存在就跳过（数据集偶尔有缺图）
        if not os.path.exists(path):
            continue

        boxes = []
        for ann in anns_by_img.get(img_id, []):
            # 只要正常标注，iscrowd=1 的那种不规则标注跳过
            if ann.get('iscrowd', 0):
                continue
            x, y, w, h = ann['bbox']
            # 宽高必须是正数才有意义
            if w <= 1 or h <= 1:
                continue
            idx = cat_id_to_idx.get(ann['category_id'])
            if idx is None:
                continue
            boxes.append((idx, float(x), float(y), float(w), float(h)))

        # 一个框都没有的图不要
        if boxes:
            result.append((path, im['width'], im['height'], boxes))

    return result


# ------------------------------------------------------------------ #
# 4. 裁一个「大头像」出来
# ------------------------------------------------------------------ #
def crop_bigface(img, boxes, box_index):
    """
    围绕第 box_index 个框裁一块区域，缩放成 320x224。

    参数：
        img        PIL 图片对象
        boxes      [(类别, x, y, w, h), ...] 像素坐标
        box_index  以哪个框为「主角」

    返回：
        (裁好的图, 新的 boxes 列表)  新 boxes 的坐标已经换算到 320x224 上
        如果这个框没法裁（越界、太小），返回 (None, None)
    """
    W, H = img.size                      # 原图尺寸
    cls, bx, by, bw, bh = boxes[box_index]

    # 画面宽高比，320/224 = 1.4286
    ratio = IMG_W / IMG_H

    # --- 计算裁剪窗口大小 ---
    # 要求：框的高占窗口高的 TARGET_RATIO，宽同理
    # 取两者算出来的较大者，保证框完整装得下
    ch = max(bh / TARGET_RATIO, (bw / TARGET_RATIO) / ratio)
    cw = ch * ratio                      # 宽按比例算出来

    # 窗口比原图还大，说明这张图本身就不够大 → 直接用整图
    if cw >= W or ch >= H:
        cw, ch = W, H
        # 整图模式下宽高比可能不对，后面 resize 会拉伸，可以接受

    # --- 让窗口以框为中心 ---
    x0 = bx + bw / 2.0 - cw / 2.0
    y0 = by + bh / 2.0 - ch / 2.0

    # --- 把窗口推回图片范围内（不能越界）---
    if x0 < 0:
        x0 = 0.0
    if y0 < 0:
        y0 = 0.0
    if x0 + cw > W:
        x0 = W - cw
    if y0 + ch > H:
        y0 = H - ch

    # 保险起见再夹一次（浮点误差可能让它变成 -0.0001）
    x0 = max(0.0, x0)
    y0 = max(0.0, y0)

    # --- 执行裁剪 ---
    # PIL 的 crop 要的是 (左, 上, 右, 下)
    patch = img.crop((int(x0), int(y0), int(x0 + cw), int(y0 + ch)))
    # 缩放到 K210 的输入尺寸
    patch = patch.resize((IMG_W, IMG_H), Image.BILINEAR)

    # --- 把所有框换算到新图坐标 ---
    new_boxes = []
    for c, x, y, w, h in boxes:
        # 先在裁剪窗口内重新算左上角
        nx = x - x0
        ny = y - y0
        # 再缩放到 320x224
        nx = nx / cw * IMG_W
        ny = ny / ch * IMG_H
        nw = w / cw * IMG_W
        nh = h / ch * IMG_H

        # 裁掉超出画面的部分
        x1 = max(0.0, nx)
        y1 = max(0.0, ny)
        x2 = min(float(IMG_W), nx + nw)
        y2 = min(float(IMG_H), ny + nh)

        fw = x2 - x1
        fh = y2 - y1

        # 裁完太小的框没意义，丢掉
        if fw < MIN_BOX_PIX or fh < MIN_BOX_PIX:
            continue

        # 主角框必须还在（否则这次裁剪就白干了）
        new_boxes.append((c, x1, y1, fw, fh))

    # 裁完一个框都没有 → 放弃
    if not new_boxes:
        return None, None

    # 裁完目标太多 → 说明这块区域还是人群，不是「大头像」，放弃
    if len(new_boxes) > MAX_OBJ_IN_CROP:
        return None, None

    return patch, new_boxes


# ------------------------------------------------------------------ #
# 5. 把框转成 YOLO 的 txt 文本
# ------------------------------------------------------------------ #
def boxes_to_yolo(boxes):
    """
    YOLO 格式每行是：类别 中心x 中心y 宽 高，全部除以图片宽高变到 0~1。
    """
    lines = []
    for c, x, y, w, h in boxes:
        cx = (x + w / 2.0) / IMG_W     # 中心点 x，归一化
        cy = (y + h / 2.0) / IMG_H     # 中心点 y，归一化
        nw = w / IMG_W                 # 宽，归一化
        nh = h / IMG_H                 # 高，归一化
        lines.append('%d %.6f %.6f %.6f %.6f' % (c, cx, cy, nw, nh))
    return '\n'.join(lines)


# ------------------------------------------------------------------ #
# 6. 主流程
# ------------------------------------------------------------------ #
def main():
    random.seed(SEED)   # 固定随机种子

    print('=' * 60)
    print('第 1 步：生成大头像数据集')
    print('=' * 60)

    # ---------- 6.1 找原始数据 ----------
    roots = find_raw_roots()
    if not roots:
        print('!! 没找到 ModelScope 缓存里的 SHWD，先跑 fetch_ms.py 下载')
        return
    print('找到 %d 个数据分片' % len(roots))
    for split, img_dir, json_path in roots:
        print('   %-6s 图:%s' % (split, img_dir))
        print('         标注:%s' % json_path)

    # ---------- 6.2 把两个分片合成一个大池子 ----------
    all_items = []          # 每个元素是 (图路径, 宽, 高, 框列表)
    seen_md5 = set()        # 已经见过的图片 MD5，用来去重
    dup = 0                 # 重复了几张

    for split, img_dir, json_path in roots:
        items = load_coco(json_path, img_dir)
        print('   %s 读到 %d 张有标注的图' % (split, len(items)))

        for path, w, h, boxes in items:
            # 算这张图的 MD5（读文件内容，不是文件名）
            with open(path, 'rb') as f:
                md5 = hashlib.md5(f.read()).hexdigest()
            # 见过就跳过 —— SHWD 是从网上爬的，确实有重复图
            if md5 in seen_md5:
                dup += 1
                continue
            seen_md5.add(md5)
            all_items.append((path, w, h, boxes))

    print('去重后剩 %d 张原图（丢掉重复 %d 张）' % (len(all_items), dup))
    if not all_items:
        print('!! 一张图都没读到，检查上面的路径')
        return

    # ---------- 6.3 划分训练集 / 验证集 ----------
    # 注意：是按【原图】划分的。
    # 同一张原图裁出来的所有大头像，只能落在同一个集合里，
    # 否则就是泄漏 —— 训练集里见过这张图的裁片，验证集里又见到它的另一张裁片，
    # 指标会虚高。
    random.shuffle(all_items)
    n_val = int(len(all_items) * VAL_RATIO)
    val_items = all_items[:n_val]
    train_items = all_items[n_val:]

    print('原图划分：训练 %d 张 / 验证 %d 张' % (len(train_items), len(val_items)))

    # ---------- 6.4 建输出目录 ----------
    # 上一次跑剩下的文件必须清掉，否则新旧样本会混在一起。
    # 注意：只删 dataset 这个输出目录，不碰任何原始数据。
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
        print('已清空旧的输出目录 %s' % OUT)

    # 重新建空目录
    for sub in ('train', 'val'):
        for kind in ('images', 'labels'):
            d = os.path.join(OUT, kind, sub)
            os.makedirs(d, exist_ok=True)

    # ---------- 6.5 逐张裁剪 ----------
    # 统计每个集合裁出来多少张、每类多少个框
    stats = {}
    for sub in ('train', 'val'):
        stats[sub] = {'imgs': 0, 'boxes': collections.Counter()}

    for sub, items in (('train', train_items), ('val', val_items)):
        print('--- 处理 %s ---' % sub)

        for i, (path, w, h, boxes) in enumerate(items):
            # 打开原图，统一转 RGB（有的图是灰度或带透明通道）
            try:
                img = Image.open(path).convert('RGB')
            except Exception:
                continue

            # 原图面积，用来算每个框占多少
            img_area = float(w * h)

            # 对这张图里的每一个框，各裁一张大头像
            for bi, (cls, bx, by, bw, bh) in enumerate(boxes):

                # 太小的框不要：面积占比不够，裁出来是糊的
                if (bw * bh) / img_area < MIN_BOX_AREA:
                    continue

                # 裁
                patch, nb = crop_bigface(img, boxes, bi)
                if patch is None:
                    continue

                # 输出文件名：原图序号 + 第几个框，保证唯一
                name = '%05d_%02d' % (i, bi)

                # 存图
                patch.save(os.path.join(OUT, 'images', sub, name + '.jpg'),
                           quality=92)

                # 存标注
                with open(os.path.join(OUT, 'labels', sub, name + '.txt'),
                          'w', encoding='utf-8') as f:
                    f.write(boxes_to_yolo(nb))

                # 记账
                stats[sub]['imgs'] += 1
                for c, *_ in nb:
                    stats[sub]['boxes'][c] += 1

            # 每 200 张原图打印一次进度
            if (i + 1) % 200 == 0:
                print('   %d/%d' % (i + 1, len(items)))

    # ---------- 6.6 打印统计 ----------
    print()
    for sub in ('train', 'val'):
        s = stats[sub]
        print('%s: %d 张裁片' % (sub, s['imgs']))
        for c in sorted(s['boxes']):
            nm = CLASS_NAMES[c] if c < len(CLASS_NAMES) else str(c)
            print('    class %d (%s): %d 个框' % (c, nm, s['boxes'][c]))

    # ---------- 6.7 写 data.yaml ----------
    # 训练脚本靠这个文件找数据、知道有几个类
    yaml_path = os.path.join(OUT, 'data.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write('# 大头像安全帽数据集（自动生成）\n')
        f.write('path: %s\n' % OUT.replace('\\', '/'))
        f.write('train: images/train\n')
        f.write('val: images/val\n')
        f.write('nc: %d\n' % len(CLASS_NAMES))
        f.write('names: %s\n' % json.dumps(CLASS_NAMES, ensure_ascii=False))
    print('\n已写 %s' % yaml_path)
    print('=' * 60)
    print('第 1 步完成')


# 只有直接运行这个文件时，才执行 main()
# 被别的脚本 import 时不会自动跑，这是 Python 的惯例
if __name__ == '__main__':
    main()
