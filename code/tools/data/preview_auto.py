"""可视化自动标注：在图上画 SAM 分割轮廓 + bbox + 类别名，拼成 2×3 大图。

用法: python preview_auto.py
输出: D:/2/preview_auto.png
"""

import os
import cv2
import numpy as np

IMG_DIR = r"D:/2/dataset_new/images/JPEGImages"
LBL_DIR = r"D:/2/dataset_new/labels_auto"
OUT = r"D:/2/preview_auto.png"

# 6 张覆盖三类别的图
FILES = [
    "firc_xianlan_99",    # thunderbolt
    "firc_xianlan_100",   # thunderbolt
    "firc_xianlan_984",   # break
    "firc_xianlan_141",   # break + wear
    "firc_xianlan_101",   # wear
    "firc_xianlan_102",   # wear
]

NAMES = {0: "break", 1: "thunderbolt", 2: "wear"}
# BGR 颜色（cv2 用 BGR）
COLORS = {0: (0, 0, 255), 1: (255, 0, 0), 2: (0, 255, 0)}  # 红/蓝/绿

CELL_W = 460
TITLE_H = 34


def load_polygons(txt_path, w, h):
    polys = []
    if not os.path.exists(txt_path):
        return polys
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            toks = line.split()
            cls = int(toks[0])
            vals = [float(t) for t in toks[1:]]
            pts = []
            for i in range(0, len(vals) - 1, 2):
                x = vals[i] * w
                y = vals[i + 1] * h
                pts.append((int(round(x)), int(round(y))))
            if len(pts) >= 3:
                polys.append((cls, pts))
    return polys


def draw_one(name):
    img_path = os.path.join(IMG_DIR, name + ".jpg")
    txt_path = os.path.join(LBL_DIR, name + ".txt")
    img = cv2.imread(img_path)
    if img is None:
        raise RuntimeError(f"读图失败: {img_path}")
    h, w = img.shape[:2]
    polys = load_polygons(txt_path, w, h)

    overlay = img.copy()
    for cls, pts in polys:
        color = COLORS[cls]
        arr = np.array([pts], dtype=np.int32)
        cv2.fillPoly(overlay, [arr], color)
        cv2.polylines(img, [arr], isClosed=True, color=color, thickness=2)
        # 外接框
        xmin = min(p[0] for p in pts); ymin = min(p[1] for p in pts)
        xmax = max(p[0] for p in pts); ymax = max(p[1] for p in pts)
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 1)
        # 标签
        label = NAMES[cls]
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = ymin - 6 if ymin - 6 > th else ymin + th + 4
        cv2.rectangle(img, (xmin, ty - th - 2), (xmin + tw + 6, ty + 2), color, -1)
        cv2.putText(img, label, (xmin + 3, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    img = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)  # 半透明填充

    # 缩放到统一宽度
    scale = CELL_W / img.shape[1]
    img = cv2.resize(img, (CELL_W, int(round(img.shape[0] * scale))))

    # 顶部标题条：文件名 + 各类数量
    counts = {}
    for cls, _ in polys:
        counts[cls] = counts.get(cls, 0) + 1
    summary = " ".join(f"{NAMES[c]}x{n}" for c, n in sorted(counts.items()))
    title = f"{name}  [{summary}]"
    bar = np.full((TITLE_H, CELL_W, 3), 245, dtype=np.uint8)
    cv2.putText(bar, title, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def main():
    cells = [draw_one(n) for n in FILES]
    # 统一高度（按最高者补齐白边）
    max_h = max(c.shape[0] for c in cells)
    for i, c in enumerate(cells):
        if c.shape[0] < max_h:
            pad = np.full((max_h - c.shape[0], CELL_W, 3), 255, dtype=np.uint8)
            cells[i] = np.vstack([c, pad])
    rows = [np.hstack(cells[0:3]), np.hstack(cells[3:6])]
    grid = np.vstack(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    ok = cv2.imwrite(OUT, grid)
    print(f"SAVED={ok} {OUT} {grid.shape[1]}x{grid.shape[0]}")
    for n in FILES:
        p = os.path.join(LBL_DIR, n + ".txt")
        cnt = sum(1 for _ in open(p, encoding="utf-8")) if os.path.exists(p) else 0
        print(f"  {n}: {cnt} 段")


if __name__ == "__main__":
    main()
