"""画「原始多小框 vs 合并大框」对比图。

左：原始自动标注（每个 bbox 单独画）
右：同类合并（同一类别的所有框取并集，画成一个外接大框）
输出: D:/2/preview_merge_compare.png
"""

import os
import cv2
import numpy as np

IMG = r"D:/2/dataset_new/images/JPEGImages/firc_xianlan_141.jpg"
BBOX = r"D:/2/dataset_new/labels_bbox/firc_xianlan_141.txt"
OUT = r"D:/2/preview_merge_compare.png"

NAMES = {0: "break", 1: "thunderbolt", 2: "wear"}
COLORS = {0: (0, 0, 255), 1: (255, 0, 0), 2: (0, 255, 0)}  # BGR 红/蓝/绿


def load_boxes(txt, w, h):
    boxes = []
    with open(txt, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cls, cx, cy, bw, bh = [float(x) for x in line.split()]
            x1 = (cx - bw / 2) * w
            y1 = (cy - bh / 2) * h
            x2 = (cx + bw / 2) * w
            y2 = (cy + bh / 2) * h
            boxes.append((int(cls), int(x1), int(y1), int(x2), int(y2)))
    return boxes


def draw_label(img, text, x, y, color):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(img, (x, y - th - 2), (x + tw + 6, y + 2), color, -1)
    cv2.putText(img, text, (x + 3, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def merge_by_class(boxes):
    """同类所有框取并集 -> 每个类别一个大框 [(cls, x1, y1, x2, y2), ...]"""
    groups = {}
    for cls, x1, y1, x2, y2 in boxes:
        g = groups.setdefault(cls, [x1, y1, x2, y2])
        g[0] = min(g[0], x1); g[1] = min(g[1], y1)
        g[2] = max(g[2], x2); g[3] = max(g[3], y2)
    return [(c, *g) for c, g in sorted(groups.items())]


def panel(title, draw_fn):
    img = cv2.imread(IMG).copy()
    draw_fn(img)
    scale = 900 / img.shape[1]
    img = cv2.resize(img, (900, int(round(img.shape[0] * scale))))
    bar = np.full((36, 900, 3), 245, dtype=np.uint8)
    cv2.putText(bar, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def main():
    img0 = cv2.imread(IMG)
    h, w = img0.shape[:2]
    boxes = load_boxes(BBOX, w, h)
    merged = merge_by_class(boxes)
    n_break = sum(1 for b in boxes if b[0] == 0)
    n_wear = sum(1 for b in boxes if b[0] == 2)

    def draw_original(img):
        for cls, x1, y1, x2, y2 in boxes:
            cv2.rectangle(img, (x1, y1), (x2, y2), COLORS[cls], 2)
            draw_label(img, NAMES[cls], x1, y1 - 4, COLORS[cls])

    def draw_merged(img):
        ov = img.copy()
        for cls, x1, y1, x2, y2 in merged:
            cv2.rectangle(ov, (x1, y1), (x2, y2), COLORS[cls], -1)
            cv2.rectangle(img, (x1, y1), (x2, y2), COLORS[cls], 3)
            draw_label(img, NAMES[cls], x1, y1 - 6, COLORS[cls])
        cv2.addWeighted(ov, 0.3, img, 0.7, 0, img)

    left = panel(f"原始自动标注: {len(boxes)} 个小框 (break x{n_break} + wear x{n_wear})", draw_original)
    right = panel(f"合并后: {len(merged)} 个大框 (同类并集)", draw_merged)

    grid = np.hstack([left, right])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    ok = cv2.imwrite(OUT, grid)
    print(f"SAVED={ok} {OUT} {grid.shape[1]}x{grid.shape[0]}")
    print("原始框:", len(boxes), " 合并后框:", len(merged))
    for c, x1, y1, x2, y2 in merged:
        print(f"  {NAMES[c]} 合并框: ({x1},{y1})-({x2},{y2}) 尺寸 {x2-x1}x{y2-y1}")


if __name__ == "__main__":
    main()
