"""把 SAM 自动标注的分割多边形 txt 转成 YOLO 检测框 bbox txt。

输入: D:/2/dataset_new/labels_auto/*.txt   (class x1 y1 x2 y2 ... 归一化多边形)
输出: D:/2/dataset_new/labels_bbox/*.txt   (class cx cy w h 归一化 bbox)

规则: 每个多边形(一行)取所有点的最小外接矩形，类别号不变。
"""

import os
from pathlib import Path

SRC = Path(r"D:/2/dataset_new/labels_auto")
DST = Path(r"D:/2/dataset_new/labels_bbox")


def poly_to_bbox(vals):
    """vals: [class, x1, y1, x2, y2, ...] -> (class, cx, cy, w, h) 或 None"""
    cls = int(vals[0])
    pts = vals[1:]
    if len(pts) < 2 or len(pts) % 2 != 0:
        return None
    xs = pts[0::2]
    ys = pts[1::2]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2
    w = xmax - xmin
    h = ymax - ymin
    return cls, cx, cy, w, h


def main():
    os.makedirs(DST, exist_ok=True)
    src_files = sorted(SRC.glob("*.txt"))
    converted = 0
    anomalies = []  # (filename, reason)

    for f in src_files:
        out_lines = []
        with open(f, "r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                toks = line.split()
                try:
                    vals = [float(t) for t in toks]
                except ValueError:
                    anomalies.append((f.name, f"第{lineno}行含非数字"))
                    continue
                bbox = poly_to_bbox(vals)
                if bbox is None:
                    anomalies.append((f.name, f"第{lineno}行坐标数异常(共{len(vals)-1}个)"))
                    continue
                cls, cx, cy, w, h = bbox
                out_lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        if out_lines:
            with open(DST / f.name, "w", encoding="utf-8") as fh:
                fh.write("\n".join(out_lines) + "\n")
            converted += 1
        else:
            # 源文件存在但全部为空/异常 -> 仍写一个空文件占位，保持一一对应
            (DST / f.name).touch()
            anomalies.append((f.name, "无有效行，输出空文件"))

    print(f"SRC 文件数: {len(src_files)}")
    print(f"转换输出文件数: {converted}")
    print(f"异常数: {len(anomalies)}")
    for a in anomalies[:50]:
        print("  ANOMALY", a)

    # 统计 bbox 总数
    total = 0
    for f in DST.glob("*.txt"):
        total += sum(1 for _ in open(f, encoding="utf-8") if _.strip())
    print(f"输出 bbox 总行数: {total}")


if __name__ == "__main__":
    main()
