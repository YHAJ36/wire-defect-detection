"""X-AnyLabeling .json -> YOLO .txt 批量转换

用法:
    python json2yolo.py [目标目录]

默认目录: D:\\2\\ultralytics_wire_main\\datasets\\wire_data_yolo\\images\\train

规则:
    - 每个 .json 生成同名 .txt（输出到同一目录）
    - 格式: class_id x_center y_center width height（归一化到 0~1）
    - 类别映射: break=0, thunderbolt=1, wear=2
    - rectangle / quadrilateral 均取外接矩形
    - 没有标签的 .json -> 生成空 .txt
"""

import json
import os
import sys
import glob

LABEL_MAP = {"break": 0, "thunderbolt": 1, "wear": 2}

DEFAULT_DIR = r"D:\2\ultralytics_wire_main\datasets\wire_data_yolo\images\train"


def convert_one(json_path):
    with open(json_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    W = data.get("imageWidth")
    H = data.get("imageHeight")
    if not W or not H:
        raise ValueError("missing imageWidth/imageHeight")

    lines = []
    stats = {"kept": 0, "unknown": 0, "degenerate": 0, "unknown_labels": {}}

    for shape in data.get("shapes", []):
        label = shape.get("label")

        if label not in LABEL_MAP:
            stats["unknown"] += 1
            stats["unknown_labels"][label] = stats["unknown_labels"].get(label, 0) + 1
            continue

        pts = shape.get("points") or []
        if len(pts) < 2:
            stats["degenerate"] += 1
            continue

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)

        x_center = (xmin + xmax) / 2.0 / W
        y_center = (ymin + ymax) / 2.0 / H
        width = (xmax - xmin) / W
        height = (ymax - ymin) / H

        # 夹到 [0,1]
        x_center = min(max(x_center, 0.0), 1.0)
        y_center = min(max(y_center, 0.0), 1.0)
        width = min(max(width, 0.0), 1.0)
        height = min(max(height, 0.0), 1.0)

        if width <= 0 or height <= 0:
            stats["degenerate"] += 1
            continue

        lines.append(
            f"{LABEL_MAP[label]} {x_center:.6f} {y_center:.6f} "
            f"{width:.6f} {height:.6f}"
        )
        stats["kept"] += 1

    return lines, stats


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DIR
    target = os.path.abspath(target)

    if not os.path.isdir(target):
        print(f"ERROR: directory not found: {target}")
        sys.exit(1)

    json_files = sorted(glob.glob(os.path.join(target, "*.json")))
    print(f"Target dir : {target}")
    print(f"JSON found : {len(json_files)}")
    print("-" * 60)

    total_kept = 0
    total_unknown = 0
    total_degenerate = 0
    all_unknown = {}
    written = 0
    empty_txt = 0
    cls_count = {0: 0, 1: 0, 2: 0}
    errors = []

    for jp in json_files:
        name = os.path.splitext(os.path.basename(jp))[0]
        try:
            lines, stats = convert_one(jp)
        except Exception as e:
            errors.append((os.path.basename(jp), str(e)))
            continue

        txt_path = os.path.join(target, name + ".txt")
        with open(txt_path, "w", encoding="utf-8", newline="\n") as fh:
            if lines:
                fh.write("\n".join(lines) + "\n")
            # 无标签 -> 空文件

        written += 1
        if not lines:
            empty_txt += 1

        total_kept += stats["kept"]
        total_unknown += stats["unknown"]
        total_degenerate += stats["degenerate"]
        for k, v in stats["unknown_labels"].items():
            all_unknown[k] = all_unknown.get(k, 0) + v
        for ln in lines:
            cls_count[int(ln.split()[0])] += 1

    print(f"TXT written      : {written}")
    print(f"  empty txt      : {empty_txt}")
    print(f"Boxes kept       : {total_kept}")
    print(f"Boxes unknown    : {total_unknown} {all_unknown}")
    print(f"Degenerate boxes : {total_degenerate}")
    print("-" * 60)
    print(f"Class 0 break       : {cls_count[0]}")
    print(f"Class 1 thunderbolt : {cls_count[1]}")
    print(f"Class 2 wear        : {cls_count[2]}")

    # 没有对应 json 的图片
    imgs = sorted(glob.glob(os.path.join(target, "*.jpg")))
    json_names = {os.path.splitext(os.path.basename(p))[0] for p in json_files}
    orphans = [os.path.basename(i) for i in imgs
               if os.path.splitext(os.path.basename(i))[0] not in json_names]
    if orphans:
        print("-" * 60)
        print(f"JPG without JSON ({len(orphans)}): {orphans}")

    if errors:
        print("-" * 60)
        print(f"ERRORS ({len(errors)}):")
        for n, e in errors:
            print(f"  {n}: {e}")


if __name__ == "__main__":
    main()
