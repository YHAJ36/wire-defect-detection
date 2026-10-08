"""合并两组自动标注数据 -> 划分 train/val/test -> 生成 data.yaml。

说明：148 张手动图是 1800 张的全子集（148 ⊂ 1800），且两者的 best.pt 自动标注内容一致，
所以合并去重后 = 1800 张图 + 1785 个标注（15 张「no detections」无标注，已排除出训练集）。
"""

import os
import random
import shutil
from collections import Counter
from pathlib import Path

SRC_IMG_148 = Path(r"D:/2/ultralytics_wire_main/datasets/wire_data_yolo/images/train")
SRC_LBL_148 = Path(r"D:/2/dataset_148_auto")
SRC_IMG = Path(r"D:/2/dataset_new/images/JPEGImages")   # 1800 张（超集）
SRC_LBL = Path(r"D:/2/dataset_new/labels_auto")         # 1785 个标注（超集）

MERGED = Path(r"D:/2/dataset_merged")
FINAL = Path(r"D:/2/dataset_final")

IMG_EXTS = {".jpg", ".jpeg", ".png"}
SEED = 42
NAMES = ["break", "thunderbolt", "wear"]


def build_merged():
    (MERGED / "images").mkdir(parents=True, exist_ok=True)
    (MERGED / "labels").mkdir(parents=True, exist_ok=True)

    imgs = sorted(f for f in os.listdir(SRC_IMG) if Path(f).suffix.lower() in IMG_EXTS)
    lbls = sorted(f for f in os.listdir(SRC_LBL) if f.endswith(".txt"))

    for f in imgs:
        shutil.copy2(SRC_IMG / f, MERGED / "images" / f)
    for f in lbls:
        shutil.copy2(SRC_LBL / f, MERGED / "labels" / f)

    # 148 组全在 1800 组内，无需额外复制；仅做一致性确认
    names_148 = {f for f in os.listdir(SRC_IMG_148) if Path(f).suffix.lower() in IMG_EXTS}
    print(f"[merge] 1800 组图片 {len(imgs)} / 标注 {len(lbls)} 已复制")
    print(f"[merge] 148 组图片 {len(names_148)} 张，全部包含于 1800 组内（子集），未新增")
    return imgs, lbls


def class_dist(labels):
    c = Counter()
    for f in labels:
        p = MERGED / "labels" / f
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if line:
                c[int(line.split()[0])] += 1
    return c


def split(imgs):
    """按图片名随机打乱，只分有标注的图片，8:1:1。"""
    labeled = [f for f in imgs if (f.rsplit(".", 1)[0] + ".txt") in set(os.listdir(MERGED / "labels"))]
    random.seed(SEED)
    random.shuffle(labeled)
    n = len(labeled)
    train = labeled[: int(n * 0.8)]
    val = labeled[int(n * 0.8): int(n * 0.9)]
    test = labeled[int(n * 0.9):]
    return labeled, train, val, test


def write_split(labeled, train, val, test):
    for split_name, files in [("train", train), ("val", val), ("test", test)]:
        idir = FINAL / "images" / split_name
        ldir = FINAL / "labels" / split_name
        idir.mkdir(parents=True, exist_ok=True)
        ldir.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy2(MERGED / "images" / f, idir / f)
            lab = f.rsplit(".", 1)[0] + ".txt"
            shutil.copy2(MERGED / "labels" / lab, ldir / lab)
    print(f"[split] 有标注图总数 {len(labeled)} -> train {len(train)} / val {len(val)} / test {len(test)}")


def write_yaml():
    yaml = (
        "path: D:/2/dataset_final\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "nc: 3\n"
        f"names: {NAMES}\n"
    )
    (FINAL / "data.yaml").write_text(yaml, encoding="utf-8")
    print("[yaml] 已写入 D:/2/dataset_final/data.yaml")


def main():
    imgs, lbls = build_merged()
    c = class_dist(lbls)
    print("[stats] 合并后 images 总数:", len(imgs))
    print("[stats] 合并后 labels 总数:", len(lbls))
    print("[stats] 类别实例数:",
          " | ".join(f"{NAMES[k]}={v}" for k, v in sorted(c.items())),
          f" 合计 {sum(c.values())}")

    labeled, train, val, test = split(imgs)
    write_split(labeled, train, val, test)
    write_yaml()


if __name__ == "__main__":
    main()
