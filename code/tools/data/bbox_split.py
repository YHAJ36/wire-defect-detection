"""用 bbox 标注（labels_bbox）划分 8:1:1，输出 dataset_final_bbox，data.yaml 加 task: detect。

源图片: D:/2/dataset_new/images/JPEGImages
源标注: D:/2/dataset_new/labels_bbox (1785 个 bbox txt, 每行 class cx cy w h)
输出:   D:/2/dataset_final_bbox/{images,labels}/{train,val,test}
"""

import os
import random
import shutil
from pathlib import Path

SRC_IMG = Path(r"D:/2/dataset_new/images/JPEGImages")
SRC_LBL = Path(r"D:/2/dataset_new/labels_bbox")
FINAL = Path(r"D:/2/dataset_final_bbox")

IMG_EXTS = {".jpg", ".jpeg", ".png"}
SEED = 42
NAMES = ["break", "thunderbolt", "wear"]


def main():
    imgs = [f for f in os.listdir(SRC_IMG) if Path(f).suffix.lower() in IMG_EXTS]
    lbl_names = {f for f in os.listdir(SRC_LBL) if f.endswith(".txt")}
    labeled = [f for f in imgs if (f.rsplit(".", 1)[0] + ".txt") in lbl_names]
    labeled.sort()

    random.seed(SEED)
    random.shuffle(labeled)
    n = len(labeled)
    splits = {
        "train": labeled[: int(n * 0.8)],
        "val": labeled[int(n * 0.8): int(n * 0.9)],
        "test": labeled[int(n * 0.9):],
    }

    for name, files in splits.items():
        idir = FINAL / "images" / name
        ldir = FINAL / "labels" / name
        idir.mkdir(parents=True, exist_ok=True)
        ldir.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy2(SRC_IMG / f, idir / f)
            shutil.copy2(SRC_LBL / (f.rsplit(".", 1)[0] + ".txt"), ldir / (f.rsplit(".", 1)[0] + ".txt"))
        print(f"[split] {name}: {len(files)} 图 + {len(files)} 标")

    yaml = (
        "path: D:/2/dataset_final_bbox\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "nc: 3\n"
        f"names: {NAMES}\n"
        "task: detect\n"
    )
    (FINAL / "data.yaml").write_text(yaml, encoding="utf-8")
    print("[yaml] 已写入 D:/2/dataset_final_bbox/data.yaml")


if __name__ == "__main__":
    main()
