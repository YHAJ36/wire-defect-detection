"""把 148 对 (jpg, txt) 按 8:1:1 划分到 train/val/test。

源目录 : D:\\2\\ultralytics_wire_main\\datasets\\wire_data_yolo\\images\\train
目标   : D:\\2\\dataset\\images\\{train,val,test}
         D:\\2\\dataset\\labels\\{train,val,test}

划分   : 118 / 15 / 15，随机打乱（固定 seed 保证可复现）
"""

import glob
import os
import random
import shutil
import sys

SRC = r"D:\2\ultralytics_wire_main\datasets\wire_data_yolo\images\train"
DST = r"D:\2\dataset"

SEED = 42
N_TRAIN, N_VAL, N_TEST = 118, 15, 15


def main():
    # 1. 收集配对
    jpgs = {os.path.splitext(os.path.basename(p))[0]
            for p in glob.glob(os.path.join(SRC, "*.jpg"))}
    txts = {os.path.splitext(os.path.basename(p))[0]
            for p in glob.glob(os.path.join(SRC, "*.txt"))}

    if jpgs != txts:
        print("ERROR: jpg/txt 不配对")
        print("  jpg only:", sorted(jpgs - txts))
        print("  txt only:", sorted(txts - jpgs))
        sys.exit(1)

    stems = sorted(jpgs)
    total = len(stems)
    print(f"Source     : {SRC}")
    print(f"Pairs found: {total}")

    if total != N_TRAIN + N_VAL + N_TEST:
        print(f"WARNING: 期望 {N_TRAIN+N_VAL+N_TEST} 对, 实际 {total} 对")

    # 2. 建目录
    for sub in ("train", "val", "test"):
        os.makedirs(os.path.join(DST, "images", sub), exist_ok=True)
        os.makedirs(os.path.join(DST, "labels", sub), exist_ok=True)

    # 3. 打乱并切分
    random.seed(SEED)
    shuffled = stems[:]
    random.shuffle(shuffled)

    splits = {
        "train": shuffled[:N_TRAIN],
        "val":   shuffled[N_TRAIN:N_TRAIN + N_VAL],
        "test":  shuffled[N_TRAIN + N_VAL:],
    }

    # 4. 复制
    print("-" * 60)
    for sub, names in splits.items():
        for stem in names:
            shutil.copy2(os.path.join(SRC, stem + ".jpg"),
                         os.path.join(DST, "images", sub, stem + ".jpg"))
            shutil.copy2(os.path.join(SRC, stem + ".txt"),
                         os.path.join(DST, "labels", sub, stem + ".txt"))
        print(f"{sub:<6} copied: {len(names)}")

    # 5. 校验
    print("-" * 60)
    print("Verification:")
    all_assigned = []
    ok = True
    for sub in ("train", "val", "test"):
        ni = len(glob.glob(os.path.join(DST, "images", sub, "*.jpg")))
        nl = len(glob.glob(os.path.join(DST, "labels", sub, "*.txt")))
        names_i = {os.path.splitext(os.path.basename(p))[0]
                   for p in glob.glob(os.path.join(DST, "images", sub, "*.jpg"))}
        names_l = {os.path.splitext(os.path.basename(p))[0]
                   for p in glob.glob(os.path.join(DST, "labels", sub, "*.txt"))}
        pair_ok = (names_i == names_l)
        all_assigned.extend(names_i)
        print(f"  {sub:<6} images={ni:<4} labels={nl:<4} paired={pair_ok}")
        if not pair_ok:
            ok = False

    # 重叠 / 遗漏检查
    overlap = len(all_assigned) - len(set(all_assigned))
    missing = set(stems) - set(all_assigned)
    print("-" * 60)
    print(f"Total assigned : {len(all_assigned)}")
    print(f"Duplicates     : {overlap}")
    print(f"Missing        : {len(missing)} {sorted(missing)[:5]}")
    print(f"ALL OK         : {ok and overlap == 0 and not missing}")


if __name__ == "__main__":
    main()
