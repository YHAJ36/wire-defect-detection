# -*- coding: utf-8 -*-
"""E2 前置件：wear 图像 2x 过采样列表。

口径
    train 共 1428 张，其中含 wear（cls=2）的 165 张。
    E2 让这 165 张在 epoch 内出现两次：1428 + 165 = 1593 行。
    break / thunderbolt 的实例数与图像数不变，只改 wear 的采样频率。
    （与 c4 的 copy-paste 不同：不生成新图、不改标签、不改总图数占比口径，
     只改 DataLoader 的采样分布。因此 iterations 会从 ceil(1428/64)=23 变为
     ceil(1593/64)=25，每 epoch 多 2 步 —— 这是过采样的必然代价，记录中注明。）

产物
    D:/2/exp_upscale/data/wear_oversample_train.txt
    D:/2/exp_upscale/data/data_wearos.yaml
    D:/2/exp_upscale/data/wear_oversample_manifest.json

用法
    生成：  D:/yolo-env/Scripts/python.exe D:/2/code/upscale/make_wear_oversample.py
    校验：  ... --verify-only
"""
import argparse
import json
from pathlib import Path

SRC = Path('D:/2/dataset_final_bbox')
OUT = Path('D:/2/exp_upscale/data')
IMG_TRAIN = SRC / 'images' / 'train'
LBL_TRAIN = SRC / 'labels' / 'train'
LIST = OUT / 'wear_oversample_train.txt'
YAML = OUT / 'data_wearos.yaml'
MANIFEST = OUT / 'wear_oversample_manifest.json'

NAMES = ['break', 'thunderbolt', 'wear']


def scan():
    imgs = sorted(IMG_TRAIN.glob('*'))
    imgs = [p for p in imgs if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}]
    wear = []
    wear_boxes = 0
    for im in imgs:
        lb = LBL_TRAIN / (im.stem + '.txt')
        has_wear = False
        if lb.exists():
            for line in lb.read_text(encoding='utf-8').splitlines():
                line = line.strip()
                if not line:
                    continue
                if int(float(line.split()[0])) == 2:
                    has_wear = True
                    wear_boxes += 1
        if has_wear:
            wear.append(im)
    return imgs, wear, wear_boxes


def write():
    imgs, wear, wear_boxes = scan()
    if len(imgs) != 1428:
        raise SystemExit('❌ train 图像数 %d，预期 1428' % len(imgs))
    if len(wear) != 165:
        raise SystemExit('❌ 含 wear 图像数 %d，预期 165' % len(wear))

    OUT.mkdir(parents=True, exist_ok=True)
    lines = [str(p).replace('\\', '/') for p in imgs]
    extra = [str(p).replace('\\', '/') for p in wear]
    LIST.write_text('\n'.join(lines + extra) + '\n', encoding='utf-8')

    YAML.write_text(
        '# E2: wear 图像 2x 过采样（1428 + 165 = 1593 行）\n'
        'path: %s\n'
        'train: %s\n'
        'val: images/val\n'
        'test: images/test\n'
        'nc: 3\n'
        "names: ['break', 'thunderbolt', 'wear']\n"
        'task: detect\n' % (SRC.as_posix(), LIST.as_posix()),
        encoding='utf-8',
    )

    manifest = {
        'source': SRC.as_posix(),
        'train_images': len(imgs),
        'wear_images': len(wear),
        'wear_boxes': wear_boxes,
        'list_lines': len(lines) + len(extra),
        'expected_epoch_iterations_batch8': (len(lines) + len(extra) + 7) // 8,
        'wear_images_duplicated': extra,
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding='utf-8')
    print('写入 %s：%d + %d = %d 行' % (LIST, len(lines), len(extra), len(lines) + len(extra)))
    print('写入 %s' % YAML)
    print('写入 %s' % MANIFEST)


def verify():
    if not LIST.exists() or not YAML.exists():
        raise SystemExit('❌ 产物不存在，先运行生成')
    lines = [x for x in LIST.read_text(encoding='utf-8').splitlines() if x.strip()]
    imgs, wear, _ = scan()
    expected = [str(p).replace('\\', '/') for p in imgs] + [str(p).replace('\\', '/') for p in wear]
    assert lines == expected, '列表内容与当前数据集不一致'
    assert len(lines) == 1593, len(lines)
    assert len(set(lines)) == 1428, len(set(lines))
    from collections import Counter
    dup = {k: v for k, v in Counter(lines).items() if v > 1}
    assert len(dup) == 165 and all(v == 2 for v in dup.values()), len(dup)
    print('OK：1593 行 / 1428 唯一图 / 165 张 wear 各出现 2 次')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--verify-only', action='store_true')
    args = p.parse_args()
    verify() if args.verify_only else write()


if __name__ == '__main__':
    main()
