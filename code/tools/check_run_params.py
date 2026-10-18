# -*- coding: utf-8 -*-
"""注入核验闸门：核对 run 的 best.pt 参数量 == 该臂的预期值。

为什么必须在**每个 run 之后**都核一遍（2026-10-15）
    `--resume` 续训时 ultralytics **不会**调 `get_model()`
    （`ultralytics/engine/model.py:794`：`if not args.get("resume")` 才重建模型），
    模型直接从 ckpt 反序列化。于是：
      · 注入层在（当初 pickle 进了 ckpt）—— 正常；
      · 但训练脚本里那句"注入 CBAM+BN …全模型参数量 X"的打印**不会出现**，
        日志里看不出注入是否还在；
      · 万一注入以任何方式丢失，续训会**静默地跑成一个普通基线**，
        事后只看 mAP 根本分不出"臂无效"和"臂根本没跑"。
    这就是 WORKLOG 答辩素材 1（CBAM 注入被静默丢弃）的同一类事故，
    所以照 `check_triplet_params.py` 的做法设一道量具闸门。

预期值（nc=3 / yolo11s，已用三个已完成的 run 的 best.pt 实测核对过）：
    baseline（无注入）          9,428,953
    +CBAM+BN @neck(层9)        9,461,821    （= baseline + 32,868）
    +Triplet  @neck(层9)       9,429,253    （= baseline + 300）

用法：
    D:/yolo-env/Scripts/python.exe D:/2/code/tools/check_run_params.py nf_v11_896_s42 ...
    （不给 run 名 ⇒ 核验 exp_upscale/runs 下全部 run）
退出码：有任一不符 ⇒ 1（供链脚本判读）
"""
import sys
from pathlib import Path

sys.path.insert(0, 'D:/2/code/yolov11')      # cbam_bn / triplet_att 反序列化需要

from ultralytics import YOLO  # noqa: E402

RUNS = Path('D:/2/exp_upscale/runs')
P_BASE = 9_428_953
P_CBAM_BN = 9_461_821
P_TRIPLET = 9_429_253


def expect_of(run):
    """从 run 名推断该臂的预期参数量。"""
    if 'cbam_bn' in run:
        return P_CBAM_BN, 'CBAM+BN'
    if 'triplet' in run:
        return P_TRIPLET, 'Triplet'
    if run.startswith('nf_') or 'baseline' in run:
        return P_BASE, 'baseline'
    return None, '未知臂'


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

    names = [a for a in sys.argv[1:] if not a.startswith('--')]
    explicit = bool(names)
    if not names:
        # 默认只扫本批（nf_* / att_*）。归档 run（wire_*）不属于本批、
        # 无法按臂核验，若一并计入会把"未知臂"当失败 ⇒ 链脚本误报注入丢失。
        names = sorted(p.name for p in RUNS.iterdir()
                       if (p / 'weights' / 'best.pt').exists()
                       and (p.name.startswith('nf_') or p.name.startswith('att_')))

    print('=' * 88)
    print('注入核验闸门 —— best.pt 参数量 == 该臂预期值')
    print('=' * 88)
    print(f'{"run":<34}{"臂":<12}{"参数量":>13}{"预期":>13}{"判定":>14}')
    print('-' * 88)

    bad = 0
    for run in names:
        w = RUNS / run / 'weights' / 'best.pt'
        exp, label = expect_of(run)
        if not w.exists():
            print(f'{run:<34}{label:<12}{"—":>13}{"—":>13}{"缺 best.pt":>14}')
            bad += 1
            continue
        n = sum(q.numel() for q in YOLO(str(w)).model.parameters())
        if exp is None:
            # 显式点名却认不出臂 ⇒ 名字给错了，算失败；
            # 扫描时遇到归档 run ⇒ 非本批，跳过，不算失败。
            if explicit:
                print(f'{run:<34}{label:<12}{n:>13,}{"—":>13}{"★ 名字不符":>14}')
                bad += 1
            else:
                print(f'{run:<34}{label:<12}{n:>13,}{"—":>13}{"非本批，跳过":>14}')
        elif n == exp:
            print(f'{run:<34}{label:<12}{n:>13,}{exp:>13,}{"OK":>14}')
        else:
            print(f'{run:<34}{label:<12}{n:>13,}{exp:>13,}{"★ 不符":>14}')
            bad += 1

    print('-' * 88)
    if bad:
        print(f'❌ {bad} 个 run 未通过核验 —— 注入可能丢失，数字不可用。')
    else:
        print(f'✅ 全部 {len(names)} 个 run 通过：注入在场，mAP 可比。')
    print('=' * 88)
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
