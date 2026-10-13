# -*- coding: utf-8 -*-
"""从评估日志里把 test 集指标抽成结构化表（同名+各类别+Speed），供报告取数。

用法：D:/yolo-env/Scripts/python.exe code/tools/eval/dump_test_metrics.py
只读，不改任何文件。
"""
import re
import sys

LOGS = [
    ('320 SGD',   'D:/2/logs/eval_test_sgd.log',
     ['wire_v11_v3', 'wire_v11_backbone_v3', 'wire_v11_neck_v3', 'wire_v11_both_v3',
      'wire_v12s_v4', 'wire_v12s_cbam_v4', 'v5_v2(320)']),
    ('320 AdamW', 'D:/2/logs/eval_test_adamw.log',
     ['wire_v11_adamw', 'wire_v11_backbone_adamw', 'wire_v11_neck_adamw', 'wire_v11_both_adamw',
      'wire_v12s_adamw', 'wire_v12s_cbam_adamw', 'v5_adamw(320)']),
    ('640 SGD',   'D:/2/logs/train_640_all.log',
     ['v11_640(见 eval_test_640.log)', 'wire_v11_640_neck', 'wire_v11_640_backbone',
      'wire_v11_640_both', 'wire_v12_640', 'wire_v12_640_cbam', 'v5_640']),
    ('640 AdamW', 'D:/2/logs/train_640_adamw_all.log',
     ['wire_v11_640_adamw', 'wire_v11_640_backbone_adamw', 'wire_v11_640_neck_adamw',
      'wire_v11_640_both_adamw', 'v5_640_adamw']),
]

ANSI = re.compile(r'\x1b\[[0-9;]*m')
HDR = re.compile(r'Class\s+Images\s+Instances')
ROW = re.compile(r'^\s*(all|break|thunderbolt|wear)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$')


def blocks(path):
    lines = [ANSI.sub('', l) for l in open(path, encoding='utf-8', errors='replace')]
    out, cur = [], None
    for ln in lines:
        if HDR.search(ln):
            cur = {}
            out.append(cur)
            continue
        m = ROW.match(ln)
        if m and cur is not None:
            cur[m.group(1)] = tuple(float(m.group(i)) for i in range(4, 8))
    return [b for b in out if 'all' in b], lines


def main():
    for setname, path, names in LOGS:
        bs, lines = blocks(path)
        print(f'\n########## {setname}  ({len(bs)} 个 val 块)  {path}')
        for i, b in enumerate(bs):
            nm = names[i] if i < len(names) else f'?[{i}]'
            a = b['all']
            print(f'{i:>2} {nm:<32} mAP50={a[2]:.4f} mAP50-95={a[3]:.4f} P={a[0]:.4f} R={a[1]:.4f}')
            if i >= len(names) - 3:
                for c in ('break', 'thunderbolt', 'wear'):
                    if c in b:
                        print(f'      {c:<12} mAP50={b[c][2]:.4f} mAP50-95={b[c][3]:.4f}')
        sp = [ANSI.sub('', l).strip() for l in lines if 'Speed:' in l]
        if sp:
            print(f'  末 {min(8, len(sp))} 条 Speed:')
            for s in sp[-8:]:
                print(f'    {s[:110]}')


if __name__ == '__main__':
    main()
