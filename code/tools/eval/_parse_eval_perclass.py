# -*- coding: utf-8 -*-
"""从 eval_test_sgd.log 提取 7 个模型 test 集各类别指标（临时脚本）。"""
import re

raw = open('D:/2/eval_test_sgd.log', 'rb').read().decode('utf-8', 'replace')
raw = re.sub(r'\x1b\[[0-9;]*m', '', raw)

# ultralytics 每段以 "test: Scanning" 或 save_dir 标记开头；简单按 "Speed:" 分段
blocks = re.split(r'^Speed:.*$', raw, flags=re.M)
names = ['wire_v11_v3', 'wire_v11_backbone_v3', 'wire_v11_neck_v3',
         'wire_v11_both_v3', 'wire_v12s_v4', 'wire_v12s_cbam_v4', 'wire_v5_v2']

pat = re.compile(r'^\s{2,}(all|break|thunderbolt|wear)\s+(\d+)\s+(\d+)\s+'
                 r'([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', re.M)

found = []
for b in blocks:
    last = {x[0]: x for x in pat.findall(b)}
    if len(last) == 4:
        found.append(last)

print(f"{'model':<22}{'class':<13}{'mAP50':>8}{'mAP50-95':>10}{'P':>8}{'R':>8}")
print('-' * 70)
for name, blk in zip(names, found):
    for k in ['all', 'break', 'thunderbolt', 'wear']:
        x = blk[k]
        print(f"{name if k == 'all' else '':<22}{k:<13}"
              f"{float(x[5]):>8.3f}{float(x[6]):>10.3f}{float(x[3]):>8.3f}{float(x[4]):>8.3f}")
    print()
print('段数 =', len(found))
