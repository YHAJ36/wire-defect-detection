# -*- coding: utf-8 -*-
"""从 train_sgd.log 里提取每次 run 最后一次验证的各类别指标（临时脚本）。"""
import re

raw = open('D:/2/train_sgd.log', 'rb').read().decode('utf-8', 'replace')
raw = re.sub(r'\x1b\[[0-9;]*m', '', raw)

SEP = '\\'
marks = [(m.start(), m.group(1)) for m in
         re.finditer(r'Logging results to (D:\\2\\runs\\[a-z0-9_]+)', raw)]
marks.append((len(raw), None))

pat = re.compile(r'^\s{2,}(all|break|thunderbolt|wear)\s+(\d+)\s+(\d+)\s+'
                 r'([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', re.M)

print(f"{'run':<22}{'class':<13}{'mAP50':>8}{'mAP50-95':>10}{'P':>8}{'R':>8}")
print('-' * 70)
for i in range(len(marks) - 1):
    seg = raw[marks[i][0]:marks[i + 1][0]]
    last = {b[0]: b for b in pat.findall(seg)}
    if len(last) < 4:
        continue
    name = marks[i][1].split(SEP)[-1]
    for k in ['all', 'break', 'thunderbolt', 'wear']:
        b = last[k]
        print(f"{name if k == 'all' else '':<22}{k:<13}"
              f"{float(b[5]):>8.3f}{float(b[6]):>10.3f}{float(b[3]):>8.3f}{float(b[4]):>8.3f}")
    print()
