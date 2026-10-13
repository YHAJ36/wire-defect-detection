# -*- coding: utf-8 -*-
"""A/B 超参 diff：修复臂@60 vs 原版@60。

为什么要查：修复臂@60 用 train_v11_640_cbamfix.py，对照是 wire_v11_640_both（原版@60）。
两者必须**逐项一致**，否则 @60 的对比不干净 —— z 均值差异分不清是"修复改动"还是"别的超参"。

方法：直接比两个 **args.yaml**（从 runs/<name>/ 读）。
     args.yaml 是 ultralytics 训练开始时自己写出的**实际生效值**（已含所有默认值），
     比读训练脚本的 kwargs 可靠：脚本里没写的项会被 ultralytics 用默认值填上，
     而那正是 A/B 最容易漏掉的差异来源。
     脚本层再补一道：AST 解析两个脚本的 model.train(...) 调用，逐项比字面量 kwargs。

输出：logs/ab_config_diff.txt
用法：D:/yolo-env/Scripts/python.exe D:/2/code/tools/check_ab_config.py
"""
import ast
import sys
from pathlib import Path

RUN_A = 'wire_v11_640_both'        # 原版@60（已完成的 60 轮）
RUN_B = 'wire_v11_640_cbamfix'     # 修复臂@60
RUNS = Path('D:/2/runs')
OUT = Path('D:/2/logs/ab_config_diff.txt')

SCRIPT_A = Path('D:/2/code/yolov11/train_v11_640.py')
SCRIPT_B = Path('D:/2/code/yolov11/train_v11_640_cbamfix.py')

# 这些键**不影响训练动力学**，允许不同；命中则不计为不一致（但仍列出，便于核对）
COSMETIC = {
    'name', 'save_dir',           # 目录名
    'save_period',                # 只影响落盘频率，不进入 loss/梯度
    'mode', 'task',               # 任务声明
}
# 这些键**必须一致**，不一致直接判 A/B 不干净
MUST = [
    'epochs', 'batch', 'imgsz', 'optimizer', 'lr0', 'lrf', 'momentum',
    'weight_decay', 'warmup_epochs', 'warmup_momentum', 'warmup_bias_lr',
    'cos_lr', 'close_mosaic', 'seed', 'deterministic', 'single_cls',
    'box', 'cls', 'dfl', 'nbs', 'dropout', 'label_smoothing',
    'mosaic', 'mixup', 'copy_paste', 'cutmix', 'erasing', 'auto_augment',
    'hsv_h', 'hsv_s', 'hsv_v', 'degrees', 'translate', 'scale', 'shear',
    'perspective', 'flipud', 'fliplr', 'bgr',
    'data', 'amp', 'fraction', 'multi_scale', 'rect', 'pretrained',
    'patience', 'workers', 'device', 'val', 'split', 'resume', 'freeze',
    'overlap_mask', 'mask_ratio', 'cache',
]


def parse_yaml_simple(p):
    """极简 yaml 解析：args.yaml 全是扁平的 `key: value`，不引 PyYAML。"""
    d = {}
    for ln in p.read_text(encoding='utf-8').splitlines():
        ln = ln.rstrip()
        if not ln or ln.startswith('#') or ':' not in ln:
            continue
        k, _, v = ln.partition(':')
        v = v.strip()
        if v in ('null', '~', ''):
            val = None
        elif v == 'true':
            val = True
        elif v == 'false':
            val = False
        else:
            try:
                val = int(v)
            except ValueError:
                try:
                    val = float(v)
                except ValueError:
                    val = v.strip("'\"")
        d[k.strip()] = val
    return d


def train_kwargs(script):
    """AST 取 model.train(...) 的关键字字面量。"""
    tree = ast.parse(script.read_text(encoding='utf-8'))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func if isinstance(node.func, ast.Attribute) else None
        if f is not None and f.attr == 'train':
            out = {}
            for kw in node.keywords:
                try:
                    out[kw.arg] = ast.literal_eval(kw.value)
                except Exception:
                    out[kw.arg] = f'<非字面量: {ast.unparse(kw.value)}>'
            return out
    return {}


def fmt(v):
    if v is None:
        return 'null'
    if isinstance(v, bool):
        return 'true' if v else 'false'
    return str(v)


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    lines = []

    def say(s=''):
        lines.append(s)
        print(s, flush=True)

    ya = parse_yaml_simple(RUNS / RUN_A / 'args.yaml')
    yb = parse_yaml_simple(RUNS / RUN_B / 'args.yaml')

    say('=' * 96)
    say(f'A/B 超参 diff：{RUN_A}（原版@60） vs {RUN_B}（修复臂@60）')
    say('=' * 96)
    say()
    say('来源：runs/<name>/args.yaml（ultralytics 自写的实际生效值，已含全部默认值）')
    say(f'      {RUNS / RUN_A / "args.yaml"}')
    say(f'      {RUNS / RUN_B / "args.yaml"}')
    say()

    # ---- 主表：两列对照 ----
    keys = list(dict.fromkeys(list(ya) + list(yb)))
    mismatch = []
    cosmetic = []
    say('-' * 96)
    say(f'{"超参":<22}{"原版@60":<26}{"修复@60":<26}{"判定"}')
    say('-' * 96)
    for k in keys:
        va, vb = ya.get(k, '<缺失>'), yb.get(k, '<缺失>')
        if va == vb:
            tag = ''
        elif k in COSMETIC:
            tag = '· 无害（不影响训练动力学）'
            cosmetic.append(k)
        else:
            tag = '★ 不一致'
            mismatch.append((k, va, vb))
        say(f'{k:<22}{fmt(va):<26}{fmt(vb):<26}{tag}')

    # ---- 必须一致项的显式核对 ----
    say()
    say('-' * 96)
    say('必须一致项逐条核对（MUST 列表）')
    say('-' * 96)
    for k in MUST:
        va, vb = ya.get(k, '<缺失>'), yb.get(k, '<缺失>')
        ok = va == vb
        say(f'  [{"OK " if ok else "!! "}] {k:<20}{fmt(va):<24}{fmt(vb):<24}')
        if not ok and (k, va, vb) not in mismatch:
            mismatch.append((k, va, vb))

    # ---- 脚本层：声明的 kwargs 是否一致 ----
    say()
    say('-' * 96)
    say('脚本层核对：train(...) 的字面量 kwargs')
    say('-' * 96)
    ka, kb = train_kwargs(SCRIPT_A), train_kwargs(SCRIPT_B)
    skeys = list(dict.fromkeys(list(ka) + list(kb)))
    sdiff = []
    for k in skeys:
        va, vb = ka.get(k, '<未声明>'), kb.get(k, '<未声明>')
        same = va == vb
        if not same:
            sdiff.append((k, va, vb))
        say(f'  [{"OK " if same else "!! "}] {k:<20}{fmt(va):<26}{fmt(vb):<26}')
    say()
    say('  脚本 diff（除注入的 CBAM 类外应无差异）：')
    say(f'    train_v11_640.py         = {SCRIPT_A}')
    say(f'    train_v11_640_cbamfix.py = {SCRIPT_B}')
    say('    预期差异仅两处：`from cbam_fixed import CBAM`、默认 --mode/--name。')
    say('    另：cbamfix 的 --mode/--name 有默认值（原版 required=True），'
        '是命令行便利性，不进 args.yaml。')

    # ---- fix_settings.py 的作用域 ----
    say()
    say('-' * 96)
    say('fix_settings.py 是否可能污染 A/B？')
    say('-' * 96)
    say('  只读写 ultralytics settings.json 的两个**路径**键：')
    say('    weights_dir  -> D:/1/models')
    say('    datasets_dir -> D:/2')
    say('  不触碰任何训练超参；且本项目脚本一律用绝对路径传 weights/data/project，')
    say('  故 settings 被重置也不影响训练/评估结果（见该脚本文件头）。')
    say('  ⇒ fix_settings.py **不构成** A/B 的差异来源。')

    # ---- 判读 ----
    say()
    say('=' * 96)
    say('判读')
    say('=' * 96)
    if mismatch:
        say(f'  [失败] 发现 {len(mismatch)} 项超参不一致，@60 对比**不干净**，需先消掉：')
        for k, va, vb in mismatch:
            say(f'      {k}: {fmt(va)}  vs  {fmt(vb)}')
    else:
        say('  [通过] 所有非无害项逐项一致，@60 对比**干净**。')
        if cosmetic:
            say(f'  仅以下项不同且无害（不影响训练动力学）：'
                f'{", ".join(f"{k}({fmt(ya.get(k))} vs {fmt(yb.get(k))})" for k in cosmetic)}')
    if sdiff:
        say()
        say(f'  [注意] 脚本层声明有 {len(sdiff)} 处差异（上表已列）：')
        for k, va, vb in sdiff:
            say(f'      {k}: {fmt(va)}  vs  {fmt(vb)}')
        say('      若这些项在 args.yaml 里已收敛（上表"通过"），则只是默认值写法不同，不影响结果。')
    else:
        say('  脚本层声明亦逐项一致。')

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'\n已写入 {OUT}')


if __name__ == '__main__':
    main()
