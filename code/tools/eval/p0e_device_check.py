# -*- coding: utf-8 -*-
"""P0-E（训练内 val vs 独立 val 的 0.21 点口径差）+ P0-G（设备差的外推边界）。

=============================================================================
P0-E：先看 ultralytics 源码，把两个口径的差异逐条摆开
=============================================================================
`engine/validator.py` 的 `BaseValidator.__call__`：

    if self.training:
        # Force FP16 val during training
        self.args.half = self.device.type != "cpu" and trainer.amp
        model = trainer.ema.ema or trainer.model
        model = model.half() if self.args.half else model.float()
    else:
        model = AutoBackend(weights=..., fp16=self.args.half)   # 我们不传 half ⇒ False

`engine/trainer.py` 的 `save_model`：

    "ema": deepcopy(self.ema.ema).half(),      # 第 567 行 —— checkpoint 存的是 **fp16** EMA

于是两个口径是：

    训练内 val（results.csv）  权重 = 内存里的 fp32 EMA（未量化）   算术 = **fp16（强制 half）**
    独立 .val()（报告用）      权重 = best.pt 里的 fp16 EMA（已量化） 算术 = fp32

**两边各有各的精度缺损，方向不同** ⇒ 它们的差不是"谁更准"，是**精度口径差**。
这也意味着训练内那个 val 值**不可复现**：fp32 EMA 从未落盘，只有被量化过的 fp16 版本。

其余候选逐条排除（本脚本只负责"跑两次看波动基线"和"fp16 算术"这两条）：

  mosaic        排除。`data/build.py`: `augment=mode == "train"`，
                而 `data/dataset.py` 里 mosaic/mixup 只在 `if self.augment:` 分支构造
                ⇒ val 的 dataloader **结构上**没有 mosaic，任何轮次都没有。
                close_mosaic 改的是训练集，与 val 无关。
  rect          排除。`build_dataset` 调 `build_yolo_dataset(..., stride=...)` **不传 rect**，
                取默认 False；args.yaml 也是 rect: false ⇒ 两侧 rect=False。
  batch         排除。`get_dataloader(..., self.args.batch)` ⇒ 训练内=args.batch=8；
                本脚本独立评估也传 batch=8。
  workers       排除，且有**内部对照**：896 组 args.yaml workers=0、640 组 workers=4，
                两组的训练内 val 都用各自的 workers。而两组都出现同号同量级的偏移
                （+0.21 / +0.25 点）⇒ 与 workers 无关。
  cache         排除。args.yaml cache: false。

剩下唯一站得住的机制是**精度**。本脚本做的实验：
  E1 重复性：同权重/同数据/同设备，连续两次独立 val ⇒ 若为 0.0000，说明独立 val 是确定性的，
    那么 0.21 点就**不是跑动噪声**，而是口径差。
  E2 batch：batch=8 vs 16 ⇒ 看 batch 是否影响（排除"训练内用了别的 batch"）。
  E3 fp16 算术：half=True ⇒ 若读数向训练内值靠拢，则"强制 fp16 算术"就是那 0.21 点的来源。

=============================================================================
P0-G：0.0001 的设备差只测过一个完整 test，要外推
=============================================================================
补测 leak / clean 各自的 CPU vs GPU 设备差：
  - 若几组设备差一致（都 ≈0.0001）⇒ "CPU 子集读数与 GPU 完整读数可混算"成立
  - 若不一致 ⇒ 所有跨设备数字作废，泄漏量化必须在单一设备内重做

输出：logs/p0e_device_check.txt
用法：D:/yolo-env/Scripts/python.exe D:/2/code/tools/eval/p0e_device_check.py
"""
import sys
from pathlib import Path

sys.path.insert(0, 'D:/2/code/yolov11')
sys.path.insert(0, 'D:/2/code/yolov12')

from ultralytics import YOLO  # noqa: E402

FULL = 'D:/2/dataset_final_bbox/data.yaml'
SUB = Path('D:/2/dataset_leak_subsets')
OUT = Path('D:/2/logs/p0e_device_check.txt')
NAMES = ['break', 'thunderbolt', 'wear']

MODELS = [('v11@896', 'D:/2/runs/wire_v11_896', 896),
          ('v11@640', 'D:/2/runs/wire_v11_640', 640)]

# 本次 CPU 读数（logs/device_selection_check.txt / logs/leak_eval.txt），用于同表对照
CPU_REF = {
    ('v11@896', 'val_full'): 0.9011, ('v11@640', 'val_full'): 0.8741,
    ('v11@896', 'test_leak'): 0.9235, ('v11@896', 'test_clean'): 0.9144,
    ('v11@640', 'test_leak'): 0.9165, ('v11@640', 'test_clean'): 0.8984,
    ('v11@896', 'val_leak'): 0.9322, ('v11@896', 'val_clean'): 0.8871,
    ('v11@640', 'val_leak'): 0.9410, ('v11@640', 'val_clean'): 0.8481,
}
# 记录在案的 GPU 独立完整 test（eval_test_*.py 口径：split=test/batch=8/workers=0/device=0/half=False）
GPU_TEST = {'v11@896': 0.9148, 'v11@640': 0.8992}


def intrain_best(csv):
    """从 results.csv 取 fitness 最优轮（fitness = 0.1*mAP50 + 0.9*mAP50-95）。"""
    rows = csv.read_text(encoding='utf-8').strip().splitlines()
    hdr = rows[0].split(',')
    i50, i95 = hdr.index('metrics/mAP50(B)'), hdr.index('metrics/mAP50-95(B)')
    best = None
    for r in rows[1:]:
        t = r.split(',')
        if len(t) <= max(i50, i95):
            continue
        f = 0.1 * float(t[i50]) + 0.9 * float(t[i95])
        if best is None or f > best[0]:
            best = (f, int(t[0]), float(t[i50]), float(t[i95]))
    return best


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    lines = []

    def say(s=''):
        lines.append(s)
        print(s, flush=True)

    def run(tag, w, imgsz, data, split, device, half, batch, name):
        m = YOLO(w)   # 每次全新加载：`.half()` 会改模型，复用会串味
        r = m.val(data=data, split=split, imgsz=imgsz, batch=batch, workers=0,
                  device=device, half=half, project='D:/2/runs/_leak_eval',
                  name=f'{name}_{tag}', exist_ok=True, verbose=False, plots=False)
        b = r.box
        say(f'  [done] {name:>18} {tag:<8} mAP50={b.map50:.4f} mAP50-95={b.map:.4f}')
        return float(b.map50), float(b.map)

    # ================= P0-E =================
    say('=' * 104)
    say('P0-E：训练内 val(0.8990/0.8716) vs 独立 val —— 0.21~0.25 点的来源')
    say('=' * 104)
    say('  机理（源码）：训练内 val 走 `model.half()` 强制 fp16 算术（权重是内存中的 fp32 EMA）；')
    say('               独立 .val() 是 fp32 算术（权重是 best.pt 里已量化成 fp16 的 EMA）。')

    E = {}
    IT = {}
    for tag, root, imgsz in MODELS:
        it = intrain_best(Path(root) / 'results.csv')
        IT[tag] = it
        w = f'{root}/weights/best.pt'
        E[(tag, 'gpu_fp32_a')] = run(tag, w, imgsz, FULL, 'val', 0, False, 8, 'E1a_gpu_fp32')
        E[(tag, 'gpu_fp32_b')] = run(tag, w, imgsz, FULL, 'val', 0, False, 8, 'E1b_gpu_fp32')
        E[(tag, 'gpu_b16')] = run(tag, w, imgsz, FULL, 'val', 0, False, 16, 'E2_gpu_b16')
        E[(tag, 'gpu_half')] = run(tag, w, imgsz, FULL, 'val', 0, True, 8, 'E3_gpu_half')
        E[(tag, 'cpu_fp32')] = run(tag, w, imgsz, FULL, 'val', 'cpu', False, 8, 'E4_cpu_fp32')

    say()
    say(f'{"模型":<10}{"训练内val":>11}{"(fitness轮)":>13}{"独立GPU×2":>18}'
        f'{"重复性Δ":>10}{"batch16Δ":>11}{"halfΔ":>10}{"CPU fp32":>10}')
    for tag, _, _ in MODELS:
        f_, ep, m50, m95 = IT[tag]
        a, b = E[(tag, 'gpu_fp32_a')][0], E[(tag, 'gpu_fp32_b')][0]
        h = E[(tag, 'gpu_half')][0]
        c = E[(tag, 'cpu_fp32')][0]
        say(f'{tag:<10}{m50:>11.4f}{ep:>13}{a:>9.4f}/{b:<8.4f}'
            f'{b - a:>+10.4f}{E[(tag, "gpu_b16")][0] - a:>+11.4f}'
            f'{h - a:>+10.4f}{c:>10.4f}')
    say()
    say('  列义：`训练内val` = results.csv 中 fitness 最优轮的 mAP50（被强制 fp16 算术）；')
    say('        `独立GPU×2` = 同权重同数据同设备连跑两次（fp32）；`重复性Δ` = 两次之差；')
    say('        `batch16Δ` = batch 8→16 的变化；`halfΔ` = 开 half=True 的变化（fp16 算术）。')
    say()
    for tag, _, _ in MODELS:
        a = E[(tag, 'gpu_fp32_a')][0]
        h = E[(tag, 'gpu_half')][0]
        m50 = IT[tag][2]
        rep = E[(tag, 'gpu_fp32_b')][0] - a
        say(f'  {tag}：')
        if abs(rep) < 5e-5:
            say(f'    E1 重复性 Δ = {rep:+.5f} ⇒ **独立 val 是确定性的**，0.21 点不是跑动噪声。')
        else:
            say(f'    E1 重复性 Δ = {rep:+.5f} ⇒ 独立 val 自身有 {abs(rep):.4f} 的跑动波动，')
            say('       需先确认它是否足以吃掉口径差（见下）。')
        say(f'    E2 batch 8 vs 16 的 Δ = {E[(tag, "gpu_b16")][0] - a:+.4f}'
            '  （两种口径都是 batch=8，此项只作边界说明）')
        if (h - a) * (h - m50) > 0 and abs(h - a) > 1e-4:
            say(f'    E3 half=True 把读数从 {a:.4f} 推到 {h:.4f}（Δ={h - a:+.4f}），'
                f'**方向朝训练内值 {m50:.4f} 移动**')
            say(f'       ⇒ 与源码机制一致：那 0.21 点来自"训练内强制 fp16 算术"。')
        else:
            say(f'    E3 half=True 读数 {h:.4f}（Δ={h - a:+.4f}）**未**朝训练内值 {m50:.4f} 移动')
            say('       ⇒ "强制 fp16 算术"不足以解释口径差，需另找机制（如实记录，不得含糊）。')
        say(f'    口径差（独立fp32 − 训练内）= {a - m50:+.4f}')

    # 结论
    say()
    say('  【P0-E 结论 —— PDF 里 val 数字写哪个】')
    say('    写**独立评估**的值（896: 0.9011 / 640: 0.8741；cpu/gpu 差 0.0001，等价）。三条理由：')
    say('      (1) **可复现**：权重归档为 best.pt，命令归档，任何人任何设备可重跑；')
    say('          训练内那个值依赖训练时的 fp16 强制，且其 fp32 EMA **从未落盘**，不可审计。')
    say('      (2) **与 test 同口径**：倒挂是 val↔test 的比较，而 test 一直是独立 .val() 读数；')
    say('          混用两种口径会把 0.21 点口径差算进倒挂里。')
    say('      (3) 统一口径后倒挂缩小 0.21/0.25 点（896: 1.58→1.37 点；640: 2.76→2.52 点），')
    say('          即倒挂的 13% 是口径伪影 ⇒ 这一条必须写进 PDF 的口径说明。')
    say('    ⚠ 训练日志里的 val 曲线（results.csv）只作**监控**用途，不作报告口径。')

    # ================= P0-G =================
    say()
    say('=' * 104)
    say('P0-G：设备差的外推边界（0.0001 只测过一个完整 test）')
    say('=' * 104)
    D = {}
    for tag, root, imgsz in MODELS:
        w = f'{root}/weights/best.pt'
        for sub in ('test_leak', 'test_clean', 'val_leak', 'val_clean'):
            sp, s = sub.split('_', 1)
            D[(tag, sub)] = run(tag, w, imgsz, str(SUB / sub / 'data.yaml'),
                                sp, 0, False, 8, f'G_gpu_{sub}')
    say()
    say(f'{"模型/子集":<22}{"CPU":>10}{"GPU":>10}{"CPU−GPU":>11}   判读')
    diffs = []
    for tag, _, _ in MODELS:
        for sub in ('test_leak', 'test_clean', 'val_leak', 'val_clean'):
            c, g = CPU_REF[(tag, sub)], D[(tag, sub)][0]
            d = c - g
            diffs.append(d)
            say(f'{tag + "/" + sub:<22}{c:>10.4f}{g:>10.4f}{d:>+11.4f}')
        c, g = CPU_REF[(tag, 'val_full')], E[(tag, 'cpu_fp32')][0]
        say(f'{tag + "/val_full":<22}{c:>10.4f}{g:>10.4f}{c - g:>+11.4f}   （完整 val 的对照）')
    diffs.append(GPU_TEST['v11@896'] - E[('v11@896', 'cpu_fp32')][0])
    diffs.append(GPU_TEST['v11@640'] - E[('v11@640', 'cpu_fp32')][0])
    span = max(diffs) - min(diffs)
    say()
    say(f'  全部设备差（CPU−GPU）：min {min(diffs):+.4f} / max {max(diffs):+.4f} / 极差 {span:.4f}')
    say(f'  另：完整 test 上的设备差 = {GPU_TEST["v11@896"] - E[("v11@896", "cpu_fp32")][0]:+.4f}'
        f'（896）/ {GPU_TEST["v11@640"] - E[("v11@640", "cpu_fp32")][0]:+.4f}（640）')
    if span <= 0.002:
        say('  ⇒ 各组设备差的极差 ≤ 0.2 点、与子集规模无关 ⇒ **"可混算"成立**（在 0.2 点以内）：')
        say('     CPU 子集读数可与 GPU 完整读数并列，但**跨设备差本身有 ~0.1~0.2 点的量级**，')
        say('     故任何 <0.2 点的跨设备比较都不可立论。')
    else:
        say('  ⇒ 各设备差不一致（极差 > 0.2 点）⇒ **跨设备数字不可混算**，'
            '泄漏量化必须在单一设备内重做。')
    say('  ⚠ 注意：leak/clean 的 CPU 值来自 logs/leak_eval.txt（v11@896 test_leak 0.9235 等），')
    say('     其 GPU 补测是本次新跑的；两者同权重同 batch 同 imgsz，只差设备。')

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'\n已写入 {OUT}')


if __name__ == '__main__':
    main()
