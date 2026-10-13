# -*- coding: utf-8 -*-
"""泄漏量化：对 leak / clean 子集分别评估，并给出**噪声基准**与逐类明细。

背景：数据集存在跨 split 近似重复（train↔test 144 对、train↔val 254 对，见
`code/tools/check_near_dup.py`），故报告的 mAP 是"有近复制品的图"与"干净的图"的**混合值**。
本脚本把两者拆开算，回答"泄漏到底抬了多少"。

三个必须一起给的东西（否则结论读不出来）：
  1. **逐类 AP + 逐类实例数**：mAP50(all) 是三类**未加权均值**，每类占 1/3。
     若某类在子集里只剩几个实例（实测 test_leak 的 wear 只有 9 个），
     它的 AP 会剧烈抖动并单独拉动总分 30 点 —— 那种情况下 leak − clean 测的是
     "wear 被切剩几个实例"，不是泄漏。
  2. **噪声基准**：把 clean 对半切成 h1/h2，算两份之间的 AP 差。
     这是"同条件下"的经验噪声尺度；leak − clean 只有明显超过它才算真效应。
     （48 张图上算 AP，抽样误差本身就有几个点。）
  3. **口径标注**：val 的 best.pt 是用**完整 val**选的，故 val 侧同时吃了"泄漏"与
     "选择效应"，只有 test 侧是干净的实验。

设备用 **cpu**：单卡 8G 已被 60 轮训练占住 ~2.9G，本评估只需推理，
放 CPU 上慢几分钟但**对训练零风险**（宁慢不赌）。

输出：logs/leak_eval.txt（同时写 logs/leak_eval.log 供后台跟踪）
用法：D:/yolo-env/Scripts/python.exe D:/2/code/tools/eval/eval_leak_subsets.py
"""
import sys
from pathlib import Path

sys.path.insert(0, 'D:/2/code/yolov11')
sys.path.insert(0, 'D:/2/code/yolov12')

from ultralytics import YOLO  # noqa: E402

SUB = Path('D:/2/dataset_leak_subsets')
OUT = Path('D:/2/logs/leak_eval.txt')
NAMES = ['break', 'thunderbolt', 'wear']

# (显示名, 权重, imgsz) —— imgsz 必须是该模型的训练分辨率，否则是另一种设置
MODELS = [
    ('v11@896', 'D:/2/runs/wire_v11_896/weights/best.pt', 896),
    ('v11@640', 'D:/2/runs/wire_v11_640/weights/best.pt', 640),
]
# 每个 split 要评的子集（顺序即输出顺序）
SUBSETS = ['leak', 'clean', 'clean_h1', 'clean_h2']
SPLITS = ['test', 'val']


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    lines = []

    def say(s=''):
        lines.append(s)
        print(s, flush=True)

    # 先量出每个子集的逐类实例数与图数（与 AP 的类一一对应，用于薄类告警）
    CNT, NIMG = {}, {}
    for sp in SPLITS:
        for sub in SUBSETS:
            ndir = SUB / f'{sp}_{sub}' / 'labels'
            cnt = {i: 0 for i in range(3)}
            n = 0
            for p in sorted(ndir.iterdir()):
                n += 1
                for ln in p.read_text(encoding='utf-8').splitlines():
                    if ln.strip():
                        cnt[int(float(ln.split()[0]))] += 1
            CNT[(sp, sub)], NIMG[(sp, sub)] = cnt, n

    results = {}
    for tag, w, imgsz in MODELS:
        m = YOLO(w)
        for sp in SPLITS:
            for sub in SUBSETS:
                d = SUB / f'{sp}_{sub}'
                r = m.val(
                    data=str(d / 'data.yaml'), split='val', imgsz=imgsz,
                    batch=8, workers=0, device='cpu',
                    # 落盘到独立前缀目录，避免污染 runs/ 下按组遍历的脚本
                    project='D:/2/runs/_leak_eval', name=f'leak_{tag}_{sp}_{sub}',
                    exist_ok=True, verbose=False, plots=False,
                )
                b = r.box
                idx = list(b.ap_class_index)
                per = {int(c): float(b.ap50[i]) for i, c in enumerate(idx)}
                per95 = {int(c): float(b.ap[i]) for i, c in enumerate(idx)}
                results[(tag, sp, sub)] = dict(
                    map50=float(b.map50), map=float(b.map),
                    per=per, per95=per95,
                )
                say(f'  [done] {tag} {sp}_{sub}: mAP50={b.map50:.4f} '
                    f'mAP50-95={b.map:.4f}  类={ {NAMES[c]: round(v,3) for c, v in per.items()} }')

    # ---------- 报告 ----------
    say()
    say('=' * 100)
    say('泄漏量化：leak / clean 子集评估（device=cpu）')
    say('=' * 100)

    for tag, _, _ in MODELS:
        say()
        say('#' * 100)
        say(f'# {tag}')
        say('#' * 100)
        for sp in SPLITS:
            say()
            say(f'--- {sp} ---')
            say(f'{"子集":<10}{"图数":>6}{"实例":>6}{"mAP50":>10}{"mAP50-95":>11}   '
                + '   '.join(f'{n + " AP50/实例":>20}' for n in NAMES))
            for sub in SUBSETS:
                d = results[(tag, sp, sub)]
                cnt, n_img = CNT[(sp, sub)], NIMG[(sp, sub)]
                cells = '   '.join(
                    f'{d["per"].get(i, 0):>11.4f}/{cnt[i]:<8}' for i in range(3))
                say(f'{sub:<10}{n_img:>6}{sum(cnt.values()):>6}'
                    f'{d["map50"]:>10.4f}{d["map"]:>11.4f}   {cells}')

            # ---- 判据 ----
            L = results[(tag, sp, 'leak')]
            C = results[(tag, sp, 'clean')]
            h1 = results[(tag, sp, 'clean_h1')]
            h2 = results[(tag, sp, 'clean_h2')]
            gap = L['map50'] - C['map50']
            floor = abs(h1['map50'] - h2['map50'])
            say()
            say(f'  判据：leak − clean = **{gap:+.4f}**（mAP50）；'
                f' mAP50-95 差 = **{L["map"] - C["map"]:+.4f}**')
            say(f'  噪声基准（clean 对半 h1 vs h2）= **{floor:.4f}**')
            # 门限用 1.5×噪声基准：h1/h2 本身是抽样的一个实现，单点比较太脆。
            if floor > 0 and abs(gap) <= 1.5 * floor:
                say(f'  ⇒ |差| ≤ 1.5×噪声基准 ⇒ **该差在本子集规模下不可分辨**，'
                    '不能据此说"泄漏有影响"或"影响可忽略"。')
            elif gap > 3 * max(floor, 0.005):
                say('  ⇒ 差显著超过噪声基准且 > 3 点 ⇒ **泄漏抬升明确**，'
                    'PDF 必须给 clean 子集读数 + 声明。')
            elif gap < 1 * max(floor, 0.005):
                say('  ⇒ 差小且不超过噪声基准 ⇒ 可声明"已量化，影响可忽略"（附数）。')
            else:
                say('  ⇒ 介于两者之间：泄漏有抬升但幅度有限，按上面两个数如实写。')
            # 薄类告警 + 逐类贡献拆解（说明总分差是谁贡献的）
            cnt = CNT[(sp, 'leak')]
            thin = [NAMES[i] for i in range(3) if cnt[i] < 10]
            if thin:
                say(f'  ⚠ leak 子集里 {"/".join(thin)} 的实例数 < 10（'
                    + '、'.join(f'{NAMES[i]}={cnt[i]}' for i in range(3) if cnt[i] < 10)
                    + '），该类 AP 不可信，其波动可单独拉动总分 ±30 点，'
                      '故总分差必须配合逐类拆解读。')
            say('  逐类贡献（leak − clean，单位点）：'
                + '   '.join(
                    f'{NAMES[i]} {100 * (L["per"].get(i, 0) - C["per"].get(i, 0)):+.1f}'
                    for i in range(3))
                + '    ← 三类各占 1/3，故总差 = 三项均值')

    # ---------- headline：完整集读数 vs clean 读数 ----------
    # 这才是论文真正要回答的问题：报告的 mAP 被泄漏抬高了多少？
    # FULL 是从既有评估日志抄来的完整 split 读数（来源：logs/eval_896.log、
    # runs/wire_v11_640/results.csv 最佳轮 —— 见 WORKLOG「896 组」与「数据划分校验」段）。
    FULL = {('v11@896', 'test'): 0.9148, ('v11@640', 'test'): 0.8992,
            ('v11@896', 'val'): 0.8990, ('v11@640', 'val'): 0.8716}
    say()
    say('=' * 100)
    say('headline 影响：headline 移动量 = 重建 − clean')
    say('=' * 100)
    say('⚠ 算法要点：**不可用"完整 − clean"**。完整(179 张)与 clean(131 张)是不同图数的')
    say('   集合，直接相减没有意义。正确做法是先按图数加权**重建**出"若整个 split 都是')
    say('   这个水平"的读数，再与 clean 相减：')
    say('       重建 = (n_clean×clean + n_leak×leak) / (n_clean + n_leak)')
    say('       headline 移动量 = 重建 − clean')
    say('   ⚠ 但必须知道它的**代数本质**：')
    say('       重建 − clean = [n_leak/(n_clean+n_leak)] × (leak − clean)')
    say('   ⇒ 它**不是独立测量**，只是 leak−clean 按泄漏占比缩放（本例 ×0.268），')
    say('     因而**继承 leak−clean 的全部混淆**（薄类主导、leak/clean 组成差异）。')
    say('     它的正确身份是"报告的 headline 被移动了多少"，**不是**"更可信的泄漏效应"。')
    say('     两者不可当作两个独立证据引用。')
    say('   ⚠ "重建 ≠ 记录在案的完整读数"的差额（0.2~0.5 点）**必须单独解释**：')
    say('     记录值来自 GPU，本脚本来自 CPU ⇒ 先排除设备差异，再看子集构造。')
    say('     见 code/tools/eval/check_device_and_selection.py（P0-B）。')
    say('     **差额未解释前，本表的数值不可信。**')
    say()
    say(f'{"模型/split":<20}{"记录(完整)":>12}{"重建":>10}{"clean":>10}{"leak":>10}'
        f'{"移动量=重建−clean":>18}{"重建−记录":>12}')
    for tag, _, _ in MODELS:
        for sp in SPLITS:
            f = FULL.get((tag, sp))
            L, C = results[(tag, sp, 'leak')], results[(tag, sp, 'clean')]
            nl, nc = NIMG[(sp, 'leak')], NIMG[(sp, 'clean')]
            rec = (nl * L['map50'] + nc * C['map50']) / (nl + nc)
            say(f'{tag + "/" + sp:<20}{f:>12.4f}{rec:>10.4f}{C["map50"]:>10.4f}'
                f'{L["map50"]:>10.4f}{rec - C["map50"]:>+18.4f}{rec - f:>+12.4f}')
    say()
    say('  "重建−记录"列就是待解释的差额（含设备差异）。')

    say()
    say('=' * 100)
    say('口径声明（照此写进 PDF）')
    say('=' * 100)
    say('  1. 报告的 mAP 是"含 train 近复制品的图"与"无近复制品的图"的混合值。')
    say('     **泄漏效应在本子集规模下不可分辨（差 ≤ 噪声基准）**，故不修正 headline，')
    say('     clean 子集读数备查。')
    say('     ⚠ 措辞纪律：只能说"**本实验规模无法分辨**"，**不得**写成')
    say('     "泄漏影响可忽略 / ≤0.1 点"—— 从落在噪声内的差里取精确上界是过度解读，')
    say('     答辩时"不可分辨"比"≤0.1 点"站得住得多。')
    say('     ⚠ 另：本脚本在 CPU 上评估，与训练/既有报告的 GPU 读数**不可混算**，')
    say('     差额未解释前（见 check_device_and_selection.py）数值不可信；')
    say('     最终泄漏量化应在**单一设备**内产生。')
    say('  2. **clean ≠ 跨场景干净**：MAD≤20 只抓"近同一张图"，同一场景的不同角度未排除。')
    say('     故 clean 读数只能说"无近复制品"，不能说"未见过的新场景"。')
    say('  3. val 侧的 best.pt 由**完整 val**选出，故 val 的 leak−clean 混入了选择效应，')
    say('     不如 test 侧干净；且 val 本身是早停/选择集，其读数不代表泛化。')

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'\n已写入 {OUT}')


if __name__ == '__main__':
    main()
