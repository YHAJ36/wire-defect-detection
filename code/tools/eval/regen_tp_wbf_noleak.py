# -*- coding: utf-8 -*-
"""给 WBF 融合结果也存一份「框架自己的 tp」，供 bootstrap 做配对比较。

为什么必须配对：单侧 CI 宽 10.4 pt，但 WBF 与单模型跑在**同一批 203 张图**上，
配对差把「这批图整体偏难/偏易」这个共同项抵掉，CI 会窄得多。
只有配对差的 CI 才能回答「+3.09 pt 是不是真的」。

复用 eval_wbf_tta.py 的 provider 机制（保证融合语义与 eval_noleak_final.tsv 一致），
只把它的 `_ORIG_UPDATE` 换成「先记 tp、再交回原函数」，这样 tp 是框架算的，
不是自己实现的 IoU 匹配。

用法：D:/yolo-env/Scripts/python.exe code/tools/eval/regen_tp_wbf_noleak.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, 'D:/2/code/tools/eval')
for _d in ('D:/2/code/yolov11', 'D:/2/code/yolov12'):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import eval_wbf_tta as W  # noqa: E402  它的顶层代码会注册 update_metrics 补丁
from ultralytics.models.yolo.detect.val import DetectionValidator  # noqa: E402

ROOT = Path('D:/2')
CACHE = ROOT / 'exp_upscale/logs/wbf_cache_noleak'      # letterbox 归一化，provider 与之配套
OUT = ROOT / 'exp_upscale/logs/tp_cache_noleak'

SEEDS = ('s42', 's43', 's44')
PLAN = [
    ('wbf_base', ['nf_v11_noleak_' + s for s in SEEDS], 3),
    ('wbf_triplet', ['att_v11_noleak_triplet_' + s for s in SEEDS], 3),
    ('wbf_mix6', ['nf_v11_noleak_' + s for s in SEEDS]
                 + ['att_v11_noleak_triplet_' + s for s in SEEDS], 6),
]
WBF_IOU = 0.55

_BASE = W._ORIG_UPDATE          # 真正的原始 update_metrics（已被 W 存下来）
# eval_wbf_tta 的默认 DATA 指向 dataset_noleak_clean（100 图 test），
# 这里是段级 203 图卷，必须显式覆盖，否则 provider 找不到缓存 key。
W.DATA = str(ROOT / 'dataset_noleak/data.yaml')
W.SPLIT = 'test'
W.IMGSZ = 896
HOOK = {'c': None}


def _after(self, preds, batch):
    """provider 已经把 preds 换成融合结果，这里记下框架的 tp 再交回原函数。"""
    if HOOK['c'] is not None:
        for si, p in enumerate(preds):
            stem = Path(batch['im_file'][si]).stem
            pbatch = self._prepare_batch(si, batch)
            predn = self._prepare_pred(p)
            tp = self._process_batch(predn, pbatch)['tp']
            HOOK['c'][stem] = (
                tp[:, 0].astype(np.float32) if getattr(tp, 'size', 0) else np.zeros(0, np.float32),
                predn['conf'].detach().cpu().numpy().astype(np.float32),
                predn['cls'].detach().cpu().numpy().astype(np.float32),
                pbatch['cls'].detach().cpu().numpy().astype(np.float32),
            )
    return _BASE(self, preds, batch)


W._ORIG_UPDATE = _after


def load_cache(path):
    z = np.load(path, allow_pickle=True)
    keys = [str(k) for k in z['__keys__']]
    return keys, {k: {'boxes': z[k][:, :4].astype(np.float32),
                      'conf': z[k][:, 4].astype(np.float32),
                      'cls': z[k][:, 5].astype(np.int64)} for k in keys}


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    OUT.mkdir(parents=True, exist_ok=True)
    for tag, runs, nmodels in PLAN:
        caches, stem_ref = [], None
        for r in runs:
            keys, d = load_cache(CACHE / f'{r}_raw.npz')
            if stem_ref is None:
                stem_ref = sorted(keys)
            elif sorted(keys) != stem_ref:
                raise SystemExit(f'{r} 图列表不一致')
            caches.append(d)
        provider = W.make_provider(caches, WBF_IOU, nmodels)
        HOOK['c'] = {}
        try:
            met = W.run_val(runs[0], augment=False, collect=None, provider=provider,
                            tag=f'tp_{tag}')
        finally:
            data = HOOK['c']
            HOOK['c'] = None
        stems = sorted(data)
        np.savez_compressed(
            OUT / f'{tag}.npz',
            __keys__=np.array(stems),
            tp50=np.array([data[s][0] for s in stems], dtype=object),
            dconf=np.array([data[s][1] for s in stems], dtype=object),
            pcls=np.array([data[s][2] for s in stems], dtype=object),
            gcls=np.array([data[s][3] for s in stems], dtype=object),
        )
        print(f'{tag:<14} n={len(data):<4} mAP50={met["mAP50"]:.4f} '
              f'mAP50-95={met["mAP50-95"]:.4f}')


if __name__ == '__main__':
    main()
