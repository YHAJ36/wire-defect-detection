"""从 ModelScope 拉取 SHWD（安全帽检测训练数据集）。

只负责把数据落到本地缓存，不做任何处理。
下载完成后缓存位置：
    C:/Users/LR/.cache/modelscope/hub/datasets/modelscope/safetyhelmet_detection_for_train/master/
"""
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from modelscope.msdatasets import MsDataset

for split in ('train', 'validation'):
    try:
        ds = MsDataset.load(
            'safetyhelmet_detection_for_train',
            namespace='modelscope',
            split=split,
        )
        print('OK split=%s len=%d' % (split, len(ds)), flush=True)
        print('  features:', list(ds.features.keys()) if hasattr(ds, 'features') else '?', flush=True)
    except Exception as e:
        print('FAIL split=%s: %s %s' % (split, type(e).__name__, str(e)[:300]), flush=True)

print('DONE', flush=True)
