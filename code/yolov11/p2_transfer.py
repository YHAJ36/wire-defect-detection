# -*- coding: utf-8 -*-
"""把官方 yolo11s 预训练权重搬到 yolo11s-p2v2.yaml 的结构上。

为什么需要这个文件
------------------
给模型加一个检测尺度（P2）时，ultralytics 的权重加载（`BaseModel.load` ->
`intersect_dicts`）按「层号 + 分支号」逐字对齐，只要层号或分支号变了就整块丢弃。

实测（v1 配置，P2 支路插在中间）：
    直接 load 只迁移 297/593，检测头 161 个张量**迁移率为 0**。
    对照：同结构基线是 493/499，检测头 121 个里只差 6 个类别卷积（nc 不同）。

这就是第一组 P2 实验失效的直接原因：每颗种子都在重学一个完全随机的检测头，
既压低了均值，也把种子间的方差放大了约 100 倍（F=101，临界 19）。

本模块做的事
------------
`yolo11s-p2v2.yaml` 已经把 P2 支路追加到末尾（23-25 层），0-22 层与官方逐层一致，
所以只有检测头需要重排（23 -> 26，分支右移 1 位）。本模块补齐这部分：

    分支 1/2/3  <- 预训练分支 0/1/2   形状完全一致，直接搬
    dfl         <- 预训练 dfl         直接搬
    分支 0      <- 由分支 1 折叠初始化（输入 128 -> 64，相邻通道取均值）

为什么新增分支要折叠初始化而不是随机
------------------------------------
TAL 分配正样本时按「对齐度量」在所有层之间竞争。一个完全随机的 P2 分支打分很低，
永远争不过已经训练好的其它三层，于是分不到正样本、也就永远不更新。
用相邻尺度的分支做初值，能让它一开始就有可用的响应，避免这种自锁。

仍为随机初始化的部分（无法继承，属于新增结构）
----------------------------------------------
    第 23 层  nn.Upsample   无参数
    第 25 层  C3k2 [128]    P2 支路融合块
    检测头分支 0 的第二层起（若通道宽度与供体不同）

用法
----
只核对、不训练、不写文件：

    D:/yolo-env/Scripts/python.exe code/yolov11/p2_transfer.py
"""

import sys

import torch


def _layer_index(key):
    part = key.split('.')
    if len(part) < 3 or part[0] != 'model':
        return None
    try:
        return int(part[1])
    except ValueError:
        return None


def _branch_count(sd, layer, module):
    """数某个检测头模块有几个分支。"""
    prefix = 'model.%d.%s.' % (layer, module)
    return len({k[len(prefix):].split('.')[0] for k in sd if k.startswith(prefix)})


def _adapt(tensor, target_shape):
    """把张量调整成目标形状。

    优先同形直用；输入通道成整数倍则成对折叠；都比目标大则按重叠区裁剪。
    目标更大时无法凭空补出来，返回 None 交给随机初始化。
    """
    if tuple(tensor.shape) == tuple(target_shape):
        return tensor
    if tensor.dim() == 4 and len(target_shape) == 4 and target_shape[1] > 0:
        src_in, dst_in = tensor.shape[1], target_shape[1]
        if src_in % dst_in == 0 and tensor.shape[0] == target_shape[0]:
            ratio = src_in // dst_in
            folded = tensor.reshape(tensor.shape[0], dst_in, ratio, *tensor.shape[2:]).mean(dim=2)
            if tuple(folded.shape) == tuple(target_shape):
                return folded
    # 只沿输入通道裁剪：输出通道数不同意味着这一层语义已经变了
    # （典型是类别卷积 nc=80 -> nc=3），那种情况交给随机初始化，不要硬凑。
    if tensor.dim() == len(target_shape) and tensor.shape[0] == target_shape[0] and all(
            t <= s for t, s in zip(target_shape[1:], tensor.shape[1:])):
        return tensor[tuple([slice(None)] + [slice(0, t) for t in target_shape[1:]])].clone()
    return None


def _branch_in_channels(sd, layer, module, branch):
    """取某分支第一个「真卷积」的输入通道数，用来判断能否直接搬。

    注意：检测头分类分支的第一层是深度卷积（weight 形状 (C, 1, k, k)），
    它的 shape[1] 恒为 1，不能拿来当输入通道；要顺着往后找第一个
    输入通道大于 1 的卷积（即紧随其后的 1x1 混合卷积）。
    """
    prefix = 'model.%d.%s.%d.' % (layer, module, branch)
    for key in sorted(sd):
        if not (key.startswith(prefix) and key.endswith('.weight')):
            continue
        tensor = sd[key]
        if tensor.dim() != 4:
            continue
        if tensor.shape[1] > 1:
            return tensor.shape[1]
    return None


def build_plan(pre_sd, p2_sd):
    """返回 {目标键: 源键}，只处理检测头；其余层由 model.load 同名同形直装。

    供体选择按「输入通道相等」优先，而不是简单的序号偏移：
    新增的 P2 分支与 P3 分支输入通道相同，应当直接继承 P3 分支的权重，
    这样它一开始就具备可用的响应，在 TAL 里不会因为随机打分而争不到正样本。
    """
    pre_layers = sorted({i for i in map(_layer_index, pre_sd) if i is not None})
    p2_layers = sorted({i for i in map(_layer_index, p2_sd) if i is not None})
    src_det, dst_det = pre_layers[-1], p2_layers[-1]

    n_src = _branch_count(pre_sd, src_det, 'cv2')
    n_dst = _branch_count(p2_sd, dst_det, 'cv2')
    shift = n_dst - n_src

    plan, mapping = {}, {}
    for module in ('cv2', 'cv3'):
        src_in = [_branch_in_channels(pre_sd, src_det, module, j) for j in range(n_src)]
        for j in range(n_dst):
            dst_c = _branch_in_channels(p2_sd, dst_det, module, j)

            # 1) 同名序号且输入通道一致 -> 直接对应
            donor = j if (j < n_src and src_in[j] == dst_c) else None
            # 2) 否则取第一个输入通道一致的分支（新增尺度通常落到这一条）
            if donor is None:
                same = [i for i in range(n_src) if src_in[i] == dst_c]
                donor = same[0] if same else None
            # 3) 仍然找不到，退回按序号右移，交给 _adapt 折叠或裁剪
            if donor is None:
                donor = max(0, min(n_src - 1, j - shift))

            mapping['%s.%d' % (module, j)] = donor
            src_pat = 'model.%d.%s.%d.' % (src_det, module, donor)
            dst_pat = 'model.%d.%s.%d.' % (dst_det, module, j)
            for key in pre_sd:
                if key.startswith(src_pat):
                    plan[key.replace(src_pat, dst_pat, 1)] = key

    dfl_pat = 'model.%d.dfl.' % src_det
    for key in pre_sd:
        if key.startswith(dfl_pat):
            plan[key.replace(dfl_pat, 'model.%d.dfl.' % dst_det, 1)] = key

    return plan, (src_det, dst_det, n_src, n_dst, mapping)


def transfer(model, ckpt, verbose=True):
    """把 ckpt 权重搬到 model 上；model 应已先做过一次 model.load(ckpt)。"""
    src_model = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    pre_sd = src_model.state_dict()
    p2_sd = model.state_dict()
    plan, info = build_plan(pre_sd, p2_sd)

    new_sd, n_copy, n_fold, n_skip = {}, 0, 0, 0
    for dst_key, src_key in plan.items():
        if dst_key not in p2_sd:
            n_skip += 1
            continue
        adapted = _adapt(pre_sd[src_key], p2_sd[dst_key].shape)
        if adapted is None:
            n_skip += 1
            continue
        new_sd[dst_key] = adapted.clone()
        if tuple(adapted.shape) == tuple(pre_sd[src_key].shape):
            n_copy += 1
        else:
            n_fold += 1

    model.load_state_dict(new_sd, strict=False)

    covered = set()
    for key, val in p2_sd.items():
        if key in pre_sd and tuple(pre_sd[key].shape) == tuple(val.shape):
            covered.add(key)
    covered |= set(new_sd)

    random_layers = {}
    for key in p2_sd:
        layer = _layer_index(key)
        if layer is None or key in covered:
            continue
        random_layers[layer] = random_layers.get(layer, 0) + 1

    report = {
        'total': len(p2_sd),
        'loaded': len(covered),
        'copied': n_copy,
        'folded': n_fold,
        'skipped': n_skip,
        'detect_shift': (info[0], info[1]),
        'branch_map': info[4],
        'random_layers': dict(sorted(random_layers.items())),
    }
    if verbose:
        print('[p2_transfer] 检测头 %d -> %d 层，分支 %d -> %d 个' % (
            info[0], info[1], info[2], info[3]))
        for mod in ('cv2', 'cv3'):
            pairs = ['%d<-%d' % (j, info[4]['%s.%d' % (mod, j)]) for j in range(info[3])]
            print('[p2_transfer]   %s 分支映射 %s' % (mod, ' '.join(pairs)))
        print('[p2_transfer] %d/%d 张量已继承（直搬 %d，折叠 %d，跳过 %d）' % (
            report['loaded'], report['total'], n_copy, n_fold, n_skip))
        if random_layers:
            detail = ', '.join('第%d层 %d个' % (k, v) for k, v in sorted(random_layers.items()))
            print('[p2_transfer] 仍为随机初始化：%s' % detail)
        else:
            print('[p2_transfer] 全部继承')
    return report


def _main():
    """只核对：建模型、搬权重、打印覆盖率。不训练、不写文件。"""
    sys.path.insert(0, 'D:/2/code/yolov11')
    from ultralytics import YOLO
    from ultralytics.nn.tasks import attempt_load_one_weight

    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else 'D:/1/models/yolo11s.pt'
    cfg = sys.argv[2] if len(sys.argv) > 2 else 'D:/2/code/yolov11/yolo11s-p2v2.yaml'

    model = YOLO(cfg).model
    ckpt, _ = attempt_load_one_weight(ckpt_path)
    ck_sd = ckpt.state_dict()

    plain = YOLO(cfg).model
    plain.load(ckpt)
    plain_sd = plain.state_dict()
    base_n = sum(1 for k, v in plain_sd.items()
                 if k in ck_sd and tuple(v.shape) == tuple(ck_sd[k].shape))

    print('配置：%s' % cfg)
    print('权重：%s' % ckpt_path)
    print('旧做法直接 load：%d/%d' % (base_n, len(plain_sd)))
    model.load(ckpt)
    rep = transfer(model, ckpt)
    print('补搬后：%d/%d，多拿到 %d 个张量' % (rep['loaded'], rep['total'], rep['loaded'] - base_n))

    det = rep['detect_shift'][1]
    head_keys = [k for k in model.state_dict() if _layer_index(k) == det]
    nonzero = sum(1 for k in head_keys if model.state_dict()[k].abs().sum() > 0)
    print('检测头 %d 个张量，非零 %d 个' % (len(head_keys), nonzero))


if __name__ == '__main__':
    _main()
