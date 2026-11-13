# -*- coding: utf-8 -*-
"""Uniform weight averaging (model soup) over checkpoints that share one init.

Only float tensors are averaged; integer buffers (num_batches_tracked) are taken
from the first checkpoint.  Averaging is done in float32 and written back in the
dtype of the anchor checkpoint (ultralytics strips loaded weights to fp16).

Usage:
    python code/tools/fuse/make_soup.py \
        --ckpts exp_upscale/runs/nf_v11_896_s42/weights/best.pt \
                exp_upscale/runs/nf_v11_896_s43/weights/best.pt \
        --out report/_phase1_probe/soup_s42_s43.pt
"""
import argparse
import os
import sys

import torch

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(BASE, 'code', 'yolov11'))


def load_ckpt(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpts', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--weights', nargs='+', type=float, default=None,
                    help='per-checkpoint weights, default uniform')
    args = ap.parse_args()

    n = len(args.ckpts)
    w = args.weights if args.weights else [1.0 / n] * n
    assert len(w) == n, 'weights length mismatch'
    tot = sum(w)
    w = [x / tot for x in w]

    anchor = load_ckpt(args.ckpts[0])
    sd0 = anchor['model'].state_dict()

    acc = {}
    ref_dtype = {}
    for path, wi in zip(args.ckpts, w):
        sd = load_ckpt(path)['model'].state_dict()
        if set(sd.keys()) != set(sd0.keys()):
            raise RuntimeError('state_dict keys differ: %s' % path)
        for k, t in sd.items():
            if not torch.is_floating_point(t):
                continue
            ref_dtype[k] = t.dtype
            v = t.detach().to(torch.float32) * wi
            acc[k] = acc[k] + v if k in acc else v
        print('  merged %-70s w=%.4f' % (path, wi))

    merged = {}
    for k, v in acc.items():
        merged[k] = v.to(ref_dtype[k])
    for k, t in sd0.items():
        if not torch.is_floating_point(t):
            merged[k] = t.clone()

    missing, unexpected = anchor['model'].load_state_dict(merged, strict=True)
    print('load_state_dict ->', missing, unexpected)

    # record provenance so the .pt is self-describing
    anchor['soup'] = {'ckpts': [os.path.basename(os.path.dirname(os.path.dirname(p))) for p in args.ckpts],
                      'weights': w}
    anchor['epoch'] = -1
    anchor['best_fitness'] = None
    anchor['optimizer'] = None
    anchor['ema'] = None

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(anchor, args.out)
    sz = os.path.getsize(args.out) / 1e6
    print('saved %s (%.1f MB)' % (args.out, sz))


if __name__ == '__main__':
    main()
