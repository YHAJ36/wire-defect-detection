# -*- coding: utf-8 -*-
"""ultralytics settings.json 前置检查 / 修复。

背景
   本机 ultralytics settings.json 会被不定期重置（2026-10-10 当天实测 3 次，无外因：
   未联网、未升级包、未改配置）。重置后 weights_dir 从 D:/1/models 掉回相对路径 "weights"，
   datasets_dir 变成 cwd 派生值（如 D:\\2\\code\\yolov11\\datasets）。
   本项目脚本一律用绝对路径传 weights/data/project，所以重置不会污染训练或评估结果；
   但一旦 ultralytics 需要按 weights_dir 去找权重，就会尝试联网下载 —— GitHub 直连不通，
   会静默卡住或报错，且报错信息与"配置被重置"毫无关联，排查成本很高。

用法
    run_all_*.sh 开头已前置调用；手动跑训练前也建议先执行一次。
        D:/yolo-env/Scripts/python.exe D:/2/scripts/fix_settings.py

输出
    一行：`settings OK`（无需修复）/ `settings fixed ...`（已修复）
退出码
    0 正常（含已修复）；1 修复失败

设计说明
    先用 json 直接读文件判断，**不 import ultralytics** —— 因为 import ultralytics 本身
    就可能触发这次重置，用它来"检查"等于先制造问题再修复。只有确实需要写回时才 import
    并用 settings.update()（官方写入口，会顺带补全缺失键与版本号）。
"""
import json
import os
import sys
from pathlib import Path

EXPECT = {
    'weights_dir': 'D:/1/models',
    'datasets_dir': 'D:/2',
}


def settings_path() -> Path:
    appdata = os.environ.get('APPDATA')
    base = Path(appdata) if appdata else Path.home() / 'AppData' / 'Roaming'
    return base / 'Ultralytics' / 'settings.json'


def read_raw(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return {}


def main() -> int:
    path = settings_path()
    cur = read_raw(path)

    wrong = {k: cur.get(k) for k, v in EXPECT.items() if cur.get(k) != v}
    if not wrong:
        print('settings OK')
        return 0

    try:
        from ultralytics import settings as ul_settings
        ul_settings.update(dict(EXPECT))
    except Exception as e:
        print(f'settings fixed 失败：{type(e).__name__}: {e}')
        return 1

    still = [k for k, v in EXPECT.items() if read_raw(path).get(k) != v]
    if still:
        print(f'settings fixed 失败，写回后仍不符：{still}')
        return 1

    detail = ', '.join(f'{k}: {wrong[k]!r} -> {v!r}' for k, v in EXPECT.items() if k in wrong)
    print(f'settings fixed  ({detail})')
    return 0


if __name__ == '__main__':
    sys.exit(main())
