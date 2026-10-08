"""把 X-AnyLabeling .json 里的 "bench" 标签批量替换为 "break"。

用法:
    python replace_bench.py [目标目录]

默认目录: D:\\2\\ultralytics_wire_main\\datasets\\wire_data_yolo\\images\\train

只替换 shapes[].label 字段，其他字段原样保留。
"""

import json
import os
import sys
import glob

OLD_LABEL = "bench"
NEW_LABEL = "break"

DEFAULT_DIR = r"D:\2\ultralytics_wire_main\datasets\wire_data_yolo\images\train"


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DIR
    target = os.path.abspath(target)

    if not os.path.isdir(target):
        print(f"ERROR: directory not found: {target}")
        sys.exit(1)

    json_files = sorted(glob.glob(os.path.join(target, "*.json")))
    print(f"Target dir  : {target}")
    print(f"JSON found  : {len(json_files)}")
    print("-" * 60)

    total_replaced = 0
    files_changed = 0
    errors = []

    for jp in json_files:
        try:
            with open(jp, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as e:
            errors.append((os.path.basename(jp), f"read: {e}"))
            continue

        n = 0
        for shape in data.get("shapes", []):
            if shape.get("label") == OLD_LABEL:
                shape["label"] = NEW_LABEL
                n += 1

        if n:
            try:
                with open(jp, "w", encoding="utf-8", newline="\n") as fh:
                    json.dump(data, fh, ensure_ascii=False, indent=2)
                    fh.write("\n")
            except Exception as e:
                errors.append((os.path.basename(jp), f"write: {e}"))
                continue
            total_replaced += n
            files_changed += 1

    print(f'Replaced "{OLD_LABEL}" -> "{NEW_LABEL}"')
    print(f"Files changed   : {files_changed}")
    print(f"Labels replaced : {total_replaced}")
    if errors:
        print("-" * 60)
        print(f"ERRORS ({len(errors)}):")
        for nm, e in errors:
            print(f"  {nm}: {e}")


if __name__ == "__main__":
    main()
