"""测试 main.py 的检测功能：加载模型 -> 打开图片 -> 截图。

用法: python detect_test.py <输出png> [候选图片数]

不修改 main.py。做法是把 QFileDialog.getOpenFileName patch 成返回固定路径，
再调用按钮的 .click()（真实触发 clicked 信号 -> 槽函数），走完整 UI 流程。

注意 import 顺序：必须先 import main（它内部先 import ultralytics/torch 再 import PyQt5），
否则先加载 Qt DLL 会导致 torch 的 c10.dll 加载失败（WinError 1114）。
"""

import os
import sys

MAIN_DIR = r"D:\2\ultralytics_wire_main"
WEIGHTS = os.path.join(MAIN_DIR, "weights", "best.pt")
IMG_DIR = r"D:\2\dataset\images\train"


def main():
    out_png = os.path.abspath(sys.argv[1])
    n_try = int(sys.argv[2]) if len(sys.argv) > 2 else 6

    os.chdir(MAIN_DIR)
    sys.path.insert(0, MAIN_DIR)

    print("[1] cwd =", os.getcwd(), flush=True)
    import main as app_mod  # 先于任何 PyQt5 直接 import，避开 DLL 顺序坑

    print("[2] main 模块导入完成（ultralytics + PyQt5 均已加载）", flush=True)

    app = app_mod.QApplication(sys.argv)
    window = app_mod.DefectMonitorSystem()
    window.show()
    app.processEvents()
    print("[3] 窗口已显示", flush=True)

    # ---- 步骤 2：模拟点击「加载模型」 ----
    app_mod.QFileDialog.getOpenFileName = staticmethod(
        lambda *a, **k: (WEIGHTS, "PyTorch Model (*.pt)")
    )
    window.btn_load_model.click()          # 真实触发 clicked 信号
    app.processEvents()
    print("[4] 已点击「加载模型」", flush=True)
    print("    模型标签:", window.model_label.text(), flush=True)
    print("    model 对象:", type(window.model).__name__ if window.model else None, flush=True)
    if window.model is None:
        print("FATAL 模型未加载成功", flush=True)
        return 1
    print("    类别表:", window.model.names, flush=True)

    # ---- 步骤 3：模拟点击「打开图片」 ----
    cands = sorted(f for f in os.listdir(IMG_DIR) if f.lower().endswith(".jpg"))[:n_try]
    app_mod.QFileDialog.getOpenFileName = staticmethod(
        lambda *a, **k: (os.path.join(IMG_DIR, k_img[0]), "Image Files (*.png *.jpg *.jpeg *.bmp)")
    )

    best = None
    for name in cands:
        k_img = [name]
        window.btn_image.click()           # 真实触发 clicked 信号
        app.processEvents()
        n = window.defect_list.count()
        items = [window.defect_list.item(i).text() for i in range(n)]
        print(f"[5] {name}  ->  {n} 个检测结果  {items}", flush=True)
        if best is None or n > best[1]:
            best = (name, n, items)
        if n > 0:
            break

    # 用检测结果最好的那张重跑一次，作为最终截图内容
    if best and best[1] > 0:
        k_img = [best[0]]
        window.btn_image.click()
        app.processEvents()
        print(f"[6] 最终截图采用: {best[0]}", flush=True)
    else:
        print("[6] 候选图均无检测结果，截图为最后一张", flush=True)

    print("[7] 报警灯文字:", window.alarm_label.text(), flush=True)
    print("[8] 图像区是否有 pixmap:", bool(window.image_label.pixmap()), flush=True)

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    pixmap = window.grab()
    ok = pixmap.save(out_png)
    print(f"GRAB_OK={ok} {out_png} {pixmap.width()}x{pixmap.height()}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
