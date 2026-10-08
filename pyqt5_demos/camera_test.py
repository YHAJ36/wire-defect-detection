"""测试 main.py 的「开启摄像头」检测：加载模型 -> 点「开启摄像头」 -> 截图。

用法: python camera_test.py <输出png> [等待秒数]

不修改 main.py。用 .click() 真实触发按钮。注意先 import main（DLL 顺序）。
"""

import os
import sys

MAIN_DIR = r"D:\2\ultralytics_wire_main"
WEIGHTS = os.path.join(MAIN_DIR, "weights", "best.pt")


def main():
    out_png = os.path.abspath(sys.argv[1])
    wait_s = int(sys.argv[2]) if len(sys.argv) > 2 else 12

    os.chdir(MAIN_DIR)
    sys.path.insert(0, MAIN_DIR)

    import main as app_mod  # 必须先于直接 import PyQt5

    app = app_mod.QApplication(sys.argv)
    window = app_mod.DefectMonitorSystem()
    window.show()

    # 1) 加载模型
    app_mod.QFileDialog.getOpenFileName = staticmethod(
        lambda *a, **k: (WEIGHTS, "PyTorch Model (*.pt)")
    )
    window.btn_load_model.click()
    print("[1] 模型标签:", window.model_label.text(), flush=True)
    if window.model is None:
        print("FATAL 模型未加载", flush=True)
        return 1

    # 2) 点「开启摄像头」——线程会异步取帧，必须跑真事件循环
    window.btn_camera.click()
    print("[2] 已点击「开启摄像头」，线程对象:", window.video_thread, flush=True)

    def probe():
        t = window.video_thread
        if t is not None:
            print(f"    线程 isRunning={t.isRunning()} cap={t.cap} "
                  f"capOpened={t.cap.isOpened() if t.cap else None}", flush=True)
        print("    图像区有 pixmap:", bool(window.image_label.pixmap()), flush=True)

    app_mod.QTimer.singleShot(3000, probe)

    def grab():
        t = window.video_thread
        if t is not None:
            print(f"[3] 停止前线程 isRunning={t.isRunning()}", flush=True)
        print("[4] 报警灯:", window.alarm_label.text(), flush=True)
        n = window.defect_list.count()
        items = [window.defect_list.item(i).text() for i in range(n)]
        print(f"[5] 缺陷列表 {n} 条: {items}", flush=True)
        print("[6] 图像区有 pixmap:", bool(window.image_label.pixmap()), flush=True)

        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        pixmap = window.grab()
        ok = pixmap.save(out_png)
        print(f"GRAB_OK={ok} {out_png} {pixmap.width()}x{pixmap.height()}", flush=True)
        window.stop_detection()
        app.quit()

    app_mod.QTimer.singleShot(wait_s * 1000, grab)
    sys.exit(app.exec_())


if __name__ == "__main__":
    sys.exit(main())
