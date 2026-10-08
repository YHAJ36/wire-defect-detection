"""启动例程 -> 触发演示模式 -> 把主窗口自身渲染成 PNG。

用法: python run_and_grab.py <例程目录> <输出png> [演示模式秒数]

用 widget.grab() 直接渲染控件树，不经过窗口管理器，
因此不受窗口遮挡影响。不修改例程目录里的任何文件。
"""

import importlib
import os
import sys


def main():
    demo_dir = os.path.abspath(sys.argv[1])
    out_png = os.path.abspath(sys.argv[2])
    wait_ms = int(sys.argv[3]) * 1000 if len(sys.argv) > 3 else 8000

    # 例程用相对路径读 playdata.txt，所以工作目录必须切到例程目录；
    # import 走的是脚本所在目录，故手动把例程目录加进 sys.path。
    print("[1] 参数解析完成", flush=True)
    os.chdir(demo_dir)
    sys.path.insert(0, demo_dir)
    print("[2] cwd =", os.getcwd(), flush=True)

    from PyQt5.Qt import QApplication, QTimer
    print("[3] PyQt5 导入完成", flush=True)

    # PyQt5 里槽函数抛异常会直接 qFatal 终止进程；先用 excepthook 把异常落盘。
    def hook(exc_type, exc, tb):
        import traceback

        with open(os.path.join(os.path.dirname(out_png), "crash.log"), "a", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            traceback.print_exception(exc_type, exc, tb, file=f)
        sys.__excepthook__(exc_type, exc, tb)
        os._exit(9)

    sys.excepthook = hook

    pm_module = importlib.import_module("ParamMonitor")
    ParamMonitor = pm_module.ParamMonitor
    print("[4] ParamMonitor 导入完成", flush=True)

    # 教程代码里 drawRespWave/drawSPO2Wave/drawECG*Wave 把浮点结果直接传给 QPoint，
    # PyQt5 只接受 int，异常从定时器槽里逃出会触发 qFatal 直接杀进程。
    # 这里在自己进程内把模块的 QPoint 全局名换成会取整的版本 —— 不改动教程源文件。
    if os.environ.get("PATCH_QPOINT"):
        _RealQPoint = pm_module.QPoint

        def _qpoint_int(*args):
            if len(args) == 2:
                return _RealQPoint(int(args[0]), int(args[1]))
            return _RealQPoint(*args)

        pm_module.QPoint = _qpoint_int
        print("[4b] 已把 QPoint 替换为取整版本（仅本进程内存中）", flush=True)

    app = QApplication(sys.argv)
    print("[5] QApplication 创建完成", flush=True)
    window = ParamMonitor()
    print("[6] 主窗口构造完成", flush=True)
    window.show()
    print("[7] 窗口已显示", flush=True)

    play_file = os.path.join(demo_dir, "playdata.txt")

    def start_demo():
        print("[8] 进入演示模式", flush=True)
        # 绕过文件选择对话框，直接置标志位并载入演示数据
        window.mPlayFlag = True
        window.slot_playData(play_file)
        print("[9] 演示数据载入完成", flush=True)

    if os.environ.get("NO_DEMO"):
        start_demo = lambda: print("[8] 跳过演示模式", flush=True)

    def grab():
        print("[10] 开始抓图", flush=True)
        pixmap = window.grab()
        pixmap.save(out_png)
        print(f"GRAB_OK {out_png} {pixmap.width()}x{pixmap.height()}", flush=True)
        app.quit()

    QTimer.singleShot(1000, start_demo)
    QTimer.singleShot(wait_ms, grab)
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
