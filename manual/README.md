# 手动转换 K230 模型：四步

目标：自己把 `best.pt` 走成 `kmodel`，不依赖之前的脚本。
每一步只干一件事，跑完必须能回答"这一步看什么数字才算过"。

先记两条环境前提（Windows 特有，网上教程基本不写）：

```powershell
$env:DOTNET_ROLL_FORWARD="LatestMajor"     # 每次开新窗口都要设，否则 import nncase 直接失败
& D:/yolo-env/Scripts/python.exe <脚本>     # 必须用 yolo-env，系统 Python 里没有 ultralytics/nncase
```

---

## 第 1 步：让 nncase 能 import

```powershell
$env:DOTNET_ROLL_FORWARD="LatestMajor"; & D:/yolo-env/Scripts/python.exe -c "import nncase; print('ok')"
```

看什么：打印 `ok`。
不设那个变量会报 `Failed to initialize hostfxr: 0x80008096`，因为 nncase 的编译器主体是 C#，
它要 .NET 7.0，而机器上只有 6 和 8，.NET 默认不肯向上兼容。

---

## 第 2 步：导出 ONNX —— `01_export_onnx.py`

```powershell
$env:DOTNET_ROLL_FORWARD="LatestMajor"; & D:/yolo-env/Scripts/python.exe D:/2/manual/01_export_onnx.py
```

看什么：`输入 [1,3,640,640]`、`输出 [1,7,8400]`、大小约 10 MB。
8400 = 80² + 40² + 20²，三个尺度的格子数；7 = 4 个框参数 + 3 个类别分数。

官方写法 `YOLO.export(format='onnx')` 在本环境会失败（缺 `onnxscript`），
所以脚本里直接调 `torch.onnx.export(..., dynamo=False)`。

---

## 第 3 步：检查 ONNX 对不对 —— `02_check_onnx.py`

```powershell
$env:DOTNET_ROLL_FORWARD="LatestMajor"; & D:/yolo-env/Scripts/python.exe D:/2/manual/02_check_onnx.py
```

看什么：`两个模型的输出最大差异 0.000000`。
形状对不代表数值对——导出时漏一步，形状照样对，但框全歪。所以要和一份验证过的旧 ONNX 对数值。

---

## 第 4 步：编译 kmodel —— `03_compile_kmodel.py`

```powershell
$env:DOTNET_ROLL_FORWARD="LatestMajor"; & D:/yolo-env/Scripts/python.exe D:/2/manual/03_compile_kmodel.py
```

看什么：`compile OK` + `kmodel: ... 3.42 MB`，耗时约 90 秒。

**这一步为什么需要 50 张训练图**：权重本来是 float32，压到 int8 只有 256 个格，
必须知道"每一层数值大概在什么范围"才能决定 256 个格怎么铺。范围不是猜的，
是喂几十张真实图片统计出来的，叫校准（calibration）。标尺定歪了模型照样跑，但分数整体偏低，且不报错。

**校准方法选 NoClip 不选 Kld**：Kld 会裁长尾求整体最优，代价是把 YOLO 的高分框一起削平，
实测分数封顶在 0.544；NoClip 不舍弃高分，检测任务要的就是高分框。

---

## 第 5 步：验证 kmodel —— `04_verify_kmodel.py`

```powershell
$env:DOTNET_ROLL_FORWARD="LatestMajor"; & D:/yolo-env/Scripts/python.exe D:/2/manual/04_verify_kmodel.py
```

看什么：最高分那个框的**类别一致**、**中心差 1~2 px**、**分数比接近 1**。

不能直接上板试，因为量化误差不会报错。板上框歪了，你分不清是量化掉的还是解码写错了。
先用 nncase 模拟器跑同一个 kmodel 和 ONNX 对，误差就被隔离成"纯量化误差"。

### 2026-11-07 实测结果（本目录产物）

| 比什么 | float 输入 + NoClip（本目录） | 之前上板的 uint8 图内归一化版 |
|---|---|---|
| top1 类别 | thunderbolt = 正确 | 正确 |
| top1 中心偏差 | 1.91 / 1.01 px | 0.7 / 0.3 px |
| top1 分数比 | **0.990** | 0.927 |
| 框参数误差中位 | 1.24 px | — |

结论：这条路分数几乎不丢（0.99×），代价是板上要自己把 uint8 帧转成 float32 再喂，
多一层 Python 逐像素开销（每帧几十 ms）。省帧率还是要 uint8 那条，省分数是这条。

---

## 产物

| 文件 | 大小 | 说明 |
|---|---|---|
| `yolo11n_640.onnx` | 10.4 MB | 第 2 步产物 |
| `yolo11n_640_manual.kmodel` | 3.42 MB | 第 4 步产物 |

## 第 6 步：传到板子上，用板子跑一遍

```powershell
# 传模型 + 测试输入（目标路径可省，默认放 /sdcard/ 同名）
& C:/Users/LR/AppData/Local/Programs/Python/Python312/python.exe D:/2/deploy/upload_kmodel.py --file "D:/2/manual/yolo11n_640_manual.kmodel" --file "D:/2/deploy/board_case/testcase.bin"
# 跑（脚本在 PC 上，直接送过去执行，不用先传到板上）
& C:/Users/LR/AppData/Local/Programs/Python/Python312/python.exe -m mpremote connect COM8 run D:/2/deploy/board_infer_float.py
```

两个坑：

- `--file` 的分隔符是 **`::`** 不是 `:`，因为 Windows 路径自带冒号，`D:/2/x.kmodel:/sdcard/x` 会被切错。
- 命令是 **mpremote**，不是 mpremoto；它装在系统 Python 里，`yolo-env` 里没有。

### 2026-11-07 板端实测（对得上 PC）

| 项目 | PC 模拟器 | 板上实际 |
|---|---|---|
| top1 分数 | 0.713 | **0.713** |
| 类别 | 1 thunderbolt | **1** |
| 框中心 | 341.0 / 258.2 | **341.0 / 258.3** |
| >0.10 的 anchor | 9 | **9** |

**一位不差。手动转换这条路是通的。**

但要注意各段耗时（板上实测）：

| 段 | 耗时 |
|---|---|
| 读 SD 卡 | 990 ms |
| uint8 → float32 + 换轴 | 151 ms |
| KPU 推理 | **66 ms** |
| Python 解码 8400 个格子 | **188 ms** |

纯推理只有 66 ms（15.2 fps），**慢的是 Python 那两段**。这就是正式部署版为什么要
uint8 输入（省掉转格式）+ top-8 向量化解码（省掉逐元素循环）——不是模型不行，是胶水代码贵。

---

## 下一步（等板子插上）

把 kmodel 传到板子 `/sdcard/`，用 `D:/2/deploy/board_infer_raw.py` 那套链路跑。
注意：本 kmodel 输入是 float32 NCHW，板上喂之前要 `astype(np.float32)/255` 并转置，
和之前那个 uint8 NHWC 的版本不一样，不能直接套旧脚本。

