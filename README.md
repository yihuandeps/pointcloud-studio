# 点云工具 · PointCloud Studio

把一张普通照片变成可交互的 3D 点云 / 粒子云，并导出标准 PLY 文件。

- **⚡ 浏览器快速模式** —— 模型在你自己的浏览器里跑（WebGPU），零服务器成本，纯静态可部署
- **🎯 Python 高精度模式** —— 本机 GPU 跑 MoGe-2，输出带真实米制尺度的点云（阶段二）

技术方案与算法推导见 [PLAN.md](PLAN.md)。

---

## 快速开始

```bash
cd web
npm install
npm run fetch-model      # 预下载模型到 public/models/（约 73MB，只需一次）
npm run dev              # → http://127.0.0.1:5173/
```

打开页面后，把图片拖进去即可。也可以点「选择文件」、直接 `Ctrl+V` 粘贴、或者用页面上的内置示例图。

> **为什么要先 `fetch-model`？**
> transformers.js 默认从 `huggingface.co` 拉模型，而国内网络通常直连超时。
> 这个脚本走 `hf-mirror.com` 镜像把模型下到本地，之后运行时零网络依赖。
> 不跑也能用 —— 首次推理时会自动从镜像下载，但成功率取决于你的网络。

---

## 环境要求

| 项 | 要求 | 说明 |
|---|---|---|
| Node.js | ≥ 20 | 开发和构建 |
| 浏览器 | Chrome / Edge 113+ | WebGPU 加速；不支持会自动回退 CPU（较慢但能用） |
| 显卡 | 任意 | 8GB 显存足够；集显走 CPU 回退也能跑 |
| Python | 3.10+ | **只有高精度模式需要**，快速模式完全用不到 |

---

## 用法

### 参数面板

| 参数 | 作用 | 调参建议 |
|---|---|---|
| **点数** | 采样密度 | 20 万左右是画质与流畅的平衡点 |
| **视场角 FOV** | 拍摄相机的水平视场角 | 手机约 55–70°，长焦更小。调小会把画面「推远压平」 |
| **深度强度** | 最远/最近的距离比 | 越大立体感越强，过大会把背景拉爆成噪点 |
| **边缘剔除** | 剔除前后景交界处被拉长的「面条」 | **最影响成品观感的参数**。调小剔得更干净，调大保留更多点 |
| **远景剔除** | 丢掉天空 / 无穷远背景 | 风景照建议开大一点，室内特写可以设 0 |
| **点大小 / 不透明度 / 边缘柔和** | 视觉调整 | — |
| **实体模式** | 开=写深度缓冲遮挡正确，关=云雾感通透 | 想要「真点云」的观感就开，想要特效就关 |
| **色调** | 原图色彩 / 深度渐变 / 单色霓虹 / 原图×深度 | — |
| **呼吸 / 鼠标斥力 / 自动旋转** | 动效 | 斥力半径设 0 可关闭鼠标交互 |

### 导出

- **PLY 二进制** —— 每点 15 字节，50 万点约 7.5MB。CloudCompare / MeshLab / Open3D / Blender 均可直接打开
- **PLY ASCII** —— 可读文本，体积约 3 倍
- **保持原始尺度** —— 渲染用的是归一化坐标，勾上则还原回原始坐标（高精度模式下即真实米制）
- **截图 PNG** —— 导出当前视角的画面

> 特效（噪声扰动、鼠标斥力、聚合动画）只在 GPU 顶点着色器里做，CPU 端始终保留干净坐标。
> 所以**导出的 PLY 不会被特效扭曲**。

---

## 🎯 高精度模式（可选）

浏览器模式用的是相对深度 —— 尺度未知、FOV 靠猜。高精度模式在本机 GPU 上跑
[MoGe-2](https://github.com/microsoft/MoGe)，直接输出**带真实米制尺度的点图和相机内参**，
边缘也更干净（模型自带有效性 mask，不用靠梯度阈值硬剔）。

### 安装

```powershell
cd server
.\setup.ps1        # 装 PyTorch CUDA + MoGe + FastAPI，约 3GB
```

需要 Python 3.11（`winget install --id Python.Python.3.11 --scope user`）。

脚本做了两件容易被坑的事：

- **把 pip 的临时目录和缓存放到项目所在盘**。默认在系统盘，而 2.4GB 的 PyTorch wheel
  很容易把不宽裕的系统盘撑爆，报 `[Errno 28] No space left on device`
- **MoGe 装不上时自动走源码回退**。MoGe 的两个依赖（utils3d / pipeline）写的是
  `git+https://github.com/...`，国内网络下 `github.com:443` 常常直连超时，
  `pip install git+` 必败。回退逻辑见 `fetch_sources.py`：小仓库走 `codeload`，
  MoGe 本身走 `api.github.com` 列文件树 + `raw.githubusercontent.com` 逐个下
  （只取包本身 354KB，完整 tarball 有 10.5MB 且不支持断点续传）

### 启动

```powershell
.\start.ps1        # 监听 127.0.0.1:8000
```

然后在前端面板的「引擎 → 推理模式」切到 **🎯 本机 Python**。首次启动会下载模型权重
（走 hf-mirror 镜像，缓存在 `server/.cache/huggingface`）：

| 变体 | 参数量 | 权重体积 | 适用 |
|---|---|---|---|
| `moge-2-vitl`（默认） | 326M | **1.2 GB** | 质量最好，8GB 显存绰绰有余 |
| `moge-2-vitl-normal` | 331M | ~1.3 GB | 额外输出法线 |
| `moge-2-vitb-normal` | 104M | ~400 MB | 更快 |
| `moge-2-vits-normal` | 35M | ~140 MB | 最轻，网络差时先用这个试通 |

### 两种模式的区别

| | ⚡ 浏览器 | 🎯 高精度 |
|---|---|---|
| 模型 | Depth Anything V2-Small (25M) | MoGe-2 ViT-L (326M) |
| 深度性质 | 相对值，尺度未知 | **米制尺度 + 自动预测内参** |
| 单图耗时 | 0.2–1s | 0.1–0.3s + 传输 |
| 边缘质量 | 靠「边缘剔除」滑杆补救 | 模型自带 mask，更干净 |
| 依赖 | 无 | Python + GPU |

切到高精度模式后，**「视场角 FOV」「深度强度」「边缘剔除」「远景剔除」四个滑杆会自动置灰** ——
这些参数是给相对深度用的，MoGe 给的是真实几何，它们不参与计算。「保持原始尺度」导出的
PLY 此时就是**真实米制坐标**。

### 换模型变体

```powershell
$env:MOGE_MODEL = "Ruicheng/moge-2-vitb-normal"   # 104M，更快
.\start.ps1
```

可选：`moge-2-vitl`（默认，326M）· `moge-2-vitl-normal`（331M，带法线）·
`moge-2-vitb-normal`（104M）· `moge-2-vits-normal`（35M）

---

## 常用命令

```bash
# 前端
npm run dev                    # 开发服务器
npm run build                  # 生产构建 → dist/
npm run preview                # 预览构建产物

npm run fetch-model            # 下载 small 模型（默认）
npm run fetch-model base       # 下载 base 模型（更细腻，约 290MB）

npm test                       # 全部测试（核心算法 40 项 + 前后端契约 16 项）
npm run test:core              # 只跑核心算法（不需要模型和 Python）
npm run test:protocol          # 只跑前后端契约（需要 Python，不需要 torch）
npm run test:e2e               # 端到端：真实推理 → 点云 → 导出 PLY（需先 fetch-model）
```

**关于契约测试**：尺寸计算和二进制布局在 Python 和 JS 里各实现了一遍。两边差一个像素，
点图和颜色就整体错位，画面看着"糊了"却查不出原因；布局对不上则会解出一堆 NaN。
`test:protocol` 用真实的 Python 产出去喂真实的 JS 解析器，覆盖 952 组尺寸和完整的二进制往返。

`npm run test:e2e` 会在 `web/tests/out/e2e_test.ply` 生成一个真实点云文件，可以直接拖进
CloudCompare 验证整条链路。

---

## 部署

```bash
npm run build
```

`dist/` 是纯静态产物，直接丢到 Vercel / Netlify / GitHub Pages / 任意静态服务器即可。

**关于体积**：`dist/` 约 98MB，其中 73MB 是打包进去的模型权重。两种取舍：

- **保留模型**（默认）—— 自包含，用户零等待零网络依赖，但部署包大
- **删掉 `public/models/` 再构建** —— `dist/` 降到约 25MB，改由用户浏览器首次访问时从
  `hf-mirror.com` 下载模型（有缓存，只下一次）。代价是首次打开慢、且依赖镜像可用性

---

## 目录结构

```
点云工具/
├── web/                         前端（纯静态）
│   ├── src/
│   │   ├── config.js            模型清单 / 镜像地址 / 参数 schema
│   │   ├── core/
│   │   │   ├── depth.worker.js  Worker 里跑深度估计，不卡 UI
│   │   │   ├── depthBrowser.js  Worker 的 Promise 封装
│   │   │   ├── depthServer.js   高精度模式的 API 客户端
│   │   │   ├── imagePrep.js     解码 / 限长边 / 程序化示例图
│   │   │   ├── unproject.js     ⭐ 深度图 → XYZRGB（核心算法）
│   │   │   └── plyExport.js     PLY 写出
│   │   ├── render/              three.js 场景 + 粒子 shader
│   │   └── ui/                  参数面板 / 拖拽上传 / 状态栏
│   ├── scripts/fetch-model.mjs  模型预下载
│   ├── tests/                   回归测试
│   └── public/models/           预下载的模型（不入库）
├── server/                      Python 后端（阶段二）
├── PLAN.md                      技术方案与算法推导
└── README.md
```

---

## 常见问题

**页面打开是空的 / 控制台报 WebGPU 错误**
面板上的「引擎」状态灯会显示实际用的后端。没有 WebGPU 会自动回退到 CPU（WASM），
慢一些但能用。Chrome 里可以到 `chrome://gpu` 确认 WebGPU 状态。

**模型下不下来**
先跑 `npm run fetch-model`。如果这个脚本也失败，说明 `hf-mirror.com` 在你的网络下不通，
可以改 `web/src/config.js` 里的 `HF_ENDPOINT` 换别的镜像。

**点云边缘有很多拉长的「面条」**
把「边缘剔除」调小（比如 0.02）。这是单图转点云的固有问题 —— 深度图在遮挡边界是平滑
过渡的，反投影后就会在前后景之间拉丝。

**背景一片噪点 / 被拉得很远**
调大「远景剔除」，或调小「深度强度」。天空这类无穷远区域的深度预测本来就不可靠。

**转出来很平，没有立体感**
调大「深度强度」，或调小「视场角 FOV」。也可能是原图本身缺少深度线索（比如平面翻拍图）。

**导出的 PLY 在 CloudCompare 里是灰的**
CloudCompare 打开后需要在左侧属性面板把 `Colors` 从 `None` 切到 `RGB`。

---

## 依赖的开源项目

| 项目 | 用途 | 许可 |
|---|---|---|
| [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) | 单目深度估计 | Apache 2.0 |
| [transformers.js](https://github.com/huggingface/transformers.js) | 浏览器端推理运行时 | Apache 2.0 |
| [three.js](https://github.com/mrdoob/three.js) | 3D 渲染 | MIT |
| [MoGe](https://github.com/microsoft/MoGe) | 高精度后端（阶段二） | MIT |
| [Vite](https://github.com/vitejs/vite) | 构建工具 | MIT |
