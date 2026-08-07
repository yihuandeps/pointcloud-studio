# 点云工具 · 技术方案（PointCloud Studio）

> 单张普通照片 → 3D 点云 / 粒子云特效
> 定稿日期：2026-08-03

---

## 一、总体架构

```
                    ┌─────────────────────────────────┐
   用户上传图片 ───→ │  预处理：解码 / 限长边到 1024px  │
                    └────────────┬────────────────────┘
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
      ┌──────────────────────┐      ┌────────────────────────┐
      │  ⚡ 快速模式（默认）  │      │  🎯 高精度模式（可选）  │
      │  浏览器 Web Worker   │      │  本机 Python + GPU     │
      │  transformers.js     │      │  FastAPI + MoGe-2      │
      │  Depth Anything V2-S │      │  ViT-L 326M            │
      │  WebGPU / WASM 兜底  │      │  HTTP :8000            │
      └──────────┬───────────┘      └───────────┬────────────┘
                 │  相对逆深度(disparity)        │  米制点图 + 相机内参
                 └───────────────┬───────────────┘
                                 ▼
                    ┌─────────────────────────────────┐
                    │  反投影 → XYZRGB (Float32Array) │
                    │  + 飞边剔除 + 天空剔除 + 抽稀    │
                    └────────────┬────────────────────┘
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
      ┌──────────────────────┐      ┌────────────────────────┐
      │  three.js 粒子渲染   │      │  PLY 二进制导出        │
      │  Points + 自定义GLSL │      │  CloudCompare/Blender  │
      └──────────────────────┘      └────────────────────────┘
```

**核心设计原则**：特效变换（噪声、斥力、聚合动画）**只在 GPU 顶点着色器里做**，CPU 端始终保留一份干净的原始坐标 —— 这样导出的 PLY 才是正确的点云，而不是被特效扭曲过的数据。

---

## 二、目录结构

```
点云工具/
├── web/                              # 前端（纯静态，可独立部署）
│   ├── index.html                    # 页面骨架
│   ├── vite.config.js                # 端口 / /api 代理到 :8000 / esnext
│   ├── package.json
│   ├── public/samples/               # 内置示例图
│   └── src/
│       ├── main.js                   # 总编排：上传→推理→建云→渲染→导出
│       ├── style.css                 # 深色玻璃拟态 UI
│       ├── config.js                 # 模型清单、HF 镜像、默认参数
│       ├── core/
│       │   ├── depth.worker.js       # ⭐ Worker：加载模型 + 推理，不卡 UI
│       │   ├── depthBrowser.js       # Worker 主线程封装（含进度回调）
│       │   ├── depthServer.js        # 高精度模式：POST 到 Python API
│       │   ├── imagePrep.js          # 解码、限长边、取 RGBA
│       │   ├── unproject.js          # ⭐ 深度图 → XYZRGB（核心算法）
│       │   └── plyExport.js          # 二进制 PLY 写出
│       ├── render/
│       │   ├── viewer.js             # three.js 场景/相机/控制器/循环
│       │   ├── pointCloud.js         # Points + BufferGeometry 装配
│       │   └── shaders.js            # ⭐ GLSL：聚合/噪声/斥力/透视衰减
│       └── ui/
│           ├── panel.js              # 参数面板
│           ├── dropzone.js           # 拖拽上传 + 示例图
│           └── status.js             # 进度 / 点数 / 帧率 / 后端状态
│
├── server/                           # 后端（高精度模式，可选）
│   ├── app.py                        # FastAPI：/health /infer
│   ├── moge_runner.py                # MoGe-2 加载与推理封装
│   ├── requirements.txt
│   └── start.ps1                     # 一键启动（含 HF 镜像变量）
│
├── PLAN.md                           # 本文档
└── README.md                         # 安装 / 启动 / 部署 / FAQ
```

---

## 三、关键算法

### 3.1 视差 → 深度

Depth Anything 输出的 `predicted_depth` 是**相对逆深度（disparity）**，值越大越近，无物理单位。

先归一化：`d = (raw - min) / (max - min)`，得到 0=最远、1=最近。

然后**不能**直接用 `z = 1 - d`（线性映射，透视关系错误，出来像一张浮雕）。正确做法是在**视差空间**线性插值再取倒数：

```
disp = d * (1/zNear − 1/zFar) + 1/zFar
z    = 1 / disp
```

校验：`d=1 → z=zNear` ✓　`d=0 → z=zFar` ✓

`zFar/zNear` 比值 = UI 上的**「深度强度」**滑杆。比值越大立体感越强，过大则背景被拉爆。默认 3.5。

### 3.2 针孔相机反投影

```
fx = 0.5 * W / tan(hFOV / 2)     fy = fx        （方形像素）
cx = W / 2                       cy = H / 2

X =  (u − cx) * z / fx
Y = −(v − cy) * z / fy      ← 图像 v 轴向下，three.js Y 轴向上，取负
Z = −z                      ← three.js 相机看向 −Z
```

`hFOV` 对普通照片未知，默认 **55°**（手机/常见相机典型值），做成滑杆。

> 高精度模式跳过整节 —— MoGe-2 的 `infer()` 直接返回 `points (H,W,3)` 与 `intrinsics (3,3)`，已是正确米制坐标。

### 3.3 ⭐ 飞边剔除（决定成品好看还是难看）

单图转点云**最明显的破绽**：前景/背景交界处深度平滑过渡，反投影后会在人物边缘与背景间拉出"面条"。

**深度梯度剔除** —— 检查四邻域归一化视差的最大跳变：

```
g = max( |d(u±s,v) − d(u,v)| , |d(u,v±s) − d(u,v)| )
if (g > edgeThreshold) → 丢弃此点
```

`edgeThreshold` 默认 0.04，UI 上叫**「边缘剔除」**。

配套**「天空剔除」**：`d < skyFloor` 的点丢掉（无穷远处噪点会被拉到极远变成噪声）。默认 0.02。

### 3.4 采样密度

目标点数 N 由滑杆控制（5万 – 80万），反算网格步长：

```
stride = max(1, round( sqrt(W * H / N) ))
```

规则网格采样 → 点数可控、性能可预期，不会因为换大图就掉帧。

### 3.5 粒子着色器

**顶点属性**：`position`(目标位置) · `color`(原图RGB) · `aScatter`(随机散开位) · `aSeed`(随机种子)

```glsl
vec3 pos = mix(aScatter, position, easeOutCubic(uProgress));    // 聚合入场
pos += snoise3(pos * uNoiseFreq + uTime) * uNoiseAmp;           // 呼吸扰动

vec3 d = pos - uMouse;                                          // 鼠标斥力
pos += normalize(d) * smoothstep(uRepelRadius, 0.0, length(d)) * uRepelStrength;

vec4 mv = modelViewMatrix * vec4(pos, 1.0);
gl_PointSize = uSize * uPixelRatio * (uSizeScale / -mv.z);      // 透视衰减
gl_Position  = projectionMatrix * mv;
```

**片元**：`gl_PointCoord` 算中心距，`smoothstep` 圆形柔边，边缘外 `discard`。

**「云雾 / 实体」**两种渲染模式：
- 云雾 = `transparent + depthWrite:false` → 通透飘逸
- 实体 = `alphaTest + depthWrite:true` → 遮挡正确、更像真实点云

### 3.6 PLY 二进制格式

```
ply
format binary_little_endian 1.0
comment created by PointCloud Studio
element vertex N
property float x / y / z
property uchar red / green / blue
end_header
<N × 15 字节>
```

每点 15 字节，50 万点 ≈ 7.5 MB。CloudCompare / Open3D / MeshLab / Blender 均可直接打开。

导出时提供**「保持原始尺度」**选项：渲染用的是归一化坐标，勾选后按记录的 `center / scale` 还原回原始（高精度模式下即米制）坐标。

---

## 四、已确认的技术事实（实测，非推测）

| 事项 | 结论 |
|---|---|
| transformers.js 版本 | 实装 **v4.2.0**，网上 v3 教程写法需调整 |
| `predicted_depth` | Tensor，dims `[H, W]`，float32，**已插值回原图尺寸**，**未归一化**（原始逆深度） |
| `depth` 字段 | RawImage，Uint8 单通道，已归一化 0–255 —— 精度较低，**采用 `predicted_depth`** |
| 🚨 网络 | **`huggingface.co` 直连超时，`hf-mirror.com` 返回 200** → 必须内建 `env.remoteHost` 镜像 |
| dtype 后缀映射 | `fp32→''` `fp16→'_fp16'` `q8→'_quantized'` `q4f16→'_q4f16'` |
| 可用量化版本 | fp16 / q8 / q4f16 / int8 / uint8 / bnb4 均存在于 onnx-community 仓库 |
| 下载体积 | fp16 ≈ 50MB（WebGPU）· q8 = 25MB（CPU 兜底） |
| 本机 GPU | RTX 3070 Laptop 8GB → WebGPU 走 fp16；Python 端 MoGe-2 ViT-L 也吃得下 |
| 本机 Python | **3.9.7 太旧**，MoGe 需 3.10+ |

---

## 五、双模式对比

| | ⚡ 浏览器快速模式 | 🎯 Python 高精度模式 |
|---|---|---|
| 模型 | Depth Anything V2-Small (25M) | MoGe-2 ViT-L (326M) |
| 算力 | 用户浏览器 WebGPU | 本机 RTX 3070 |
| 首次下载 | 50MB（浏览器缓存，仅一次） | 700MB |
| 单图耗时 | 0.2 – 1s | 0.1 – 0.3s + 传输 |
| 深度性质 | 相对值，尺度未知，FOV 靠猜 | **米制尺度 + 自动预测内参** |
| 边缘质量 | 中等，靠梯度剔除补救 | 好，自带有效性 mask |
| 部署 | 纯静态，Vercel / GitHub Pages | 需带 GPU 的机器 |
| 许可 | Apache 2.0 | **MIT**（可商用） |

---

## 六、环境准备

**需要用户做的：**

1. **装 Python 3.11**（只影响高精度模式；快速模式完全不需要 Python）
   `winget install Python.Python.3.11` —— 保留现有 3.9 不冲突
2. **准备 3–5 张测试图**：人像特写 / 桌面物体 / 室外风景，放入 `web/public/samples/`
3. 网络：前端已内建 hf-mirror，Python 端启动脚本自动设 `HF_ENDPOINT`

---

## 七、分阶段交付

| 阶段 | 内容 | 验收标准 |
|---|---|---|
| **一** | 前端骨架 + 浏览器模式 + 反投影 + 粒子渲染 + PLY 导出 | 拖图进去出可旋转粒子云，能下载 PLY 并用 CloudCompare 打开 |
| **二** | Python 后端 + MoGe-2 + 前端模式切换 | 切到「高精度」出米制点云，边缘明显更干净 |
| **三** | 打磨：动效曲线、配色、示例图、响应式、构建部署 | 手机能开，`npm run build` 产物可直接部署 |

---

## 八、实施记录：踩到的坑

按发现顺序，都是文档里查不到、只有真跑一遍才会暴露的。

| # | 问题 | 根因 | 处理 |
|---|---|---|---|
| 1 | transformers.js 教程写法全对不上 | 实装是 **v4.2.0**，网上教程基本是 v3 | 直接读 `node_modules` 里的源码确认 API |
| 2 | 模型下不下来 | `huggingface.co` 直连超时；`hf-mirror` 通但权重会 307 跳到 `us.aws.cdn.hf.co` | 预下载脚本 + 本地加载优先 |
| 3 | Node 里连镜像必超时 | Node 全局 `fetch`（undici）在本机连不上，`curl` 和 `node:https` 都正常 | 下载脚本改用 `node:https` 手写重定向跟随 |
| 4 | 「点数」滑杆是跳变的 | 整数步长：步长 1 全量、步长 2 只剩四分之一，中间没档位。设 40 万实际给 78 万 | 改浮点步长，误差 ±92% → **±0.9%** |
| 5 | 粒子糊满屏幕 | `gl_PointSize` 量纲错：把视口高度当参考视距，默认参数算出 355px | 参考视距改为相机距离，让滑杆单位就是 CSS 像素，并加 64px 上限 |
| 6 | `.ps1` 脚本语法崩溃 | PowerShell 5.1 在中文系统按 GBK 读无 BOM 的 UTF-8 文件，中文全乱、全角括号破坏语法 | 脚本一律存成 **UTF-8 with BOM** |
| 7 | 装 PyTorch 报 `[Errno 28]` | pip 的 TEMP 和 cache 都在系统盘，而系统盘只剩 2.6GB，2.4GB 的 wheel 直接撑爆 | `setup.ps1` 把 `TMP`/`PIP_CACHE_DIR`/`HF_HOME` 全指到项目盘 |
| 8 | `pip install git+https://github.com/...` 失败 | `github.com:443` 连接超时，MoGe 的 utils3d / pipeline 两个依赖是 git 直连地址 | 见下方「GitHub 各域名的可达性差异」 |

### GitHub 各域名的可达性差异

同一个 github.com 下的服务，在这台机器上表现完全不同：

| 域名 | 状态 | 用途 |
|---|---|---|
| `github.com` | ❌ 443 超时 | `git clone` 走这里，所以 `pip install git+...` 必败 |
| `codeload.github.com` | ✅ 可达，**但不支持 Range** | 发 tarball。小仓库没问题；大仓库断了无法续传，只能重下 |
| `raw.githubusercontent.com` | ✅ 可达且快（13KB / 0.77s） | 单文件 |
| `api.github.com` | ✅ 可达 | 列文件树 |
| `ghproxy.net` | ⚠️ 可达但只有 ~12KB/s | 兜底 |

据此设计了 `fetch_sources.py` 的回退策略：

- **utils3d（156KB）/ pipeline（12KB）** → codeload 抓 tarball，小到不会断
- **MoGe** → 完整 tarball 有 10.5MB（大部分是 demo 素材），实测多次下到 3.5MB 就断且续传无效。
  改成用 `api.github.com` 列出文件树，只挑 `moge/` 包本身的 60 个文件，
  从 `raw.githubusercontent.com` 逐个下 —— **354KB，一次成功**。

装 MoGe 时必须 `--no-deps`：它的 `pyproject.toml` 里那两个依赖写的是 `git+` 直连地址，
不加这个参数 pip 会无视已装好的版本、再去 clone 一次，然后再次失败。

### 两条值得记住的经验

**前后端各写一遍同一个算法 = 迟早对不上。**
尺寸计算里 Python 的 `round()` 是银行家舍入（`round(2.5)==2`），JS 的 `Math.round` 是
`.5` 向上（`3`）。差一个像素，点图和颜色就整体错位，画面看着"糊了"却完全查不出原因。
所以专门写了跨语言契约测试（`web/tests/protocol.test.mjs`），用真实的 Python 产出去喂
真实的 JS 解析器，覆盖 952 组尺寸 + 完整二进制往返。

**纯函数要能脱离重依赖单独测。**
`imaging.py` 刻意不 import torch —— 尺寸计算和二进制打包恰恰是最容易出错、
症状最隐蔽的部分，不该因为没装 3GB 的深度学习依赖就测不了。

---

## 九、参考的开源项目

| 项目 | 用途 | 许可 |
|---|---|---|
| [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) | 浏览器端深度估计（ONNX） | Apache 2.0 |
| [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) | 备选高精度后端 | Apache 2.0 / CC-BY-NC |
| [MoGe](https://github.com/microsoft/MoGe) | 高精度后端主选 | MIT |
| [transformers.js](https://github.com/huggingface/transformers.js) | 浏览器推理运行时 | Apache 2.0 |
| [three.js](https://github.com/mrdoob/three.js) | 渲染 | MIT |
| [MapAnything](https://github.com/facebookresearch/map-anything) | 未来扩展多视角重建 | Apache 2.0 |
| [FoundationStereo](https://github.com/NVlabs/FoundationStereo) | 未来扩展双目输入 | NVIDIA Source Code License |
| [Spark](https://github.com/sparkjsdev/spark) | 未来扩展 3DGS 渲染 | MIT |
