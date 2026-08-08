# 点云工具 · PointCloud Studio

把一张普通照片变成可交互的 3D 点云 / 粒子云，并导出标准 PLY 文件。

- **⚡ 浏览器快速模式** —— 模型在你自己的浏览器里跑（WebGPU），零服务器成本，纯静态可部署
- **🎯 Python 高精度模式** —— 本机 GPU 跑 MoGe-2，输出带真实米制尺度的点云
- **🧊 生成式 3D · 单图** —— 本机 GPU 跑 TripoSR，一张图补全背面，快（约 3 秒）
- **🎭 生成式 3D · 多视图** —— 本机 GPU 跑 Hunyuan3D-2mv，**给它三视图，背面照着你的图重建**

技术方案与算法推导见 [PLAN.md](PLAN.md)。

> **在线试用：https://yihuandeps.github.io/pointcloud-studio/**
> 线上版**只有 ⚡ 浏览器模式能真正出结果**，它完全在你的浏览器里跑，打开就能用。
> 另外三种要在你自己的电脑上跑一个 Python + GPU 服务，纯静态托管提供不了 ——
> 探测不到后端时，页面会给它们加上「· 需本机运行」的标注（仍然可以选中，
> 选中后会给出启用步骤）。想用就按下面的说明在本机装，装好后展开下拉标注会自动消失。

> **先说清楚一件事：前两种模式做不出"能转一圈"的效果。**
> 单张照片只记录了朝向镜头的那一层表面，深度估计再准也只能把这层皮推出去 ——
> 得到的是**浮雕（2.5D）**，正面看很立体，转到侧面就是一张弯曲的纸，背后是空的。
> 这不是参数没调好，是单图信息量的上限。想要真正的立体，用生成式 3D 模式。

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

### 四种模式的区别

| | ⚡ 浏览器 | 🎯 高精度 | 🧊 生成式·单图 | 🎭 生成式·多视图 |
|---|---|---|---|---|
| 模型 | Depth Anything V2-S (25M) | MoGe-2 ViT-L (326M) | TripoSR (1.4GB) | Hunyuan3D-2mv (4.9GB) |
| 输入 | 一张图 | 一张图 | 一张图 | **1–4 张视图** |
| 产物 | 浮雕（一层皮） | 浮雕（一层皮，米制） | 完整 360° 形体 | **完整 360° 形体** |
| 转到背面 | 空的 | 空的 | 有内容，但常糊 | **结构清晰** |
| 背面来源 | — | — | 模型凭空推断 | **你给的背面图** |
| 耗时 | 0.2–1s | 0.1–0.3s | 约 3s | 11–40s |
| 显存 | — | 1.5GB | 1.5GB | 5.4GB |
| 适合 | 快速出特效 | 要真实尺度 | 只有一张图时 | **人物 / 商品，要能转一圈** |

**"测量"和"生成"的区别很重要**：前两种模式的几何来自照片里真实可见的像素；
后两种的背面是模型**生成**的。区别在于生成的依据：

- **单图（TripoSR）** 只"看"过正面一次，背面是纯外推。正面很好，
  但转到 135°–180° 通常糊成一团 —— 这是原理性上限，调参数救不回来。
- **多视图（Hunyuan3D-2mv）** 把你给的每一张视图当条件输入，
  给了背面图，背面就是照着它重建的。三视图下后脑勺、衣褶、鞋子都是清楚的。

> **关于"三视图"的一个澄清**：三视图**不能**用来做摄影测量
> （Luma AI / Polycam / COLMAP 那条路）—— 那类方法靠在多张照片里匹配同一个特征点做
> 三角定位，要求相邻照片有 60–80% 画面重叠，三视图之间几乎零重叠，匹配不上任何东西。
> 但多视图**生成**模型是另一套机制：视图是喂给扩散模型的条件，不是用来三角定位的。
> 所以在这条路上，三视图恰恰是标准用法。

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

## 🧊 生成式 3D 模式（补全背面）

用 [TripoSR](https://github.com/VAST-AI-Research/TripoSR) 从单张图推断**完整的 360° 形体**，
包括照片里完全看不见的背面。这是三种模式里唯一能"转一圈都不穿帮"的。

和高精度模式共用同一个后端（`server/start.ps1`，依赖也由同一个 `setup.ps1` 装好），
在前端面板「引擎 → 推理模式」切到 **🧊 生成式 3D** 即可。

### 怎么用效果最好

TripoSR 是按**居中的单个物体**训练的，后端会先用 rembg 自动抠掉背景。所以：

- **好使**：商品、玩具、家具、雕塑、人物半身像 —— 主体明确、边界清楚
- **不好使**：风景、街景、多个物体、大场景 —— 模型不知道该把哪个当主体

风景照请用前两种模式（那种场景本来也不需要看背面）。

### 参数与耗时

切到这个模式后，**「视场角 FOV」「深度强度」「边缘剔除」「远景剔除」会自动置灰** ——
这些是给深度图反投影用的，生成式模式直接产出三维形体，它们不参与计算。
「点数」滑杆仍然有效，后端一次采样 60 万点，前端本地抽稀，**改点数不用重跑模型**。

RTX 3070 Laptop（8GB）实测：模型冷加载约 12 秒，之后每张图 5–20 秒。

### 模型权重与冷启动

两种后端模式各有一份权重，都缓存在 `server/.cache/huggingface`（不入库，也不会碰系统盘）：

| 模式 | 仓库 | 体积 |
|---|---|---|
| 🎯 高精度 | `Ruicheng/moge-2-vitl` | 1.2 GB |
| 🧊 生成式·单图 | `stabilityai/TripoSR` + `facebook/dino-vitb16`（只要 config） | 1.4 GB |
| 🎭 生成式·多视图 | `tencent/Hunyuan3D-2mv`（只下 turbo 子目录） | 4.9 GB |

> 只下 `hunyuan3d-dit-v2-mv-turbo` 一个子目录。整个仓库有 29.6GB（三个变体全算），
> 别用 `snapshot_download` 不带 `allow_patterns` 去拉。

**下载中断会留下坏缓存，而且症状具有误导性** —— HuggingFace 把没下完的文件留成
`blobs/*.incomplete`，`snapshots/` 里则是空的。这时候界面不会报错，只会一直转圈，
实际上是在反复重试一个下不动的链接。确认权重是否完整：

```powershell
# snapshots 目录里应该有真实文件；只有 .incomplete 说明没下完
Get-ChildItem server\.cache\huggingface\hub\models--*\snapshots -Recurse -File
```

没下完就把对应的 `models--*` 整个目录删掉重下。国内网络下 `hf-mirror.com` 时好时坏，
挂代理直连 `huggingface.co` 往往更稳：

```powershell
$env:HTTPS_PROXY = "http://127.0.0.1:7897"   # 换成你自己的代理端口
$env:HF_HOME = "$PWD\server\.cache\huggingface"
$env:HF_HUB_DISABLE_SYMLINKS = "1"           # 否则 Windows 非管理员会 WinError 1314
server\.venv\Scripts\python.exe -c "from huggingface_hub import hf_hub_download; hf_hub_download('Ruicheng/moge-2-vitl','model.pt')"
```

前端的引擎状态灯会区分「模型未下载」和「后端未启动」，不会让你对着转圈猜。

**冷启动耗时**：权重已缓存时，MoGe 约 9 秒、TripoSR 约 12 秒进显存。
两个 runner 都刻意先用 `local_files_only=True` 走本地缓存 —— 否则每次加载都要对
每个文件做一次联网 HEAD 校验，走 hf-mirror 时实测要多花 60 秒以上，
而下面的显存互斥会让这个代价在每次切模式时重复付。

---

## 🎭 多视图模式（人物 / 商品首选）

用 [Hunyuan3D-2mv](https://github.com/Tencent-Hunyuan/Hunyuan3D-2) 从**最多四张视图**重建完整形体。
和其他后端模式共用同一个服务，在「引擎 → 推理模式」切到 **🎭 生成式 3D · 多视图**。

切过去之后拖拽区会变成四个上传槽：

| 槽位 | 必填 | 对应方位 |
|---|---|---|
| 正面 | **是** | 相机在 +Z |
| 左侧 | 否 | 相机在 +X |
| 背面 | 否 | 相机在 −Z |
| 右侧 | 否 | 相机在 −X |

只给正面也能跑（退化成模型自己编背面）；**每多给一张，那一侧就是照着你的原图重建的**。
人物三视图（正面 / 左侧 / 背面）是最典型的用法，也是官方样例的组合。

图可以点击选择，也可以直接拖到某个槽上；拖到页面空白处会自动填进第一个空槽。

### 给几张图？按表面积加权的实测覆盖率

| 输入 | 无任何视图覆盖 | 掠射采样（颜色不可靠） | 良好 |
|---|---|---|---|
| 只有正面 | 48.50% | 18.51% | 32.98% |
| 三视图 | 0.00% | 22.42% | 77.58% |
| **四视图** | **0.00%** | **10.29%** | **89.71%** |

三视图已经没有覆盖空洞了 —— 法线朝右的表面总能被正面或背面以掠射角擦到。
**第四张图的价值不是"补洞"，是把掠射区砍掉一半**，颜色可靠的表面从 77.6% 提到 89.7%。
左右不对称的角色尤其建议四张都给。

四视图之后剩下的唯一短板：**21.1% 的表面法线主要朝上或朝下**（头顶、肩膀、鞋面），
四张水平视图都只能掠射它们。模型只接受这四个方位，没有顶视图选项。

### 生成出来是一块板 / 一个方块？

模型按「**居中的单个物体、周围留白**」训练。输入不符合这个前提时，
它会把整个矩形画面当成物体，产出一块板。后端会对每张图做体检并把问题回报到界面上：

| 症状 | 原因 | 怎么改 |
|---|---|---|
| 主体占画面 > 92% | 背景没抠掉，或图片本身就是满幅 | 换一张主体周围有留白的图 |
| 前景宽高比 > 2.2 | **多半是把三个视图拼在了一张图里** | 拆成三张，分别放进对应槽位 |
| 抠图后什么都不剩 | 主体不明确 / 对比度太低 | 换一张主体清晰的图 |

**最常见的一种**：中文语境里「三视图」经常指**一张**画着三个视图的图纸。
这个工具要的是**三个独立的图片文件**，每张只有一个视角的完整人物。
把一张拼版图放进「正面」槽，模型看到的是三个并排的小人，结果自然是一块板。

背景本身不用你手动抠 —— rembg 会自动处理，实测连杂乱彩色背景都能抠干净
（同一组图在「带 alpha / 白底 / 杂乱背景」三种输入下生成的形体跨度差异 < 0.03）。

### 颜色是怎么来的

Hunyuan3D 的形状管线**只出几何、不带顶点色**（带纹理的那条管线要编译 CUDA 扩展，
Windows 上装不上）。所以颜色由 `hunyuan_runner.py` 自己做：
按每个点的法线挑最正对的那张输入图，正交投影采样。

对多视图输入来说这反而是优点 —— **颜色直接来自你给的原图，不是模型脑补的**。

相邻视图之间按 `(n·d)⁴` 加权混合。指数取得高，所以绝大多数点实际就等于「用最正对的那张图」，
只有在两个视图的交界带上才真正混合 —— 既没有硬接缝，又把投影对不齐可能带来的重影
限制在很窄的一条带里。

投影用的是「轮廓包围盒对齐」：把网格在该视角下的投影包围盒线性映射到该图的前景包围盒，
不用去猜模型内部的归一化和留边参数，两边剪影直接对齐。

两级兜底：掠射角下投影容易落到剪影外被 alpha 挡掉，
① 放宽符号限制再取最接近的视图碰一次 ② 仍未着色的取三维空间里最近的已着色点。
实测最终未着色率 0.01%。

### 视图方位是实测定的，不是推的

`hy3dgen` 的 `MVImageProcessorV2` 只写了 `"front, front clockwise 90, back, front clockwise 270"`，
但"顺时针"从哪个视角算存在歧义。所以 `VIEW_DIRS` 的轴向是**实测**确定的：
把官方样例的 `left.png` 与网格在 +X / −X 两个相机位的渲染逐一比对，
+X 完全吻合（脸朝左、发髻在右），−X 是镜像。改这块前先重做这个比对。

Hunyuan3D 的坐标系（Y 朝上、正面 +Z）与 three.js 一致，所以不需要换轴。

### ⚠️ 许可提醒

其他依赖都是 MIT / Apache / BSD，**只有 Hunyuan3D-2 是
[腾讯混元非商业许可](https://github.com/Tencent-Hunyuan/Hunyuan3D-2/blob/main/LICENSE)**。
它同时限制商业用途和部分地区的使用，权重和模型代码都受约束。
本仓库不分发它的权重（运行时才下载），但如果你要把这个模式用于商业项目，
请先读一遍那份许可。其他三种模式不受此限制。

### 耗时与显存

RTX 3070 Laptop（8GB）实测：权重进显存约 43 秒（4.9GB），之后每次生成 11–40 秒，
显存峰值 5.38GB。体素分辨率默认 256，OOM 时自动降级（224 → 192 → 160）。

想要更高质量可以换非蒸馏的标准变体（要 50 步，慢很多）：

```powershell
$env:HUNYUAN_SUBFOLDER = "hunyuan3d-dit-v2-mv"
$env:HUNYUAN_STEPS = "50"
.\start.ps1
```

### 显存说明（8GB 卡请留意）

MoGe 1.5GB、TripoSR 1.5GB、Hunyuan3D-2mv 峰值 5.4GB，同时驻留必 OOM，
所以后端做了**显存互斥**：切换模式时自动把其余模型踢出显存，切回来重新加载。
实测切换一次 7–15 秒（Hunyuan 约 43 秒，权重大得多），前提是权重已缓存（见上一节）。

marching cubes 的体素分辨率默认 192。**显存不足时会自动降级重试**
（160 → 128 → 96），不会直接失败。想要更高细节且显存充裕：

```powershell
$env:TRIPO_MC_RES = "256"   # 细节更好，需要约 2GB 额外显存
.\start.ps1
```

如果生成时报显存不足，先关掉浏览器等占显存的程序 —— 桌面应用通常已经吃掉 1–2GB。

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

npm run test:gen3d             # 单图生成式全链路（需先启动 server/start.ps1）
npm run test:mv                # 多视图全链路（同上）

# 后端（在 server/ 下，用 .venv 里的 python）
.venv\Scripts\python.exe test_backend.py    # 高精度：MoGe 推理 → HTTP → 二进制打包
.venv\Scripts\python.exe test_gen3d.py      # 单图生成式：TripoSR → 立体性检验 → HTTP
.venv\Scripts\python.exe test_mv.py         # 多视图：Hunyuan3D → 着色检验 → HTTP
```

`npm run test:gen3d` 和 `test_gen3d.py` 的分工值得说明：后者验证后端自己，
前者验证**浏览器真正走的那条路** —— 同一份 `gen3dServer.js` 解析二进制、
同一份 `unproject.js` 抽稀归一化、同一份 `plyExport.js` 写文件。
后端全绿但前端解析错位这类问题，只有前者能测出来。产物 `tests/out/e2e_gen3d.ply`
可以直接拖进 CloudCompare 转一圈验背面。

**两个后端测试要分开跑** —— 各自都要把模型加载进显存，8GB 的卡同时跑会 OOM。

`test_gen3d.py` 里有两项断言值得说明，它们是这个模式存在意义的直接检验：
**三轴跨度比 > 0.4**（浮雕的最薄轴向接近 0，实测球体 0.96）、
**背面点占比 > 25%**（纯正面浮雕接近 0，实测 49.2%）。

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
│   │   │   ├── gen3dServer.js   生成式 3D 的 API 客户端
│   │   │   ├── imagePrep.js     解码 / 限长边 / 程序化示例图
│   │   │   ├── unproject.js     ⭐ 深度图 → XYZRGB（核心算法）
│   │   │   └── plyExport.js     PLY 写出
│   │   ├── render/              three.js 场景 + 粒子 shader
│   │   └── ui/                  参数面板 / 拖拽上传 / 状态栏
│   ├── scripts/fetch-model.mjs  模型预下载
│   ├── tests/                   回归测试
│   └── public/models/           预下载的模型（不入库）
├── server/                      Python 后端（高精度 + 生成式 3D）
│   ├── app.py                   FastAPI 接口 + 两个模型的显存互斥
│   ├── moge_runner.py           MoGe-2 封装（高精度）
│   ├── tripo_runner.py          TripoSR 封装（生成式 3D）
│   ├── tsr/                     vendored TripoSR 源码（见下）
│   ├── test_backend.py          高精度模式端到端测试
│   └── test_gen3d.py            生成式 3D 端到端测试
├── PLAN.md                      技术方案与算法推导
└── README.md
```

**关于 `server/tsr/`**：TripoSR 官方不发 PyPI 包，只能装源码，所以直接 vendor 进来了
（MIT 协议，LICENSE 已一并保留）。相对上游改了一处：`tsr/models/isosurface.py` 里的
`torchmcubes` 换成了 `scikit-image` 的 `marching_cubes`。原因是 torchmcubes 只提供
`git+https://` 源码依赖且需要本地编译 C++/CUDA 扩展，Windows 上装不上。
两者顶点轴序不同（torchmcubes 是 zyx，skimage 是 xyz），所以上游那句 `[2,1,0]` 翻转也一并去掉了。

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

**转个角度点云就没了 / 看起来是平的**
前两种模式的产物本来就是浮雕，只有朝向镜头的一层表面 —— 这是单图深度估计的固有边界，
不是 bug 也调不出来。想要能转一圈的立体效果，切到 **🧊 生成式 3D 模式**。

**生成式模式出来的东西不像原图 / 形状很怪**
TripoSR 按"居中的单个物体"训练。风景、街景、多个物体、大场景都不适用 ——
它不知道该把哪个当主体。换一张主体明确、背景干净的图片试试。

**切到后端模式后一直转圈，既不报错也不出结果**
多半是权重没下完。看「模型权重与冷启动」一节确认 `snapshots/` 里有没有真实文件；
只有 `.incomplete` 就是没下完，删掉重下。

**生成时报显存不足**
先关掉浏览器等占显存的程序（桌面应用通常已占 1–2GB）。后端已经做了自动降级
（192 → 160 → 128 → 96），如果降到底还失败，说明可用显存确实太少了。

**导出的 PLY 在 CloudCompare 里是灰的**
CloudCompare 打开后需要在左侧属性面板把 `Colors` 从 `None` 切到 `RGB`。

---

## 依赖的开源项目

| 项目 | 用途 | 许可 |
|---|---|---|
| [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) | 单目深度估计 | Apache 2.0 |
| [transformers.js](https://github.com/huggingface/transformers.js) | 浏览器端推理运行时 | Apache 2.0 |
| [three.js](https://github.com/mrdoob/three.js) | 3D 渲染 | MIT |
| [MoGe](https://github.com/microsoft/MoGe) | 高精度后端 | MIT |
| [TripoSR](https://github.com/VAST-AI-Research/TripoSR) | 单图生成式后端（源码 vendor 在 `server/tsr/`） | MIT |
| [Hunyuan3D-2](https://github.com/Tencent-Hunyuan/Hunyuan3D-2) | 多视图生成式后端 | 腾讯混元非商业许可 |
| [rembg](https://github.com/danielgatis/rembg) | 生成前的自动抠图 | MIT |
| [scikit-image](https://github.com/scikit-image/scikit-image) | marching cubes（替代需编译的 torchmcubes） | BSD-3 |
| [Vite](https://github.com/vitejs/vite) | 构建工具 | MIT |
