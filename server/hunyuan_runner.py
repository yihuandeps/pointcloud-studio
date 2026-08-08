"""
Hunyuan3D-2mv 推理封装（多视图生成式 3D）。

相比 TripoSR：TripoSR 只"看"过正面一次，背面靠一次性外推，转过去就是一团糊；
Hunyuan3D-2mv 接受最多四张视图作为条件，背面是**依据你给的背面图**重建的，
所以能真正经得起 360° 环绕。实测同一个角色，TripoSR 背面糊成一片，
这个模型的后脑勺、背带、腿部结构都在。

一个必须处理的差异：Hunyuan3D 的形状管线**只出几何，不带顶点色**
（带纹理的那条管线要编译 CUDA 扩展，Windows 上装不上）。
所以颜色由本模块自己做：按顶点法线挑最"正对"的那张输入图，正交投影采样。
对多视图输入来说这反而更忠实 —— 颜色直接来自用户给的原图，不是模型脑补的。
"""

from __future__ import annotations

import io
import os
import threading
import time

import numpy as np
import torch
from PIL import Image

import bg_removal
import surface

DEFAULT_MODEL = os.environ.get("HUNYUAN_MODEL", "tencent/Hunyuan3D-2mv")
# turbo 是步数蒸馏版，5 步即可（标准版要 50 步）。想要更高质量换 hunyuan3d-dit-v2-mv
DEFAULT_SUBFOLDER = os.environ.get("HUNYUAN_SUBFOLDER", "hunyuan3d-dit-v2-mv-turbo")
# 体素分辨率。256 在 RTX 3070(8GB) 上显存峰值约 5.4GB、耗时约 33 秒
DEFAULT_OCTREE = int(os.environ.get("HUNYUAN_OCTREE", "256"))
FALLBACK_OCTREE = (224, 192, 160)
DEFAULT_STEPS = int(os.environ.get("HUNYUAN_STEPS", "5"))
DEFAULT_POINTS = 600_000
MAX_POINTS = 1_000_000

# 视图名 → 相机所在方向（由物体指向相机）。
# 权威定义见 hy3dgen/shapegen/preprocessors.py 的 MVImageProcessorV2：
# "front, front clockwise 90, back, front clockwise 270"，front/left/back/right。
# 但"顺时针"从哪个视角算有歧义，所以这里的轴向是**实测**定下来的：
# 把官方样例的 left.png 与网格在 +X / -X 两个相机位的渲染逐一比对，
# +X 完全吻合（脸朝左、发髻在右），-X 是镜像。
VIEW_DIRS = {
    "front": (0.0, 0.0, 1.0),
    "left": (1.0, 0.0, 0.0),
    "back": (0.0, 0.0, -1.0),
    "right": (-1.0, 0.0, 0.0),
}
UP = np.array([0.0, 1.0, 0.0])


def _screen_basis(d: np.ndarray):
    """相机方向 d（物体→相机）对应的屏幕右向量与上向量。"""
    right = np.cross(UP, d)
    right /= np.linalg.norm(right) + 1e-9
    up = np.cross(d, right)
    return right, up


class _View:
    """一张已抠好背景的输入图，连同它的前景包围盒。"""

    def __init__(self, name: str, image: Image.Image):
        self.name = name
        self.dir = np.array(VIEW_DIRS[name], dtype=np.float64)
        rgba = np.asarray(image.convert("RGBA"))
        self.rgb = rgba[:, :, :3]
        self.alpha = rgba[:, :, 3]

        ys, xs = np.nonzero(self.alpha > 8)
        if len(xs) == 0:  # 整张全透明，退化成整图
            self.box = (0, 0, rgba.shape[1] - 1, rgba.shape[0] - 1)
        else:
            self.box = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


class HunyuanRunner:
    def __init__(self, model_id: str = DEFAULT_MODEL, subfolder: str = DEFAULT_SUBFOLDER,
                 device: str | None = None):
        self.model_id = model_id
        self.subfolder = subfolder
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.pipeline = None
        self._lock = threading.Lock()
        self.load_error: str | None = None
        self.load_seconds: float | None = None

    # ---------------- 加载 ----------------

    def _resolve_weights(self) -> str:
        """
        解析出权重所在的本地目录，已缓存时**完全不联网**。

        hy3dgen 的 smart_load_model 会先探 $HY3DGEN_MODELS（默认 ~/.cache/hy3dgen，
        在系统盘上），探不到才回落 snapshot_download —— 那条路每次加载都要联网校验，
        走 hf-mirror 时能把一次请求拖到几分钟。
        它内部是 os.path.join(base_dir, model_path, subfolder)，而 os.path.join
        碰到绝对路径会丢弃前面的部分，所以把 snapshot 目录当 model_path 传进去，
        就直接落到本地文件上，既不碰系统盘也不发请求。
        """
        from huggingface_hub import hf_hub_download

        rel = f"{self.subfolder}/model.fp16.safetensors"
        try:
            path = hf_hub_download(repo_id=self.model_id, filename=rel,
                                   local_files_only=True)
        except Exception:
            return self.model_id  # 没下过，交给 hy3dgen 自己去下（走 HF_HOME）
        # .../snapshots/<rev>/<subfolder>/model.fp16.safetensors → .../snapshots/<rev>
        return os.path.dirname(os.path.dirname(path))

    def load(self):
        if self.pipeline is not None:
            return self.pipeline
        with self._lock:
            if self.pipeline is not None:
                return self.pipeline
            from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

            t0 = time.time()
            pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                self._resolve_weights(), subfolder=self.subfolder, variant="fp16"
            )
            try:
                # FlashVDM：官方的快速体素解码，turbo 变体配套用
                pipe.enable_flashvdm()
            except Exception as exc:  # 没有也能跑，只是慢一些
                print(f"[Hunyuan] FlashVDM 不可用（{type(exc).__name__}），用常规解码")
            self.pipeline = pipe
            self.load_seconds = round(time.time() - t0, 2)
            return self.pipeline

    def unload(self):
        """踢出显存。8GB 卡上和 MoGe / TripoSR 同时驻留必 OOM，切模式时由 app.py 调用。"""
        with self._lock:
            if self.pipeline is None:
                return False
            self.pipeline = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return True

    # ---------------- 信息 ----------------

    def weights_cached(self) -> bool:
        """权重（约 4.9GB）是否已在本地。只查缓存，不发网络请求。"""
        from huggingface_hub import hf_hub_download

        try:
            hf_hub_download(
                repo_id=self.model_id,
                filename=f"{self.subfolder}/model.fp16.safetensors",
                local_files_only=True,
            )
            return True
        except Exception:
            return False

    def info(self) -> dict:
        d = {
            "ok": self.load_error is None,
            "kind": "generative-mv",
            "model": f"{self.model_id}/{self.subfolder}",
            "device": str(self.device),
            "loaded": self.pipeline is not None,
            "weightsCached": self.pipeline is not None or self.weights_cached(),
            "views": list(VIEW_DIRS),
            "torch": torch.__version__,
        }
        if self.load_error:
            d["error"] = self.load_error
        if self.device.type == "cuda" and torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            d["gpu"] = props.name
            d["vram"] = f"{props.total_memory / 1024**3:.1f} GB"
        return d

    # ---------------- 着色 ----------------

    @staticmethod
    def _sample_colors(points: np.ndarray, normals: np.ndarray,
                       views: list[_View]) -> np.ndarray:
        """
        按法线朝向从各视图采样颜色，相邻视图之间做加权混合。

        每个点对每个视图算出"正对程度" n·d，以 n·d 的 BLEND_POWER 次方为权重
        把各视图采到的颜色加权平均。指数取得高，所以绝大部分点实际上就等于
        "用最正对的那个视图"，只有在两个视图的交界带上才真正混合 ——
        既避免硬接缝，又把投影对不齐可能带来的重影限制在很窄的一条带里。

        投影用的是**轮廓包围盒对齐**：把网格在该视角下的投影包围盒线性映射到
        该图的前景包围盒。这样不用去猜模型内部的归一化/留边参数，
        两边的剪影直接对齐，天然自洽。

        两级兜底。按面积加权实测（角色样例）：三视图无覆盖 0.00%、掠射 22.4%；
        四视图无覆盖 0.00%、掠射 10.3%。所以缺的从来不是"没有视图看得见"，
        而是掠射角下投影容易落到剪影外被 alpha 挡掉：
          1. 落在剪影外的点，放宽符号限制再取最接近的视图碰一次
          2. 仍然没着上的，取三维空间里最近的已着色点
        """
        n_pts = len(points)
        BLEND_POWER = 4.0
        ALPHA_MIN = 8

        acc = np.zeros((n_pts, 3), dtype=np.float64)
        wsum = np.zeros(n_pts, dtype=np.float64)
        facing = np.stack([normals @ v.dir for v in views], axis=1)  # (N, V)

        def project(view: _View):
            """点 → 该视图的像素坐标，以及是否落在前景剪影内。"""
            right, up = _screen_basis(view.dir)
            sx = points[:, :3] @ right
            sy = points[:, :3] @ up
            x0, y0, x1, y1 = view.box
            px = x0 + (sx - sx.min()) / (sx.max() - sx.min() + 1e-9) * (x1 - x0)
            # 图像 v 轴向下，世界 y 向上
            py = y1 - (sy - sy.min()) / (sy.max() - sy.min() + 1e-9) * (y1 - y0)
            px = np.clip(px.astype(int), 0, view.rgb.shape[1] - 1)
            py = np.clip(py.astype(int), 0, view.rgb.shape[0] - 1)
            return px, py, view.alpha[py, px] > ALPHA_MIN

        projected = []
        for k, view in enumerate(views):
            px, py, hit = project(view)
            projected.append((px, py, hit))
            # 背对该视图的点不能采，否则会把正面的颜色贴到背面去
            w = np.clip(facing[:, k], 0.0, None) ** BLEND_POWER * hit
            acc += w[:, None] * view.rgb[py, px]
            wsum += w

        filled = wsum > 1e-12
        colors = np.full((n_pts, 3), 200, dtype=np.uint8)
        colors[filled] = np.clip(
            acc[filled] / wsum[filled][:, None], 0, 255
        ).astype(np.uint8)

        # 兜底一：掠射角下投影落到剪影外的点，放宽符号取最接近的视图
        todo = ~filled
        if todo.any():
            best = np.argmax(facing, axis=1)
            for k, view in enumerate(views):
                sel = np.flatnonzero(todo & (best == k))
                if len(sel) == 0:
                    continue
                px, py, hit = projected[k]
                good = sel[hit[sel]]
                colors[good] = view.rgb[py[good], px[good]]
                filled[good] = True

        # 兜底二：仍然没着上的，取空间最近的已着色点
        if not filled.all() and filled.any():
            from scipy.spatial import cKDTree

            tree = cKDTree(points[filled])
            _, idx = tree.query(points[~filled], k=1)
            colors[~filled] = colors[filled][idx]

        return colors

    # ---------------- 生成 ----------------

    def generate(self, images: dict[str, bytes], n_points: int = DEFAULT_POINTS,
                 octree_resolution: int = DEFAULT_OCTREE,
                 steps: int = DEFAULT_STEPS) -> dict:
        import trimesh

        pipe = self.load()
        n_points = max(10_000, min(int(n_points), MAX_POINTS))

        if "front" not in images:
            raise RuntimeError("至少要提供正面图（front）")
        unknown = set(images) - set(VIEW_DIRS)
        if unknown:
            raise RuntimeError(f"不认识的视图名：{sorted(unknown)}，只支持 {list(VIEW_DIRS)}")

        t0 = time.time()

        # 1) 抠背景。模型按"居中的单个物体"训练，带背景效果明显变差
        cut: dict[str, Image.Image] = {}
        warnings: list[str] = []
        for name, raw in images.items():
            img = Image.open(io.BytesIO(raw))
            cut[name] = bg_removal.cutout(img)

            # 体检：这两种输入会让模型把整个画面当物体，生成出来是一块板
            st = bg_removal.foreground_stats(cut[name])
            if st["empty"]:
                warnings.append(f"「{name}」抠图后没有剩下任何主体，这张多半没法用")
            elif st["coverage"] > 0.92:
                warnings.append(
                    f"「{name}」主体占了画面 {st['coverage']*100:.0f}%，"
                    "等于没抠掉背景 —— 生成结果会是一块板，建议换一张主体周围留白的图"
                )
            elif st["aspect"] > 2.2:
                warnings.append(
                    f"「{name}」前景宽高比 {st['aspect']:.1f}，太扁了 —— "
                    "如果这是一张把三个视图拼在一起的图，请拆成三张分别上传"
                )

        # 传给模型时按 front/left/back/right 的固定顺序，别依赖 dict 的插入序
        ordered = {k: cut[k] for k in VIEW_DIRS if k in cut}

        # 2) 多视图 → 网格
        with self._lock, torch.no_grad():
            mesh = None
            last_oom = None
            for res in (octree_resolution, *FALLBACK_OCTREE):
                if res > octree_resolution:
                    continue
                try:
                    mesh = pipe(
                        image=ordered,
                        num_inference_steps=steps,
                        octree_resolution=res,
                        num_chunks=20000,
                        generator=torch.manual_seed(12345),
                        output_type="trimesh",
                    )[0]
                    used_octree = res
                    break
                except torch.cuda.OutOfMemoryError as exc:
                    last_oom = exc
                    torch.cuda.empty_cache()
                    continue

            if mesh is None:
                raise RuntimeError(
                    f"显存不足，降到 {FALLBACK_OCTREE[-1]} 分辨率仍失败。"
                    "关掉占显存的程序（浏览器/剪辑软件）后重试"
                ) from last_oom

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 3) 表面采样。法线取所在面的法线：既用来决定从哪张图取色，
        #    也随点云一起送到前端做光照 —— 没有法线就没有明暗，形体看着是平的
        samples, face_idx = trimesh.sample.sample_surface(mesh, n_points)
        normals = surface.sample_normals(mesh, face_idx)

        views = [_View(name, img) for name, img in ordered.items()]
        colors = self._sample_colors(np.asarray(samples), normals, views)
        ao = surface.estimate_ao(np.asarray(samples), normals)

        # 4) Hunyuan3D 是 Y 朝上、正面 +Z，与 three.js 一致，不用换轴
        pts = np.asarray(samples, dtype=np.float32)

        ms = int((time.time() - t0) * 1000)

        return {
            "positions": np.ascontiguousarray(pts, dtype=np.float32),
            "colors": np.ascontiguousarray(colors, dtype=np.uint8),
            "normals": np.ascontiguousarray(surface.pack_normals(normals)),
            "ao": np.ascontiguousarray(ao),
            "count": int(len(pts)),
            "resolution": used_octree,
            "viewsUsed": list(ordered),
            "warnings": warnings,
            "ms": ms,
            "model": f"{self.model_id}/{self.subfolder}",
            "device": str(self.device),
        }
