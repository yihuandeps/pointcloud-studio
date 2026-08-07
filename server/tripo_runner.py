"""
TripoSR 推理封装（生成式 3D 模式）。

与 MoGe 的本质区别：MoGe 是"测量"——把照片里看得见的表面变成点；
TripoSR 是"生成"——依据单张图推断完整的 360° 形体，包括照片里看不见的背面。
背面是模型基于先验的合理想象，不保证与实物一致。

流程：抠除背景（rembg）→ TripoSR 得到 triplane 隐式场 →
      marching cubes 提取带顶点色的网格 → 表面均匀采样成点云。

vendored 的 tsr/ 相对上游改了一处：torchmcubes（需编译）换成 scikit-image，
见 tsr/models/isosurface.py。
"""

from __future__ import annotations

import io
import os
import threading
import time

import numpy as np
import torch
from PIL import Image

DEFAULT_MODEL = os.environ.get("TRIPO_MODEL", "stabilityai/TripoSR")
# marching cubes 的体素分辨率。256³ 要查询 1677 万个点，8GB 卡上（还要和桌面
# 应用抢显存）很容易 OOM，实测 192 是画质与稳定性的平衡点；OOM 时按 FALLBACK
# 逐级降级重试，见 generate()
DEFAULT_RESOLUTION = int(os.environ.get("TRIPO_MC_RES", "192"))
FALLBACK_RESOLUTIONS = (160, 128, 96)
# 网格面数上限。喂进噪声图之类的输入时密度场会极其杂乱，
# 生成数千万面的垃圾网格，能把显存和内存一起撑爆
MAX_FACES = 4_000_000
# 一次采样的点数上限比前端滑杆最大值（80 万）略高，前端在本地抽稀，改点数不用重跑模型
DEFAULT_POINTS = int(os.environ.get("TRIPO_POINTS", "600000"))
MAX_POINTS = 1_000_000


class TripoRunner:
    def __init__(self, model_id: str = DEFAULT_MODEL, device: str | None = None,
                 chunk_size: int = 8192):
        self.model_id = model_id
        self.chunk_size = chunk_size  # triplane 查询分块，防止 256³ 网格一次性撑爆显存
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = None
        self.rembg_session = None
        self._lock = threading.Lock()
        self.load_error: str | None = None

    # ---------------- 加载 ----------------

    def load(self):
        if self.model is not None:
            return self.model
        with self._lock:
            if self.model is not None:
                return self.model
            from tsr.system import TSR

            t0 = time.time()
            model = TSR.from_pretrained(
                self.model_id, config_name="config.yaml", weight_name="model.ckpt"
            )
            model.renderer.set_chunk_size(self.chunk_size)
            model.to(self.device)
            self.model = model
            self.load_seconds = round(time.time() - t0, 2)
            return self.model

    def unload(self):
        """把模型踢出显存。8GB 卡上和 MoGe 同时驻留会 OOM，切模式时由 app.py 调用。"""
        with self._lock:
            if self.model is None:
                return False
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return True

    def _rembg(self):
        if self.rembg_session is None:
            import rembg

            self.rembg_session = rembg.new_session("u2net")
        return self.rembg_session

    # ---------------- 信息 ----------------

    def info(self) -> dict:
        d = {
            "ok": self.load_error is None,
            "kind": "generative",
            "model": self.model_id,
            "device": str(self.device),
            "loaded": self.model is not None,
            "torch": torch.__version__,
        }
        if self.load_error:
            d["error"] = self.load_error
        if self.device.type == "cuda" and torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            d["gpu"] = props.name
            d["vram"] = f"{props.total_memory / 1024**3:.1f} GB"
        return d

    # ---------------- 生成 ----------------

    def generate(self, image_bytes: bytes, n_points: int = DEFAULT_POINTS,
                 resolution: int = DEFAULT_RESOLUTION) -> dict:
        import trimesh
        from tsr.utils import remove_background, resize_foreground

        model = self.load()
        n_points = max(10_000, min(int(n_points), MAX_POINTS))

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        t0 = time.time()

        # 1) 抠背景。TripoSR 按"居中的单个物体"训练，带背景直接喂效果很差
        img = remove_background(img, self._rembg())
        img = resize_foreground(img, 0.85)
        # 透明背景合成到中灰（与官方 run.py 一致，模型就是这么训练的）
        arr = np.array(img).astype(np.float32) / 255.0
        arr = arr[:, :, :3] * arr[:, :, 3:4] + (1 - arr[:, :, 3:4]) * 0.5
        img = Image.fromarray((arr * 255.0).astype(np.uint8))

        # 2) 隐式场 → 网格（带顶点色）
        with self._lock, torch.no_grad():
            scene_codes = model([img], device=str(self.device))

            # 显存不够就降分辨率重试。这台机器上桌面应用会占掉 1–2GB，
            # 同样的分辨率有时成有时不成，硬失败对用户毫无意义
            mesh = None
            last_oom = None
            for res in (resolution, *FALLBACK_RESOLUTIONS):
                if res > resolution:
                    continue
                try:
                    mesh = model.extract_mesh(scene_codes, True, resolution=res)[0]
                    used_resolution = res
                    break
                except torch.cuda.OutOfMemoryError as exc:
                    last_oom = exc
                    torch.cuda.empty_cache()
                    continue
                except ValueError as exc:
                    # skimage 的 marching_cubes 在密度场没有穿过阈值面时会抛 ValueError
                    raise RuntimeError(
                        "模型没有生成有效形体——试试主体更清晰、背景更干净的图片"
                    ) from exc

            if mesh is None:
                raise RuntimeError(
                    f"显存不足，降到 {FALLBACK_RESOLUTIONS[-1]} 分辨率仍失败。"
                    "关掉占显存的程序（浏览器/剪辑软件）后重试"
                ) from last_oom

            del scene_codes
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        if len(mesh.faces) > MAX_FACES:
            raise RuntimeError(
                f"生成的网格异常复杂（{len(mesh.faces):,} 面）——"
                "输入多半不是清晰的单个物体，换一张主体明确的图片试试"
            )

        # 3) 表面均匀采样 + 顶点色的重心坐标插值
        samples, face_idx = trimesh.sample.sample_surface(mesh, n_points)
        bary = trimesh.triangles.points_to_barycentric(
            mesh.triangles[face_idx], samples
        )
        vc = np.asarray(mesh.visual.vertex_colors)[:, :3].astype(np.float32)
        cols = (vc[mesh.faces[face_idx]] * bary[:, :, None]).sum(axis=1)

        # 4) TSR 世界是 z 朝上、条件相机在 +x 轴；three.js 是 y 朝上、相机看向 -Z。
        #    (x,y,z) → (y,z,x) 是行列式 +1 的纯旋转，且让生成物正面朝向初始视角
        pts = samples[:, [1, 2, 0]].astype(np.float32)

        ms = int((time.time() - t0) * 1000)

        return {
            "positions": np.ascontiguousarray(pts, dtype=np.float32),
            "colors": np.ascontiguousarray(
                np.clip(cols, 0, 255).astype(np.uint8)
            ),
            "count": int(len(pts)),
            "resolution": used_resolution,
            "ms": ms,
            "model": self.model_id,
            "device": str(self.device),
        }
