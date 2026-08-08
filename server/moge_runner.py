"""
MoGe-2 推理封装。

相比浏览器端的 Depth Anything，MoGe 的优势在于：
  - 直接输出米制尺度的点图 points (H,W,3)，不需要视差→深度的转换
  - 自己预测相机内参，不需要用户猜 FOV
  - 自带有效性 mask，天空/无效区域直接标出来，不用靠梯度阈值硬剔
"""

from __future__ import annotations

import io
import os
import threading
import time

import numpy as np
import torch
from PIL import Image

from imaging import fit_size, js_round  # noqa: F401  (js_round 供外部复用)

# 可用变体（显存 8GB 跑 vitl 绰绰有余）：
#   Ruicheng/moge-2-vitl          326M  米制尺度
#   Ruicheng/moge-2-vitl-normal   331M  额外输出法线
#   Ruicheng/moge-2-vitb-normal   104M
#   Ruicheng/moge-2-vits-normal    35M
DEFAULT_MODEL = os.environ.get("MOGE_MODEL", "Ruicheng/moge-2-vitl")
DEFAULT_MAX_SIDE = int(os.environ.get("MOGE_MAX_SIDE", "1024"))


class MoGeRunner:
    def __init__(self, model_id: str = DEFAULT_MODEL, device: str | None = None, fp16: bool = True):
        self.model_id = model_id
        self.fp16 = fp16
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = None
        self._lock = threading.Lock()  # GPU 推理串行化
        self.load_error: str | None = None

    # ---------------- 加载 ----------------

    def load(self):
        if self.model is not None:
            return self.model
        with self._lock:
            if self.model is not None:
                return self.model
            # 延迟到这里 import，让没装依赖时 /api/health 仍能返回可读的错误
            from moge.model.v2 import MoGeModel

            t0 = time.time()
            # 已缓存就别联网校验。from_pretrained 把多余的 kwargs 透传给
            # hf_hub_download，走 hf-mirror 时这一次 HEAD 要几十秒，
            # 而 app.py 的显存互斥会在每次切模式时重新加载，代价要反复付。
            try:
                model = MoGeModel.from_pretrained(self.model_id, local_files_only=True)
            except Exception:
                model = MoGeModel.from_pretrained(self.model_id)
            model = model.to(self.device).eval()
            self.model = model
            self.load_seconds = round(time.time() - t0, 2)
            return self.model

    def unload(self):
        """
        把模型踢出显存。8GB 卡上 MoGe 和 TripoSR 同时驻留会 OOM，
        app.py 在两种模式间切换时调这个（见 _switch_to）。
        """
        with self._lock:
            if self.model is None:
                return False
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return True

    # ---------------- 信息 ----------------

    def weights_cached(self) -> bool:
        """
        权重是否已在本地。没有的话首次使用要下 1.2GB，而 hf-mirror 不通时
        请求会挂在重试里十几分钟且毫无反馈 —— 前端据此提前给出提示。
        只查本地缓存，不发网络请求。
        """
        from huggingface_hub import hf_hub_download

        try:
            hf_hub_download(
                repo_id=self.model_id, filename="model.pt", local_files_only=True
            )
            return True
        except Exception:
            return False

    def info(self) -> dict:
        d = {
            "ok": self.load_error is None,
            "model": self.model_id,
            "device": str(self.device),
            "loaded": self.model is not None,
            "weightsCached": self.model is not None or self.weights_cached(),
            "fp16": self.fp16,
            "torch": torch.__version__,
        }
        if self.load_error:
            d["error"] = self.load_error
        if self.device.type == "cuda" and torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            d["gpu"] = props.name
            d["vram"] = f"{props.total_memory / 1024**3:.1f} GB"
        return d

    # ---------------- 推理 ----------------

    def infer(self, image_bytes: bytes, max_side: int = DEFAULT_MAX_SIDE) -> dict:
        model = self.load()

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        w, h = fit_size(img.width, img.height, max_side)
        if (w, h) != (img.width, img.height):
            img = img.resize((w, h), Image.LANCZOS)

        rgb = np.asarray(img, dtype=np.uint8)
        tensor = (
            torch.from_numpy(rgb).to(self.device).permute(2, 0, 1).float() / 255.0
        )

        t0 = time.time()
        with self._lock, torch.no_grad():
            out = model.infer(tensor, use_fp16=self.fp16)
        ms = int((time.time() - t0) * 1000)

        points = out["points"].detach().cpu().numpy().astype(np.float32)  # (H,W,3)
        mask = out["mask"].detach().cpu().numpy()                          # (H,W) bool
        intr = out.get("intrinsics")
        intrinsics = intr.detach().cpu().numpy().tolist() if intr is not None else None

        # 理论上 MoGe 输出与输入同尺寸；万一不是，强行对齐，否则前端的颜色会错位
        if points.shape[0] != h or points.shape[1] != w:
            import cv2

            points = cv2.resize(points, (w, h), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(
                mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
            ).astype(bool)

        # 无效区域可能是 inf/nan，清成 0 由 mask 兜底，避免前端拿到脏数据
        bad = ~np.isfinite(points).all(axis=2)
        if bad.any():
            points[bad] = 0.0
            mask = mask & ~bad

        return {
            "points": np.ascontiguousarray(points, dtype=np.float32),
            "mask": np.ascontiguousarray(mask.astype(np.uint8)),
            "width": w,
            "height": h,
            "intrinsics": intrinsics,
            "ms": ms,
            "model": self.model_id,
            "device": str(self.device),
        }
