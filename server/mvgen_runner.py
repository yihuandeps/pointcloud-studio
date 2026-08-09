"""
MV-Adapter 视图生成：一张图 → 一组互相一致的多视图。

这是给「只有一张图」的场景补上的一环。单图重建的根本困难是背面没有任何信息，
模型只能外推，转过去就糊。解法是先把缺的视角**生成**出来，再走多视图重建 ——
所以这里的产物直接喂给 HunyuanRunner。

为什么是 MV-Adapter 而不是别的：
  - 它一次生成**一组**互相对齐的视图，视图一致性是模型结构保证的。
    普通的文生图/图生图 API 做不到这点 —— 它会给你编出另一个角色，
    衣服细节和比例都对不上，拿去重建只会更糟。
  - 它的相机约定是 **仰角全 0、方位角含 0/90/180/270**，
    正好就是 Hunyuan3D-2mv 要的 front/left/back/right。
    （Zero123++ 因此被排除：它的方位角是 30/90/150/…、仰角在 +20/−10 间交替，对不上。）

用的是 SD2.1 变体而不是 SDXL：官方说 SDXL 版 image-to-multiview 要约 14GB 显存，
这台机器只有 8GB。SD2.1 版官方标注 <6GB，代价是质量「稍低一些」。
"""

from __future__ import annotations

import io
import os
import sys
import threading
import time
import types
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import bg_removal

ROOT = Path(__file__).parent
SRC = ROOT / ".cache" / "src" / "MV-Adapter"

BASE_MODEL = os.environ.get("MVGEN_BASE", "Manojb/stable-diffusion-2-1-base")
ADAPTER_REPO = os.environ.get("MVGEN_ADAPTER_REPO", "huanngzh/mv-adapter")
ADAPTER_FILE = os.environ.get("MVGEN_ADAPTER", "mvadapter_i2mv_sd21.safetensors")

#: 要生成的方位角。上游示例默认是六个 [0,45,90,180,270,315]，
#: 但视角数是可配的（脚本里 num_views = len(azimuth_deg)），而我们只需要这四个 ——
#: 它们正好是 Hunyuan3D-2mv 的 front/left/back/right。
#: 砍到 4 个还有个现实原因：6 个视角 × 512 分辨率、再被 CFG 翻倍，8GB 显存装不下。
AZIMUTH_DEG = [0, 90, 180, 270]
NUM_VIEWS = len(AZIMUTH_DEG)

#: MV-Adapter 的方位角 → Hunyuan3D-2mv 的视图名。
#: **这个映射是实测定的**，见 tools/verify_mvgen_mapping 的说明：
#: 拿官方样例图跑一遍，把生成的各方位角与样例自带的 left/back 真图逐一比对。
AZIMUTH_TO_VIEW = {0: "front", 90: "left", 180: "back", 270: "right"}

#: CPU 卸载默认**关闭**，别轻易打开。
#: 这条管线的参考条件是先跑一遍参考图、把各注意力层的 hidden states 按
#: 「模块名 → 张量」缓存起来，再在多视图那一遍里查表。accelerate 的卸载 hook 会
#: 改变模块路径，于是查表时报
#: KeyError: 'down_blocks.0.attentions.0.transformer_blocks.0.attn1.processor'。
#: 显存问题另有解法：不开注意力分块（走 SDPA）+ 下面的分辨率降级阶梯。
LOW_VRAM = os.environ.get("MVGEN_LOWVRAM", "0") != "0"

#: 生成分辨率。384 是 8GB 卡上的实际可用点：
#: 多视图注意力把各视角的 token 拼在一起算，序列长度 = 视角数 ×(px/8)²。
#: 512 时是 4×4096=16384，实测显存吃到 7.9/8.0GB，GPU 满载但一直在腾挪，慢到不可用；
#: 384 是 4×2304=9216，注意力开销约降到三分之一。
DEFAULT_SIZE = int(os.environ.get("MVGEN_SIZE", "384"))
DEFAULT_STEPS = int(os.environ.get("MVGEN_STEPS", "30"))
DEFAULT_GUIDANCE = float(os.environ.get("MVGEN_GUIDANCE", "3.0"))


def _stub_unavailable():
    """
    给 nvdiffrast / triton 塞占位模块。

    mvadapter 的 utils.mesh_utils 包在 __init__ 里把网格渲染那套一起导入了，
    而我们只用其中一个相机函数；nvdiffrast 要编译 CUDA 扩展，Windows 上装不上，
    triton 只有 Linux 版。两者在 i2mv 这条路上都不会被真正调用。

    占位模块必须带一个合法的 __spec__：diffusers 会去读 triton.__spec__ 判断可用性，
    裸 ModuleType 的 __spec__ 是 None，会让它直接抛 "triton.__spec__ is None"。
    """
    import importlib.machinery

    for name in ("nvdiffrast", "nvdiffrast.torch", "triton", "triton.language"):
        if name in sys.modules:
            continue
        mod = types.ModuleType(name)
        mod.__spec__ = importlib.machinery.ModuleSpec(name, None)
        sys.modules[name] = mod


class MvGenRunner:
    def __init__(self, device: str | None = None):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.pipe = None
        self._lock = threading.Lock()
        self.load_error: str | None = None
        self.load_seconds: float | None = None

    # ---------------- 加载 ----------------

    def load(self):
        if self.pipe is not None:
            return self.pipe
        with self._lock:
            if self.pipe is not None:
                return self.pipe
            if not (SRC / "mvadapter").is_dir():
                raise RuntimeError(
                    f"找不到 MV-Adapter 源码（{SRC}），重跑 setup.ps1 会拉下来"
                )
            if str(SRC) not in sys.path:
                sys.path.insert(0, str(SRC))
            _stub_unavailable()

            from diffusers import AutoencoderKL, DDPMScheduler
            from mvadapter.pipelines.pipeline_mvadapter_i2mv_sd import (
                MVAdapterI2MVSDPipeline,
            )
            from mvadapter.schedulers.scheduling_shift_snr import ShiftSNRScheduler

            t0 = time.time()
            pipe = MVAdapterI2MVSDPipeline.from_pretrained(
                BASE_MODEL, torch_dtype=torch.float16, variant="fp16",
                safety_checker=None, requires_safety_checker=False,
            )
            pipe.scheduler = ShiftSNRScheduler.from_scheduler(
                pipe.scheduler, shift_mode="interpolated", shift_scale=8.0,
                scheduler_class=DDPMScheduler,
            )
            pipe.init_custom_adapter(num_views=NUM_VIEWS)
            pipe.load_custom_adapter(ADAPTER_REPO, weight_name=ADAPTER_FILE)
            pipe.set_progress_bar_config(disable=True)

            # 只开 VAE 分块解码（逐张解，省显存且不改结果）。
            #
            # **千万别开 enable_attention_slicing()**：它会把注意力切回
            # get_attention_scores 那条老路径，而那条路径要把完整的注意力矩阵
            # 展开成实体张量。多视图注意力是把各视角的 token 拼在一起算的，
            # 序列长度是单视角的 4 倍，实测直接要 39GB。
            # 默认的 SDPA 从不展开这个矩阵，反而省得多。
            try:
                pipe.enable_vae_slicing()
            except Exception:
                pass

            # 8GB 卡上整条管线放不下：即便砍到 4 视角、320 分辨率、开了分块，
            # 参考条件那一步仍会 OOM（实测剩 2.88GB 时就爆）。
            # CPU 卸载让 UNet / VAE / 文本编码器按需在显存与内存之间搬，
            # 峰值降到 3GB 上下，代价是慢一些 —— 对一次几十秒的生成可以接受。
            # 注意：开了卸载就不能再 pipe.to(device)，那会把 accelerate 的 hook 打乱。
            # 先只转精度、不搬设备：适配器是 from_pretrained 之后才加载的，
            # 它的权重还是 fp32，不统一会在第一个矩阵乘上报
            # "expected mat1 and mat2 to have the same dtype: Half != float"。
            pipe.to(dtype=torch.float16)

            if LOW_VRAM:
                pipe.enable_model_cpu_offload(device=str(self.device))
                # cond_encoder 不在 accelerate 托管的组件表里，手动放上去（很小）
                pipe.cond_encoder.to(device=self.device, dtype=torch.float16)
            else:
                pipe.to(device=self.device)
                pipe.cond_encoder.to(device=self.device, dtype=torch.float16)

            self.pipe = pipe
            self.load_seconds = round(time.time() - t0, 2)
            return self.pipe

    def unload(self):
        """踢出显存。和 Hunyuan 加起来远超 8GB，两者必须串行。"""
        with self._lock:
            if self.pipe is None:
                return False
            self.pipe = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return True

    # ---------------- 信息 ----------------

    def weights_cached(self) -> bool:
        from huggingface_hub import hf_hub_download

        try:
            hf_hub_download(repo_id=ADAPTER_REPO, filename=ADAPTER_FILE,
                            local_files_only=True)
            hf_hub_download(repo_id=BASE_MODEL, filename="model_index.json",
                            local_files_only=True)
            return True
        except Exception:
            return False

    def info(self) -> dict:
        d = {
            "ok": self.load_error is None,
            "kind": "view-generator",
            "model": f"{BASE_MODEL} + {ADAPTER_FILE}",
            "device": str(self.device),
            "loaded": self.pipe is not None,
            "weightsCached": self.pipe is not None or self.weights_cached(),
            "azimuths": AZIMUTH_DEG,
            "produces": list(AZIMUTH_TO_VIEW.values()),
            "sourceReady": (SRC / "mvadapter").is_dir(),
        }
        if self.load_error:
            d["error"] = self.load_error
        return d

    # ---------------- 生成 ----------------

    def generate_views(self, image_bytes: bytes, size: int = DEFAULT_SIZE,
                       steps: int = DEFAULT_STEPS,
                       guidance: float = DEFAULT_GUIDANCE,
                       seed: int = 42) -> dict:
        """
        单张图 → { 'front': PIL, 'left': PIL, 'back': PIL, 'right': PIL }。

        返回的图都是 RGBA、背景已抠掉，可以直接交给 HunyuanRunner.generate。
        """
        pipe = self.load()

        from mvadapter.utils.geometry import get_plucker_embeds_from_cameras_ortho
        from mvadapter.utils.mesh_utils import get_orthogonal_camera

        # 参考图先抠背景：模型按居中的单个物体训练，带背景生成质量会明显下滑
        ref = bg_removal.cutout(Image.open(io.BytesIO(image_bytes)))
        ref = self._fit(ref, size)

        cameras = get_orthogonal_camera(
            elevation_deg=[0] * NUM_VIEWS,
            distance=[1.8] * NUM_VIEWS,
            left=-0.55, right=0.55, bottom=-0.55, top=0.55,
            # 上游脚本里就是减 90，保持一致
            azimuth_deg=[a - 90 for a in AZIMUTH_DEG],
            device=str(self.device),
        )
        self._plucker = get_plucker_embeds_from_cameras_ortho

        t0 = time.time()
        with self._lock, torch.no_grad():
            out = None
            last_oom = None
            # 显存不够就降分辨率重试。桌面应用会占掉 1–2GB，同样的尺寸有时成有时不成
            for px in (size, 448, 384, 320):
                if px > size:
                    continue
                try:
                    out = pipe(
                        "high quality",
                        height=px, width=px,
                        num_inference_steps=steps,
                        guidance_scale=guidance,
                        num_images_per_prompt=NUM_VIEWS,
                        control_image=self._control(cameras, px),
                        control_conditioning_scale=1.0,
                        reference_image=self._fit(ref, px),
                        reference_conditioning_scale=1.0,
                        negative_prompt=(
                            "watermark, ugly, deformed, noisy, blurry, low contrast"
                        ),
                        generator=torch.Generator(self.device).manual_seed(seed),
                    ).images
                    used_size = px
                    break
                except torch.cuda.OutOfMemoryError as exc:
                    last_oom = exc
                    torch.cuda.empty_cache()
                    continue

            if out is None:
                raise RuntimeError(
                    "显存不足，降到 320 分辨率仍失败。"
                    "关掉占显存的程序（浏览器/剪辑软件）后重试"
                ) from last_oom
        ms = int((time.time() - t0) * 1000)

        views: dict[str, Image.Image] = {}
        for az, img in zip(AZIMUTH_DEG, out):
            name = AZIMUTH_TO_VIEW.get(az)
            if name:
                views[name] = bg_removal.cutout(img)

        return {"views": views, "all": dict(zip(AZIMUTH_DEG, out)),
                "size": used_size, "ms": ms}

    def _control(self, cameras, px: int):
        """相机 → plücker 嵌入，归一化到 [0,1] 作为条件图。分辨率随重试变化。"""
        plucker = self._plucker(cameras.c2w, [1.1] * NUM_VIEWS, px)
        return ((plucker + 1.0) / 2.0).clamp(0, 1)

    @staticmethod
    def _fit(img: Image.Image, size: int) -> Image.Image:
        """等比缩放并居中放进方形画布，留一点边 —— 与上游预处理一致。"""
        img = img.convert("RGBA")
        w, h = img.size
        if w > h:
            nw, nh = int(size * 0.9), int(h * (size * 0.9) / w)
        else:
            nh, nw = int(size * 0.9), int(w * (size * 0.9) / h)
        img = img.resize((max(1, nw), max(1, nh)), Image.LANCZOS)
        canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        canvas.paste(img, ((size - nw) // 2, (size - nh) // 2), img)
        return canvas
