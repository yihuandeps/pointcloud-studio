"""
点云工具 · 高精度模式后端

启动：
    server\\start.ps1
或：
    uvicorn app:app --host 127.0.0.1 --port 8000

前端通过 Vite 的 /api 代理访问（见 web/vite.config.js）。
"""

from __future__ import annotations

import os
import traceback

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from hunyuan_runner import HunyuanRunner
from imaging import pack_cloud, pack_result
from moge_runner import DEFAULT_MAX_SIDE, MoGeRunner
from tripo_runner import DEFAULT_POINTS, TripoRunner

app = FastAPI(title="PointCloud Studio · MoGe backend", version="0.1.0")

# 直接开 dev server（不走 Vite 代理）时也能调
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

runner = MoGeRunner()
gen_runner = TripoRunner()
mv_runner = HunyuanRunner()

_RUNNERS = {"depth": ("MoGe", runner), "gen": ("TripoSR", gen_runner),
            "mv": ("Hunyuan3D-2mv", mv_runner)}


def _switch_to(mode: str):
    """
    显存互斥：MoGe 1.5GB、TripoSR 1.5GB、Hunyuan3D-2mv 峰值 5.4GB，
    8GB 的卡还要和桌面应用抢，两个一起驻留必 OOM。切模式时把其余的踢出去。
    切回来会重新加载（权重已缓存时 7–15 秒），比推理途中崩掉强。
    """
    for key, (name, r) in _RUNNERS.items():
        if key != mode and r.unload():
            print(f"[显存] 已卸载 {name} 腾出空间")


@app.get("/api/health")
def health():
    return runner.info()


@app.post("/api/warmup")
def warmup():
    """提前把模型加载进显存，避免第一张图等太久。"""
    try:
        _switch_to("depth")
        runner.load()
        return runner.info()
    except Exception as exc:
        runner.load_error = str(exc)
        raise HTTPException(status_code=500, detail=f"模型加载失败：{exc}") from exc


@app.post("/api/infer")
def infer(image: UploadFile = File(...), max_side: str = Form(default="")):
    """
    返回紧凑二进制（几百万浮点走 JSON 又慢又占内存）：

        [0..3]      uint32 LE   头部 JSON 字节长度 L
        [4..4+L)    UTF-8 JSON  { width, height, hasMask, intrinsics, model, device, ms }
        接着        float32[W*H*3]  点图（OpenCV 坐标系，米）
        再接着      uint8[W*H]      有效性 mask

    布局必须与 web/src/core/depthServer.js 的解析逻辑一致。
    """
    try:
        raw = image.file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="收到空文件")

        try:
            side = int(max_side) if max_side.strip() else DEFAULT_MAX_SIDE
        except ValueError:
            side = DEFAULT_MAX_SIDE

        _switch_to("depth")
        result = runner.infer(raw, max_side=side)

        body = pack_result(
            result["points"].tobytes(),
            result["mask"].tobytes(),
            {
                "width": result["width"],
                "height": result["height"],
                "hasMask": True,
                "intrinsics": result["intrinsics"],
                "model": result["model"],
                "device": result["device"],
                "ms": result["ms"],
            },
        )
        return Response(content=body, media_type="application/octet-stream")

    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"推理失败：{exc}") from exc


#: 单图模式用哪个引擎。
#:
#: 默认 Hunyuan（只喂正面）。同一张图并排渲染过：TripoSR 的五官是涂抹的、
#: 发丝一片噪点；Hunyuan 干净得多。TripoSR 是 2024 年初的模型，差距是代际的。
#: 代价是慢一些（约 12s vs 3–6s）、显存多用一些（5.4GB vs 1.5GB）——
#: 显存吃紧或要快时设 SINGLE_ENGINE=triposr 换回去。
SINGLE_ENGINE = os.environ.get("SINGLE_ENGINE", "hunyuan").lower()
_SINGLE_IS_HUNYUAN = SINGLE_ENGINE != "triposr"
_SINGLE_KEY = "mv" if _SINGLE_IS_HUNYUAN else "gen"


def _single_runner():
    return mv_runner if _SINGLE_IS_HUNYUAN else gen_runner


@app.get("/api/gen/health")
def gen_health():
    d = _single_runner().info()
    d["engine"] = "hunyuan-front-only" if _SINGLE_IS_HUNYUAN else "triposr"
    return d


@app.post("/api/gen/warmup")
def gen_warmup():
    """提前把单图模式的模型加载进显存。"""
    try:
        _switch_to(_SINGLE_KEY)
        _single_runner().load()
        return gen_health()
    except Exception as exc:
        _single_runner().load_error = str(exc)
        raise HTTPException(status_code=500, detail=f"模型加载失败：{exc}") from exc


@app.post("/api/generate")
def generate(image: UploadFile = File(...), points: str = Form(default="")):
    """
    生成式 3D：单图 → 完整 360° 点云（背面由模型补全）。

    默认走 Hunyuan3D-2mv 的「只给正面」路径 —— 它本来就接受 1–4 张视图，
    只给一张时退化成模型自己编背面，但几何和正面纹理都明显好过 TripoSR。

    返回 imaging.pack_cloud 约定的二进制，与多视图模式完全一致。
    """
    try:
        raw = image.file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="收到空文件")

        try:
            n = int(points) if points.strip() else DEFAULT_POINTS
        except ValueError:
            n = DEFAULT_POINTS

        _switch_to(_SINGLE_KEY)
        if _SINGLE_IS_HUNYUAN:
            result = mv_runner.generate({"front": raw}, n_points=n)
        else:
            result = gen_runner.generate(raw, n_points=n)

        head = {
            "count": result["count"],
            "model": result["model"],
            "device": result["device"],
            "resolution": result["resolution"],
            "engine": "hunyuan-front-only" if _SINGLE_IS_HUNYUAN else "triposr",
            "hasNormals": True,
            "hasAO": True,
            "ms": result["ms"],
        }
        # 走 Hunyuan 时它还会回报输入图的体检结果，一并带给前端
        if result.get("warnings"):
            head["warnings"] = result["warnings"]

        body = pack_cloud(
            result["positions"].tobytes(),
            result["colors"].tobytes(),
            head,
            result["normals"].tobytes(),
            result["ao"].tobytes(),
        )
        return Response(content=body, media_type="application/octet-stream")

    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"生成失败：{exc}") from exc


@app.get("/api/mv/health")
def mv_health():
    return mv_runner.info()


@app.post("/api/mv/warmup")
def mv_warmup():
    """提前把 Hunyuan3D-2mv 加载进显存（4.9GB 权重，冷加载约 15 秒）。"""
    try:
        _switch_to("mv")
        mv_runner.load()
        return mv_runner.info()
    except Exception as exc:
        mv_runner.load_error = str(exc)
        raise HTTPException(status_code=500, detail=f"模型加载失败：{exc}") from exc


@app.post("/api/mv/generate")
async def mv_generate(
    front: UploadFile = File(...),
    left: UploadFile | None = File(default=None),
    back: UploadFile | None = File(default=None),
    right: UploadFile | None = File(default=None),
    points: str = Form(default=""),
):
    """
    多视图生成式 3D：1–4 张视图 → 完整 360° 点云。

    正面必填，其余可选 —— 给得越多背面越忠实于你的原图，
    只给正面时退化成"模型自己编背面"，效果接近但优于 TripoSR。
    响应格式与 /api/generate 相同（imaging.pack_cloud）。
    """
    try:
        images: dict[str, bytes] = {}
        for name, up in (("front", front), ("left", left),
                         ("back", back), ("right", right)):
            if up is None:
                continue
            raw = await up.read()
            if raw:
                images[name] = raw

        if "front" not in images:
            raise HTTPException(status_code=400, detail="至少要上传正面图")

        try:
            n = int(points) if points.strip() else DEFAULT_POINTS
        except ValueError:
            n = DEFAULT_POINTS

        _switch_to("mv")
        result = mv_runner.generate(images, n_points=n)

        body = pack_cloud(
            result["positions"].tobytes(),
            result["colors"].tobytes(),
            {
                "count": result["count"],
                "model": result["model"],
                "device": result["device"],
                "resolution": result["resolution"],
                "viewsUsed": result["viewsUsed"],
                "warnings": result["warnings"],
                "hasNormals": True,
                "hasAO": True,
                "ms": result["ms"],
            },
            result["normals"].tobytes(),
            result["ao"].tobytes(),
        )
        return Response(content=body, media_type="application/octet-stream")

    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"生成失败：{exc}") from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "8000")),
    )
