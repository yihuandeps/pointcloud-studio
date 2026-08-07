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

from imaging import pack_result
from moge_runner import DEFAULT_MAX_SIDE, MoGeRunner

app = FastAPI(title="PointCloud Studio · MoGe backend", version="0.1.0")

# 直接开 dev server（不走 Vite 代理）时也能调
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

runner = MoGeRunner()


@app.get("/api/health")
def health():
    return runner.info()


@app.post("/api/warmup")
def warmup():
    """提前把模型加载进显存，避免第一张图等太久。"""
    try:
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "8000")),
    )
