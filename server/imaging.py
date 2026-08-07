"""
纯函数：尺寸计算与二进制打包。

刻意不 import torch —— 这样这些函数在没装深度学习依赖的环境里也能被测试，
而它们恰恰是最容易和前端对不上、且出错后症状最诡异的部分。
"""

from __future__ import annotations

import json
import math
import struct


def js_round(x: float) -> int:
    """
    复刻 JavaScript 的 Math.round（.5 一律向上取）。

    Python 内置 round() 是银行家舍入：round(2.5) == 2、round(0.5) == 0，
    而 JS 里分别是 3 和 1。前后端必须算出完全一样的图像尺寸，
    否则点图和颜色会整体错位一行/一列，画面看着"糊了"却查不出原因。
    """
    return int(math.floor(x + 0.5))


def fit_size(w: int, h: int, max_side: int) -> tuple[int, int]:
    """与前端 web/src/core/imagePrep.js 的 fitSize() 逐位对齐。"""
    if max_side <= 0:
        return w, h
    longest = max(w, h)
    if longest <= max_side:
        return w, h
    k = max_side / longest
    return max(1, js_round(w * k)), max(1, js_round(h * k))


def pack_cloud(positions_bytes: bytes, colors_bytes: bytes, header: dict) -> bytes:
    """
    生成式 3D 模式的点云包，与 pack_result 同族的紧凑二进制：

        [0..3]      uint32 LE   头部 JSON 字节长度 L
        [4..4+L)    UTF-8 JSON  { count, model, device, ms }
        接着        float32[count*3]  XYZ（three.js 坐标系，y 朝上）
        再接着      uint8[count*3]    RGB

    布局必须与 web/src/core/gen3dServer.js 的解析逻辑一致。
    """
    head = json.dumps(header, ensure_ascii=False).encode("utf-8")
    return b"".join(
        [struct.pack("<I", len(head)), head, positions_bytes, colors_bytes]
    )


def pack_result(points_bytes: bytes, mask_bytes: bytes, header: dict) -> bytes:
    """
    打成前端 depthServer.js 约定的紧凑二进制：

        [0..3]      uint32 LE   头部 JSON 字节长度 L
        [4..4+L)    UTF-8 JSON
        接着        float32[W*H*3]  点图
        再接着      uint8[W*H]      mask（header.hasMask 为 true 时）

    几百万个浮点走 JSON 又慢又占内存，所以不用 JSON 传数据体。
    """
    head = json.dumps(header, ensure_ascii=False).encode("utf-8")
    parts = [struct.pack("<I", len(head)), head, points_bytes]
    if header.get("hasMask"):
        parts.append(mask_bytes)
    return b"".join(parts)
