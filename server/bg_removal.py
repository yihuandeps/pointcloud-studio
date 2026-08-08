"""
共享的 rembg 会话。

TripoSR 和 Hunyuan3D 都要先抠背景，各建一个会话会在内存里放两份 u2net（每份约 168MB），
所以在这里做进程级缓存。模型文件位置由 U2NET_HOME 控制（start.ps1 指到项目盘）。
"""

from __future__ import annotations

import threading

_session = None
_lock = threading.Lock()


def session():
    global _session
    if _session is None:
        with _lock:
            if _session is None:
                import rembg

                _session = rembg.new_session("u2net")
    return _session


def cutout(image):
    """
    PIL RGB/RGBA → 去背景后的 RGBA。

    已经带透明通道（且真的有透明像素）的图直接放行 —— 用户自己抠好的边缘
    通常比 u2net 更干净，没必要再抠一遍。
    """
    import rembg

    if image.mode == "RGBA" and image.getextrema()[3][0] < 255:
        return image
    return rembg.remove(image.convert("RGB"), session=session())
