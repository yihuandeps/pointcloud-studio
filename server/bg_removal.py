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


#: 判定"这张图已经自己抠好了"所需的全透明像素占比。
#: 曾经用的是"存在任意一个非全不透明像素"，太松了 —— 一条软边、一个半透明
#: 水印就能让整张带背景的图被当成已抠好直接放行，模型于是把整个矩形画面
#: 当成物体，生成出来是一块板。真正抠过的图，主体周围会有大片全透明区域。
PRECUT_MIN_TRANSPARENT = 0.05


def cutout(image):
    """
    PIL RGB/RGBA → 去背景后的 RGBA。

    自己抠好的图直接放行（人工边缘通常比 u2net 干净），但判定要够严，
    见 PRECUT_MIN_TRANSPARENT。
    """
    import numpy as np
    import rembg

    if image.mode == "RGBA":
        alpha = np.asarray(image.getchannel("A"))
        if (alpha < 8).mean() >= PRECUT_MIN_TRANSPARENT:
            return image
    return rembg.remove(image.convert("RGB"), session=session())


def foreground_stats(image) -> dict:
    """
    抠完之后给这张图做个体检，用来提示"这张图多半生成不出好结果"。

    两个已知会把结果搞成方块/板的输入：
      - 主体几乎铺满整幅画面（说明什么都没抠掉），模型把整个矩形当物体
      - 前景又扁又宽（多半是把三视图拼在一张图里，或者截图带了大片界面）
    """
    import numpy as np

    a = np.asarray(image.convert("RGBA").getchannel("A"))
    solid = a > 8
    coverage = float(solid.mean())

    ys, xs = np.nonzero(solid)
    if len(xs) == 0:
        return {"coverage": 0.0, "aspect": 0.0, "empty": True}

    w = float(xs.max() - xs.min() + 1)
    h = float(ys.max() - ys.min() + 1)
    return {"coverage": coverage, "aspect": w / max(h, 1.0), "empty": False}
