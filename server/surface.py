"""
点云的表面属性：逐点法线打包 + 凹陷度（环境光遮蔽）。

为什么需要这两样：前端把每个点按它自己的颜色平涂上去，没有任何明暗变化。
没有高光和阴影，人眼读不出表面朝向，再准确的三维形体看起来也是平的。
有了法线就能做真实光照，有了凹陷度就能把褶皱、缝隙压暗 —— 立体感主要来自这里。
"""

from __future__ import annotations

import numpy as np

#: 估计凹陷度时看多少个邻居。太小容易被采样噪声带偏，太大会把细节抹平。
AO_NEIGHBORS = 24


def pack_normals(normals: np.ndarray) -> np.ndarray:
    """
    单位法线 → int8，每点 3 字节。

    着色只需要方向，int8 的角度精度（约 0.9°）绰绰有余，
    比 float32 省 4 倍带宽 —— 60 万点是 7MB 和 1.8MB 的差别。
    """
    n = np.asarray(normals, dtype=np.float32)
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    n = n / np.maximum(norm, 1e-9)
    return np.clip(np.rint(n * 127.0), -127, 127).astype(np.int8)


def estimate_ao(points: np.ndarray, normals: np.ndarray,
                k: int = AO_NEIGHBORS) -> np.ndarray:
    """
    估计每个点的凹陷程度 → uint8，0=完全被遮蔽（深凹），255=完全开阔（凸起）。

    做法是「邻域质心相对法线的偏移」这个经典技巧：
    取每个点的 k 个最近邻，求质心。凸起处邻居都在切面下方，质心会落在法线的反方向；
    凹陷处邻居环绕在四周偏上，质心落在法线正方向。
    把这个投影量按邻域尺度归一化，就得到一个与曲率同号的信号。

    比起真正的光线投射式 AO 廉价得多（一次 kNN 查询而已），
    但要的就是「缝隙和褶皱压暗」这个效果，够用。
    """
    from scipy.spatial import cKDTree

    pts = np.ascontiguousarray(points, dtype=np.float64)
    n = np.asarray(normals, dtype=np.float64)
    n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)

    k = min(k, max(2, len(pts) - 1))
    tree = cKDTree(pts)
    # 第 0 个邻居是自己，多取一个再丢掉
    dist, idx = tree.query(pts, k=k + 1, workers=-1)
    dist, idx = dist[:, 1:], idx[:, 1:]

    centroid = pts[idx].mean(axis=1)
    offset = centroid - pts
    # 邻域半径做尺度归一化，这样模型整体缩放不会改变结果
    scale = np.maximum(dist.mean(axis=1), 1e-9)
    signal = np.einsum("ij,ij->i", offset, n) / scale  # >0 凹陷，<0 凸起

    # 经验区间：signal 落在 ±0.6 之间就够区分了，超出的截断
    ao = 1.0 - np.clip(signal / 0.6, 0.0, 1.0)
    return np.clip(np.rint(ao * 255.0), 0, 255).astype(np.uint8)


def sample_normals(mesh, face_idx: np.ndarray) -> np.ndarray:
    """表面采样点的法线 = 它所在三角面的法线。"""
    return np.asarray(mesh.face_normals)[face_idx]
