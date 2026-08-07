from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# 上游用的 torchmcubes 是 git+ 源码依赖且需要本地编译 C++ 扩展，Windows 上装不上。
# 换成 scikit-image 的 marching_cubes（纯 wheel）。注意两者顶点轴序不同：
# torchmcubes 返回 zyx（原代码有 [2,1,0] 翻转），skimage 直接是 xyz，不再翻转。
from skimage.measure import marching_cubes as _sk_marching_cubes


def marching_cubes(level: torch.Tensor, thresh: float):
    vol = level.detach().cpu().numpy()
    verts, faces, _, _ = _sk_marching_cubes(vol, level=thresh)
    v_pos = torch.from_numpy(verts.astype(np.float32))
    t_pos_idx = torch.from_numpy(faces.astype(np.int64))
    return v_pos, t_pos_idx


class IsosurfaceHelper(nn.Module):
    points_range: Tuple[float, float] = (0, 1)

    @property
    def grid_vertices(self) -> torch.FloatTensor:
        raise NotImplementedError


class MarchingCubeHelper(IsosurfaceHelper):
    def __init__(self, resolution: int) -> None:
        super().__init__()
        self.resolution = resolution
        self.mc_func: Callable = marching_cubes
        self._grid_vertices: Optional[torch.FloatTensor] = None

    @property
    def grid_vertices(self) -> torch.FloatTensor:
        if self._grid_vertices is None:
            # keep the vertices on CPU so that we can support very large resolution
            x, y, z = (
                torch.linspace(*self.points_range, self.resolution),
                torch.linspace(*self.points_range, self.resolution),
                torch.linspace(*self.points_range, self.resolution),
            )
            x, y, z = torch.meshgrid(x, y, z, indexing="ij")
            verts = torch.cat(
                [x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)], dim=-1
            ).reshape(-1, 3)
            self._grid_vertices = verts
        return self._grid_vertices

    def forward(
        self,
        level: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        level = -level.view(self.resolution, self.resolution, self.resolution)
        # skimage 版 marching_cubes 顶点已是 xyz 轴序，无需 torchmcubes 的 [2,1,0] 翻转
        v_pos, t_pos_idx = self.mc_func(level.detach(), 0.0)
        v_pos = v_pos / (self.resolution - 1.0)
        return v_pos.to(level.device), t_pos_idx.to(level.device)
