/**
 * 深度图 → XYZRGB 点云（本项目最核心的一步）
 *
 * Depth Anything 输出的是**相对逆深度（disparity）**：值越大越近，无物理单位。
 * 三个容易做错的地方：
 *   1. 不能用 z = 1 - d 这种线性映射，透视关系是错的，出来像浮雕。
 *      必须在视差空间线性插值后取倒数（见 disparityToDepth）。
 *   2. 前景/背景交界处深度是平滑过渡的，直接反投影会拉出一条条"面条"。
 *      靠深度梯度剔除干掉（见 edgeThreshold）。
 *   3. 图像 v 轴向下、three.js Y 轴向上，且相机看向 -Z，符号别搞反。
 */

import { CLOUD_WORLD_SIZE } from '../config.js';

/**
 * 视差 → 深度。
 * 在视差空间线性插值再取倒数，才是几何正确的：
 *   d = 1 → z = zNear（最近）， d = 0 → z = zFar（最远）
 */
function disparityToDepth(d, invNear, invFar) {
  return 1 / (d * (invNear - invFar) + invFar);
}

/**
 * @param {object} arg
 * @param {Float32Array} arg.depth      原始逆深度，长度 width*height
 * @param {number} arg.width
 * @param {number} arg.height
 * @param {Uint8ClampedArray} arg.rgba  同尺寸的 RGBA 像素
 * @param {object} arg.options          见 config.js 的 PARAM_SCHEMA
 */
export function buildPointCloud({ depth, width, height, rgba, options }) {
  const {
    targetPoints = 220000,
    fovDeg = 55,
    depthStrength = 3.5,
    edgeThreshold = 0.04,
    skyFloor = 0.02,
  } = options ?? {};

  const n = width * height;
  if (!depth || depth.length < n) throw new Error('深度图尺寸与图像不匹配');

  /* ---- 1. 归一化视差到 [0,1]，1 = 最近 ---- */
  let dmin = Infinity;
  let dmax = -Infinity;
  for (let i = 0; i < n; i++) {
    const v = depth[i];
    if (!Number.isFinite(v)) continue;
    if (v < dmin) dmin = v;
    if (v > dmax) dmax = v;
  }
  if (!Number.isFinite(dmin) || !Number.isFinite(dmax)) {
    throw new Error('深度图无有效值');
  }
  const range = dmax - dmin || 1;

  const dn = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const v = depth[i];
    dn[i] = Number.isFinite(v) ? (v - dmin) / range : 0;
  }

  /* ---- 2. 采样步长 ----
     必须用浮点步长。整数步长（Math.round）会让点数跳变：
     步长 1→全量、步长 2→四分之一，中间没有档位，
     用户拖到「40 万点」可能实际拿到 78 万，白白吃掉一倍性能。 */
  const step = Math.max(1, Math.sqrt(n / Math.max(1, targetPoints)));
  const cols = Math.ceil(width / step);
  const rows = Math.ceil(height / step);
  const cap = cols * rows;
  // 梯度检测的邻域偏移仍需整数像素
  const nb = Math.max(1, Math.round(step));

  /* ---- 3. 相机内参（FOV 未知，用用户给的估计值）---- */
  const fx = (0.5 * width) / Math.tan((fovDeg * Math.PI) / 360);
  const fy = fx; // 方形像素
  const cx = width / 2;
  const cy = height / 2;

  const zNear = 1;
  const zFar = zNear * depthStrength;
  const invNear = 1 / zNear;
  const invFar = 1 / zFar;

  /* ---- 4. 逐点反投影 + 剔除 ---- */
  const pos = new Float32Array(cap * 3);
  const col = new Float32Array(cap * 3);
  const dep = new Float32Array(cap);

  let k = 0;
  let culledEdge = 0;
  let culledSky = 0;

  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;

  const doEdge = edgeThreshold < 1;
  const nbRow = nb * width;

  for (let fv = 0; fv < height; fv += step) {
    const v = Math.floor(fv);
    const rowBase = v * width;
    for (let fu = 0; fu < width; fu += step) {
      const u = Math.floor(fu);
      const idx = rowBase + u;
      const d = dn[idx];

      if (d < skyFloor) { culledSky++; continue; }

      // 深度梯度剔除：四邻域跳变过大 → 位于遮挡边界，丢弃
      if (doEdge) {
        let g = 0;
        if (u >= nb)          g = Math.max(g, Math.abs(dn[idx - nb] - d));
        if (u + nb < width)   g = Math.max(g, Math.abs(dn[idx + nb] - d));
        if (v >= nb)          g = Math.max(g, Math.abs(dn[idx - nbRow] - d));
        if (v + nb < height)  g = Math.max(g, Math.abs(dn[idx + nbRow] - d));
        if (g > edgeThreshold) { culledEdge++; continue; }
      }

      const z = disparityToDepth(d, invNear, invFar);
      const X = ((u - cx) * z) / fx;
      const Y = -((v - cy) * z) / fy; // 图像 v 向下 → 世界 Y 向上
      const Z = -z;                    // three.js 相机看向 -Z

      const p = k * 3;
      pos[p] = X; pos[p + 1] = Y; pos[p + 2] = Z;

      const c = idx * 4;
      col[p] = rgba[c] / 255;
      col[p + 1] = rgba[c + 1] / 255;
      col[p + 2] = rgba[c + 2] / 255;

      dep[k] = d;

      if (X < minX) minX = X; if (X > maxX) maxX = X;
      if (Y < minY) minY = Y; if (Y > maxY) maxY = Y;
      if (Z < minZ) minZ = Z; if (Z > maxZ) maxZ = Z;

      k++;
    }
  }

  if (k === 0) {
    throw new Error('没有点存活下来 —— 试试把「边缘剔除」调大、或把「远景剔除」调小');
  }

  /* ---- 5. 归一化到世界原点附近，便于相机取景 ---- */
  const center = [(minX + maxX) / 2, (minY + maxY) / 2, (minZ + maxZ) / 2];
  const extent = Math.max(maxX - minX, maxY - minY, maxZ - minZ) || 1;
  const scale = CLOUD_WORLD_SIZE / extent;

  for (let i = 0; i < k; i++) {
    const p = i * 3;
    pos[p]     = (pos[p]     - center[0]) * scale;
    pos[p + 1] = (pos[p + 1] - center[1]) * scale;
    pos[p + 2] = (pos[p + 2] - center[2]) * scale;
  }

  return {
    positions: pos.subarray(0, k * 3),
    colors: col.subarray(0, k * 3),
    depths: dep.subarray(0, k),
    count: k,
    // 导出时若要还原原始（高精度模式下即米制）坐标：orig = p / scale + center
    transform: { center, scale },
    stats: {
      step: Number(step.toFixed(3)),
      candidates: cap,
      culledEdge,
      culledSky,
      sourceWidth: width,
      sourceHeight: height,
    },
  };
}

/**
 * 阶段二用：MoGe 直接给出米制点图 (H,W,3) 与有效性 mask，
 * 无需视差转换和 FOV 猜测，只做剔除 + 抽稀 + 归一化。
 * 保留同样的返回结构，渲染与导出代码可以完全复用。
 */
/**
 * 生成式 3D 模式用：TripoSR 返回的已是三维散点云（three.js 坐标系、带 RGB），
 * 不与图片像素对齐，只做抽稀 + 归一化。
 * 服务端是网格表面随机采样，点序天然乱序，取前 N 个就是均匀抽稀。
 * 保留同样的返回结构，渲染与导出代码可以完全复用。
 */
export function buildPointCloudFromCloud({ positions, colors, count, options }) {
  const { targetPoints = 220000 } = options ?? {};
  const k = Math.min(count, Math.max(1, Math.floor(targetPoints)));
  if (!k) throw new Error('点云为空');

  const pos = new Float32Array(k * 3);
  const col = new Float32Array(k * 3);
  const dep = new Float32Array(k);

  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;

  for (let i = 0; i < k; i++) {
    const p = i * 3;
    const X = positions[p];
    const Y = positions[p + 1];
    const Z = positions[p + 2];
    pos[p] = X; pos[p + 1] = Y; pos[p + 2] = Z;

    col[p] = colors[p] / 255;
    col[p + 1] = colors[p + 1] / 255;
    col[p + 2] = colors[p + 2] / 255;

    if (X < minX) minX = X; if (X > maxX) maxX = X;
    if (Y < minY) minY = Y; if (Y > maxY) maxY = Y;
    if (Z < minZ) minZ = Z; if (Z > maxZ) maxZ = Z;
  }

  const center = [(minX + maxX) / 2, (minY + maxY) / 2, (minZ + maxZ) / 2];
  const extent = Math.max(maxX - minX, maxY - minY, maxZ - minZ) || 1;
  const scale = CLOUD_WORLD_SIZE / extent;

  for (let i = 0; i < k; i++) {
    const p = i * 3;
    pos[p]     = (pos[p]     - center[0]) * scale;
    pos[p + 1] = (pos[p + 1] - center[1]) * scale;
    pos[p + 2] = (pos[p + 2] - center[2]) * scale;
  }

  // 深度属性给色调/动效用：+Z 朝向初始视角，1 = 离观察者最近，与其他模式语义一致
  const zRange = (maxZ - minZ) || 1;
  for (let i = 0; i < k; i++) {
    dep[i] = (pos[i * 3 + 2] / scale + center[2] - minZ) / zRange;
  }

  return {
    positions: pos,
    colors: col,
    depths: dep,
    count: k,
    transform: { center, scale },
    stats: {
      step: 1,
      candidates: count,
      culledEdge: 0,
      culledSky: 0,
      sourceWidth: 0,
      sourceHeight: 0,
    },
  };
}

export function buildPointCloudFromPointMap({ points, mask, width, height, rgba, options }) {
  const { targetPoints = 220000 } = options ?? {};
  const n = width * height;

  const step = Math.max(1, Math.sqrt(n / Math.max(1, targetPoints)));
  const cap = Math.ceil(width / step) * Math.ceil(height / step);

  const pos = new Float32Array(cap * 3);
  const col = new Float32Array(cap * 3);
  const dep = new Float32Array(cap);

  let k = 0;
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  let zMin = Infinity, zMax = -Infinity;

  for (let fv = 0; fv < height; fv += step) {
    const v = Math.floor(fv);
    for (let fu = 0; fu < width; fu += step) {
      const u = Math.floor(fu);
      const idx = v * width + u;
      if (mask && !mask[idx]) continue;

      const s = idx * 3;
      const X = points[s];
      const Y = -points[s + 1]; // MoGe 用 OpenCV 坐标系：Y 向下
      const Z = -points[s + 2]; // Z 向前 → three.js 看向 -Z
      if (!Number.isFinite(X) || !Number.isFinite(Y) || !Number.isFinite(Z)) continue;

      const p = k * 3;
      pos[p] = X; pos[p + 1] = Y; pos[p + 2] = Z;

      const c = idx * 4;
      col[p] = rgba[c] / 255;
      col[p + 1] = rgba[c + 1] / 255;
      col[p + 2] = rgba[c + 2] / 255;

      dep[k] = points[s + 2];
      if (dep[k] < zMin) zMin = dep[k];
      if (dep[k] > zMax) zMax = dep[k];

      if (X < minX) minX = X; if (X > maxX) maxX = X;
      if (Y < minY) minY = Y; if (Y > maxY) maxY = Y;
      if (Z < minZ) minZ = Z; if (Z > maxZ) maxZ = Z;

      k++;
    }
  }

  if (k === 0) throw new Error('点图中没有有效点');

  const center = [(minX + maxX) / 2, (minY + maxY) / 2, (minZ + maxZ) / 2];
  const extent = Math.max(maxX - minX, maxY - minY, maxZ - minZ) || 1;
  const scale = CLOUD_WORLD_SIZE / extent;

  for (let i = 0; i < k; i++) {
    const p = i * 3;
    pos[p]     = (pos[p]     - center[0]) * scale;
    pos[p + 1] = (pos[p + 1] - center[1]) * scale;
    pos[p + 2] = (pos[p + 2] - center[2]) * scale;
  }

  // 深度属性归一化到 [0,1]，1 = 最近，与浏览器模式保持一致
  const zRange = zMax - zMin || 1;
  for (let i = 0; i < k; i++) dep[i] = 1 - (dep[i] - zMin) / zRange;

  return {
    positions: pos.subarray(0, k * 3),
    colors: col.subarray(0, k * 3),
    depths: dep.subarray(0, k),
    count: k,
    transform: { center, scale },
    stats: {
      step: Number(step.toFixed(3)),
      candidates: cap,
      culledEdge: 0,
      culledSky: 0,
      sourceWidth: width,
      sourceHeight: height,
    },
  };
}
