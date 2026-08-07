/**
 * PLY 点云导出。
 *
 * 二进制格式每点 15 字节（3×float32 + 3×uint8），50 万点约 7.5MB。
 * CloudCompare / Open3D / MeshLab / Blender 均可直接打开。
 *
 * 注意：渲染用的坐标是归一化过的（见 unproject.js 第 5 步）。
 * keepScale=true 时按 transform 还原回原始坐标 —— 高精度模式下即米制。
 */

const HEADER_COMMENT = 'created by PointCloud Studio (点云工具)';

function restore(positions, count, transform, keepScale) {
  if (!keepScale || !transform) return positions;
  const { center, scale } = transform;
  const out = new Float32Array(count * 3);
  const inv = 1 / scale;
  for (let i = 0; i < count; i++) {
    const p = i * 3;
    out[p]     = positions[p]     * inv + center[0];
    out[p + 1] = positions[p + 1] * inv + center[1];
    out[p + 2] = positions[p + 2] * inv + center[2];
  }
  return out;
}

function header(count, binary) {
  return [
    'ply',
    binary ? 'format binary_little_endian 1.0' : 'format ascii 1.0',
    `comment ${HEADER_COMMENT}`,
    `element vertex ${count}`,
    'property float x',
    'property float y',
    'property float z',
    'property uchar red',
    'property uchar green',
    'property uchar blue',
    'end_header',
    '',
  ].join('\n');
}

/**
 * @param {object} cloud  buildPointCloud() 的返回值
 * @param {{format?: 'binary'|'ascii', keepScale?: boolean}} opts
 * @returns {Blob}
 */
export function buildPLY(cloud, opts = {}) {
  const { format = 'binary', keepScale = false } = opts;
  const { colors, count, transform } = cloud;
  const positions = restore(cloud.positions, count, transform, keepScale);

  if (format === 'ascii') {
    // 用分块拼接，避免几十万行一次性 join 撑爆内存
    const CHUNK = 20000;
    const parts = [header(count, false)];
    let buf = '';
    for (let i = 0; i < count; i++) {
      const p = i * 3;
      buf +=
        `${positions[p].toFixed(6)} ${positions[p + 1].toFixed(6)} ${positions[p + 2].toFixed(6)} ` +
        `${Math.round(colors[p] * 255)} ${Math.round(colors[p + 1] * 255)} ${Math.round(colors[p + 2] * 255)}\n`;
      if (i % CHUNK === CHUNK - 1) { parts.push(buf); buf = ''; }
    }
    if (buf) parts.push(buf);
    return new Blob(parts, { type: 'application/octet-stream' });
  }

  const head = new TextEncoder().encode(header(count, true));
  const STRIDE = 15; // 3*4 + 3*1
  const body = new ArrayBuffer(count * STRIDE);
  const view = new DataView(body);
  const bytes = new Uint8Array(body);

  for (let i = 0; i < count; i++) {
    const p = i * 3;
    const o = i * STRIDE;
    view.setFloat32(o, positions[p], true);
    view.setFloat32(o + 4, positions[p + 1], true);
    view.setFloat32(o + 8, positions[p + 2], true);
    bytes[o + 12] = Math.max(0, Math.min(255, Math.round(colors[p] * 255)));
    bytes[o + 13] = Math.max(0, Math.min(255, Math.round(colors[p + 1] * 255)));
    bytes[o + 14] = Math.max(0, Math.min(255, Math.round(colors[p + 2] * 255)));
  }

  return new Blob([head, body], { type: 'application/octet-stream' });
}

/** 估算导出体积（字节），用于在按钮上提前显示大小。 */
export function estimateSize(count, format = 'binary') {
  return format === 'binary' ? count * 15 + 200 : count * 46;
}

export function formatBytes(n) {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1048576).toFixed(1)} MB`;
}

export function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  // 立刻 revoke 在部分浏览器会打断下载，延后释放
  setTimeout(() => URL.revokeObjectURL(url), 30000);
}

export function timestampName(prefix = 'pointcloud', ext = 'ply') {
  const d = new Date();
  const pad = (v) => String(v).padStart(2, '0');
  return `${prefix}_${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}_${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}.${ext}`;
}
