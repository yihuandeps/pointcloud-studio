/**
 * 生成式 3D 模式：调用本机 Python 后端的 TripoSR（server/tripo_runner.py）。
 *
 * 与高精度模式（depthServer.js）的本质区别：返回的不是与图片逐像素对齐的
 * 点图，而是一份完整 360° 的散点云 —— 照片里看不见的背面由模型生成补全。
 *
 * 二进制布局（必须与 server/imaging.py 的 pack_cloud 一致）：
 *
 *   [0..3]           uint32 LE  头部 JSON 的字节长度 L
 *   [4..4+L)         UTF-8 JSON { count, model, device, ms }
 *   接着             float32[count*3]  XYZ（three.js 坐标系，y 朝上）
 *   再接着           uint8[count*3]    RGB
 */

const API = {
  health: '/api/gen/health',
  warmup: '/api/gen/warmup',
  generate: '/api/generate',
};

export async function warmupGenServer() {
  try {
    const res = await fetch(API.warmup, { method: 'POST' });
    return res.ok ? await res.json() : null;
  } catch {
    return null;
  }
}

export async function checkGenServer(timeoutMs = 2500) {
  try {
    const ctrl = new AbortController();
    const t = setTimeout(() => ctrl.abort(), timeoutMs);
    const res = await fetch(API.health, { signal: ctrl.signal });
    clearTimeout(t);
    if (!res.ok) return null;
    return await res.json(); // { ok, kind, model, device, gpu, vram, loaded }
  } catch {
    return null;
  }
}

/**
 * 解析后端的二进制点云包。
 * 独立函数，供跨语言契约测试使用（web/tests/protocol.test.mjs）。
 *
 * @param {ArrayBuffer} buf
 */
export function parseCloudResponse(buf) {
  if (buf.byteLength < 4) throw new Error('响应过短，不是合法的点云包');

  const view = new DataView(buf);
  const headLen = view.getUint32(0, true);
  if (4 + headLen > buf.byteLength) throw new Error('头部长度越界，响应已损坏');

  const head = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, headLen)));
  const { count, hasNormals, hasAO } = head;

  let offset = 4 + headLen;
  const need = count * 3 * 4 + count * 3
    + (hasNormals ? count * 3 : 0) + (hasAO ? count : 0);
  if (offset + need > buf.byteLength) {
    throw new Error(`数据体长度不足：需要 ${need} 字节，实际只剩 ${buf.byteLength - offset}`);
  }

  // 字节偏移不一定 4 对齐，不能直接 new Float32Array(buf, offset)
  const positions = new Float32Array(buf.slice(offset, offset + count * 3 * 4));
  offset += count * 3 * 4;

  const colors = new Uint8Array(buf.slice(offset, offset + count * 3));
  offset += count * 3;

  // 法线用 int8 存的（各分量 ×127），着色前还原成单位向量
  let normals = null;
  if (hasNormals) {
    normals = new Int8Array(buf.slice(offset, offset + count * 3));
    offset += count * 3;
  }

  // 凹陷度：0=深凹（要压暗），255=开阔
  let ao = null;
  if (hasAO) {
    ao = new Uint8Array(buf.slice(offset, offset + count));
    offset += count;
  }

  return { positions, colors, normals, ao, count, meta: head };
}

/**
 * 生成完整 3D 点云。TripoSR 在 RTX 3070 上一张图约 10–20 秒
 * （抠背景 + 隐式场推理 + marching cubes + 采样），比深度估计慢一个量级。
 */
/* ---------------- 多视图（Hunyuan3D-2mv）---------------- */

const MV_API = {
  health: '/api/mv/health',
  warmup: '/api/mv/warmup',
  generate: '/api/mv/generate',
};

export async function checkMvServer(timeoutMs = 2500) {
  try {
    const ctrl = new AbortController();
    const t = setTimeout(() => ctrl.abort(), timeoutMs);
    const res = await fetch(MV_API.health, { signal: ctrl.signal });
    clearTimeout(t);
    if (!res.ok) return null;
    return await res.json(); // { ok, kind, model, views, weightsCached, ... }
  } catch {
    return null;
  }
}

export async function warmupMvServer() {
  try {
    const res = await fetch(MV_API.warmup, { method: 'POST' });
    return res.ok ? await res.json() : null;
  } catch {
    return null;
  }
}

/**
 * 多视图生成。views 是 { front, left?, back?, right? } → File/Blob，front 必填。
 * 响应格式与单图模式完全相同，所以复用 parseCloudResponse。
 * RTX 3070 上三张图约 40 秒。
 */
export async function generateMvOnServer(views, { points } = {}) {
  if (!views?.front) throw new Error('至少要提供正面图');

  const form = new FormData();
  for (const [name, file] of Object.entries(views)) {
    if (file) form.append(name, file, `${name}.png`);
  }
  if (points) form.append('points', String(points));

  const res = await fetch(MV_API.generate, { method: 'POST', body: form });
  if (!res.ok) {
    const detail = await res.text().catch(() => '');
    throw new Error(`后端返回 ${res.status}${detail ? `：${detail.slice(0, 200)}` : ''}`);
  }

  return parseCloudResponse(await res.arrayBuffer());
}

export async function generateOnServer(blob, { points } = {}) {
  const form = new FormData();
  form.append('image', blob, 'input.png');
  if (points) form.append('points', String(points));

  const res = await fetch(API.generate, { method: 'POST', body: form });
  if (!res.ok) {
    const detail = await res.text().catch(() => '');
    throw new Error(`后端返回 ${res.status}${detail ? `：${detail.slice(0, 200)}` : ''}`);
  }

  return parseCloudResponse(await res.arrayBuffer());
}
