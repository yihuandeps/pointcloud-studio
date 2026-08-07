/**
 * 高精度模式：调用本机 Python 后端（server/app.py）。
 *
 * 几百万个浮点走 JSON 又慢又占内存，所以约定一个紧凑的二进制响应：
 *
 *   [0..3]           uint32 LE  头部 JSON 的字节长度 L
 *   [4..4+L)         UTF-8 JSON { width, height, hasMask, intrinsics, model, ms }
 *   接着             float32[width*height*3]  点图（OpenCV 坐标系，米）
 *   再接着（可选）    uint8[width*height]      有效性 mask
 *
 * Python 端必须按同样的布局写出。
 */

const API = {
  health: '/api/health',
  warmup: '/api/warmup',
  infer: '/api/infer',
};

/** 提前把模型加载进显存，免得第一张图卡半天。失败不影响后续推理，只是慢一点。 */
export async function warmupServer() {
  try {
    const res = await fetch(API.warmup, { method: 'POST' });
    return res.ok ? await res.json() : null;
  } catch {
    return null;
  }
}

export async function checkServer(timeoutMs = 2500) {
  try {
    const ctrl = new AbortController();
    const t = setTimeout(() => ctrl.abort(), timeoutMs);
    const res = await fetch(API.health, { signal: ctrl.signal });
    clearTimeout(t);
    if (!res.ok) return null;
    return await res.json(); // { ok, model, device, vram }
  } catch {
    return null;
  }
}

/**
 * 解析后端的二进制响应。
 * 抽成独立函数是为了能跨语言做契约测试 —— Python 打包、这里解包，
 * 两边任何一方改了布局都会立刻被测出来（见 web/tests/protocol.test.mjs）。
 *
 * @param {ArrayBuffer} buf
 */
export function parseInferResponse(buf) {
  if (buf.byteLength < 4) throw new Error('响应过短，不是合法的点图包');

  const view = new DataView(buf);
  const headLen = view.getUint32(0, true);
  if (4 + headLen > buf.byteLength) throw new Error('头部长度越界，响应已损坏');

  const head = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, headLen)));
  const { width, height, hasMask } = head;
  const n = width * height;

  let offset = 4 + headLen;
  const need = n * 3 * 4 + (hasMask ? n : 0);
  if (offset + need > buf.byteLength) {
    throw new Error(`数据体长度不足：需要 ${need} 字节，实际只剩 ${buf.byteLength - offset}`);
  }

  // 字节偏移不一定 4 对齐，不能直接 new Float32Array(buf, offset)
  const points = new Float32Array(buf.slice(offset, offset + n * 3 * 4));
  offset += n * 3 * 4;

  const mask = hasMask ? new Uint8Array(buf.slice(offset, offset + n)) : null;

  return { points, mask, width, height, meta: head };
}

export async function inferOnServer(blob, { maxSide } = {}) {
  const form = new FormData();
  form.append('image', blob, 'input.png');
  if (maxSide) form.append('max_side', String(maxSide));

  const res = await fetch(API.infer, { method: 'POST', body: form });
  if (!res.ok) {
    const detail = await res.text().catch(() => '');
    throw new Error(`后端返回 ${res.status}${detail ? `：${detail.slice(0, 200)}` : ''}`);
  }

  return parseInferResponse(await res.arrayBuffer());
}
