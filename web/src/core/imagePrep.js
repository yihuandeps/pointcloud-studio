/**
 * 图像预处理：解码 → 限长边 → 取 RGBA 像素。
 *
 * 为什么要限长边：Depth Anything 内部固定跑在 518px，喂 4000px 的原图
 * 只会让「插值回原尺寸」这一步白白吃掉几百毫秒和几十 MB 内存。
 */

import { MAX_INPUT_SIDE } from '../config.js';

/** 等比缩放后的尺寸（长边不超过 maxSide，且不放大）。 */
export function fitSize(w, h, maxSide) {
  const longest = Math.max(w, h);
  if (longest <= maxSide) return { width: w, height: h, scaled: false };
  const k = maxSide / longest;
  return {
    width: Math.max(1, Math.round(w * k)),
    height: Math.max(1, Math.round(h * k)),
    scaled: true,
  };
}

/** File / Blob → ImageBitmap */
export async function decodeImage(source) {
  if (source instanceof ImageBitmap) return source;
  if (source instanceof Blob) return await createImageBitmap(source);
  if (typeof source === 'string') {
    const res = await fetch(source);
    if (!res.ok) throw new Error(`图片加载失败：${res.status}`);
    return await createImageBitmap(await res.blob());
  }
  throw new Error('不支持的图片来源');
}

/**
 * ImageBitmap → { data: Uint8ClampedArray(RGBA), width, height }
 * 同时返回一个可直接贴到 DOM 的缩略 canvas。
 */
export function bitmapToRGBA(bitmap, maxSide = MAX_INPUT_SIDE) {
  const { width, height } = fitSize(bitmap.width, bitmap.height, maxSide);
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(bitmap, 0, 0, width, height);
  const { data } = ctx.getImageData(0, 0, width, height);
  return { data, width, height, canvas };
}

/**
 * canvas → PNG Blob。
 * 高精度模式发给后端的是**已经缩放好的**像素，不是原图 ——
 * 否则前后端各算一次目标尺寸，只要有一位不一致，点图和颜色就整体错位。
 */
export function canvasToBlob(canvas, type = 'image/png') {
  return new Promise((resolve, reject) => {
    canvas.toBlob((b) => (b ? resolve(b) : reject(new Error('canvas 编码失败'))), type);
  });
}

/** 一步到位：任意来源 → RGBA 像素 */
export async function prepareImage(source, maxSide = MAX_INPUT_SIDE) {
  const bitmap = await decodeImage(source);
  const out = bitmapToRGBA(bitmap, maxSide);
  out.originalWidth = bitmap.width;
  out.originalHeight = bitmap.height;
  if (bitmap.close) bitmap.close();
  return out;
}

/* ============================================================
   程序化示例图
   还没有真实素材时用来打通链路。三张图都刻意做出明确的深度层次，
   让深度模型有东西可"看"。等你把真图放进 public/samples/ 再替换。
   ============================================================ */

export const SAMPLE_KINDS = [
  { id: 'sphere', name: '球体' },
  { id: 'blocks', name: '方块' },
  { id: 'corridor', name: '走廊' },
];

export function makeSample(kind, size = 640) {
  const c = document.createElement('canvas');
  c.width = c.height = size;
  const g = c.getContext('2d');
  const S = size;

  if (kind === 'sphere') {
    // 渐变背景 + 地面
    const bg = g.createLinearGradient(0, 0, 0, S);
    bg.addColorStop(0, '#2b3a55');
    bg.addColorStop(0.62, '#4a5c74');
    bg.addColorStop(0.63, '#6b6357');
    bg.addColorStop(1, '#3a352e');
    g.fillStyle = bg;
    g.fillRect(0, 0, S, S);

    // 接触阴影
    g.save();
    g.translate(S * 0.5, S * 0.75);
    g.scale(1, 0.26);
    const sh = g.createRadialGradient(0, 0, 0, 0, 0, S * 0.3);
    sh.addColorStop(0, 'rgba(0,0,0,0.55)');
    sh.addColorStop(1, 'rgba(0,0,0,0)');
    g.fillStyle = sh;
    g.beginPath();
    g.arc(0, 0, S * 0.3, 0, Math.PI * 2);
    g.fill();
    g.restore();

    // 球体：主光 + 环境反弹光 + 高光
    const R = S * 0.24;
    const cx = S * 0.5;
    const cy = S * 0.56;
    const ball = g.createRadialGradient(cx - R * 0.35, cy - R * 0.42, R * 0.05, cx, cy, R);
    ball.addColorStop(0, '#ffd9a8');
    ball.addColorStop(0.42, '#e08a4c');
    ball.addColorStop(0.82, '#8c3f22');
    ball.addColorStop(1, '#4a1d12');
    g.fillStyle = ball;
    g.beginPath();
    g.arc(cx, cy, R, 0, Math.PI * 2);
    g.fill();

    const spec = g.createRadialGradient(cx - R * 0.4, cy - R * 0.48, 0, cx - R * 0.4, cy - R * 0.48, R * 0.3);
    spec.addColorStop(0, 'rgba(255,255,255,0.85)');
    spec.addColorStop(1, 'rgba(255,255,255,0)');
    g.fillStyle = spec;
    g.beginPath();
    g.arc(cx, cy, R, 0, Math.PI * 2);
    g.fill();
  }

  if (kind === 'blocks') {
    const bg = g.createLinearGradient(0, 0, 0, S);
    bg.addColorStop(0, '#1d2430');
    bg.addColorStop(1, '#39404d');
    g.fillStyle = bg;
    g.fillRect(0, 0, S, S);

    // 由远及近画方块：越近越大、越亮、越靠下
    const boxes = [
      { x: 0.20, y: 0.42, w: 0.15, tone: 0.38 },
      { x: 0.62, y: 0.40, w: 0.17, tone: 0.44 },
      { x: 0.38, y: 0.52, w: 0.24, tone: 0.62 },
      { x: 0.10, y: 0.66, w: 0.28, tone: 0.82 },
      { x: 0.68, y: 0.70, w: 0.30, tone: 1.0 },
    ];
    for (const b of boxes) {
      const x = b.x * S, y = b.y * S, w = b.w * S, h = w * 1.15;
      const dep = w * 0.34;
      const base = Math.round(70 + 120 * b.tone);
      // 顶面
      g.fillStyle = `rgb(${base + 42},${base + 34},${base + 22})`;
      g.beginPath();
      g.moveTo(x, y);
      g.lineTo(x + dep, y - dep * 0.6);
      g.lineTo(x + w + dep, y - dep * 0.6);
      g.lineTo(x + w, y);
      g.closePath();
      g.fill();
      // 右侧面
      g.fillStyle = `rgb(${base - 32},${base - 28},${base - 20})`;
      g.beginPath();
      g.moveTo(x + w, y);
      g.lineTo(x + w + dep, y - dep * 0.6);
      g.lineTo(x + w + dep, y + h - dep * 0.6);
      g.lineTo(x + w, y + h);
      g.closePath();
      g.fill();
      // 正面
      g.fillStyle = `rgb(${base},${base - 6},${base - 14})`;
      g.fillRect(x, y, w, h);
    }
  }

  if (kind === 'corridor') {
    g.fillStyle = '#0d1018';
    g.fillRect(0, 0, S, S);
    const vx = S * 0.5, vy = S * 0.48;

    // 一层层向内收缩的矩形框，形成强透视深度线索
    const rings = 16;
    for (let i = rings; i >= 1; i--) {
      const t = i / rings;
      const w = S * 0.98 * t;
      const h = S * 0.98 * t;
      const lum = Math.round(18 + 200 * (1 - t) ** 1.5);
      g.strokeStyle = `rgb(${lum},${Math.round(lum * 0.94)},${Math.round(lum * 1.08)})`;
      g.lineWidth = Math.max(1, S * 0.012 * t);
      g.strokeRect(vx - w / 2, vy - h / 2, w, h);
    }
    // 四条透视引导线
    g.strokeStyle = 'rgba(120,190,255,0.28)';
    g.lineWidth = 1.2;
    for (const [px, py] of [[0, 0], [S, 0], [0, S], [S, S]]) {
      g.beginPath();
      g.moveTo(px, py);
      g.lineTo(vx, vy);
      g.stroke();
    }
    // 尽头光晕
    const glow = g.createRadialGradient(vx, vy, 0, vx, vy, S * 0.16);
    glow.addColorStop(0, 'rgba(200,235,255,0.95)');
    glow.addColorStop(1, 'rgba(200,235,255,0)');
    g.fillStyle = glow;
    g.fillRect(0, 0, S, S);
  }

  return c;
}

/** 生成示例图并转成 Blob，走和真实上传完全相同的链路。 */
export function sampleToBlob(kind, size = 640) {
  return new Promise((resolve) => {
    makeSample(kind, size).toBlob((b) => resolve(b), 'image/png');
  });
}
