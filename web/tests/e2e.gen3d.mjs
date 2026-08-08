/**
 * 生成式 3D 端到端验证（Node 版）：真实后端 → 前端解析 → 前端建云 → 导出 PLY
 *
 * server/test_gen3d.py 验证的是后端自己；这份验证的是**浏览器真正走的那条路**：
 * 同一份 gen3dServer.js 解析二进制、同一份 unproject.js 抽稀归一化、
 * 同一份 plyExport.js 写文件。后端测试全绿但前端解析错位这种事，只有这里能测出来。
 *
 * 需要先启动后端：server\start.ps1
 * 运行：cd web/tests && node e2e.gen3d.mjs
 * 产物：web/tests/out/e2e_gen3d.ply（可直接拖进 CloudCompare 转一圈看背面）
 */

import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';

import { parseCloudResponse } from '../src/core/gen3dServer.js';
import { buildPointCloudFromCloud } from '../src/core/unproject.js';
import { buildPLY } from '../src/core/plyExport.js';
import { defaultParams } from '../src/config.js';

const API = process.env.PCS_API ?? 'http://127.0.0.1:8000';

let pass = 0, fail = 0;
const ok = (name, cond, extra = '') => {
  if (cond) { pass++; console.log(`  ✓ ${name}${extra ? '  ' + extra : ''}`); }
  else { fail++; console.log(`  ✗ ${name}  ${extra}`); }
};

/* ---------------- 最小 PNG 编码器 ---------------- */
// Node 没有内置图片编码。只需要一张能喂给后端的真图，所以手写：
// 每行前缀一个 filter 0 字节，整体 deflate，套上 IHDR/IDAT/IEND 三个块。

function crc32(buf) {
  let c, crc = 0xffffffff;
  for (let n = 0; n < buf.length; n++) {
    c = (crc ^ buf[n]) & 0xff;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    crc = c ^ (crc >>> 8);
  }
  return (crc ^ 0xffffffff) >>> 0;
}

function chunk(type, data) {
  const len = Buffer.alloc(4);
  len.writeUInt32BE(data.length);
  const body = Buffer.concat([Buffer.from(type, 'ascii'), data]);
  const crc = Buffer.alloc(4);
  crc.writeUInt32BE(crc32(body));
  return Buffer.concat([len, body, crc]);
}

function encodePNG(width, height, rgb) {
  const raw = Buffer.alloc(height * (1 + width * 3));
  for (let y = 0; y < height; y++) {
    const o = y * (1 + width * 3);
    raw[o] = 0; // filter: none
    rgb.copy(raw, o + 1, y * width * 3, (y + 1) * width * 3);
  }
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(width, 0);
  ihdr.writeUInt32BE(height, 4);
  ihdr[8] = 8;  // bit depth
  ihdr[9] = 2;  // color type: truecolor
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk('IHDR', ihdr),
    chunk('IDAT', zlib.deflateSync(raw)),
    chunk('IEND', Buffer.alloc(0)),
  ]);
}

/** 白底 + 居中的朗伯着色橙球 —— 对 rembg 抠图和 TripoSR 都友好的输入 */
function spherePNG(W = 512, H = 512) {
  const rgb = Buffer.alloc(W * H * 3);
  for (let v = 0; v < H; v++) {
    for (let u = 0; u < W; u++) {
      let r = 245, g = 245, b = 245;
      const dx = (u - W * 0.5) / (W * 0.3);
      const dy = (v - H * 0.5) / (W * 0.3);
      const d2 = dx * dx + dy * dy;
      if (d2 < 1) {
        const sh = Math.sqrt(1 - d2);
        const lit = Math.max(0, 0.3 + 0.8 * sh - 0.25 * dx);
        r = 235 * lit; g = 120 * lit; b = 40 * lit;
      }
      const i = (v * W + u) * 3;
      rgb[i] = Math.min(255, r);
      rgb[i + 1] = Math.min(255, g);
      rgb[i + 2] = Math.min(255, b);
    }
  }
  return encodePNG(W, H, rgb);
}

/* ---------------- 探测后端 ---------------- */

console.log(`\n后端：${API}`);
let health;
try {
  const r = await fetch(`${API}/api/gen/health`, {
    signal: AbortSignal.timeout(3000),
  });
  health = await r.json();
} catch {
  console.log('\n后端没起来，跳过。先运行 server\\start.ps1 再重跑这个测试。\n');
  process.exit(0);
}
console.log(`模型：${health.model} · ${health.device}${health.gpu ? ` (${health.gpu})` : ''}\n`);

/* ---------------- [1] 真实生成 ---------------- */

console.log('[1] 调用后端生成');
const png = spherePNG();
ok('测试图已编码为合法 PNG', png.length > 1000 && png.subarray(1, 4).toString() === 'PNG',
  `${(png.length / 1024).toFixed(1)} KB`);

const REQUEST_POINTS = 300000;
const form = new FormData();
form.append('image', new Blob([png], { type: 'image/png' }), 'input.png');
form.append('points', String(REQUEST_POINTS));

const t0 = Date.now();
const res = await fetch(`${API}/api/generate`, { method: 'POST', body: form });
ok('POST /api/generate 返回 200', res.ok,
  res.ok ? `${((Date.now() - t0) / 1000).toFixed(1)}s` : await res.text().then((t) => t.slice(0, 200)));
if (!res.ok) process.exit(1);

/* ---------------- [2] 前端解析器 ---------------- */

console.log('\n[2] 用前端的 parseCloudResponse 解析');
const buf = await res.arrayBuffer();
const cloud = parseCloudResponse(buf);

ok('解析成功', true, `${(buf.byteLength / 1048576).toFixed(2)} MB`);
ok('点数与请求一致', cloud.count === REQUEST_POINTS, String(cloud.count));
ok('坐标数组长度 = count*3', cloud.positions.length === cloud.count * 3);
ok('颜色数组长度 = count*3', cloud.colors.length === cloud.count * 3);
ok('坐标全部有限', cloud.positions.every(Number.isFinite));
ok('头部带回实际使用的体素分辨率', Number.isInteger(cloud.meta.resolution),
  `res=${cloud.meta.resolution}, ${cloud.meta.ms}ms`);

/* ---------------- [3] 前端建云 ---------------- */

console.log('\n[3] 用前端的 buildPointCloudFromCloud 建云');
const params = { ...defaultParams(), targetPoints: 220000 };
const built = buildPointCloudFromCloud({
  positions: cloud.positions,
  colors: cloud.colors,
  count: cloud.count,
  options: params,
});

ok('按「点数」滑杆抽稀', built.count === 220000, `${cloud.count} → ${built.count}`);
ok('抽稀不重跑模型（源点数守恒）', built.stats.candidates === cloud.count);
ok('输出坐标全部有限', built.positions.every(Number.isFinite));
ok('颜色已归一化到 [0,1]',
  built.colors.every((c) => c >= 0 && c <= 1));
ok('深度属性在 [0,1]', built.depths.every((d) => d >= -1e-6 && d <= 1 + 1e-6));

/* ---- 归一化：最长边应当正好是 CLOUD_WORLD_SIZE ---- */
const ext = [0, 1, 2].map((a) => {
  let lo = Infinity, hi = -Infinity;
  for (let i = a; i < built.positions.length; i += 3) {
    if (built.positions[i] < lo) lo = built.positions[i];
    if (built.positions[i] > hi) hi = built.positions[i];
  }
  return hi - lo;
});
ok('最长边归一化到 2.4 世界单位', Math.abs(Math.max(...ext) - 2.4) < 1e-3,
  `x=${ext[0].toFixed(2)} y=${ext[1].toFixed(2)} z=${ext[2].toFixed(2)}`);

/* ---- 这个模式存在的意义：立体 + 有背面 ---- */
console.log('\n[4] 立体性（这才是这个模式存在的理由）');
const ratio = Math.min(...ext) / Math.max(...ext);
ok('三轴跨度同量级 —— 是形体不是浮雕（浮雕接近 0）', ratio > 0.4, `比值 ${ratio.toFixed(2)}`);

let back = 0;
let zLo = Infinity, zHi = -Infinity;
for (let i = 2; i < built.positions.length; i += 3) {
  if (built.positions[i] < zLo) zLo = built.positions[i];
  if (built.positions[i] > zHi) zHi = built.positions[i];
}
const zMid = (zLo + zHi) / 2;
for (let i = 2; i < built.positions.length; i += 3) if (built.positions[i] < zMid) back++;
const backFrac = back / built.count;
ok('背面有实质内容 —— 照片看不见的部分被补全了', backFrac > 0.25,
  `背面点占 ${(backFrac * 100).toFixed(1)}%`);

/* ---------------- [5] 导出 PLY ---------------- */

console.log('\n[5] 用前端的 buildPLY 导出');
const OUT = path.join(import.meta.dirname, 'out');
fs.mkdirSync(OUT, { recursive: true });

const plyBlob = buildPLY(built, { format: 'binary', keepScale: false });
const plyBuf = Buffer.from(await plyBlob.arrayBuffer());
const plyPath = path.join(OUT, 'e2e_gen3d.ply');
fs.writeFileSync(plyPath, plyBuf);

const header = plyBuf.subarray(0, 200).toString('latin1');
ok('PLY 头部合法', header.startsWith('ply\n'));
ok('声明的点数正确', header.includes(`element vertex ${built.count}`));
ok('二进制体积符合每点 15 字节', plyBuf.length > built.count * 15,
  `${(plyBuf.length / 1048576).toFixed(2)} MB`);
console.log(`     已写出：${plyPath}`);
console.log('     （拖进 CloudCompare 转一圈，背面应当是实心的）');

console.log(`\n${'─'.repeat(54)}`);
console.log(`  通过 ${pass} 项，失败 ${fail} 项`);
console.log(`${'─'.repeat(54)}\n`);
process.exit(fail ? 1 : 0);
