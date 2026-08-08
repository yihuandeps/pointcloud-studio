/**
 * 多视图端到端验证（Node 版）：真实后端 → 前端解析 → 前端建云 → 导出 PLY
 *
 * 走的是浏览器真正的那条路：同一份 gen3dServer.js 解析、
 * 同一份 unproject.js 建云、同一份 plyExport.js 写文件。
 *
 * 需要先启动后端：server\start.ps1
 * 运行：cd web/tests && node e2e.mv.mjs
 * 产物：web/tests/out/e2e_mv.ply（拖进 CloudCompare 转一圈看背面）
 */

import fs from 'node:fs';
import path from 'node:path';

import { parseCloudResponse } from '../src/core/gen3dServer.js';
import { buildPointCloudFromCloud } from '../src/core/unproject.js';
import { buildPLY } from '../src/core/plyExport.js';
import { defaultParams } from '../src/config.js';

const API = process.env.PCS_API ?? 'http://127.0.0.1:8000';
// 官方样例三视图，setup 克隆源码时一并带下来的
const EX = path.join(
  import.meta.dirname, '..', '..', 'server', '.cache', 'src', 'Hunyuan3D-2',
  'assets', 'example_mv_images', '1',
);

let pass = 0, fail = 0;
const ok = (name, cond, extra = '') => {
  if (cond) { pass++; console.log(`  ✓ ${name}${extra ? '  ' + extra : ''}`); }
  else { fail++; console.log(`  ✗ ${name}  ${extra}`); }
};

if (!fs.existsSync(path.join(EX, 'front.png'))) {
  console.log(`\n找不到样例三视图（${EX}），跳过。重跑 setup.ps1 会拉下来。\n`);
  process.exit(0);
}

console.log(`\n后端：${API}`);
let health;
try {
  const r = await fetch(`${API}/api/mv/health`, { signal: AbortSignal.timeout(3000) });
  health = await r.json();
} catch {
  console.log('\n后端没起来，跳过。先运行 server\\start.ps1。\n');
  process.exit(0);
}
if (health.weightsCached === false) {
  console.log('\n权重还没下载完（约 4.9GB），跳过。\n');
  process.exit(0);
}
console.log(`模型：${health.model} · ${health.device}\n`);

/* ---- [1] 三视图生成 ---- */
console.log('[1] 三视图 → 后端生成');
const VIEWS = ['front', 'left', 'back'];
const form = new FormData();
for (const v of VIEWS) {
  const buf = fs.readFileSync(path.join(EX, `${v}.png`));
  form.append(v, new Blob([buf], { type: 'image/png' }), `${v}.png`);
}
const REQUEST_POINTS = 300000;
form.append('points', String(REQUEST_POINTS));

const t0 = Date.now();
const res = await fetch(`${API}/api/mv/generate`, { method: 'POST', body: form });
ok('POST /api/mv/generate 返回 200', res.ok,
  res.ok ? `${((Date.now() - t0) / 1000).toFixed(1)}s` : (await res.text()).slice(0, 200));
if (!res.ok) process.exit(1);

/* ---- [2] 前端解析 ---- */
console.log('\n[2] 前端 parseCloudResponse');
const cloud = parseCloudResponse(await res.arrayBuffer());
ok('点数与请求一致', cloud.count === REQUEST_POINTS, String(cloud.count));
ok('坐标全部有限', cloud.positions.every(Number.isFinite));
ok('回报了实际使用的视图', Array.isArray(cloud.meta.viewsUsed)
  && cloud.meta.viewsUsed.length === 3, JSON.stringify(cloud.meta.viewsUsed));
ok('回报了体素分辨率', Number.isInteger(cloud.meta.resolution),
  `res=${cloud.meta.resolution}, ${cloud.meta.ms}ms`);

/* ---- [3] 前端建云 ---- */
console.log('\n[3] 前端 buildPointCloudFromCloud');
const params = { ...defaultParams(), targetPoints: 220000 };
const built = buildPointCloudFromCloud({
  positions: cloud.positions, colors: cloud.colors, count: cloud.count, options: params,
});
ok('按点数滑杆抽稀', built.count === 220000, `${cloud.count} → ${built.count}`);
ok('颜色归一化到 [0,1]', built.colors.every((c) => c >= 0 && c <= 1));

const ext = [0, 1, 2].map((a) => {
  let lo = Infinity, hi = -Infinity;
  for (let i = a; i < built.positions.length; i += 3) {
    if (built.positions[i] < lo) lo = built.positions[i];
    if (built.positions[i] > hi) hi = built.positions[i];
  }
  return hi - lo;
});
ok('最长边归一化到 2.4', Math.abs(Math.max(...ext) - 2.4) < 1e-3,
  `x=${ext[0].toFixed(2)} y=${ext[1].toFixed(2)} z=${ext[2].toFixed(2)}`);

/* ---- [4] 立体性与着色 ---- */
console.log('\n[4] 立体性与着色');
ok('三轴都有实质厚度（浮雕的最薄轴接近 0）',
  Math.min(...ext) / Math.max(...ext) > 0.3,
  `比值 ${(Math.min(...ext) / Math.max(...ext)).toFixed(2)}`);

let back = 0, zLo = Infinity, zHi = -Infinity;
for (let i = 2; i < built.positions.length; i += 3) {
  if (built.positions[i] < zLo) zLo = built.positions[i];
  if (built.positions[i] > zHi) zHi = built.positions[i];
}
const zMid = (zLo + zHi) / 2;
for (let i = 2; i < built.positions.length; i += 3) if (built.positions[i] < zMid) back++;
ok('背面有实质内容', back / built.count > 0.25,
  `背面点占 ${(back / built.count * 100).toFixed(1)}%`);

// 背面颜色来自 back.png，若与正面统计完全一致说明视图选择没生效
const stat = (which) => {
  let r = 0, g = 0, b = 0, n = 0;
  for (let i = 0; i < built.count; i++) {
    const isBack = built.positions[i * 3 + 2] < zMid;
    if (isBack !== which) continue;
    r += built.colors[i * 3]; g += built.colors[i * 3 + 1]; b += built.colors[i * 3 + 2]; n++;
  }
  return [r / n, g / n, b / n];
};
const [fr, fg, fb] = stat(false);
const [br, bg, bb] = stat(true);
const dif = Math.max(Math.abs(fr - br), Math.abs(fg - bg), Math.abs(fb - bb));
ok('正/背面配色有差异（着色确实按视图选的）', dif > 0.005, `均值差 ${dif.toFixed(3)}`);

let gray = 0;
for (let i = 0; i < built.count; i++) {
  const r = built.colors[i * 3], g = built.colors[i * 3 + 1], b = built.colors[i * 3 + 2];
  if (Math.abs(r - 200 / 255) < 1e-6 && Math.abs(g - 200 / 255) < 1e-6
      && Math.abs(b - 200 / 255) < 1e-6) gray++;
}
ok('几乎没有未着色的点', gray / built.count < 0.01,
  `未着色 ${(gray / built.count * 100).toFixed(2)}%`);

/* ---- [5] 导出 ---- */
console.log('\n[5] 导出 PLY');
const OUT = path.join(import.meta.dirname, 'out');
fs.mkdirSync(OUT, { recursive: true });
const blob = buildPLY(built, { format: 'binary', keepScale: false });
const buf = Buffer.from(await blob.arrayBuffer());
const out = path.join(OUT, 'e2e_mv.ply');
fs.writeFileSync(out, buf);
ok('PLY 头部合法', buf.subarray(0, 200).toString('latin1').startsWith('ply\n'));
ok('点数声明正确',
  buf.subarray(0, 200).toString('latin1').includes(`element vertex ${built.count}`));
console.log(`     已写出：${out}`);

console.log(`\n${'─'.repeat(54)}`);
console.log(`  通过 ${pass} 项，失败 ${fail} 项`);
console.log(`${'─'.repeat(54)}\n`);
process.exit(fail ? 1 : 0);
