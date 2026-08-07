/**
 * 前后端契约测试（跨语言）
 *
 * 尺寸计算和二进制布局在 Python 和 JS 里各实现了一遍。两边一旦不一致：
 *   - 尺寸差一像素 → 点图和颜色整体错位，画面"糊了"但看不出原因
 *   - 布局对不上  → 解出来一堆 NaN 或者直接越界
 * 这两类 bug 症状都很隐蔽，所以用真实的 Python 产出去喂真实的 JS 解析器。
 *
 * 运行：cd web/tests && node protocol.test.mjs
 */

import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { execFileSync } from 'node:child_process';

import { fitSize } from '../src/core/imagePrep.js';
import { parseInferResponse } from '../src/core/depthServer.js';
import { parseCloudResponse } from '../src/core/gen3dServer.js';

let pass = 0, fail = 0;
const ok = (name, cond, extra = '') => {
  if (cond) { pass++; console.log(`  ✓ ${name}${extra ? '  ' + extra : ''}`); }
  else { fail++; console.log(`  ✗ ${name}  ${extra}`); }
};

/* ---- 找一个能用的 Python（imaging.py 只用标准库，任意 3.9+ 都行）---- */
const SERVER = path.join(import.meta.dirname, '..', '..', 'server');
const candidates = [
  path.join(SERVER, '.venv', 'Scripts', 'python.exe'),
  path.join(SERVER, '.venv', 'bin', 'python'),
  path.join(os.homedir(), 'AppData', 'Local', 'Programs', 'Python', 'Python311', 'python.exe'),
  'python3',
  'python',
];

let PY = null;
for (const c of candidates) {
  try {
    execFileSync(c, ['-c', 'import sys; assert sys.version_info >= (3, 9)'], { stdio: 'ignore' });
    PY = c;
    break;
  } catch { /* 试下一个 */ }
}
if (!PY) {
  console.log('\n找不到可用的 Python（>=3.9），跳过契约测试。');
  console.log('装好 Python 后重跑：node tests/protocol.test.mjs\n');
  process.exit(0);
}
console.log(`\nPython：${PY}`);

/* ---- 让 Python 生成测试向量 ---- */
const OUT = path.join(import.meta.dirname, 'out', 'protocol');
fs.mkdirSync(OUT, { recursive: true });

let emitLog;
try {
  emitLog = execFileSync(PY, [path.join(SERVER, 'emit_vectors.py'), OUT], {
    cwd: SERVER,
    encoding: 'utf8',
  }).trim();
} catch (err) {
  console.log(`\n生成测试向量失败：${err.stderr || err.message}\n`);
  process.exit(1);
}
console.log(`向量：${emitLog}\n`);

const V = JSON.parse(fs.readFileSync(path.join(OUT, 'vectors.json'), 'utf8'));

/* ---- [1] Math.round 语义 ---- */
console.log('[1] js_round 是否真的等价于 JS 的 Math.round');
{
  const bad = V.round.filter((c) => Math.round(c.x) !== c.out);
  ok('全部一致', bad.length === 0,
    bad.length ? `不一致：${bad.slice(0, 5).map((c) => `${c.x}→py:${c.out} js:${Math.round(c.x)}`).join('  ')}`
               : `${V.round.length} 个取值`);

  // 反向确认这个测试确实有区分度：Python 内置 round() 会挂在这里
  const halfCases = V.round.filter((c) => Math.abs(c.x % 1) === 0.5);
  ok('测试集覆盖了 .5 边界（否则测了等于没测）', halfCases.length >= 6,
    `${halfCases.length} 个 .5 用例`);
}

/* ---- [2] fitSize 跨语言一致 ---- */
console.log('\n[2] fitSize 前后端逐位一致');
{
  const mismatches = [];
  for (const c of V.fitSize) {
    const js = fitSize(c.w, c.h, c.maxSide);
    if (js.width !== c.out[0] || js.height !== c.out[1]) {
      mismatches.push(`${c.w}×${c.h}@${c.maxSide}: py=${c.out.join('×')} js=${js.width}×${js.height}`);
    }
  }
  ok('全部一致', mismatches.length === 0,
    mismatches.length ? `${mismatches.length} 处不一致，前 3 个：\n     ${mismatches.slice(0, 3).join('\n     ')}`
                      : `${V.fitSize.length} 组尺寸`);

  const scaled = V.fitSize.filter((c) => Math.max(c.w, c.h) > c.maxSide).length;
  ok('测试集包含大量需要缩放的用例', scaled > 400, `${scaled} / ${V.fitSize.length} 组触发缩放`);
}

/* ---- [3] 二进制协议：Python 打包 → JS 解包 ---- */
console.log('\n[3] 二进制协议往返');
{
  const buf = fs.readFileSync(path.join(OUT, 'payload.bin'));
  const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);

  let parsed;
  try {
    parsed = parseInferResponse(ab);
    ok('JS 能解析 Python 产出的包', true, `${buf.length} 字节`);
  } catch (err) {
    ok('JS 能解析 Python 产出的包', false, err.message);
    throw err;
  }

  const { width, height, header } = V.payload;
  ok('宽高正确', parsed.width === width && parsed.height === height,
    `${parsed.width}×${parsed.height}`);
  ok('头部元信息完整透传',
    parsed.meta.model === header.model && parsed.meta.ms === header.ms
      && JSON.stringify(parsed.meta.intrinsics) === JSON.stringify(header.intrinsics),
    `model=${parsed.meta.model} ms=${parsed.meta.ms}`);

  const n = width * height;
  ok('点图长度 = W*H*3', parsed.points.length === n * 3, `${parsed.points.length}`);
  ok('mask 长度 = W*H', parsed.mask?.length === n, `${parsed.mask?.length}`);

  // 按 Python 里同样的公式重算，逐值比对
  let pBad = 0, pMaxErr = 0;
  for (let i = 0; i < n * 3; i++) {
    const expect = Math.fround((i % 101) * 0.5 - 25.0);
    const err = Math.abs(parsed.points[i] - expect);
    if (err > 0) { pBad++; pMaxErr = Math.max(pMaxErr, err); }
  }
  ok('每一个 float32 都逐位相同', pBad === 0, `不符 ${pBad} 个，最大误差 ${pMaxErr}`);

  let mBad = 0;
  for (let i = 0; i < n; i++) if (parsed.mask[i] !== (i * 7 + 3) % 2) mBad++;
  ok('mask 逐字节相同', mBad === 0, `不符 ${mBad} 个`);

  // 非方形且质数尺寸，行列搞反会立刻炸
  ok('宽高没被交换', width !== height && parsed.width === width,
    `${width}×${height}（质数，行列反了会露馅）`);
}

/* ---- [3b] 生成式 3D 的散点云协议：Python pack_cloud → JS parseCloudResponse ---- */
console.log('\n[3b] 散点云协议往返（生成式 3D）');
{
  const buf = fs.readFileSync(path.join(OUT, 'cloud_payload.bin'));
  const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);

  let parsed;
  try {
    parsed = parseCloudResponse(ab);
    ok('JS 能解析 Python 产出的包', true, `${buf.length} 字节`);
  } catch (err) {
    ok('JS 能解析 Python 产出的包', false, err.message);
    throw err;
  }

  const { count, header } = V.cloudPayload;
  ok('点数正确', parsed.count === count, `${parsed.count}（质数，偏移错一个元素就露馅）`);
  ok('头部元信息完整透传',
    parsed.meta.model === header.model && parsed.meta.ms === header.ms,
    `model=${parsed.meta.model} ms=${parsed.meta.ms}`);
  ok('坐标长度 = count*3', parsed.positions.length === count * 3, `${parsed.positions.length}`);
  ok('颜色长度 = count*3', parsed.colors.length === count * 3, `${parsed.colors.length}`);

  // 按 Python 里同样的公式重算，逐值比对
  let pBad = 0;
  for (let i = 0; i < count * 3; i++) {
    if (parsed.positions[i] !== Math.fround(((i * 13) % 251) * 0.02 - 2.5)) pBad++;
  }
  ok('每一个 float32 都逐位相同', pBad === 0, `不符 ${pBad} 个`);

  let cBad = 0;
  for (let i = 0; i < count * 3; i++) {
    if (parsed.colors[i] !== (i * 31 + 7) % 256) cBad++;
  }
  ok('颜色逐字节相同', cBad === 0, `不符 ${cBad} 个`);

  const throwsCloud = (b) => {
    try { parseCloudResponse(b); return false; } catch { return true; }
  };
  const truncated = buf.subarray(0, buf.length - 100);
  ok('数据体被截断 → 抛错',
    throwsCloud(truncated.buffer.slice(truncated.byteOffset, truncated.byteOffset + truncated.byteLength)));
}

/* ---- [4] 损坏输入不应静默通过 ---- */
console.log('\n[4] 异常响应的处理');
{
  const throws = (buf) => {
    try { parseInferResponse(buf); return false; } catch { return true; }
  };
  ok('空响应 → 抛错', throws(new ArrayBuffer(0)));
  ok('只有半个长度字段 → 抛错', throws(new ArrayBuffer(2)));

  const full = fs.readFileSync(path.join(OUT, 'payload.bin'));
  const truncated = full.subarray(0, full.length - 500);
  ok('数据体被截断 → 抛错（而不是解出一堆垃圾）',
    throws(truncated.buffer.slice(truncated.byteOffset, truncated.byteOffset + truncated.byteLength)));

  const badLen = Buffer.from(full);
  badLen.writeUInt32LE(0xfffffff, 0);
  ok('头部长度越界 → 抛错',
    throws(badLen.buffer.slice(badLen.byteOffset, badLen.byteOffset + badLen.byteLength)));
}

console.log(`\n${'─'.repeat(54)}`);
console.log(`  通过 ${pass} 项，失败 ${fail} 项`);
console.log(`${'─'.repeat(54)}\n`);
process.exit(fail ? 1 : 0);
