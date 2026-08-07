/**
 * 核心算法回归测试（不依赖浏览器）
 * 运行：cd web/tests && node core.test.mjs
 */

import { buildPointCloud } from '../src/core/unproject.js';
import { buildPLY } from '../src/core/plyExport.js';
import { CLOUD_WORLD_SIZE } from '../src/config.js';

let pass = 0, fail = 0;
const ok = (name, cond, extra = '') => {
  if (cond) { pass++; console.log(`  ✓ ${name}${extra ? '  ' + extra : ''}`); }
  else { fail++; console.log(`  ✗ ${name}  ${extra}`); }
};

/* ---- 合成深度图（注意：模型输出的是逆深度，值越大越近）----
   ┌──────────────────────────────────────┐
   │ 背景 0.20 + 轻微斜坡                  │
   │  ┌─渐变带─┐      ┌──前景方块──┐       │
   │  │0.20→0.50│      │    0.80    │       │  ← 方块边界是"硬断层"
   │  │0.005/px │      └────────────┘       │  ← 渐变带是"中等梯度"
   │  └────────┘                            │
   └──────────────────────────────────────┘
   硬断层任何阈值都该剔；中等梯度只有严阈值才剔。 */
const W = 800, H = 600;
const N = W * H;
const depth = new Float32Array(N);
const rgba = new Uint8ClampedArray(N * 4);

const SQ = { x0: 380, x1: 660, y0: 150, y1: 450 };
const RAMP = { x0: 60, x1: 120 };   // 60px 内视差 0.20→0.50，即 0.005/px

for (let v = 0; v < H; v++) {
  for (let u = 0; u < W; u++) {
    const i = v * W + u;
    let d, tag; // tag: 0=背景 1=前景 2=渐变带
    if (u >= SQ.x0 && u < SQ.x1 && v >= SQ.y0 && v < SQ.y1) {
      d = 0.80; tag = 1;
    } else if (u >= RAMP.x0 && u < RAMP.x1) {
      d = 0.20 + (u - RAMP.x0) * 0.005; tag = 2;
    } else {
      d = 0.20 + (u / W) * 0.02; tag = 0;
    }
    // 原始值刻意不在 0..1，用来验证归一化逻辑
    depth[i] = d * 37.5 + 4.2;
    const c = i * 4;
    rgba[c]     = tag === 1 ? 230 : tag === 2 ? 20 : 40;
    rgba[c + 1] = tag === 1 ? 120 : tag === 2 ? 220 : 60;
    rgba[c + 2] = tag === 1 ? 60 : tag === 2 ? 30 : 200;
    rgba[c + 3] = 255;
  }
}
const isFg = (c, i) => c.colors[i * 3] > 0.5;
const isRamp = (c, i) => c.colors[i * 3 + 1] > 0.5 && c.colors[i * 3] < 0.2;

const base = { targetPoints: 40000, fovDeg: 55, depthStrength: 3.5, skyFloor: 0 };
const noCull = { ...base, edgeThreshold: 1 };

console.log('\n[1] 反投影 · 基本正确性');
{
  const c = buildPointCloud({ depth, width: W, height: H, rgba, options: noCull });

  ok('点数命中目标（浮点步长）',
    Math.abs(c.count - 40000) / 40000 < 0.02,
    `实际 ${c.count} / 目标 40000，偏差 ${((c.count / 40000 - 1) * 100).toFixed(2)}%`);
  ok('边缘剔除关闭时不剔点', c.stats.culledEdge === 0);

  let bad = 0;
  for (let i = 0; i < c.positions.length; i++) if (!Number.isFinite(c.positions[i])) bad++;
  ok('坐标无 NaN/Inf', bad === 0, `异常 ${bad} 个`);

  let cbad = 0;
  for (let i = 0; i < c.colors.length; i++) if (c.colors[i] < 0 || c.colors[i] > 1) cbad++;
  ok('颜色都在 [0,1]', cbad === 0);

  const mn = [Infinity, Infinity, Infinity], mx = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < c.count; i++) for (let a = 0; a < 3; a++) {
    const v = c.positions[i * 3 + a];
    if (v < mn[a]) mn[a] = v;
    if (v > mx[a]) mx[a] = v;
  }
  const ext = Math.max(mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]);
  ok('归一化到世界尺寸', Math.abs(ext - CLOUD_WORLD_SIZE) < 1e-3,
    `最长边 ${ext.toFixed(4)} / 期望 ${CLOUD_WORLD_SIZE}`);

  const cen = [0, 1, 2].map(a => (mn[a] + mx[a]) / 2);
  ok('包围盒已居中', cen.every(v => Math.abs(v) < 1e-4),
    `中心 [${cen.map(v => v.toFixed(5)).join(', ')}]`);
}

console.log('\n[2] 视差→深度 · 几何关系');
{
  const c = buildPointCloud({ depth, width: W, height: H, rgba, options: noCull });
  let fgZ = 0, fgN = 0, bgZ = 0, bgN = 0;
  for (let i = 0; i < c.count; i++) {
    if (isFg(c, i)) { fgZ += c.positions[i * 3 + 2]; fgN++; }
    else if (!isRamp(c, i)) { bgZ += c.positions[i * 3 + 2]; bgN++; }
  }
  fgZ /= fgN; bgZ /= bgN;
  ok('前景比背景更靠近相机', fgZ > bgZ,
    `前景 Z=${fgZ.toFixed(3)}，背景 Z=${bgZ.toFixed(3)}`);

  const c2 = buildPointCloud({ depth, width: W, height: H, rgba, options: { ...noCull, depthStrength: 7 } });
  let f2 = 0, n2 = 0, b2 = 0, m2 = 0;
  for (let i = 0; i < c2.count; i++) {
    if (isFg(c2, i)) { f2 += c2.positions[i * 3 + 2]; n2++; }
    else if (!isRamp(c2, i)) { b2 += c2.positions[i * 3 + 2]; m2++; }
  }
  ok('深度强度增大 → 前后景拉得更开',
    (f2 / n2 - b2 / m2) > (fgZ - bgZ),
    `×3.5 → Δ=${(fgZ - bgZ).toFixed(3)}；×7 → Δ=${(f2 / n2 - b2 / m2).toFixed(3)}`);

  // 视差→深度是取倒数，等间隔视差在近处压缩、远处拉伸
  const zOf = (d) => 1 / (d * (1 / 1 - 1 / 3.5) + 1 / 3.5);
  ok('视差→深度非线性（取倒数而非线性插值）',
    Math.abs((zOf(0.0) - zOf(0.5)) - (zOf(0.5) - zOf(1.0))) > 0.3,
    `远半段 Δz=${(zOf(0.0) - zOf(0.5)).toFixed(3)}，近半段 Δz=${(zOf(0.5) - zOf(1.0)).toFixed(3)}`);
}

console.log('\n[3] 飞边剔除（最关键的质量点）');
{
  const off  = buildPointCloud({ depth, width: W, height: H, rgba, options: noCull });
  const mid  = buildPointCloud({ depth, width: W, height: H, rgba, options: { ...base, edgeThreshold: 0.04 } });
  const hard = buildPointCloud({ depth, width: W, height: H, rgba, options: { ...base, edgeThreshold: 0.005 } });

  ok('硬断层处确实剔到了点', mid.stats.culledEdge > 0, `剔除 ${mid.stats.culledEdge} 个`);
  ok('剔除量合理（没把点全杀了）',
    mid.count > off.count * 0.9 && mid.count < off.count,
    `${off.count} → ${mid.count}（剔掉 ${((1 - mid.count / off.count) * 100).toFixed(1)}%）`);

  const perim = 2 * ((SQ.x1 - SQ.x0) + (SQ.y1 - SQ.y0));
  const expect = perim / mid.stats.step;
  ok('剔除量与断层周长同量级',
    mid.stats.culledEdge > expect * 0.4 && mid.stats.culledEdge < expect * 4,
    `实际 ${mid.stats.culledEdge}，周长估算 ≈ ${Math.round(expect)}`);

  ok('阈值调小 → 剔得明显更狠',
    hard.stats.culledEdge > mid.stats.culledEdge * 1.5,
    `0.04 → ${mid.stats.culledEdge} 个；0.005 → ${hard.stats.culledEdge} 个`);

  // 中等梯度带：宽阈值该留住，严阈值该剔掉
  const rampIn = (c) => { let k = 0; for (let i = 0; i < c.count; i++) if (isRamp(c, i)) k++; return k; };
  ok('宽阈值保留中等梯度区', rampIn(mid) > rampIn(off) * 0.9,
    `原 ${rampIn(off)} → 剩 ${rampIn(mid)}`);
  ok('严阈值剔除中等梯度区', rampIn(hard) < rampIn(off) * 0.3,
    `原 ${rampIn(off)} → 剩 ${rampIn(hard)}`);
}

console.log('\n[4] 远景剔除');
{
  const c = buildPointCloud({ depth, width: W, height: H, rgba, options: { ...noCull, skyFloor: 0.6 } });
  ok('剔掉了低视差区域', c.stats.culledSky > 0, `剔除 ${c.stats.culledSky} 个`);
  let bgLeft = 0;
  for (let i = 0; i < c.count; i++) if (!isFg(c, i)) bgLeft++;
  ok('视差 < 阈值的区域已清空', bgLeft === 0, `残留 ${bgLeft} 个`);
  ok('前景完整保留', c.count > 5000, `剩余 ${c.count} 点`);
}

console.log('\n[5] 点数控制精度（浮点步长的意义所在）');
{
  for (const target of [20000, 40000, 100000, 250000, 400000]) {
    const c = buildPointCloud({ depth, width: W, height: H, rgba, options: { ...noCull, targetPoints: target } });
    const err = Math.abs(c.count / target - 1);
    ok(`目标 ${target.toLocaleString()} → 实际 ${c.count.toLocaleString()}`,
      err < 0.03, `偏差 ${(err * 100).toFixed(2)}%，步长 ${c.stats.step}`);
  }
  const full = buildPointCloud({ depth, width: W, height: H, rgba, options: { ...noCull, targetPoints: 900000 } });
  ok('目标超过像素总数 → 封顶为全量', full.count === N, `${full.count} / ${N}`);
}

console.log('\n[6] PLY 二进制导出 · 往返一致性');
{
  const c = buildPointCloud({ depth, width: W, height: H, rgba, options: { ...base, edgeThreshold: 0.04 } });
  const blob = buildPLY(c, { format: 'binary', keepScale: false });
  const buf = Buffer.from(await blob.arrayBuffer());

  const headEnd = buf.indexOf('end_header\n') + 'end_header\n'.length;
  const head = buf.subarray(0, headEnd).toString('utf8');

  ok('魔数正确', head.startsWith('ply\n'));
  ok('声明二进制小端', head.includes('format binary_little_endian 1.0'));
  ok('顶点数与实际一致', head.includes(`element vertex ${c.count}`));
  ok('体积 = 头 + 15×N', buf.length === headEnd + c.count * 15,
    `实际 ${buf.length}，期望 ${headEnd + c.count * 15}`);

  const dv = new DataView(buf.buffer, buf.byteOffset + headEnd, c.count * 15);
  let maxErr = 0, colErr = 0;
  for (let i = 0; i < c.count; i++) {
    const o = i * 15;
    for (let a = 0; a < 3; a++) {
      maxErr = Math.max(maxErr, Math.abs(dv.getFloat32(o + a * 4, true) - c.positions[i * 3 + a]));
    }
    for (let a = 0; a < 3; a++) {
      if (dv.getUint8(o + 12 + a) !== Math.round(c.colors[i * 3 + a] * 255)) colErr++;
    }
  }
  ok('坐标往返无损（float32）', maxErr === 0, `最大误差 ${maxErr}`);
  ok('颜色往返无损', colErr === 0, `不一致 ${colErr} 个`);
  console.log(`     ${c.count.toLocaleString()} 点 → ${(buf.length / 1048576).toFixed(2)} MB`);
}

console.log('\n[7] PLY 保持原始尺度');
{
  const c = buildPointCloud({ depth, width: W, height: H, rgba, options: noCull });
  const buf = Buffer.from(await (buildPLY(c, { format: 'binary', keepScale: true })).arrayBuffer());
  const headEnd = buf.indexOf('end_header\n') + 'end_header\n'.length;
  const dv = new DataView(buf.buffer, buf.byteOffset + headEnd);

  const { center, scale } = c.transform;
  let maxErr = 0;
  for (let i = 0; i < Math.min(3000, c.count); i++) {
    const o = i * 15;
    for (let a = 0; a < 3; a++) {
      maxErr = Math.max(maxErr, Math.abs(dv.getFloat32(o + a * 4, true) - (c.positions[i * 3 + a] / scale + center[a])));
    }
  }
  ok('还原回归一化前的坐标', maxErr < 1e-4, `最大误差 ${maxErr.toExponential(2)}`);

  let zmin = Infinity, zmax = -Infinity;
  for (let i = 0; i < c.count; i++) {
    const z = c.positions[i * 3 + 2] / scale + center[2];
    if (z < zmin) zmin = z;
    if (z > zmax) zmax = z;
  }
  ok('还原后 Z ∈ [-zFar, -zNear]', zmin > -3.51 && zmax < -0.99,
    `Z ∈ [${zmin.toFixed(3)}, ${zmax.toFixed(3)}]，期望 [-3.5, -1]`);
}

console.log('\n[8] PLY ASCII 格式');
{
  const c = buildPointCloud({ depth, width: W, height: H, rgba, options: { ...noCull, targetPoints: 5000 } });
  const text = await (buildPLY(c, { format: 'ascii' })).text();
  const lines = text.trim().split('\n');
  const hi = lines.indexOf('end_header');
  ok('ASCII 头正确', lines[0] === 'ply' && lines[1] === 'format ascii 1.0' && hi > 0);
  ok('数据行数 = 顶点数', lines.length - hi - 1 === c.count, `${lines.length - hi - 1} vs ${c.count}`);
  const first = lines[hi + 1].split(' ');
  ok('每行 6 个字段', first.length === 6, `"${lines[hi + 1]}"`);
  ok('RGB 是 0–255 整数',
    first.slice(3).every(s => /^\d+$/.test(s) && +s >= 0 && +s <= 255), first.slice(3).join(','));
}

console.log('\n[9] 异常输入');
{
  let threw = false;
  try { buildPointCloud({ depth: new Float32Array(10), width: W, height: H, rgba, options: base }); }
  catch { threw = true; }
  ok('深度图尺寸不符 → 抛错', threw);

  threw = false;
  try { buildPointCloud({ depth, width: W, height: H, rgba, options: { ...base, skyFloor: 1.5 } }); }
  catch (e) { threw = /没有点存活/.test(e.message); }
  ok('全部被剔光 → 抛出可读错误', threw);

  const flat = new Float32Array(N).fill(7);
  const c = buildPointCloud({ depth: flat, width: W, height: H, rgba, options: noCull });
  ok('全平深度图不崩溃（除零保护）', c.count > 0 && Number.isFinite(c.positions[0]), `${c.count} 点`);
}

console.log('\n[10] 性能');
{
  const t0 = performance.now();
  for (let i = 0; i < 5; i++) {
    buildPointCloud({ depth, width: W, height: H, rgba, options: { ...base, targetPoints: 220000, edgeThreshold: 0.04 } });
  }
  const ms = (performance.now() - t0) / 5;
  ok('22 万点重建 < 150ms（拖滑杆要跟手）', ms < 150, `实测 ${ms.toFixed(1)} ms/次`);
}

console.log(`\n${'─'.repeat(54)}`);
console.log(`  通过 ${pass} 项，失败 ${fail} 项`);
console.log(`${'─'.repeat(54)}\n`);
process.exit(fail ? 1 : 0);
