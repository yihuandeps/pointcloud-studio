/**
 * 端到端验证（Node 版）：镜像下载 → 真实推理 → 反投影 → 导出 PLY
 *
 * 目的是验证三件浏览器里同样依赖的事：
 *   1. hf-mirror 镜像能不能真下到模型文件
 *   2. transformers.js v4 的 predicted_depth 格式是否如假设（float32 / [H,W] / 未归一化）
 *   3. 整条链路能否产出可被 CloudCompare 打开的 PLY
 *
 * 运行：cd web/tests && node e2e.model.mjs
 * 产物：web/tests/out/e2e_test.ply
 */

import fs from 'node:fs';
import path from 'node:path';
import { pipeline, env, RawImage } from '@huggingface/transformers';
import { buildPointCloud } from '../src/core/unproject.js';
import { buildPLY } from '../src/core/plyExport.js';
import { HF_ENDPOINT, MODELS } from '../src/config.js';

// 浏览器里 localModelPath 是 URL 路径 '/models/'，Node 里得给文件系统路径
const PUBLIC_MODELS = path.join(import.meta.dirname, '..', 'public', 'models');
env.allowLocalModels = true;
env.localModelPath = PUBLIC_MODELS;
env.allowRemoteModels = true;
env.remoteHost = HF_ENDPOINT;

let pass = 0, fail = 0;
const ok = (name, cond, extra = '') => {
  if (cond) { pass++; console.log(`  ✓ ${name}${extra ? '  ' + extra : ''}`); }
  else { fail++; console.log(`  ✗ ${name}  ${extra}`); }
};

/* ---- 合成一张有明确深度线索的 RGB 图 ---- */
const W = 512, H = 384;
const rgb = new Uint8ClampedArray(W * H * 3);
const rgba = new Uint8ClampedArray(W * H * 4);
for (let v = 0; v < H; v++) {
  for (let u = 0; u < W; u++) {
    const i = v * W + u;
    // 背景：上暗下亮的渐变（模拟天空→地面）
    let r = 30 + (v / H) * 90, g = 40 + (v / H) * 80, b = 70 + (v / H) * 60;
    // 前景：一个带高光的球
    const dx = (u - W * 0.5) / (W * 0.22);
    const dy = (v - H * 0.55) / (W * 0.22);
    const d2 = dx * dx + dy * dy;
    if (d2 < 1) {
      const sh = Math.sqrt(1 - d2);                   // 法线 z 分量，做朗伯着色
      const lit = Math.max(0, 0.25 + 0.85 * sh - 0.3 * dx + 0.25 * dy);
      r = 240 * lit; g = 150 * lit; b = 90 * lit;
    }
    rgb[i * 3] = r; rgb[i * 3 + 1] = g; rgb[i * 3 + 2] = b;
    rgba[i * 4] = r; rgba[i * 4 + 1] = g; rgba[i * 4 + 2] = b; rgba[i * 4 + 3] = 255;
  }
}

const meta = MODELS.small;
console.log(`\n本地模型目录：${PUBLIC_MODELS}`);
console.log(`回落镜像：${HF_ENDPOINT}`);
console.log(`模型：${meta.id}（q8 量化，约 ${meta.sizeMB.q8}MB）\n`);

console.log('[1] 加载模型');
let pipe;
{
  const t0 = Date.now();
  let sawDownload = false;
  try {
    pipe = await pipeline('depth-estimation', meta.id, {
      dtype: 'q8',
      progress_callback: (p) => {
        if (p.status === 'progress' && p.total) {
          sawDownload = true;
          process.stdout.write(
            `\r     下载 ${p.file} ${(p.loaded / 1048576).toFixed(1)}/${(p.total / 1048576).toFixed(1)} MB   `,
          );
        }
      },
    });
    if (sawDownload) process.stdout.write('\r' + ' '.repeat(70) + '\r');
    ok('镜像可用，模型加载成功', true, `耗时 ${((Date.now() - t0) / 1000).toFixed(1)}s`);
  } catch (err) {
    ok('镜像可用，模型加载成功', false, String(err?.message ?? err));
    console.log('\n  → 后续测试无法继续。检查网络或改 config.js 的 HF_ENDPOINT。\n');
    process.exit(1);
  }
}

console.log('\n[2] 推理输出格式（浏览器端依赖同样的假设）');
let depth, dw, dh;
{
  const image = new RawImage(rgb, W, H, 3);
  const t0 = Date.now();
  const out = await pipe(image);
  const ms = Date.now() - t0;

  ok('返回 predicted_depth 与 depth 两个字段',
    out.predicted_depth != null && out.depth != null);

  const t = out.predicted_depth;
  ok('predicted_depth 是二维 [H, W]', t.dims.length === 2, `dims = [${t.dims.join(', ')}]`);
  ok('已插值回输入尺寸', t.dims[0] === H && t.dims[1] === W,
    `${t.dims[0]}×${t.dims[1]} vs 输入 ${H}×${W}`);
  ok('类型为 float32', t.type === 'float32', `实际 ${t.type}`);

  depth = t.data instanceof Float32Array ? t.data : Float32Array.from(t.data);
  dh = t.dims[0]; dw = t.dims[1];

  let mn = Infinity, mx = -Infinity, nan = 0;
  for (const v of depth) {
    if (!Number.isFinite(v)) { nan++; continue; }
    if (v < mn) mn = v;
    if (v > mx) mx = v;
  }
  ok('无 NaN/Inf', nan === 0, `异常 ${nan} 个`);
  ok('输出未归一化（不是 0–1，说明是原始逆深度）',
    !(mn >= 0 && mx <= 1.0001), `范围 [${mn.toFixed(3)}, ${mx.toFixed(3)}]`);
  ok('有真实深度层次（不是一片死值）', mx - mn > 1e-3, `跨度 ${(mx - mn).toFixed(3)}`);

  // 球在图像中心且更近 → 中心的逆深度应大于四角
  const at = (u, v) => depth[Math.round(v) * dw + Math.round(u)];
  const center = at(dw / 2, dh * 0.55);
  const corners = (at(4, 4) + at(dw - 5, 4) + at(4, dh - 5) + at(dw - 5, dh - 5)) / 4;
  ok('中心的球比四角更近（逆深度更大）', center > corners,
    `中心 ${center.toFixed(3)} vs 四角均值 ${corners.toFixed(3)}`);

  ok('depth 字段是 Uint8 单通道 RawImage',
    out.depth.channels === 1 && out.depth.data instanceof Uint8Array,
    `channels=${out.depth.channels}`);

  console.log(`     推理耗时 ${ms} ms（Node/CPU；浏览器 WebGPU 会快得多）`);
}

console.log('\n[3] 真实深度图 → 点云');
let cloud;
{
  cloud = buildPointCloud({
    depth, width: dw, height: dh, rgba,
    options: { targetPoints: 150000, fovDeg: 55, depthStrength: 3.5, edgeThreshold: 0.04, skyFloor: 0.02 },
  });

  ok('成功生成点云', cloud.count > 1000, `${cloud.count.toLocaleString()} 点`);
  ok('剔除量在合理范围（<40%）',
    (cloud.stats.culledEdge + cloud.stats.culledSky) < cloud.stats.candidates * 0.4,
    `边缘 ${cloud.stats.culledEdge} + 远景 ${cloud.stats.culledSky} / 候选 ${cloud.stats.candidates}`);

  let bad = 0;
  for (let i = 0; i < cloud.positions.length; i++) if (!Number.isFinite(cloud.positions[i])) bad++;
  ok('坐标全部有限', bad === 0);

  // 球（暖色）应该比背景（冷色）更靠近相机
  let warmZ = 0, warmN = 0, coolZ = 0, coolN = 0;
  for (let i = 0; i < cloud.count; i++) {
    const r = cloud.colors[i * 3], b = cloud.colors[i * 3 + 2];
    if (r > b + 0.15) { warmZ += cloud.positions[i * 3 + 2]; warmN++; }
    else if (b > r + 0.05) { coolZ += cloud.positions[i * 3 + 2]; coolN++; }
  }
  ok('模型正确识别出球在前、背景在后',
    warmN > 100 && coolN > 100 && warmZ / warmN > coolZ / coolN,
    `球 Z=${(warmZ / warmN).toFixed(3)}（${warmN} 点） vs 背景 Z=${(coolZ / coolN).toFixed(3)}（${coolN} 点）`);
}

console.log('\n[4] 导出 PLY');
{
  const outDir = path.join(import.meta.dirname, 'out');
  fs.mkdirSync(outDir, { recursive: true });
  const file = path.join(outDir, 'e2e_test.ply');

  const blob = buildPLY(cloud, { format: 'binary', keepScale: false });
  fs.writeFileSync(file, Buffer.from(await blob.arrayBuffer()));

  const st = fs.statSync(file);
  ok('PLY 已写入磁盘', st.size > 0, `${(st.size / 1048576).toFixed(2)} MB`);

  const fd = fs.openSync(file, 'r');
  const headBuf = Buffer.alloc(256);
  fs.readSync(fd, headBuf, 0, 256, 0);
  fs.closeSync(fd);
  const head = headBuf.toString('utf8');
  ok('文件头是合法 PLY', head.startsWith('ply\nformat binary_little_endian 1.0'));
  ok('顶点数正确', head.includes(`element vertex ${cloud.count}`));

  console.log(`\n     产物：${file}`);
  console.log('     可以直接用 CloudCompare / MeshLab / Blender 打开验证');
}

console.log(`\n${'─'.repeat(54)}`);
console.log(`  通过 ${pass} 项，失败 ${fail} 项`);
console.log(`${'─'.repeat(54)}\n`);
process.exit(fail ? 1 : 0);
