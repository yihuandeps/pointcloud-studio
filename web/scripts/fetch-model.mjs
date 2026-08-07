/**
 * 预下载模型到 public/models/，让运行时零网络依赖。
 *
 * 为什么需要这个脚本：
 *   - 本机直连 huggingface.co 超时，必须走 hf-mirror 镜像；
 *   - 而镜像对权重文件只返回 307，真正的字节要去 us.aws.cdn.hf.co 拿；
 *   - Node 的全局 fetch（undici）在本机连不上镜像，node:https 却可以。
 * 所以这里手写 https + 跟随重定向，绕开上面所有坑。
 *
 * 用法：
 *   npm run fetch-model                     # 下载默认的 small 模型
 *   node scripts/fetch-model.mjs base       # 下载 base 模型
 *   node scripts/fetch-model.mjs small fp32 # 额外带上 fp32 权重
 */

import fs from 'node:fs';
import path from 'node:path';
import https from 'node:https';
import { fileURLToPath } from 'node:url';
import { HF_ENDPOINT, MODELS } from '../src/config.js';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const OUT_ROOT = path.join(ROOT, 'public', 'models');

const modelKey = process.argv[2] || 'small';
const extraDtypes = process.argv.slice(3);

const meta = MODELS[modelKey];
if (!meta) {
  console.error(`未知模型 "${modelKey}"，可选：${Object.keys(MODELS).join(' / ')}`);
  process.exit(1);
}

// transformers.js 会按 <localModelPath>/<repo>/<file> 去取
const FILES = [
  'config.json',
  'preprocessor_config.json',
  'onnx/model_fp16.onnx',       // WebGPU 走这个
  'onnx/model_quantized.onnx',  // WASM 兜底走这个
  ...extraDtypes.map((d) => (d === 'fp32' ? 'onnx/model.onnx' : `onnx/model_${d}.onnx`)),
];

function get(url, redirects = 0) {
  return new Promise((resolve, reject) => {
    if (redirects > 6) return reject(new Error('重定向次数过多'));
    const req = https.get(url, { timeout: 120000 }, (res) => {
      const { statusCode, headers } = res;
      if (statusCode >= 300 && statusCode < 400 && headers.location) {
        res.resume();
        const next = new URL(headers.location, url).toString();
        return resolve(get(next, redirects + 1));
      }
      if (statusCode !== 200) {
        res.resume();
        return reject(new Error(`HTTP ${statusCode}`));
      }
      resolve(res);
    });
    req.on('timeout', () => { req.destroy(new Error('连接超时')); });
    req.on('error', reject);
  });
}

function human(n) {
  if (n < 1024) return `${n} B`;
  if (n < 1048576) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1048576).toFixed(1)} MB`;
}

async function download(rel) {
  const url = `${HF_ENDPOINT.replace(/\/$/, '')}/${meta.id}/resolve/main/${rel}`;
  const dest = path.join(OUT_ROOT, ...meta.id.split('/'), ...rel.split('/'));

  if (fs.existsSync(dest) && fs.statSync(dest).size > 0) {
    console.log(`  · ${rel}  已存在，跳过（${human(fs.statSync(dest).size)}）`);
    return true;
  }

  fs.mkdirSync(path.dirname(dest), { recursive: true });
  const tmp = `${dest}.part`;

  process.stdout.write(`  ↓ ${rel}  连接中…`);
  let res;
  try {
    res = await get(url);
  } catch (err) {
    process.stdout.write(`\r  ✗ ${rel}  ${err.message}${' '.repeat(30)}\n`);
    return false;
  }

  const total = Number(res.headers['content-length'] || 0);
  let done = 0;
  let lastPaint = 0;
  // 非 TTY（比如日志重定向）时 \r 不会覆盖，刷进度只会刷屏，直接关掉
  const live = process.stdout.isTTY;

  await new Promise((resolve, reject) => {
    const out = fs.createWriteStream(tmp);
    res.on('data', (chunk) => {
      done += chunk.length;
      const now = Date.now();
      if (live && now - lastPaint > 200) {
        lastPaint = now;
        const pct = total ? ` ${((done / total) * 100).toFixed(1)}%` : '';
        process.stdout.write(`\r  ↓ ${rel}  ${human(done)}${total ? ' / ' + human(total) : ''}${pct}${' '.repeat(12)}`);
      }
    });
    res.on('error', reject);
    out.on('error', reject);
    out.on('finish', resolve);
    res.pipe(out);
  }).catch((err) => {
    fs.rmSync(tmp, { force: true });
    process.stdout.write(`\r  ✗ ${rel}  ${err.message}${' '.repeat(30)}\n`);
    throw err;
  });

  if (total && done !== total) {
    fs.rmSync(tmp, { force: true });
    process.stdout.write(`\r  ✗ ${rel}  下载不完整 ${human(done)}/${human(total)}\n`);
    return false;
  }

  fs.renameSync(tmp, dest);
  process.stdout.write(`\r  ✓ ${rel}  ${human(done)}${' '.repeat(28)}\n`);
  return true;
}

console.log(`\n模型：${meta.label}（${meta.id}）`);
console.log(`来源：${HF_ENDPOINT}`);
console.log(`目标：public/models/${meta.id}/\n`);

let okCount = 0;
for (const f of FILES) {
  try {
    if (await download(f)) okCount++;
  } catch { /* 已打印 */ }
}

console.log(`\n完成 ${okCount}/${FILES.length} 个文件。`);
if (okCount < FILES.length) {
  console.log('有文件没下下来，重跑本脚本会跳过已完成的部分继续下。\n');
  process.exit(1);
}
console.log('运行时将直接从本地加载，不再需要联网。\n');
