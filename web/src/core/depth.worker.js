/**
 * 深度估计 Worker
 *
 * 放在 Worker 里跑有两个理由：
 *   1. 首次加载要下 25–50MB 模型，主线程会卡死；
 *   2. WASM 兜底路径推理可能要好几秒。
 *
 * 消息协议（均带 id 用于配对）：
 *   ← { id, type: 'load',  modelKey }
 *   ← { id, type: 'infer', rgba, width, height }
 *   → { id, type: 'progress', payload }
 *   → { id, type: 'loaded',   payload: { device, dtype, modelId } }
 *   → { id, type: 'result',   payload: { depth, width, height, ms } }
 *   → { id, type: 'error',    payload: { message } }
 */

import { pipeline, env, RawImage } from '@huggingface/transformers';
import { HF_ENDPOINT, LOCAL_MODEL_PATH, MODELS } from '../config.js';

// 优先读 public/models/ 下预下载的模型（npm run fetch-model），
// 本地没有再回落到镜像。直连 huggingface.co 在国内网络会超时，所以不用官方源。
env.allowLocalModels = true;
env.localModelPath = LOCAL_MODEL_PATH;
env.allowRemoteModels = true;
env.remoteHost = HF_ENDPOINT;

let pipe = null;
let loadedKey = null;
let backend = { device: 'unknown', dtype: 'unknown' };

/**
 * 决定后端与量化精度，按优先级给出候选链，逐个尝试。
 * WebGPU 上不用 q8：量化模型在 WebGPU 后端经常算错或更慢。
 */
async function candidates() {
  const list = [];
  if (typeof navigator !== 'undefined' && navigator.gpu) {
    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (adapter) {
        if (adapter.features?.has('shader-f16')) list.push({ device: 'webgpu', dtype: 'fp16' });
        list.push({ device: 'webgpu', dtype: 'fp32' });
      }
    } catch { /* 没有 WebGPU 就往下走 */ }
  }
  list.push({ device: 'wasm', dtype: 'q8' });
  return list;
}

async function load(modelKey, report) {
  if (pipe && loadedKey === modelKey) return backend;

  const meta = MODELS[modelKey];
  if (!meta) throw new Error(`未知模型：${modelKey}`);

  if (pipe) {
    try { await pipe.dispose(); } catch { /* 忽略 */ }
    pipe = null;
    loadedKey = null;
  }

  const tries = await candidates();
  let lastErr = null;

  for (const cand of tries) {
    try {
      report({ status: 'init', message: `正在初始化 ${cand.device.toUpperCase()} / ${cand.dtype}` });
      pipe = await pipeline('depth-estimation', meta.id, {
        device: cand.device,
        dtype: cand.dtype,
        progress_callback: report,
      });
      loadedKey = modelKey;
      backend = { ...cand, modelId: meta.id };
      return backend;
    } catch (err) {
      lastErr = err;
      pipe = null;
      report({
        status: 'fallback',
        message: `${cand.device}/${cand.dtype} 不可用，换下一个：${err?.message ?? err}`,
      });
    }
  }

  throw new Error(`所有后端都加载失败。最后一次错误：${lastErr?.message ?? lastErr}`);
}

async function infer(rgba, width, height) {
  if (!pipe) throw new Error('模型尚未加载');

  const t0 = performance.now();

  // canvas 给的是 RGBA，模型要 RGB
  const image = new RawImage(new Uint8ClampedArray(rgba), width, height, 4).rgb();
  const out = await pipe(image);

  // predicted_depth：float32，dims [H, W]，已插值回输入尺寸，未归一化。
  // 比 out.depth（Uint8 0–255）精度高得多，反投影必须用这个。
  const t = out.predicted_depth;
  const [h, w] = t.dims.slice(-2);
  const depth = t.data instanceof Float32Array ? t.data : Float32Array.from(t.data);

  return {
    depth,
    width: w,
    height: h,
    ms: Math.round(performance.now() - t0),
  };
}

self.onmessage = async (e) => {
  const { id, type } = e.data;
  const report = (payload) => self.postMessage({ id, type: 'progress', payload });

  try {
    if (type === 'load') {
      const info = await load(e.data.modelKey, report);
      self.postMessage({ id, type: 'loaded', payload: info });
      return;
    }

    if (type === 'infer') {
      const res = await infer(e.data.rgba, e.data.width, e.data.height);
      self.postMessage({ id, type: 'result', payload: res }, [res.depth.buffer]);
      return;
    }

    throw new Error(`未知消息类型：${type}`);
  } catch (err) {
    self.postMessage({
      id,
      type: 'error',
      payload: { message: err?.message ?? String(err) },
    });
  }
};
