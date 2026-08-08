/**
 * 点云工具 · 总编排
 * 上传 → 深度估计 → 反投影 → 粒子渲染 → 导出
 */

import './style.css';

import {
  PARAM_SCHEMA, defaultParams, MODELS, DEFAULT_MODEL, MAX_INPUT_SIDE, LOCAL_MODEL_PATH,
  BROWSER_ONLY_KEYS,
} from './config.js';
import { prepareImage, canvasToBlob } from './core/imagePrep.js';
import { BrowserDepthEngine } from './core/depthBrowser.js';
import { checkServer, warmupServer, inferOnServer } from './core/depthServer.js';
import { checkGenServer, warmupGenServer, generateOnServer } from './core/gen3dServer.js';
import {
  buildPointCloud, buildPointCloudFromPointMap, buildPointCloudFromCloud,
} from './core/unproject.js';
import {
  buildPLY, downloadBlob, timestampName, estimateSize, formatBytes,
} from './core/plyExport.js';
import { Viewer } from './render/viewer.js';
import { buildPanel } from './ui/panel.js';
import { createStatus, createEngineBadge } from './ui/status.js';
import { createDropzone } from './ui/dropzone.js';

/* ---------------- 状态 ---------------- */

const params = defaultParams();

const state = {
  mode: 'browser',
  modelKey: DEFAULT_MODEL,
  image: null,     // { data, width, height }
  preparedBlob: null, // 缩放后的 PNG，高精度模式发这个（不是原图）
  source: null,    // { depth } 或 { points, mask }，缓存下来供改参数时重建
  cloud: null,
  name: 'pointcloud',
  busy: false,
  engineReady: false,
  localModels: null, // { small: true, base: false }
};

/* ---------------- 初始化 ---------------- */

const status = createStatus();
const badge = createEngineBadge();
const viewer = new Viewer(document.getElementById('stage'));
const engine = new BrowserDepthEngine();

viewer.onFps = (v) => status.setFps(v);
viewer.applyParams(params);

const panel = buildPanel(document.getElementById('paramGroups'), PARAM_SCHEMA, params, onParamChange);

const dz = createDropzone(handleImage);

/* --- 引擎下拉 --- */

const modeSel = document.getElementById('modeSel');
const modelSel = document.getElementById('modelSel');
const modelRow = document.getElementById('modelRow');

for (const [key, m] of Object.entries(MODELS)) {
  const o = document.createElement('option');
  o.value = key;
  o.textContent = `${m.label}（${m.params}）`;
  modelSel.appendChild(o);
}
modelSel.value = state.modelKey;

/**
 * 探测模型是否已经预下载到 public/models/。
 * 不能只看 HTTP 状态码 —— dev server 对不存在的路径会回落到 index.html 且返回 200，
 * 所以还要确认 content-type 真的是 JSON。
 */
async function hasLocalModel(id) {
  try {
    const r = await fetch(`${LOCAL_MODEL_PATH}${id}/config.json`, { cache: 'no-store' });
    if (!r.ok) return false;
    return (r.headers.get('content-type') || '').includes('json');
  } catch {
    return false;
  }
}

async function labelModels() {
  const results = {};
  for (const [key, m] of Object.entries(MODELS)) {
    const local = await hasLocalModel(m.id);
    results[key] = local;
    const opt = modelSel.querySelector(`option[value="${key}"]`);
    if (opt) opt.textContent = `${m.label}（${m.params}）${local ? ' · 已就绪' : ' · 需下载'}`;
  }
  return results;
}

modeSel.addEventListener('change', async () => {
  state.mode = modeSel.value;
  applyModeUI();
  state.source = null;
  await refreshEngineBadge();
  if (state.image) rerunInference();
});

/** 切模式时把当前模式下失效的控件置灰，避免用户拖了没反应还以为是 bug。 */
function applyModeUI() {
  const isBrowser = state.mode === 'browser';
  modelRow.style.display = isBrowser ? '' : 'none';
  panel.setDisabled(
    BROWSER_ONLY_KEYS,
    !isBrowser,
    state.mode === 'gen3d'
      ? '生成式 3D 直接输出完整三维形体，与深度图转换相关的参数不参与计算'
      : '高精度模式使用 MoGe 预测的真实内参与米制点图，此参数不参与计算',
  );
}

modelSel.addEventListener('change', async () => {
  state.modelKey = modelSel.value;
  state.engineReady = false;
  state.source = null;
  badge.set('idle', '模型已切换，下次推理时加载');
  if (state.image) rerunInference();
});

/* --- 面板折叠 --- */

const panelEl = document.getElementById('panel');
document.getElementById('panelToggle').addEventListener('click', () => {
  const collapsed = panelEl.dataset.collapsed === 'true';
  panelEl.dataset.collapsed = String(!collapsed);
});

/* --- 导出 / 动作按钮 --- */

const btnExport = document.getElementById('exportPly');
const btnSnap = document.getElementById('snapshot');
const btnReplay = document.getElementById('replay');
const btnReset = document.getElementById('resetView');
const plyFormat = document.getElementById('plyFormat');
const keepScale = document.getElementById('keepScale');

btnExport.addEventListener('click', onExport);
btnSnap.addEventListener('click', onSnapshot);
btnReplay.addEventListener('click', () => viewer.replay());
btnReset.addEventListener('click', () => viewer.resetView());
plyFormat.addEventListener('change', updateExportLabel);

/* ---------------- 引擎状态 ---------------- */

/**
 * 后端缺失时的提示。
 * 线上那份是纯静态托管，永远不会有后端 —— 访客手上没有仓库，
 * 跟他说「去跑 server/start.ps1」等于没说，所以要讲清楚这是本机功能。
 */
const BACKEND_HINT = (name) =>
  `${name}需要在你自己的电脑上跑一个本机服务（Python + GPU），` +
  '这个网页版提供不了。想用的话按仓库 README 装好后运行 server/start.ps1；' +
  '否则请用「⚡ 浏览器」模式，它完全在本页面里跑。';

async function refreshEngineBadge() {
  if (state.mode === 'gen3d') {
    badge.set('loading', '正在探测本机后端…');
    const info = await checkGenServer();
    if (info?.ok) {
      badge.set('ready', `TripoSR · ${info.device ?? 'cuda'}`);
      if (info.weightsCached === false) {
        // 权重没下过时预热会卡在下载上（1.4GB，镜像不通还会长时间重试），
        // 先说清楚，别让用户对着转圈的界面猜
        badge.set('idle', '模型未下载（约 1.4GB）');
        dz.note('生成式 3D：模型权重还没下载（约 1.4GB），首次使用需联网拉取，可能很慢');
      } else {
        dz.note('生成式 3D 就绪 · 单图生成完整 360° 形体，背面由模型补全（约 10–20 秒）');
        if (!info.loaded) {
          badge.set('loading', '正在预热模型…');
          warmupGenServer().then((r) => {
            badge.set(r?.ok ? 'ready' : 'error',
              r?.ok ? `TripoSR · ${r.device ?? ''}`.trim() : '预热失败，首图会较慢');
          });
        }
      }
    } else {
      badge.set('error', '需要本机后端');
      dz.note(BACKEND_HINT('生成式 3D'));
    }
    return;
  }

  if (state.mode === 'server') {
    badge.set('loading', '正在探测本机后端…');
    const info = await checkServer();
    if (info?.ok && info.weightsCached === false) {
      badge.set('idle', '模型未下载（约 1.2GB）');
      dz.note('高精度模式：MoGe 权重还没下载（约 1.2GB），首次使用需联网拉取，可能很慢');
      return;
    }
    if (info?.ok) {
      badge.set('ready', `${info.model ?? 'MoGe'} · ${info.device ?? 'cuda'}`);
      dz.note(`高精度模式就绪 · ${info.gpu ?? info.device ?? ''}`.trim());
      // 模型还没进显存就先预热，别让第一张图等模型加载
      if (!info.loaded) {
        badge.set('loading', '正在预热模型…');
        warmupServer().then((r) => {
          badge.set(r?.ok ? 'ready' : 'error',
            r?.ok ? `${r.model ?? 'MoGe'} · ${r.device ?? ''}`.trim() : '预热失败，首图会较慢');
        });
      }
    } else {
      badge.set('error', '需要本机后端');
      dz.note(BACKEND_HINT('高精度模式'));
    }
    return;
  }

  if (state.engineReady && engine.backend) {
    const b = engine.backend;
    badge.set('ready', `${b.device.toUpperCase()} · ${b.dtype}`);
  } else {
    const m = MODELS[state.modelKey];
    badge.set('idle', state.localModels?.[state.modelKey]
      ? '未加载（本地已就绪）'
      : `未加载（首次约 ${m.sizeMB.fp16}MB）`);
  }
}

async function probeWebGPU() {
  if (!navigator.gpu) return 'WebGPU 不可用，将用 CPU（WASM）推理';
  try {
    const a = await navigator.gpu.requestAdapter();
    if (!a) return '拿不到 WebGPU 适配器，将用 CPU 推理';
    return a.features?.has('shader-f16') ? 'WebGPU 就绪（fp16）' : 'WebGPU 就绪（fp32）';
  } catch {
    return 'WebGPU 探测失败，将用 CPU 推理';
  }
}

async function boot() {
  applyModeUI();
  const [gpuNote, localMap] = await Promise.all([probeWebGPU(), labelModels()]);
  state.localModels = localMap;
  dz.note([
    gpuNote,
    localMap[state.modelKey]
      ? '模型已本地就绪，无需联网'
      : `模型未预下载，首次使用需联网拉取约 ${MODELS[state.modelKey].sizeMB.fp16}MB（建议先跑 npm run fetch-model）`,
  ].join(' · '));
  await refreshEngineBadge();
}

/* ---------------- 主流程 ---------------- */

async function handleImage(fileOrBlob, name = 'pointcloud') {
  if (state.busy) return;
  state.name = name.replace(/\.[^.]+$/, '') || 'pointcloud';

  try {
    state.busy = true;
    status.show('正在解码图片…');

    const img = await prepareImage(fileOrBlob, MAX_INPUT_SIDE);
    state.image = img;
    // 后端收到的必须是这份已缩放的像素，尺寸才能和前端逐位一致
    state.preparedBlob = await canvasToBlob(img.canvas);
    status.setSize(img.width, img.height);

    await runInference();
  } catch (err) {
    console.error(err);
    status.error(String(err?.message ?? err));
  } finally {
    state.busy = false;
  }
}

async function rerunInference() {
  if (state.busy || !state.image) return;
  try {
    state.busy = true;
    await runInference();
  } catch (err) {
    console.error(err);
    status.error(String(err?.message ?? err));
  } finally {
    state.busy = false;
  }
}

async function runInference() {
  const t0 = performance.now();

  if (state.mode === 'gen3d') {
    status.show('正在生成 3D 形体（含背面补全，约 10–20 秒）…');
    // 多要一点余量，让「点数」滑杆后续加密时不用重跑模型
    const res = await generateOnServer(state.preparedBlob, {
      points: Math.max(600000, params.targetPoints),
    });
    state.source = { kind: 'cloud', ...res };
    badge.set('ready', `TripoSR · ${res.meta?.device ?? 'cuda'}`);
  } else if (state.mode === 'server') {
    status.show('正在调用本机后端…');
    const res = await inferOnServer(state.preparedBlob, { maxSide: MAX_INPUT_SIDE });
    // 尺寸对不上意味着颜色会整体错位，宁可报错也别默默出一张花图
    if (res.width !== state.image.width || res.height !== state.image.height) {
      throw new Error(
        `后端返回 ${res.width}×${res.height}，与前端 ${state.image.width}×${state.image.height} 不一致，颜色会错位`,
      );
    }
    state.source = { kind: 'pointmap', ...res };
    badge.set('ready', `${res.meta?.model ?? 'MoGe'} · ${res.meta?.device ?? 'cuda'}`);
  } else {
    if (!state.engineReady) {
      badge.set('loading', '正在加载模型…');
      await engine.load(state.modelKey, onModelProgress);
      state.engineReady = true;
      const b = engine.backend;
      badge.set('ready', `${b.device.toUpperCase()} · ${b.dtype}`);
    }

    status.show('正在估计深度…');
    const res = await engine.infer(state.image);
    state.source = { kind: 'depth', ...res };
  }

  status.show('正在生成点云…');
  rebuildCloud();

  dz.hide();
  status.showBar();
  status.setTime(Math.round(performance.now() - t0));
  status.done(`完成 · ${state.cloud.count.toLocaleString('zh-CN')} 点`);

  viewer.replay();
  for (const b of [btnExport, btnSnap, btnReplay, btnReset]) b.disabled = false;
  updateExportLabel();
}

/** 只重算点云，不重跑模型 —— 改 FOV / 点数 / 剔除阈值走这条路，很快。 */
function rebuildCloud(replay = false) {
  if (!state.source || !state.image) return;

  const t0 = performance.now();
  const cloud = state.source.kind === 'cloud'
    ? buildPointCloudFromCloud({
        positions: state.source.positions,
        colors: state.source.colors,
        count: state.source.count,
        options: params,
      })
    : state.source.kind === 'pointmap'
    ? buildPointCloudFromPointMap({
        points: state.source.points,
        mask: state.source.mask,
        width: state.source.width,
        height: state.source.height,
        rgba: state.image.data,
        options: params,
      })
    : buildPointCloud({
        depth: state.source.depth,
        width: state.source.width,
        height: state.source.height,
        rgba: state.image.data,
        options: params,
      });

  state.cloud = cloud;
  viewer.setCloud(cloud);
  viewer.applyParams(params);
  if (replay) viewer.replay();
  else viewer.cloud.snapToEnd();

  status.setPoints(cloud.count);
  console.debug(
    `[点云] ${cloud.count} 点 · 步长 ${cloud.stats.step} · ` +
    `边缘剔除 ${cloud.stats.culledEdge} · 远景剔除 ${cloud.stats.culledSky} · ` +
    `${Math.round(performance.now() - t0)}ms`,
  );
  updateExportLabel();
}

/* ---------------- 参数变更 ---------------- */

let rebuildTimer = 0;

function onParamChange(key, value, needsRebuild) {
  if (needsRebuild) {
    // 拖滑杆时别每一帧都重算，防抖
    clearTimeout(rebuildTimer);
    rebuildTimer = setTimeout(() => {
      try {
        rebuildCloud();
      } catch (err) {
        status.error(String(err?.message ?? err));
      }
    }, 130);
  } else {
    viewer.applyParams(params);
  }
}

/* ---------------- 模型下载进度 ---------------- */

function onModelProgress(p) {
  if (p.status === 'progress' && p.total) {
    const pct = p.loaded / p.total;
    status.show(
      `正在下载模型 ${p.file ?? ''} ${(p.loaded / 1048576).toFixed(1)} / ${(p.total / 1048576).toFixed(1)} MB`,
      pct,
    );
    badge.set('loading', `下载中 ${Math.round(pct * 100)}%`);
  } else if (p.status === 'init' || p.status === 'initiate') {
    status.show(p.message ?? '正在准备模型…');
  } else if (p.status === 'fallback') {
    console.warn(p.message);
    status.show(p.message);
  } else if (p.status === 'ready' || p.status === 'done') {
    status.show('模型就绪');
  }
}

/* ---------------- 导出 ---------------- */

function updateExportLabel() {
  if (!state.cloud) return;
  const bytes = estimateSize(state.cloud.count, plyFormat.value);
  btnExport.textContent = `导出 PLY（${formatBytes(bytes)}）`;
}

function onExport() {
  if (!state.cloud) return;
  try {
    status.show('正在打包 PLY…');
    const blob = buildPLY(state.cloud, {
      format: plyFormat.value,
      keepScale: keepScale.checked,
    });
    downloadBlob(blob, timestampName(state.name, 'ply'));
    status.done(`已导出 ${formatBytes(blob.size)}`);
  } catch (err) {
    console.error(err);
    status.error(String(err?.message ?? err));
  }
}

async function onSnapshot() {
  try {
    const blob = await viewer.snapshot();
    if (!blob) throw new Error('截图失败');
    downloadBlob(blob, timestampName(state.name, 'png'));
    status.done('已保存截图');
  } catch (err) {
    status.error(String(err?.message ?? err));
  }
}

/* ---------------- 启动 ---------------- */

boot();

// 方便在控制台调试
window.__pcs = { state, params, viewer, engine, rebuildCloud };
