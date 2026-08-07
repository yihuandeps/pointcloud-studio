/**
 * 全局配置：模型清单、镜像地址、参数默认值与取值范围。
 */

/**
 * HuggingFace 镜像。
 * 实测本机环境 huggingface.co 直连超时、hf-mirror.com 可达，故默认走镜像。
 * 若你的网络能直连，把这里改回 'https://huggingface.co/' 即可（结尾斜杠不能少）。
 */
export const HF_ENDPOINT = 'https://hf-mirror.com/';

/**
 * 本地模型目录（相对站点根，对应 public/models/）。
 * 跑过 `npm run fetch-model` 之后模型就在这里，运行时零网络依赖。
 * 本地找不到会自动回落到上面的镜像。
 */
export const LOCAL_MODEL_PATH = import.meta.env.BASE_URL + 'models/';

/** 可选的浏览器端深度模型。体积为 fp16 / q8 两种量化的大致值。 */
export const MODELS = {
  small: {
    id: 'onnx-community/depth-anything-v2-small',
    label: 'Depth Anything V2 · Small',
    params: '25M',
    note: '首选，速度最快',
    sizeMB: { fp16: 50, q8: 25 },
  },
  base: {
    id: 'onnx-community/depth-anything-v2-base',
    label: 'Depth Anything V2 · Base',
    params: '97M',
    note: '更细腻，下载更大',
    sizeMB: { fp16: 195, q8: 98 },
  },
};

export const DEFAULT_MODEL = 'small';

/** 送入模型前把图片长边限制到此值：模型内部只用 518px，喂太大纯属浪费。 */
export const MAX_INPUT_SIDE = 1024;

/** 点云归一化后的目标尺寸（最长边占多少世界单位）。 */
export const CLOUD_WORLD_SIZE = 2.4;

/**
 * 参数定义。panel.js 直接读这张表来生成 UI，
 * unproject.js / pointCloud.js 读同名字段取值。
 */
export const PARAM_SCHEMA = [
  {
    group: '点云重建',
    items: [
      {
        key: 'targetPoints', label: '点数', type: 'range',
        min: 20000, max: 800000, step: 10000, value: 220000,
        format: (v) => (v >= 10000 ? (v / 10000).toFixed(1) + ' 万' : v),
        hint: '越多越细腻，越吃显存。20 万左右是画质与流畅的平衡点。',
        rebuild: true,
      },
      {
        key: 'fovDeg', label: '视场角 FOV', type: 'range',
        min: 20, max: 100, step: 1, value: 55,
        format: (v) => v + '°',
        hint: '拍摄相机的水平视场角。普通手机约 55–70°，长焦更小。数值越小，画面被"推远压平"。',
        rebuild: true, browserOnly: true,
      },
      {
        key: 'depthStrength', label: '深度强度', type: 'range',
        min: 1.2, max: 8, step: 0.1, value: 3.5,
        format: (v) => '×' + v.toFixed(1),
        hint: '最远/最近的距离比。越大立体感越强，过大会把背景拉爆成噪点。',
        rebuild: true, browserOnly: true,
      },
      {
        key: 'edgeThreshold', label: '边缘剔除', type: 'range',
        min: 0.005, max: 0.3, step: 0.005, value: 0.04,
        format: (v) => v.toFixed(3),
        hint: '剔除前景/背景交界处被拉长的"面条"点。调小=剔得更狠更干净，调大=保留更多点。',
        rebuild: true, browserOnly: true,
      },
      {
        key: 'skyFloor', label: '远景剔除', type: 'range',
        min: 0, max: 0.4, step: 0.005, value: 0.02,
        format: (v) => v.toFixed(3),
        hint: '丢掉视差低于此值的点（天空、无穷远背景）。0 = 全部保留。',
        rebuild: true, browserOnly: true,
      },
    ],
  },
  {
    group: '视觉',
    items: [
      {
        key: 'pointSize', label: '点大小', type: 'range',
        min: 0.2, max: 6, step: 0.05, value: 1.6,
        format: (v) => v.toFixed(2),
      },
      {
        key: 'opacity', label: '不透明度', type: 'range',
        min: 0.05, max: 1, step: 0.01, value: 0.95,
        format: (v) => Math.round(v * 100) + '%',
      },
      {
        key: 'softness', label: '边缘柔和', type: 'range',
        min: 0, max: 0.5, step: 0.01, value: 0.18,
        format: (v) => v.toFixed(2),
        hint: '0 = 硬边方形感，越大越像柔和光点。',
      },
      {
        key: 'solidMode', label: '实体模式', type: 'toggle', value: false,
        hint: '开 = 写深度缓冲，遮挡关系正确，像真实点云；关 = 云雾感，通透飘逸。',
      },
      {
        key: 'tint', label: '色调', type: 'select', value: 'origin',
        options: [
          { v: 'origin', label: '原图色彩' },
          { v: 'depth', label: '深度渐变' },
          { v: 'mono', label: '单色霓虹' },
          { v: 'mix', label: '原图 × 深度' },
        ],
      },
    ],
  },
  {
    group: '动效',
    items: [
      {
        key: 'noiseAmp', label: '呼吸幅度', type: 'range',
        min: 0, max: 0.12, step: 0.001, value: 0.012,
        format: (v) => v.toFixed(3),
        hint: '粒子随时间的有机浮动。0 = 完全静止。',
      },
      {
        key: 'noiseFreq', label: '呼吸频率', type: 'range',
        min: 0.2, max: 6, step: 0.1, value: 1.6,
        format: (v) => v.toFixed(1),
      },
      {
        key: 'repelRadius', label: '鼠标斥力半径', type: 'range',
        min: 0, max: 1.5, step: 0.01, value: 0.45,
        format: (v) => v.toFixed(2),
        hint: '0 = 关闭鼠标交互。',
      },
      {
        key: 'repelStrength', label: '鼠标斥力强度', type: 'range',
        min: 0, max: 0.8, step: 0.01, value: 0.22,
        format: (v) => v.toFixed(2),
      },
      {
        key: 'autoRotate', label: '自动旋转', type: 'toggle', value: false,
      },
    ],
  },
];

/** 从 schema 抽出扁平的默认参数对象。 */
export function defaultParams() {
  const out = {};
  for (const g of PARAM_SCHEMA) {
    for (const it of g.items) out[it.key] = it.value;
  }
  return out;
}

/** 哪些参数变化需要重建点云（而非仅改 uniform）。 */
export const REBUILD_KEYS = new Set(
  PARAM_SCHEMA.flatMap((g) => g.items.filter((i) => i.rebuild).map((i) => i.key)),
);

/**
 * 只在浏览器模式下有意义的参数。
 * 高精度模式用 MoGe 直接给的米制点图和相机内参，不需要猜 FOV、
 * 不需要视差→深度转换，有效性也由模型自带的 mask 决定，这些滑杆都是空转。
 */
export const BROWSER_ONLY_KEYS = PARAM_SCHEMA
  .flatMap((g) => g.items.filter((i) => i.browserOnly).map((i) => i.key));
