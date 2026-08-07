/**
 * 粒子云着色器。
 *
 * 所有特效（聚合动画、呼吸扰动、鼠标斥力）都只在顶点着色器里做，
 * CPU 端的 positions 始终保持干净 —— 这样导出的 PLY 才是正确的点云。
 *
 * 写的是 GLSL1 语法（attribute / varying / gl_FragColor），
 * three.js 会自动转成 WebGL2 的 GLSL ES 3.00。
 */

/* Ashima 3D simplex noise —— 用来让粒子有机地飘动 */
const SIMPLEX_3D = /* glsl */ `
vec3 mod289(vec3 x){ return x - floor(x * (1.0/289.0)) * 289.0; }
vec4 mod289(vec4 x){ return x - floor(x * (1.0/289.0)) * 289.0; }
vec4 permute(vec4 x){ return mod289(((x*34.0)+1.0)*x); }
vec4 taylorInvSqrt(vec4 r){ return 1.79284291400159 - 0.85373472095314 * r; }

float snoise(vec3 v){
  const vec2 C = vec2(1.0/6.0, 1.0/3.0);
  const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);

  vec3 i  = floor(v + dot(v, C.yyy));
  vec3 x0 = v - i + dot(i, C.xxx);

  vec3 g = step(x0.yzx, x0.xyz);
  vec3 l = 1.0 - g;
  vec3 i1 = min(g.xyz, l.zxy);
  vec3 i2 = max(g.xyz, l.zxy);

  vec3 x1 = x0 - i1 + C.xxx;
  vec3 x2 = x0 - i2 + C.yyy;
  vec3 x3 = x0 - D.yyy;

  i = mod289(i);
  vec4 p = permute(permute(permute(
             i.z + vec4(0.0, i1.z, i2.z, 1.0))
           + i.y + vec4(0.0, i1.y, i2.y, 1.0))
           + i.x + vec4(0.0, i1.x, i2.x, 1.0));

  float n_ = 0.142857142857;
  vec3 ns = n_ * D.wyz - D.xzx;

  vec4 j = p - 49.0 * floor(p * ns.z * ns.z);

  vec4 x_ = floor(j * ns.z);
  vec4 y_ = floor(j - 7.0 * x_);

  vec4 x = x_ * ns.x + ns.yyyy;
  vec4 y = y_ * ns.x + ns.yyyy;
  vec4 h = 1.0 - abs(x) - abs(y);

  vec4 b0 = vec4(x.xy, y.xy);
  vec4 b1 = vec4(x.zw, y.zw);

  vec4 s0 = floor(b0) * 2.0 + 1.0;
  vec4 s1 = floor(b1) * 2.0 + 1.0;
  vec4 sh = -step(h, vec4(0.0));

  vec4 a0 = b0.xzyw + s0.xzyw * sh.xxyy;
  vec4 a1 = b1.xzyw + s1.xzyw * sh.zzww;

  vec3 p0 = vec3(a0.xy, h.x);
  vec3 p1 = vec3(a0.zw, h.y);
  vec3 p2 = vec3(a1.xy, h.z);
  vec3 p3 = vec3(a1.zw, h.w);

  vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2,p2), dot(p3,p3)));
  p0 *= norm.x; p1 *= norm.y; p2 *= norm.z; p3 *= norm.w;

  vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
  m = m * m;
  return 42.0 * dot(m * m, vec4(dot(p0,x0), dot(p1,x1), dot(p2,x2), dot(p3,x3)));
}
`;

export const POINT_VERT = /* glsl */ `
attribute vec3  color;      // 原图 RGB（three 只自带 position/normal/uv，color 要自己声明）
attribute vec3  aScatter;   // 入场动画的起始散开位置
attribute float aSeed;      // 每点随机种子，错开噪声相位
attribute float aDepth;     // 归一化深度 0..1（1=最近），用于深度配色

uniform float uTime;
uniform float uProgress;        // 0→1 聚合进度
uniform float uSize;
uniform float uSizeScale;
uniform float uPixelRatio;
uniform float uNoiseAmp;
uniform float uNoiseFreq;
uniform vec3  uMouse;
uniform float uRepelRadius;
uniform float uRepelStrength;
uniform int   uTint;            // 0=原图 1=深度渐变 2=单色霓虹 3=原图×深度
uniform vec3  uTintA;           // 远端色
uniform vec3  uTintB;           // 近端色

varying vec3  vColor;
varying float vFade;

${SIMPLEX_3D}

float easeOutCubic(float t){ return 1.0 - pow(1.0 - t, 3.0); }

void main() {
  float t = easeOutCubic(clamp(uProgress, 0.0, 1.0));
  vec3 pos = mix(aScatter, position, t);

  // 呼吸扰动：位置相近的点会一起流动，不会像电视雪花
  if (uNoiseAmp > 0.0) {
    vec3 q = pos * uNoiseFreq + uTime * 0.25 + aSeed * 7.0;
    pos += vec3(snoise(q), snoise(q + 31.416), snoise(q + 74.132)) * uNoiseAmp;
  }

  // 鼠标斥力
  if (uRepelRadius > 0.0 && uRepelStrength > 0.0) {
    vec3 diff = pos - uMouse;
    float dist = length(diff);
    float f = smoothstep(uRepelRadius, 0.0, dist);
    vec3 dir = dist > 1e-4 ? diff / dist : vec3(0.0, 0.0, 1.0);
    pos += dir * f * uRepelStrength;
  }

  vec3 grad = mix(uTintA, uTintB, aDepth);
  if (uTint == 0) {
    vColor = color;
  } else if (uTint == 1) {
    vColor = grad;
  } else if (uTint == 2) {
    float lum = dot(color, vec3(0.299, 0.587, 0.114));
    vColor = mix(uTintA, uTintB, lum);
  } else {
    vColor = color * (0.45 + 0.9 * grad);
  }

  vFade = t;

  vec4 mv = modelViewMatrix * vec4(pos, 1.0);
  // uSizeScale = 参考视距，使得 uSize 的单位就是「默认视距下的 CSS 像素」。
  // 再乘 dpr 换算到设备像素。上限 64 防止贴到镜头前时点撑爆（WebGL 有实现上限）。
  gl_PointSize = clamp(uSize * uPixelRatio * (uSizeScale / max(0.001, -mv.z)), 1.0, 64.0);
  gl_Position = projectionMatrix * mv;
}
`;

export const POINT_FRAG = /* glsl */ `
precision highp float;

uniform float uOpacity;
uniform float uSoftness;
uniform float uSolid;   // 1 = 实体模式（写深度、遮挡正确）

varying vec3  vColor;
varying float vFade;

void main() {
  float r = length(gl_PointCoord - 0.5);
  float alpha = 1.0 - smoothstep(0.5 - uSoftness - 0.002, 0.5, r);

  if (uSolid > 0.5) {
    if (alpha < 0.5) discard;
    gl_FragColor = vec4(vColor, 1.0);
  } else {
    float a = alpha * uOpacity * vFade;
    if (a < 0.01) discard;
    gl_FragColor = vec4(vColor, a);
  }
}
`;

/** 配色方案：[远端色, 近端色] */
export const TINT_PALETTE = {
  origin: ['#1b2a6b', '#5ff7d8'],
  depth:  ['#16215c', '#5ff7d8'],
  mono:   ['#0a1626', '#35e0d0'],
  mix:    ['#243a7d', '#7ff9e2'],
};

export const TINT_INDEX = { origin: 0, depth: 1, mono: 2, mix: 3 };
