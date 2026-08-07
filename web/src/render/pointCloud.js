/**
 * 点云对象：把 buildPointCloud() 的结果装配成 three.js Points。
 */

import * as THREE from 'three';
import { POINT_VERT, POINT_FRAG, TINT_PALETTE, TINT_INDEX } from './shaders.js';

/** 入场动画的起点：球壳上的随机位置 */
function makeScatter(count) {
  const out = new Float32Array(count * 3);
  for (let i = 0; i < count; i++) {
    // 球面均匀采样
    const u = Math.random() * 2 - 1;
    const th = Math.random() * Math.PI * 2;
    const s = Math.sqrt(1 - u * u);
    const r = 2.6 + Math.random() * 1.9;
    const p = i * 3;
    out[p] = s * Math.cos(th) * r;
    out[p + 1] = s * Math.sin(th) * r;
    out[p + 2] = u * r;
  }
  return out;
}

function makeSeeds(count) {
  const out = new Float32Array(count);
  for (let i = 0; i < count; i++) out[i] = Math.random();
  return out;
}

export class PointCloudObject {
  object = null;
  geometry = null;
  material = null;
  count = 0;

  #progress = 0;
  #target = 0;

  constructor() {
    this.material = new THREE.ShaderMaterial({
      vertexShader: POINT_VERT,
      fragmentShader: POINT_FRAG,
      transparent: true,
      depthWrite: false,
      depthTest: true,
      uniforms: {
        uTime: { value: 0 },
        uProgress: { value: 0 },
        uSize: { value: 1.6 },
        // 参考视距，见 shaders.js 里 gl_PointSize 的注释。由 Viewer 用实际相机距离覆盖。
        uSizeScale: { value: 3.4 },
        uPixelRatio: { value: 1 },
        uNoiseAmp: { value: 0.012 },
        uNoiseFreq: { value: 1.6 },
        uMouse: { value: new THREE.Vector3(1e9, 1e9, 1e9) },
        uRepelRadius: { value: 0.45 },
        uRepelStrength: { value: 0.22 },
        uTint: { value: 0 },
        uTintA: { value: new THREE.Color('#1b2a6b') },
        uTintB: { value: new THREE.Color('#5ff7d8') },
        uOpacity: { value: 0.95 },
        uSoftness: { value: 0.18 },
        uSolid: { value: 0 },
      },
    });
  }

  /** 用新的点云数据重建几何体（点数变了必须走这里）。 */
  build(cloud) {
    this.geometry?.dispose();

    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(cloud.positions, 3));
    g.setAttribute('color', new THREE.BufferAttribute(cloud.colors, 3));
    g.setAttribute('aDepth', new THREE.BufferAttribute(cloud.depths, 1));
    g.setAttribute('aScatter', new THREE.BufferAttribute(makeScatter(cloud.count), 3));
    g.setAttribute('aSeed', new THREE.BufferAttribute(makeSeeds(cloud.count), 1));
    // 点云是散点，包围球自己算比让 three 遍历更省事
    g.boundingSphere = new THREE.Sphere(new THREE.Vector3(0, 0, 0), 3.2);

    this.geometry = g;
    this.count = cloud.count;

    if (this.object) {
      this.object.geometry = g;
    } else {
      this.object = new THREE.Points(g, this.material);
      this.object.frustumCulled = false;
    }
    return this.object;
  }

  /** 只改 uniform 的参数（不需要重建几何体）。 */
  applyParams(p) {
    const u = this.material.uniforms;
    u.uSize.value = p.pointSize;
    u.uOpacity.value = p.opacity;
    u.uSoftness.value = p.softness;
    u.uNoiseAmp.value = p.noiseAmp;
    u.uNoiseFreq.value = p.noiseFreq;
    u.uRepelRadius.value = p.repelRadius;
    u.uRepelStrength.value = p.repelStrength;

    const solid = p.solidMode ? 1 : 0;
    u.uSolid.value = solid;
    this.material.depthWrite = !!p.solidMode;
    this.material.transparent = !p.solidMode;

    const tint = p.tint ?? 'origin';
    u.uTint.value = TINT_INDEX[tint] ?? 0;
    const pal = TINT_PALETTE[tint] ?? TINT_PALETTE.origin;
    u.uTintA.value.set(pal[0]);
    u.uTintB.value.set(pal[1]);

    this.material.needsUpdate = true;
  }

  setPixelRatio(r) {
    this.material.uniforms.uPixelRatio.value = r;
  }

  setMouse(v) {
    this.material.uniforms.uMouse.value.copy(v);
  }

  clearMouse() {
    this.material.uniforms.uMouse.value.set(1e9, 1e9, 1e9);
  }

  /** 重播聚合动画 */
  replay() {
    this.#progress = 0;
    this.#target = 1;
  }

  /** 直接跳到最终状态（切参数重建时用，避免每次都重播动画） */
  snapToEnd() {
    this.#progress = 1;
    this.#target = 1;
    this.material.uniforms.uProgress.value = 1;
  }

  update(elapsed, delta) {
    const u = this.material.uniforms;
    u.uTime.value = elapsed;
    if (this.#progress < this.#target) {
      // 约 1.6s 走完
      this.#progress = Math.min(this.#target, this.#progress + delta / 1.6);
      u.uProgress.value = this.#progress;
    }
  }

  dispose() {
    this.geometry?.dispose();
    this.material?.dispose();
    this.geometry = null;
    this.object = null;
  }
}
