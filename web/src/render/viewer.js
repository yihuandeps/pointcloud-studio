/**
 * three.js 场景：相机、控制器、渲染循环、鼠标世界坐标、截图。
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { PointCloudObject } from './pointCloud.js';

export class Viewer {
  /** 默认视距。同时作为点大小的参考距离，让「点大小」滑杆的单位就是 CSS 像素。 */
  static CAM_DIST = 3.4;

  constructor(canvas) {
    this.canvas = canvas;

    this.renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: true,
      alpha: true,
      // 截图需要能读回像素
      preserveDrawingBuffer: true,
      powerPreference: 'high-performance',
    });
    this.renderer.setClearColor(0x000000, 0);

    this.scene = new THREE.Scene();

    this.camera = new THREE.PerspectiveCamera(50, 1, 0.01, 100);
    this.camera.position.set(0, 0, Viewer.CAM_DIST);

    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.rotateSpeed = 0.65;
    this.controls.zoomSpeed = 0.8;
    this.controls.minDistance = 0.6;
    this.controls.maxDistance = 14;
    this.controls.autoRotateSpeed = 0.7;

    this.cloud = new PointCloudObject();

    this.clock = new THREE.Clock();
    this.raycaster = new THREE.Raycaster();
    this.pointer = new THREE.Vector2();
    this.mouseWorld = new THREE.Vector3();
    this.plane = new THREE.Plane();
    this.camDir = new THREE.Vector3();
    this.hasPointer = false;

    this.fps = 0;
    this.#frames = 0;
    this.#fpsAt = 0;
    this.onFps = null;

    this.#bind();
    this.resize();
    this.#loop();
  }

  #frames = 0;
  #fpsAt = 0;
  #raf = 0;

  #bind() {
    this._onResize = () => this.resize();
    window.addEventListener('resize', this._onResize);

    this._onPointerMove = (e) => {
      const r = this.canvas.getBoundingClientRect();
      this.pointer.x = ((e.clientX - r.left) / r.width) * 2 - 1;
      this.pointer.y = -((e.clientY - r.top) / r.height) * 2 + 1;
      this.hasPointer = true;
    };
    this._onPointerLeave = () => {
      this.hasPointer = false;
      this.cloud.clearMouse();
    };

    this.canvas.addEventListener('pointermove', this._onPointerMove);
    this.canvas.addEventListener('pointerleave', this._onPointerLeave);
  }

  resize() {
    const w = this.canvas.clientWidth || window.innerWidth;
    const h = this.canvas.clientHeight || window.innerHeight;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);

    this.renderer.setPixelRatio(dpr);
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();

    this.cloud.setPixelRatio(dpr);
    this.cloud.material.uniforms.uSizeScale.value = Viewer.CAM_DIST;
  }

  setCloud(data) {
    const obj = this.cloud.build(data);
    if (!obj.parent) this.scene.add(obj);
  }

  applyParams(p) {
    this.cloud.applyParams(p);
    this.controls.autoRotate = !!p.autoRotate;
  }

  resetView() {
    this.camera.position.set(0, 0, Viewer.CAM_DIST);
    this.controls.target.set(0, 0, 0);
    this.controls.update();
  }

  replay() {
    this.cloud.replay();
  }

  /** 把鼠标 NDC 投到过原点、朝向相机的平面上，得到斥力中心 */
  #updateMouseWorld() {
    if (!this.hasPointer || this.cloud.material.uniforms.uRepelRadius.value <= 0) {
      this.cloud.clearMouse();
      return;
    }
    this.camera.getWorldDirection(this.camDir);
    this.plane.setFromNormalAndCoplanarPoint(this.camDir, this.controls.target);
    this.raycaster.setFromCamera(this.pointer, this.camera);
    if (this.raycaster.ray.intersectPlane(this.plane, this.mouseWorld)) {
      this.cloud.setMouse(this.mouseWorld);
    }
  }

  #loop = () => {
    this.#raf = requestAnimationFrame(this.#loop);

    const delta = Math.min(this.clock.getDelta(), 0.1);
    const elapsed = this.clock.elapsedTime;

    this.controls.update();
    this.#updateMouseWorld();
    this.cloud.update(elapsed, delta);
    this.renderer.render(this.scene, this.camera);

    this.#frames++;
    if (elapsed - this.#fpsAt >= 0.5) {
      this.fps = Math.round(this.#frames / (elapsed - this.#fpsAt));
      this.#frames = 0;
      this.#fpsAt = elapsed;
      this.onFps?.(this.fps);
    }
  };

  /** 截图：需要在渲染后立刻读，否则 drawingBuffer 可能已被清 */
  snapshot() {
    this.renderer.render(this.scene, this.camera);
    return new Promise((resolve) => {
      this.canvas.toBlob((b) => resolve(b), 'image/png');
    });
  }

  dispose() {
    cancelAnimationFrame(this.#raf);
    window.removeEventListener('resize', this._onResize);
    this.canvas.removeEventListener('pointermove', this._onPointerMove);
    this.canvas.removeEventListener('pointerleave', this._onPointerLeave);
    this.controls.dispose();
    this.cloud.dispose();
    this.renderer.dispose();
  }
}
