/**
 * 浏览器端深度引擎：depth.worker.js 的主线程封装。
 * 把 postMessage 往返包装成 Promise，并把加载进度透出去。
 */

export class BrowserDepthEngine {
  #worker = null;
  #seq = 0;
  #pending = new Map();

  backend = null; // { device, dtype, modelId }

  #ensure() {
    if (this.#worker) return this.#worker;

    this.#worker = new Worker(new URL('./depth.worker.js', import.meta.url), {
      type: 'module',
    });

    this.#worker.onmessage = (e) => {
      const { id, type, payload } = e.data;
      const task = this.#pending.get(id);
      if (!task) return;

      if (type === 'progress') {
        task.onProgress?.(payload);
        return;
      }

      this.#pending.delete(id);
      if (type === 'error') task.reject(new Error(payload.message));
      else task.resolve(payload);
    };

    this.#worker.onerror = (e) => {
      const err = new Error(`Worker 崩溃：${e.message || '未知错误'}`);
      for (const task of this.#pending.values()) task.reject(err);
      this.#pending.clear();
    };

    return this.#worker;
  }

  #send(message, transfer = [], onProgress) {
    const worker = this.#ensure();
    const id = ++this.#seq;
    return new Promise((resolve, reject) => {
      this.#pending.set(id, { resolve, reject, onProgress });
      worker.postMessage({ ...message, id }, transfer);
    });
  }

  /** 加载（或切换）模型。重复调用同一个 modelKey 会命中 Worker 内缓存。 */
  async load(modelKey, onProgress) {
    this.backend = await this.#send({ type: 'load', modelKey }, [], onProgress);
    return this.backend;
  }

  /**
   * @param {{data: Uint8ClampedArray, width: number, height: number}} image
   * @returns {Promise<{depth: Float32Array, width: number, height: number, ms: number}>}
   */
  async infer(image) {
    // 拷一份再转移，避免把调用方的像素数据搬空
    const rgba = new Uint8ClampedArray(image.data);
    return await this.#send(
      { type: 'infer', rgba, width: image.width, height: image.height },
      [rgba.buffer],
    );
  }

  dispose() {
    this.#worker?.terminate();
    this.#worker = null;
    this.#pending.clear();
    this.backend = null;
  }
}
