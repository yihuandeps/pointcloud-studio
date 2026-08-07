/**
 * 上传入口：拖拽 / 文件选择 / 剪贴板粘贴 / 内置示例图。
 */

import { SAMPLE_KINDS, makeSample, sampleToBlob } from '../core/imagePrep.js';

export function createDropzone(onImage) {
  const zone = document.getElementById('dropzone');
  const pickBtn = document.getElementById('pickBtn');
  const fileInput = document.getElementById('fileInput');
  const samples = document.getElementById('samples');

  /* --- 内置示例图缩略 --- */
  for (const kind of SAMPLE_KINDS) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'sample';
    btn.title = `示例：${kind.name}`;

    const thumb = makeSample(kind.id, 128);
    thumb.style.width = '100%';
    thumb.style.height = '100%';
    btn.appendChild(thumb);

    const cap = document.createElement('span');
    cap.textContent = kind.name;
    btn.appendChild(cap);

    btn.addEventListener('click', async () => {
      const blob = await sampleToBlob(kind.id, 768);
      onImage(blob, `示例·${kind.name}`);
    });
    samples.appendChild(btn);
  }

  /* --- 文件选择 --- */
  pickBtn.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', () => {
    const f = fileInput.files?.[0];
    if (f) onImage(f, f.name);
    fileInput.value = '';
  });

  /* --- 拖拽（挂在 window 上，收起空态后依然能拖）--- */
  let depth = 0;
  const isFile = (e) => Array.from(e.dataTransfer?.types ?? []).includes('Files');

  window.addEventListener('dragenter', (e) => {
    if (!isFile(e)) return;
    e.preventDefault();
    depth++;
    zone.classList.add('is-over');
    zone.hidden = false;
  });

  window.addEventListener('dragover', (e) => {
    if (isFile(e)) e.preventDefault();
  });

  window.addEventListener('dragleave', (e) => {
    if (!isFile(e)) return;
    depth = Math.max(0, depth - 1);
    if (depth === 0) zone.classList.remove('is-over');
  });

  window.addEventListener('drop', (e) => {
    if (!isFile(e)) return;
    e.preventDefault();
    depth = 0;
    zone.classList.remove('is-over');
    const f = Array.from(e.dataTransfer.files).find((x) => x.type.startsWith('image/'));
    if (f) onImage(f, f.name);
  });

  /* --- 剪贴板粘贴 --- */
  window.addEventListener('paste', (e) => {
    const item = Array.from(e.clipboardData?.items ?? [])
      .find((x) => x.type.startsWith('image/'));
    if (!item) return;
    const f = item.getAsFile();
    if (f) onImage(f, '剪贴板图片');
  });

  return {
    hide() { zone.hidden = true; },
    show() { zone.hidden = false; },
    note(text) { document.getElementById('engineNote').textContent = text; },
  };
}
