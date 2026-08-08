/**
 * 多视图上传槽：正面 / 左侧 / 背面 / 右侧。
 *
 * 只有正面是必填的 —— 后端能只用一张图跑（退化成模型自己编背面），
 * 但每多给一张，那一侧就是照着你的原图重建的，而不是猜的。
 *
 * 视图名必须和后端 hunyuan_runner.VIEW_DIRS 的键一致。
 */

export const MV_VIEWS = [
  { key: 'front', label: '正面', required: true },
  { key: 'left', label: '左侧', required: false },
  { key: 'back', label: '背面', required: false },
  { key: 'right', label: '右侧', required: false },
];

export function createMvSlots(onGenerate) {
  const wrap = document.getElementById('mvSlots');
  const grid = document.getElementById('mvSlotGrid');
  const goBtn = document.getElementById('mvGenerate');

  /** @type {Record<string, File|Blob|null>} */
  const files = Object.create(null);
  const urls = Object.create(null); // 预览用的 objectURL，换图/清除时要回收

  function refresh() {
    const n = Object.values(files).filter(Boolean).length;
    const hasFront = !!files.front;
    goBtn.disabled = !hasFront;
    goBtn.textContent = hasFront
      ? `用这 ${n} 张图生成（约 40 秒）`
      : '先放一张正面图';
  }

  function setFile(key, file) {
    if (urls[key]) {
      URL.revokeObjectURL(urls[key]);
      urls[key] = null;
    }
    files[key] = file ?? null;

    const slot = grid.querySelector(`[data-view="${key}"]`);
    const img = slot.querySelector('img');
    if (file) {
      urls[key] = URL.createObjectURL(file);
      img.src = urls[key];
      img.hidden = false;
      slot.classList.add('is-filled');
    } else {
      img.removeAttribute('src');
      img.hidden = true;
      slot.classList.remove('is-filled');
    }
    refresh();
  }

  for (const v of MV_VIEWS) {
    const slot = document.createElement('button');
    slot.type = 'button';
    slot.className = 'mvslot' + (v.required ? ' is-required' : '');
    slot.dataset.view = v.key;
    slot.title = v.required ? `${v.label}（必填）` : `${v.label}（可选）`;

    const img = document.createElement('img');
    img.hidden = true;
    img.alt = '';
    slot.appendChild(img);

    const plus = document.createElement('span');
    plus.textContent = '＋';
    plus.style.fontSize = '17px';
    slot.appendChild(plus);

    const tag = document.createElement('span');
    tag.className = 'mvslot__tag';
    tag.textContent = v.required ? `${v.label} *` : v.label;
    slot.appendChild(tag);

    const clear = document.createElement('button');
    clear.type = 'button';
    clear.className = 'mvslot__clear';
    clear.textContent = '×';
    clear.title = '清除';
    clear.addEventListener('click', (e) => {
      e.stopPropagation(); // 别触发外层 slot 的选文件
      setFile(v.key, null);
    });
    slot.appendChild(clear);

    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'image/*';
    input.hidden = true;
    input.addEventListener('change', () => {
      if (input.files?.[0]) setFile(v.key, input.files[0]);
      input.value = '';
    });
    slot.appendChild(input);

    slot.addEventListener('click', () => input.click());

    // 支持直接把图拖到某个槽上
    slot.addEventListener('dragover', (e) => {
      e.preventDefault();
      e.stopPropagation();
    });
    slot.addEventListener('drop', (e) => {
      e.preventDefault();
      e.stopPropagation();
      const f = Array.from(e.dataTransfer?.files ?? [])
        .find((x) => x.type.startsWith('image/'));
      if (f) setFile(v.key, f);
    });

    grid.appendChild(slot);
  }

  goBtn.addEventListener('click', () => {
    if (!files.front) return;
    const payload = {};
    for (const v of MV_VIEWS) if (files[v.key]) payload[v.key] = files[v.key];
    onGenerate(payload);
  });

  refresh();

  return {
    show(on) { wrap.hidden = !on; },
    /** 拖到页面上（而不是某个槽）的图，填进第一个空槽 */
    accept(file) {
      const empty = MV_VIEWS.find((v) => !files[v.key]);
      if (empty) setFile(empty.key, file);
    },
    hasFront: () => !!files.front,
    count: () => Object.values(files).filter(Boolean).length,
  };
}
