/**
 * 参数面板：直接读 config.js 的 PARAM_SCHEMA 生成控件。
 * 加参数只需要改 schema，这里不用动。
 */

import { REBUILD_KEYS } from '../config.js';

function el(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
}

function paintTrack(input) {
  const min = Number(input.min);
  const max = Number(input.max);
  const pct = ((Number(input.value) - min) / (max - min)) * 100;
  input.style.setProperty('--pct', `${pct}%`);
}

export function buildPanel(container, schema, params, onChange) {
  container.innerHTML = '';
  const setters = new Map();
  const controls = new Map(); // key -> { row, input }

  for (const group of schema) {
    const section = el('section', 'group');
    section.appendChild(el('h3', 'group__title', group.group));

    for (const item of group.items) {
      const needsRebuild = REBUILD_KEYS.has(item.key);
      const row = el('div', 'row');

      if (item.type === 'range') {
        const head = el('div', 'row__head');
        head.appendChild(el('label', 'row__label', item.label));
        const val = el('span', 'row__val');
        head.appendChild(val);
        row.appendChild(head);

        const input = el('input');
        input.type = 'range';
        input.min = item.min;
        input.max = item.max;
        input.step = item.step;
        input.value = params[item.key];

        const paint = (v) => {
          val.textContent = item.format ? item.format(v) : v;
          paintTrack(input);
        };
        paint(Number(input.value));

        input.addEventListener('input', () => {
          const v = Number(input.value);
          params[item.key] = v;
          paint(v);
          onChange(item.key, v, needsRebuild);
        });

        row.appendChild(input);
        setters.set(item.key, (v) => { input.value = v; paint(Number(v)); });
        controls.set(item.key, { row, input });
      }

      if (item.type === 'toggle') {
        row.classList.add('row--inline');
        row.appendChild(el('label', 'row__label', item.label));

        const wrap = el('label', 'switch');
        const input = el('input');
        input.type = 'checkbox';
        input.checked = !!params[item.key];
        wrap.appendChild(input);
        wrap.appendChild(el('i'));
        row.appendChild(wrap);

        input.addEventListener('change', () => {
          params[item.key] = input.checked;
          onChange(item.key, input.checked, needsRebuild);
        });
        setters.set(item.key, (v) => { input.checked = !!v; });
        controls.set(item.key, { row, input });
      }

      if (item.type === 'select') {
        row.appendChild(el('label', 'row__label', item.label));
        const sel = el('select', 'select');
        for (const o of item.options) {
          const opt = el('option', null, o.label);
          opt.value = o.v;
          sel.appendChild(opt);
        }
        sel.value = params[item.key];
        sel.addEventListener('change', () => {
          params[item.key] = sel.value;
          onChange(item.key, sel.value, needsRebuild);
        });
        row.appendChild(sel);
        setters.set(item.key, (v) => { sel.value = v; });
        controls.set(item.key, { row, input: sel });
      }

      if (item.hint) {
        const hint = el('p', 'row__hint', item.hint);
        row.appendChild(hint);
      }

      section.appendChild(row);
    }

    container.appendChild(section);
  }

  return {
    set(key, value) {
      params[key] = value;
      setters.get(key)?.(value);
    },
    /** 置灰一批控件（例如高精度模式下失效的那些） */
    setDisabled(keys, disabled, reason = '') {
      for (const k of keys) {
        const c = controls.get(k);
        if (!c) continue;
        c.input.disabled = disabled;
        c.row.classList.toggle('is-disabled', disabled);
        c.row.title = disabled ? reason : '';
      }
    },
    values: params,
  };
}
