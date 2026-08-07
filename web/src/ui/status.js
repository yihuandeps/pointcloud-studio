/**
 * 状态栏 + 顶部进度提示。
 */

export function createStatus() {
  const bar = document.getElementById('statusbar');
  const elPoints = document.getElementById('statPoints');
  const elSize = document.getElementById('statSize');
  const elTime = document.getElementById('statTime');
  const elFps = document.getElementById('statFps');

  const toast = document.getElementById('toast');
  const toastBar = document.getElementById('toastBar');
  const toastTrack = toast.querySelector('.toast__bar');
  const toastText = document.getElementById('toastText');

  let hideTimer = 0;

  return {
    showBar() { bar.classList.add('is-on'); },

    setPoints(n) {
      elPoints.innerHTML = `<b>${n.toLocaleString('zh-CN')}</b> 点`;
    },
    setSize(w, h) { elSize.textContent = `${w}×${h}`; },
    setTime(ms) { elTime.textContent = `${ms} ms`; },
    setFps(v) { elFps.innerHTML = `<b>${v}</b> FPS`; },

    /**
     * @param {string} text
     * @param {number|null} progress 0..1；null = 不确定进度（跑马灯）
     */
    show(text, progress = null) {
      clearTimeout(hideTimer);
      toast.hidden = false;
      toast.classList.remove('is-error');
      toastText.textContent = text;
      if (progress == null) {
        toastTrack.classList.add('is-indeterminate');
        toastBar.style.width = '35%';
      } else {
        toastTrack.classList.remove('is-indeterminate');
        toastBar.style.width = `${Math.max(0, Math.min(1, progress)) * 100}%`;
      }
    },

    done(text, delay = 1400) {
      clearTimeout(hideTimer);
      toast.hidden = false;
      toast.classList.remove('is-error');
      toastTrack.classList.remove('is-indeterminate');
      toastBar.style.width = '100%';
      toastText.textContent = text;
      hideTimer = setTimeout(() => { toast.hidden = true; }, delay);
    },

    error(text) {
      clearTimeout(hideTimer);
      toast.hidden = false;
      toast.classList.add('is-error');
      toastTrack.classList.remove('is-indeterminate');
      toastBar.style.width = '100%';
      toastText.textContent = text;
      hideTimer = setTimeout(() => { toast.hidden = true; }, 8000);
    },

    hide() {
      clearTimeout(hideTimer);
      toast.hidden = true;
    },
  };
}

/** 引擎状态徽标 */
export function createEngineBadge() {
  const box = document.getElementById('engineBadge');
  const dot = box.querySelector('.engine__dot');
  const txt = box.querySelector('.engine__text');
  return {
    set(state, text) {
      dot.dataset.state = state;
      txt.textContent = text;
      txt.title = text;
    },
  };
}
