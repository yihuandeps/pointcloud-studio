"""
在 github.com 直连不通时，抓取 MoGe 及其两个 git 依赖的源码。

本机实测的网络状况：
  github.com              连接超时（git clone 直接失败）
  codeload.github.com     可达，但不支持 Range，大仓库容易半路断且无法续传
  raw.githubusercontent   可达且快
  api.github.com          可达

所以：小仓库（utils3d / pipeline）走 codeload 抓 tarball；
MoGe 仓库有 10.5MB（大部分是 demo 素材），改成用 API 列出文件树、
只挑 moge/ 包本身逐个从 raw 下载（约 354KB）。

只用标准库，不需要先装 requests。

用法：python fetch_sources.py [输出目录]
"""

from __future__ import annotations

import json
import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path

UA = {"User-Agent": "pointcloud-studio-setup"}

# 版本与 MoGe 的 pyproject.toml 中钉的 commit 保持一致
TARBALLS = {
    "utils3d": ("EasternJournalist/utils3d", "3fab839f0be9931dac7c8488eb0e1600c236e183"),
    "pipeline": ("EasternJournalist/pipeline", "866f059d2a05cde05e4a52211ec5051fd5f276d6"),
}
MOGE_REPO = "microsoft/MoGe"
MOGE_COMMIT = "925b8ed835a7a9cdb7578ba15c658a0afc969030"
MOGE_KEEP_ROOT = {"pyproject.toml", "requirements.txt", "README.md", "LICENSE"}

CTX = ssl.create_default_context()


def get(url: str, timeout: int = 90) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout, context=CTX) as r:
        return r.read()


def retry_get(url: str, tries: int = 3, timeout: int = 90) -> bytes:
    last = None
    for i in range(tries):
        try:
            return get(url, timeout)
        except Exception as exc:  # noqa: BLE001
            last = exc
    raise RuntimeError(f"{url} 下载失败：{last}")


def fetch_tarballs(out: Path) -> list[Path]:
    paths = []
    for name, (repo, ref) in TARBALLS.items():
        dest = out / f"{name}.tar.gz"
        if dest.exists() and dest.stat().st_size > 1024:
            print(f"  · {name}.tar.gz 已存在，跳过")
            paths.append(dest)
            continue
        print(f"  ↓ {name} ...", end="", flush=True)
        data = retry_get(f"https://codeload.github.com/{repo}/tar.gz/{ref}")
        dest.write_bytes(data)
        print(f" {len(data) / 1024:.0f} KB")
        paths.append(dest)
    return paths


def fetch_moge(out: Path) -> Path:
    dest = out / "moge-src"
    if (dest / "pyproject.toml").exists() and (dest / "moge" / "model" / "v2.py").exists():
        print("  · moge-src 已存在，跳过")
        return dest

    print("  ↓ 读取 MoGe 文件树 ...", end="", flush=True)
    tree = json.loads(
        retry_get(
            f"https://api.github.com/repos/{MOGE_REPO}/git/trees/{MOGE_COMMIT}?recursive=1"
        )
    )
    if tree.get("truncated"):
        raise RuntimeError("文件树被截断，需要改用 tarball 方式")

    wanted = [
        t["path"]
        for t in tree["tree"]
        if t["type"] == "blob"
        and (t["path"].startswith("moge/") or t["path"] in MOGE_KEEP_ROOT)
    ]
    total_kb = sum(t["size"] for t in tree["tree"] if t["path"] in set(wanted)) / 1024
    print(f" {len(wanted)} 个文件，{total_kb:.0f} KB")

    for i, rel in enumerate(wanted, 1):
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(
            retry_get(f"https://raw.githubusercontent.com/{MOGE_REPO}/{MOGE_COMMIT}/{rel}")
        )
        if i % 10 == 0 or i == len(wanted):
            print(f"    {i}/{len(wanted)}")

    return dest


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / ".cache" / "src")
    out.mkdir(parents=True, exist_ok=True)
    print(f"输出目录：{out}\n")

    print("[1/2] utils3d + pipeline")
    tars = fetch_tarballs(out)

    print("\n[2/2] MoGe")
    moge = fetch_moge(out)

    manifest = {
        "tarballs": [str(p) for p in tars],
        "moge_src": str(moge),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n源码就绪：{out}")


if __name__ == "__main__":
    main()
