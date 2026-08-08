"""
生成跨语言契约测试向量，供 web/tests/protocol.test.mjs 校验。

只依赖标准库，没装 torch 也能跑。

用法：python emit_vectors.py <输出目录>
"""

from __future__ import annotations

import json
import random
import struct
import sys
from pathlib import Path

from imaging import fit_size, js_round, pack_cloud, pack_result


def build_size_cases() -> list[dict]:
    cases: list[dict] = []

    # 1) 定向扫描：让 k 正好等于 0.5，这样每个奇数宽度都会撞上 .5 舍入。
    #    Python 内置 round() 在这里会给出和 JS Math.round() 不同的结果，
    #    是最能暴露问题的一组。
    for max_side in (1024, 1000, 512):
        h = max_side * 2
        for w in range(1, 1200, 7):
            cases.append({"w": w, "h": h, "maxSide": max_side})

    # 2) 常见真实分辨率
    for w, h in [
        (4032, 3024), (3024, 4032), (1920, 1080), (1080, 1920),
        (2048, 2048), (800, 600), (1024, 1024), (1023, 1025),
        (1, 1), (1, 4000), (4000, 1), (1025, 1024),
    ]:
        for max_side in (1024, 512, 2048):
            cases.append({"w": w, "h": h, "maxSide": max_side})

    # 3) 随机撒一批
    rng = random.Random(20260803)
    for _ in range(400):
        cases.append({
            "w": rng.randint(1, 6000),
            "h": rng.randint(1, 6000),
            "maxSide": rng.choice([256, 512, 1024, 1536, 2048]),
        })

    for c in cases:
        c["out"] = list(fit_size(c["w"], c["h"], c["maxSide"]))
    return cases


def build_round_cases() -> list[dict]:
    vals = [-2.5, -1.5, -0.5, 0.0, 0.5, 1.5, 2.5, 3.5, 0.49999, 0.5000001,
            1.4999999, 2.0, 100.5, 101.5, 682.5, 683.5]
    return [{"x": v, "out": js_round(v)} for v in vals]


def build_payload(out_dir: Path) -> dict:
    """确定性的点图 + mask，Node 端会按同样公式重算并逐值比对。"""
    width, height = 37, 23          # 故意用质数，暴露行列搞反的问题
    n = width * height

    points = bytearray()
    for i in range(n * 3):
        points += struct.pack("<f", (i % 101) * 0.5 - 25.0)

    mask = bytes((i * 7 + 3) % 2 for i in range(n))

    header = {
        "width": width,
        "height": height,
        "hasMask": True,
        "intrinsics": [[1.2, 0.0, 0.5], [0.0, 1.6, 0.5], [0.0, 0.0, 1.0]],
        "model": "Ruicheng/moge-2-vitl",
        "device": "cuda",
        "ms": 123,
    }

    blob = pack_result(bytes(points), mask, header)
    (out_dir / "payload.bin").write_bytes(blob)

    return {"width": width, "height": height, "bytes": len(blob), "header": header}


def build_cloud_payload(out_dir: Path) -> dict:
    """生成式 3D 的散点云包（pack_cloud），Node 端用同公式重算逐值比对。"""
    count = 977                     # 质数，偏移算错一个元素就全体错位

    positions = bytearray()
    for i in range(count * 3):
        positions += struct.pack("<f", ((i * 13) % 251) * 0.02 - 2.5)

    colors = bytes((i * 31 + 7) % 256 for i in range(count * 3))
    # 法线是 int8（各分量 ×127），凹陷度是 uint8
    normals = bytes((((i * 17 + 5) % 255) - 127) & 0xFF for i in range(count * 3))
    ao = bytes((i * 43 + 11) % 256 for i in range(count))

    header = {
        "count": count,
        "model": "stabilityai/TripoSR",
        "device": "cuda",
        "hasNormals": True,
        "hasAO": True,
        "ms": 4567,
    }

    blob = pack_cloud(bytes(positions), colors, header, normals, ao)
    (out_dir / "cloud_payload.bin").write_bytes(blob)

    # 不带法线/AO 的老式包也要留一份，确认可选段真的是可选的
    bare_header = {k: v for k, v in header.items()
                   if k not in ("hasNormals", "hasAO")}
    bare = pack_cloud(bytes(positions), colors, bare_header)
    (out_dir / "cloud_payload_bare.bin").write_bytes(bare)

    return {"count": count, "bytes": len(blob), "bareBytes": len(bare),
            "header": header}


def main():
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    out_dir.mkdir(parents=True, exist_ok=True)

    data = {
        "round": build_round_cases(),
        "fitSize": build_size_cases(),
        "payload": build_payload(out_dir),
        "cloudPayload": build_cloud_payload(out_dir),
    }
    (out_dir / "vectors.json").write_text(
        json.dumps(data, ensure_ascii=False), encoding="utf-8"
    )
    print(f"round {len(data['round'])} 条, fitSize {len(data['fitSize'])} 条, "
          f"payload {data['payload']['bytes']} 字节, "
          f"cloudPayload {data['cloudPayload']['bytes']} 字节 → {out_dir}")


if __name__ == "__main__":
    main()
