"""Tile-geometry informal checks.

Run with ``PYTHONPATH=src python tests/test_geometry.py``. No pytest wiring
yet; the script exits non-zero on any failure.
"""

from __future__ import annotations

import sys
from collections import Counter

from analysis_pipeline.gradient_test.seams import compute_seam_positions
from analysis_pipeline.gradient_test.tiles import enumerate_tiles


def _expect(cond: bool, msg: str) -> None:
    if not cond:
        print(f"FAIL: {msg}")
        sys.exit(1)


def main() -> None:
    # 100x100 image, tile_size=32, overlap=0 → step=32, 3 seams per axis → 4×4 grid
    H = W = 100
    TS = 32
    OV = 0
    seams_y = compute_seam_positions(H, TS, OV)
    seams_x = compute_seam_positions(W, TS, OV)
    _expect(list(seams_y) == [32, 64, 96], f"seams_y={list(seams_y)}")
    _expect(list(seams_x) == [32, 64, 96], f"seams_x={list(seams_x)}")

    tiles = enumerate_tiles((H, W), [TS, TS], [OV, OV])
    _expect(len(tiles) == 16, f"n_tiles={len(tiles)}")

    counts = Counter(t.n_seams for t in tiles)
    _expect(counts == {2: 4, 3: 8, 4: 4}, f"n_seams counts: {dict(counts)}")

    # Corner (0,0) covers rows [0, 32) and cols [0, 32); owns the right seam
    # (axis 1, x=32) and the bottom seam (axis 0, y=32).
    corner = tiles[0]
    _expect(corner.coord == (0, 0), f"coord={corner.coord}")
    _expect(corner.ranges == ((0, 32), (0, 32)), f"ranges={corner.ranges}")
    seam_pixels = sorted((s.axis, s.pixel) for s in corner.seams)
    _expect(seam_pixels == [(0, 32), (1, 32)], f"corner seams={seam_pixels}")

    # An interior tile at (1, 1) should own 4 seams.
    interior = next(t for t in tiles if t.coord == (1, 1))
    _expect(interior.n_seams == 4, f"interior n_seams={interior.n_seams}")
    _expect(
        interior.ranges == ((32, 64), (32, 64)),
        f"interior ranges={interior.ranges}",
    )

    # 3D: (1, 1, 64, 256, 256)-style geometry with non-zero overlap.
    tiles_3d = enumerate_tiles((64, 256, 256), [16, 64, 64], [8, 32, 32])
    counts_3d = Counter(t.n_seams for t in tiles_3d)
    # step_z=8, axis_size=64 → N = ceil((64-8)/8) = 7 → 6 seams along z, 7 regions
    # step_y/x=32, axis_size=256 → N = ceil((256-32)/32) = 7 → 6 seams, 7 regions
    _expect(
        len(tiles_3d) == 7 * 7 * 7,
        f"n_tiles_3d={len(tiles_3d)} (expected {7 * 7 * 7})",
    )
    # Interior tiles (all 3 axes interior) own 6 seams; one of each of these exists for
    # each interior 3D coord.
    n_interior_3d = sum(1 for t in tiles_3d if t.n_seams == 6)
    _expect(
        n_interior_3d == 5 * 5 * 5,
        f"interior 3D count={n_interior_3d} (expected {5 * 5 * 5})",
    )

    print("OK: tile geometry")


if __name__ == "__main__":
    main()
