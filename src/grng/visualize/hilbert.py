"""Hilbert curve bitmap visualization."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from .base import to_uint8


def _xy_to_hilbert(n: int, x: int, y: int) -> int:
    """Convert (x, y) to Hilbert curve index for an n×n grid."""
    d = 0
    s = n // 2
    while s > 0:
        rx = 1 if (x & s) > 0 else 0
        ry = 1 if (y & s) > 0 else 0
        d += s * s * ((3 * rx) ^ ry)
        # Rotate
        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x
        s //= 2
    return d


def _hilbert_to_xy(n: int, d: int) -> tuple[int, int]:
    """Convert Hilbert curve index to (x, y) for an n×n grid."""
    x = y = 0
    s = 1
    while s < n:
        rx = 1 if (d & 2) else 0
        ry = 1 if (d & 1) ^ rx else 0
        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x
        x += s * rx
        y += s * ry
        d //= 4
        s *= 2
    return x, y


def plot_hilbert(data: bytes | np.ndarray, order: int = 7) -> Figure:
    """Render bytes onto a Hilbert curve as a grayscale bitmap.

    Maps bytes onto a Hilbert curve of given order, producing an
    n×n image where n=2^order. Preserves spatial locality — any
    patterns in the byte sequence show up as visible structure.
    A truly random source produces uniform noise with no structure.

    Order 7 produces a 128×128 image using 16,384 bytes.
    Order 8 produces a 256×256 image using 65,536 bytes.

    Args:
        data: raw bytes from the entropy source.
        order: Hilbert curve order (default 7 → 128×128 image).

    Returns:
        matplotlib Figure.
    """
    arr = to_uint8(data)
    n = 2 ** order
    n_pixels = n * n

    if len(arr) < n_pixels:
        raise ValueError(
            f"Need at least {n_pixels:,} bytes for order {order}, "
            f"got {len(arr):,}"
        )

    arr = arr[:n_pixels]
    image = np.zeros((n, n), dtype=np.uint8)

    # Build lookup table: Hilbert index -> (x, y)
    xs = np.zeros(n_pixels, dtype=np.int32)
    ys = np.zeros(n_pixels, dtype=np.int32)
    for d in range(n_pixels):
        x, y = _hilbert_to_xy(n, d)
        xs[d] = x
        ys[d] = y

    image[ys, xs] = arr

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(image, cmap="gray", interpolation="nearest", vmin=0, vmax=255)
    ax.set_title(
        f"Hilbert Curve Bitmap (order {order}, {n}×{n}, {n_pixels:,} bytes)"
    )
    ax.axis("off")
    fig.tight_layout()
    return fig