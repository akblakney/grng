"""1D and 2D random walk visualizations."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from .base import to_steps, to_uint8, apply_dark_style


def plot_1d(data: bytes | np.ndarray, max_steps: int = 100_000) -> Figure:
    """Plot a 1D random walk from bit stream.

    Each bit is interpreted as a +1 or -1 step. The cumulative sum
    produces a Brownian motion path.

    Args:
        data: raw bytes from the entropy source.
        max_steps: number of bits to use (default 100k).

    Returns:
        matplotlib Figure.
    """
    apply_dark_style()
    steps = to_steps(data)[:max_steps]
    walk = np.cumsum(steps)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(walk, linewidth=0.5, color="steelblue")
    ax.axhline(0, color="white", linewidth=0.5, linestyle="--", alpha=0.4)
    ax.set_xlabel("Step")
    ax.set_ylabel("Position")
    ax.set_title(f"1D Random Walk ({len(steps):,} steps)")
    fig.tight_layout()
    return fig


def plot_2d(data: bytes | np.ndarray, max_steps: int = 50_000) -> Figure:
    """Plot a 2D random walk from byte stream.

    Consecutive byte pairs are used to determine x and y steps.
    Each byte is mapped to a step in [-1, +1] by thresholding at 128.

    Args:
        data: raw bytes from the entropy source.
        max_steps: number of steps to use (default 50k).

    Returns:
        matplotlib Figure.
    """
    apply_dark_style()
    arr = to_uint8(data)
    # Each byte -> +1 if >= 128, else -1
    steps = (arr.astype(np.int16) >= 128).astype(np.int8) * 2 - 1
    x_steps = steps[0::2][:max_steps]
    y_steps = steps[1::2][:max_steps]
    n = min(len(x_steps), len(y_steps))
    x = np.cumsum(x_steps[:n])
    y = np.cumsum(y_steps[:n])

    fig, ax = plt.subplots(figsize=(8, 8))
    # Color by progress through the walk
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    from matplotlib.collections import LineCollection
    from matplotlib.cm import viridis
    norm = plt.Normalize(0, len(x))
    lc = LineCollection(segments, cmap=viridis, norm=norm, linewidth=0.4, alpha=0.8)
    lc.set_array(np.arange(len(x)))
    ax.add_collection(lc)
    ax.autoscale()
    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(f"2D Random Walk ({n:,} steps)")
    fig.colorbar(lc, ax=ax, label="Step number")
    fig.tight_layout()
    return fig