"""Scatter plot visualization — byte[i] vs byte[i+1]."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from .base import to_uint8


def plot_scatter(data: bytes | np.ndarray, max_points: int = 50_000) -> Figure:
    """Scatter plot of consecutive byte pairs.

    Plots byte[i] vs byte[i+1]. For truly random data this produces a
    uniform cloud over the 256x256 grid with no visible structure.
    Any correlation between consecutive bytes shows up as diagonal
    lines or clustering.

    Args:
        data: raw bytes from the entropy source.
        max_points: number of pairs to plot (default 50k).

    Returns:
        matplotlib Figure.
    """
    arr = to_uint8(data)
    x = arr[:-1][:max_points].astype(float)
    y = arr[1:][:max_points].astype(float)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(x, y, s=0.5, alpha=0.3, color="steelblue", rasterized=True)
    ax.set_xlim(0, 255)
    ax.set_ylim(0, 255)
    ax.set_xlabel("byte[i]")
    ax.set_ylabel("byte[i+1]")
    ax.set_title(f"Consecutive Byte Scatter Plot ({max_points:,} pairs)")
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig