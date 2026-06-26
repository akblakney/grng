"""Monte Carlo pi estimation visualization."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from .base import to_uint8


def plot_monte_carlo(data: bytes | np.ndarray, max_points: int = 10_000) -> Figure:
    """Monte Carlo pi estimation scatter plot.

    Interprets consecutive byte pairs as (x, y) coordinates in [0, 255]^2,
    normalized to the unit square. Points inside the unit circle are colored
    differently from those outside. The ratio of inside to total points
    estimates pi/4.

    Args:
        data: raw bytes from the entropy source.
        max_points: number of points to plot (default 10k).

    Returns:
        matplotlib Figure.
    """
    arr = to_uint8(data)
    n = min(max_points, len(arr) // 2)
    x = arr[0::2][:n].astype(float) / 255.0
    y = arr[1::2][:n].astype(float) / 255.0

    inside = (x ** 2 + y ** 2) <= 1.0
    pi_estimate = 4 * inside.sum() / n

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(x[inside],  y[inside],  s=1.0, color="steelblue", alpha=0.5,
               label="Inside", rasterized=True)
    ax.scatter(x[~inside], y[~inside], s=1.0, color="salmon",    alpha=0.5,
               label="Outside", rasterized=True)

    # Draw quarter circle
    theta = np.linspace(0, np.pi / 2, 300)
    ax.plot(np.cos(theta), np.sin(theta), color="black", linewidth=1.5)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(
        f"Monte Carlo Pi Estimation\n"
        f"{n:,} points — π ≈ {pi_estimate:.6f} (actual: 3.141593)"
    )
    ax.legend(loc="lower left", markerscale=5)
    fig.tight_layout()
    return fig