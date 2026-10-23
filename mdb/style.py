"""Matplotlib style and model colors."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MODEL_LABELS = {"sdxl_turbo": "SDXL-Turbo", "sd15": "SD 1.5"}
MODEL_COLORS = {"sdxl_turbo": "#2a78d6", "sd15": "#eb6834"}
INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#fcfcfb"
GOOD, BAD = "#1baf7a", "#e34948"


def label(m):
    return MODEL_LABELS.get(m, m)


def color(m):
    return MODEL_COLORS.get(m, "#1baf7a")


def apply():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.size": 13, "axes.titlesize": 15, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelcolor": INK_2, "axes.edgecolor": MUTED, "xtick.color": INK_2, "ytick.color": INK_2,
        "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "legend.frameon": False, "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
    })
