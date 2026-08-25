"""
plot_wolf.py  —  Visualize wolf pack trajectory from wolf_trajectory.npy

Usage:
    python plot_wolf.py                        # looks for wolf_trajectory.npy in cwd
    python plot_wolf.py path/to/file.npy       # explicit path

Produces two figures:
  1. Spatial trajectory  (Cx, Cy) coloured by time, with pack size z as marker size
  2. Time series of Cx, Cy, and z on shared time axis
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ── load ──────────────────────────────────────────────────────────────────────
path = sys.argv[1] if len(sys.argv) > 1 else "wolf_trajectory.npy"
data = np.load(path)          # shape (n_steps, 4)
t, Cx, Cy, z = data[:, 0], data[:, 1], data[:, 2], data[:, 3]

# ── Figure 1: spatial trajectory ──────────────────────────────────────────────
fig1, ax = plt.subplots(figsize=(7, 6))

# colour = time, size = pack size z (clipped so tiny z stays visible)
sc = ax.scatter(Cx, Cy,
                c=t, cmap="plasma",
                s=np.clip(z * 30, 10, 300),
                alpha=0.7, linewidths=0)
cb = fig1.colorbar(sc, ax=ax, label="Time (years)")

# mark start and end
ax.plot(Cx[0],  Cy[0],  "go", ms=10, label="Start", zorder=5)
ax.plot(Cx[-1], Cy[-1], "r*", ms=12, label="End",   zorder=5)

# faint line connecting steps
ax.plot(Cx, Cy, "k-", lw=0.4, alpha=0.3)

ax.set_xlabel("Pack centre  Cx")
ax.set_ylabel("Pack centre  Cy")
ax.set_title("Wolf pack spatial trajectory\n(marker size ∝ z, colour = time)")
ax.legend()
fig1.tight_layout()
fig1.savefig("wolf_trajectory_map.png", dpi=150)
print("Saved wolf_trajectory_map.png")

# ── Figure 2: time series ──────────────────────────────────────────────────────
fig2, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)

axes[0].plot(t, Cx, color="steelblue")
axes[0].set_ylabel("Cx")
axes[0].set_title("Pack centre x")

axes[1].plot(t, Cy, color="darkorange")
axes[1].set_ylabel("Cy")
axes[1].set_title("Pack centre y")

axes[2].plot(t, z, color="seagreen")
axes[2].set_ylabel("z  (pack size scalar)")
axes[2].set_xlabel("Time (years)")
axes[2].set_title("Pack size z")

fig2.suptitle("Wolf pack dynamics", fontsize=13)
fig2.tight_layout()
fig2.savefig("wolf_trajectory_timeseries.png", dpi=150)
print("Saved wolf_trajectory_timeseries.png")

plt.show()