import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# --- Settings ---
dat_folder = "coord_seligFmt"   # folder containing .dat files
outdir = "airfoils_png"
txt_outdir = "airfoils_txt"
os.makedirs(outdir, exist_ok=True)
os.makedirs(txt_outdir, exist_ok=True)

n_points = 40
dpi = 100
img_size = (2.56, 2.56)  # 2.56 inches * 100 dpi = 256 pixels

dat_files = sorted([os.path.join(dat_folder, f) for f in os.listdir(dat_folder) if f.endswith(".dat")])
print(f"Found {len(dat_files)} airfoil files")

count = 0
for file_path in dat_files:
    try:
        # Read coordinate data
        with open(file_path, "r", errors="ignore") as f:
            lines = f.readlines()

        coords = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 2:
                try:
                    coords.append([float(parts[0]), float(parts[1])])
                except:
                    pass
        coords = np.array(coords)
        if len(coords) < 10:
            continue

        # Find leading edge (minimum x)
        le_idx = np.argmin(coords[:, 0])
        upper = coords[:le_idx + 1]
        lower = coords[le_idx:]

        # Sort by x
        upper = upper[np.argsort(upper[:, 0])]
        lower = lower[np.argsort(lower[:, 0])]

        #Interpolates
        x_common = np.linspace(0, 1, n_points)
        yu = np.interp(x_common, upper[:, 0], upper[:, 1])
        yl = np.interp(x_common, lower[:, 0], lower[:, 1])

        # Normalize and center vertically
        x = np.concatenate([x_common, x_common[::-1]])
        y = np.concatenate([yu, yl[::-1]])
        x = (x - np.min(x)) / (np.max(x) - np.min(x))
        y -= np.mean(y)

        # --- Plotting ---
        fig, ax = plt.subplots(figsize=img_size, dpi=dpi)
        ax.plot(x_common, yu, 'k', linewidth=2)
        ax.plot(x_common, yl, 'k', linewidth=2)
        ax.axis('off')
        ax.set_aspect('equal')
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.3, 0.3)

        # Save PNG
        name = os.path.splitext(os.path.basename(file_path))[0]
        save_path = os.path.join(outdir, f"{name}.png")
        fig.savefig(save_path, dpi=dpi, bbox_inches=None, pad_inches=0, facecolor='white')
        plt.close(fig)

        # Save coordinates to TXT
        txt_path = os.path.join(txt_outdir, f"{name}.txt")
        # Save as two columns: x y
        np.savetxt(txt_path, np.column_stack([x, y]), fmt="%.6f", header="x y", comments='')

        count += 1
        if count % 20 == 0:
            print(f"Saved {count} images and txt files...")

    except Exception as e:
        print(f"Skipping {file_path}: {e}")

print(f"✅ Done! Saved {count} airfoil images (256x256 px) and TXT coordinates to '{outdir}' / '{txt_outdir}'")

