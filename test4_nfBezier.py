import matplotlib.pyplot as plt
import neuralfoil as nf
import aerosandbox as asb
import numpy as np
import math

# ----------------- Bezier Curve ----------------- #
def bernstein_poly(n, i, t):
    return math.comb(n, i) * (t**i) * ((1 - t)**(n - i))

def bezier_curve(A, B, n_points=1000):
    n = len(A) - 1
    t = np.linspace(0, 1, n_points)
    Px, Pz = np.zeros_like(t), np.zeros_like(t)
    for i in range(n + 1):
        basis = bernstein_poly(n, i, t)
        Px += A[i] * basis
        Pz += B[i] * basis
    return Px, Pz

# Control points
Au = [0.0, 0.3, 0.6, 0.9, 1.0]   # x from leading edge to trailing edge
Al = [0.0, 0.3, 0.6, 0.9, 1.0]  # x same as upper

Bu = [0.0, 0.06, 0.09, 0.03, 0.0]  # z thickness, positive for upper
Bl = [0.0, -0.02, -0.03, -0.01, 0.0] # z thickness, negative for lower

# Generate curves
Pxu, Pzu = bezier_curve(Au, Bu)
Pxl, Pzl = bezier_curve(Al, Bl)

coords = np.vstack([
    np.column_stack([Pxu, Pzu]),
    np.column_stack([Pxl[::-1], Pzl[::-1]])
])

# Make airfoil
airfoil = asb.Airfoil(name="BezierFoil", coordinates=coords)

# --------- Plot Airfoil Shape --------- #
plt.figure()
plt.plot(Pxu, Pzu, 'b', label="Upper Surface")
plt.plot(Pxl, Pzl, 'r', label="Lower Surface")
plt.fill(coords[:,0], coords[:,1], color="lightgray", alpha=0.5)
plt.axis('equal')
plt.xlabel("x")
plt.ylabel("z")
plt.title("Bezier Hydrofoil Shape")
plt.legend(); plt.grid(True)

# Flow conditions
rho, mu = 1000, 1e-3
chord, velocity = 0.2, 5.0
Re_water = rho * velocity * chord / mu

# NeuralFoil analysis
alphas = np.linspace(-15, 15, 40)
CL_values, CD_values = [], []

for alpha in alphas:
    result = nf.get_aero_from_airfoil(airfoil, alpha=alpha, Re=Re_water, model_size="large")
    CL_values.append(result["CL"])
    CD_values.append(result["CD"])

# --------- Plot CL vs AoA --------- #
plt.figure()
plt.plot(alphas, CL_values, label="CL")
plt.xlabel("Angle of Attack (deg)")
plt.ylabel("CL")
plt.title(f"Bezier Hydrofoil CL vs AoA (Re≈{Re_water:.1e})")
plt.legend(); plt.grid(True)

# --------- Plot CD vs AoA --------- #
plt.figure()
plt.plot(alphas, CD_values, 'r', label="CD")
plt.xlabel("Angle of Attack (deg)")
plt.ylabel("CD")
plt.title(f"Bezier Hydrofoil CD vs AoA (Re≈{Re_water:.1e})")
plt.legend(); plt.grid(True)

plt.show()
