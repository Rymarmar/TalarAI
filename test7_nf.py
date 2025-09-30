import matplotlib.pyplot as plt
import neuralfoil as nf
import aerosandbox as asb
import numpy as np
import math
from itertools import product

# ----------------- Sweep Parameters ----------------- #
sweep_params = {
    "Au": [0.0, 0.1, 0.6, 0.9, 1.0],
    # x-locations (normalized chord 0→1) of the **upper surface control points** for the Bezier curve.
    # - 0.0   → leading edge
    # - 1.0   → trailing edge
    # - middle points (0.3, 0.6, 0.9) define the shape/camber distribution
    # can change to any amount and will be ok
    "Al": [0.0, 0.2, 0.6, 0.9, 1.0],
    # x-locations (normalized chord 0→1) of the **lower surface control points** for the Bezier curve.
    # - 0.0   → leading edge
    # - 1.0   → trailing edge
    # - middle points (0.2, 0.6, 0.9) define the lower surface shape
    # Note: upper and lower control points do not have to align; this allows asymmetric camber

    "increments": np.arange(0.05, 0.11, 0.025), #change bu and bl by 0.025 start 0.05 to 0.11
    "chord": 5, #meters
    "velocity": 5.0, #m/s
    "rho": 1000, #kg/m^3
    "mu": 1e-3, #viscosity (kg/(ms))
    "fps": 0.15, #meters
    "alphas": np.linspace(-10, 15, 30), #angles of attack (start,end, how many in between)
    "max_thickness": 0.3, # percent of the chord
    "model_size": "large",
    "plot_step": 10  # every nth hydrofoil
}

# ----------------- Bezier Curve ----------------- #
def bernstein_poly(n, i, t):
    return math.comb(n, i) * (t**i) * ((1 - t)**(n - i))

def bezier_curve(A, B, n_points=300):
    n = len(A) - 1
    t = np.linspace(0, 1, n_points)
    Px, Pz = np.zeros_like(t), np.zeros_like(t)
    for i in range(n + 1):
        basis = bernstein_poly(n, i, t)
        Px += A[i] * basis
        Pz += B[i] * basis
    return Px, Pz

# ----------------- Free-Surface Wave Drag ----------------- #
def add_free_surface_drag(CL_values, velocity, chord, fps, g=9.81):
    CL_values = np.array(CL_values)
    FNh = velocity * fps / np.sqrt(g * chord)
    CDw = 0.5 * CL_values**2 / (FNh**2 * np.exp(2 / FNh**2))
    #drag_foil = 0.5 * rho * (velocity * fps)**2 * CD_total * S  # N
    #do i need to add this part if we don't need lbf? or do we need it in lbf?

    return CDw

# ----------------- Hydrofoil Sweep Function ----------------- #
def sweep_hydrofoils(params):
    Au = params["Au"]
    Al = params["Al"]
    increments = params["increments"]
    chord = params["chord"]
    velocity = params["velocity"]
    rho = params["rho"]
    mu = params["mu"]
    fps = params["fps"]
    alphas = params["alphas"]
    max_thickness = params["max_thickness"]
    model_size = params["model_size"]
    plot_step = params["plot_step"]

    Re_water = rho * velocity * chord / mu
    all_CL, all_CD = {}, {}

    for middle_Bu in product(increments, repeat=len(Au)-2):
        for middle_Bl in product(increments, repeat=len(Al)-2):
            Bu = [0.0] + [round(b,3) for b in middle_Bu] + [0.0]
            Bl = [0.0] + [-round(b,3) for b in middle_Bl] + [0.0]

            Pxu, Pzu = bezier_curve(Au, Bu)
            Pxl, Pzl = bezier_curve(Al, Bl)

            thickness = Pzu - Pzl
            if np.any(thickness <= -0.001) or np.max(thickness) > max_thickness:
                continue

            coords = np.vstack([
                np.column_stack([Pxu, Pzu]),
                np.column_stack([Pxl[::-1], Pzl[::-1]])
            ])
            airfoil_name = f"Bu{'_'.join(f'{b:.2f}' for b in Bu)}_Bl{'_'.join(f'{b:.2f}' for b in Bl)}"
            airfoil = asb.Airfoil(name=airfoil_name, coordinates=coords)

            CL_values, CD_values = [], []
            valid_foil = True
            for alpha in alphas:
                try:
                    result = nf.get_aero_from_airfoil(
                        airfoil,
                        alpha=alpha,
                        Re=Re_water,
                        model_size=model_size
                    )
                    CL_val = result["CL"]
                    CD_val = result["CD"]

                    if not np.isfinite(CL_val) or not np.isfinite(CD_val) or CD_val <= 0:
                        valid_foil = False
                        break

                    # Add free-surface wave drag
                    CDw = add_free_surface_drag([CL_val], velocity, chord, fps)[0]
                    CD_total = CD_val + CDw

                    CL_values.append(CL_val)
                    CD_values.append(CD_total)

                except:
                    valid_foil = False
                    break

            if not valid_foil:
                continue

            all_CL[airfoil_name] = CL_values
            all_CD[airfoil_name] = CD_values
            print("Valid foil found:", Bu, Bl)

    # ----------------- Find Best L/D ----------------- #
    best_ratio, best_airfoil = -np.inf, None
    for name in all_CL:
        CL = np.array(all_CL[name])
        CD = np.array(all_CD[name])
        mask = CD > 0
        if np.any(mask):
            LD = CL[mask] / CD[mask]
            max_LD = np.max(LD)
            if max_LD > best_ratio:
                best_ratio = max_LD
                best_airfoil = name

    # ----------------- Plot Results ----------------- #
    if best_airfoil is not None:
        plt.figure(figsize=(8,6))
        for i, (name, CL_values) in enumerate(all_CL.items()):
            if i % plot_step != 0:
                continue
            plt.plot(alphas, CL_values, label=name)
        plt.xlabel("Angle of Attack (deg)")
        plt.ylabel("Lift Coefficient CL")
        plt.title("CL vs AoA (every " + str(plot_step) + " hydrofoil)")
        plt.grid(True)

        plt.figure(figsize=(8,6))
        for i, (name, CD_values) in enumerate(all_CD.items()):
            if i % plot_step != 0:
                continue
            plt.plot(alphas, CD_values, label=name)
        plt.xlabel("Angle of Attack (deg)")
        plt.ylabel("Drag Coefficient CD (with free-surface)")
        plt.title("CD vs AoA (every " + str(plot_step) +" hydrofoil)")
        plt.grid(True)

        # Plot best foil shape
        parts = best_airfoil.split("_Bl")
        Bu_vals = [float(x) for x in parts[0][2:].split("_")]
        Bl_vals = [float(x) for x in parts[1].split("_")]
        Pxu, Pzu = bezier_curve(Au, Bu_vals)
        Pxl, Pzl = bezier_curve(Al, Bl_vals)

        plt.figure(figsize=(8,4))
        plt.plot(Pxu, Pzu, label="Upper Surface", color="blue")
        plt.plot(Pxl, Pzl, label="Lower Surface", color="red")
        plt.fill_between(Pxu, Pzu, Pzl, color="skyblue", alpha=0.3)
        plt.xlabel("x (chord)")
        plt.ylabel("z (thickness)")
        plt.title(f"Best L/D Hydrofoil Shape")
        plt.axis("equal")
        plt.grid(True)
        plt.legend()
        print("Best foil:", best_airfoil, "with L/D =", best_ratio)
        plt.show()
        

    return best_airfoil, best_ratio, all_CL, all_CD

# ----------------- Run Sweep ----------------- #
best_airfoil, best_ratio, all_CL, all_CD = sweep_hydrofoils(sweep_params)

