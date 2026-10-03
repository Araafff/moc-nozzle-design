
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



GAMMA = 1.4
ME_LIST = [1.5, 3.0, 6.0]
N_WAVES = 40
Y_THROAT = 1.0

FIG_DIR = "figures"
DATA_DIR = "data"



BLUE = "#0057ff"
RED = "#ff1f1f"
GREEN = "#00aa33"
BLACK = "#000000"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 13,
    "axes.labelsize": 15,
    "axes.titlesize": 16,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.dpi": 120
})



def prandtl_meyer(M, gamma=GAMMA):
    """
    Prandtl-Meyer function nu(M), returned in radians.
    Valid for M >= 1.
    """

    if M < 1.0:
        raise ValueError("Prandtl-Meyer function requires M >= 1.")

    a = math.sqrt((gamma + 1.0) / (gamma - 1.0))
    b = math.sqrt(((gamma - 1.0) / (gamma + 1.0)) * (M**2 - 1.0))
    c = math.sqrt(M**2 - 1.0)

    return a * math.atan(b) - math.atan(c)


def mach_from_prandtl_meyer(nu_target, gamma=GAMMA):
    """
    Invert nu(M) by bisection.
    """

    if abs(nu_target) < 1.0e-12:
        return 1.0

    M_low = 1.0
    M_high = 100.0

    for _ in range(200):
        M_mid = 0.5 * (M_low + M_high)
        nu_mid = prandtl_meyer(M_mid, gamma)

        if nu_mid < nu_target:
            M_low = M_mid
        else:
            M_high = M_mid

    return 0.5 * (M_low + M_high)


def mach_angle(M):
    """
    Mach angle mu = asin(1/M), returned in radians.
    """

    return math.asin(1.0 / M)


def area_mach_ratio(M, gamma=GAMMA):
    """
    Quasi-1D isentropic area-Mach relation A/A*.
    """

    term = (2.0 / (gamma + 1.0)) * (
        1.0 + 0.5 * (gamma - 1.0) * M**2
    )

    exponent = (gamma + 1.0) / (2.0 * (gamma - 1.0))

    return (1.0 / M) * term**exponent


def isentropic_ratios(M, gamma=GAMMA):
    """
    Isentropic ratios based on stagnation quantities.

    Returns:
        T/T0, p/p0, rho/rho0, u/a0
    """

    T_T0 = 1.0 / (1.0 + 0.5 * (gamma - 1.0) * M**2)

    p_p0 = T_T0 ** (gamma / (gamma - 1.0))

    rho_rho0 = T_T0 ** (1.0 / (gamma - 1.0))

    u_a0 = M * math.sqrt(T_T0)

    return T_T0, p_p0, rho_rho0, u_a0


def intersect_lines(x1, y1, m1, x2, y2, m2):
    """
    Intersect two lines:
        y - y1 = m1 (x - x1)
        y - y2 = m2 (x - x2)
    """

    denom = m1 - m2

    if abs(denom) < 1.0e-12:
        raise RuntimeError("Nearly parallel characteristic lines encountered.")

    x = (y2 - y1 + m1 * x1 - m2 * x2) / denom
    y = y1 + m1 * (x - x1)

    return x, y



def solve_minimum_length_nozzle(Me, N=N_WAVES, gamma=GAMMA):
    
    nu_e = prandtl_meyer(Me, gamma)
    theta_max = 0.5 * nu_e

    theta_initial = np.linspace(theta_max / N, theta_max, N)

    x = np.full((N, N), np.nan)
    y = np.full((N, N), np.nan)

    theta = np.full((N, N), np.nan)
    nu = np.full((N, N), np.nan)
    M = np.full((N, N), np.nan)
    mu = np.full((N, N), np.nan)

    K_minus = np.full((N, N), np.nan)
    K_plus = np.full((N, N), np.nan)


    for i in range(N):

        theta_i = theta_initial[i]
        nu_i = theta_i

        K_minus_i = theta_i + nu_i

        for j in range(i + 1):

            if j == i:
                theta_ij = 0.0
                nu_ij = K_minus_i

            else:
                K_plus_j = -2.0 * theta_initial[j]

                theta_ij = 0.5 * (K_minus_i + K_plus_j)
                nu_ij = 0.5 * (K_minus_i - K_plus_j)

            theta[i, j] = theta_ij
            nu[i, j] = nu_ij

            K_minus[i, j] = theta_ij + nu_ij
            K_plus[i, j] = theta_ij - nu_ij

            M[i, j] = mach_from_prandtl_meyer(nu_ij, gamma)
            mu[i, j] = mach_angle(M[i, j])


            if i == 0 and j == 0:
                M_start = mach_from_prandtl_meyer(nu_i, gamma)
                mu_start = mach_angle(M_start)

                m_minus = math.tan(theta_i - mu_start)

                x[i, j] = -Y_THROAT / m_minus
                y[i, j] = 0.0

            elif j == 0:
                M_start = mach_from_prandtl_meyer(nu_i, gamma)
                mu_start = mach_angle(M_start)

                m_minus = math.tan(
                    0.5 * (theta_i + theta[i, j])
                    -
                    0.5 * (mu_start + mu[i, j])
                )

                m_plus = math.tan(
                    0.5 * (theta[i - 1, j] + theta[i, j])
                    +
                    0.5 * (mu[i - 1, j] + mu[i, j])
                )

                x[i, j], y[i, j] = intersect_lines(
                    0.0,
                    Y_THROAT,
                    m_minus,
                    x[i - 1, j],
                    y[i - 1, j],
                    m_plus
                )

            elif j == i:
                m_minus = math.tan(
                    0.5 * (theta[i, j - 1] + theta[i, j])
                    -
                    0.5 * (mu[i, j - 1] + mu[i, j])
                )

                x[i, j] = x[i, j - 1] - y[i, j - 1] / m_minus
                y[i, j] = 0.0

            else:
                m_minus = math.tan(
                    0.5 * (theta[i, j - 1] + theta[i, j])
                    -
                    0.5 * (mu[i, j - 1] + mu[i, j])
                )

                m_plus = math.tan(
                    0.5 * (theta[i - 1, j] + theta[i, j])
                    +
                    0.5 * (mu[i - 1, j] + mu[i, j])
                )

                x[i, j], y[i, j] = intersect_lines(
                    x[i, j - 1],
                    y[i, j - 1],
                    m_minus,
                    x[i - 1, j],
                    y[i - 1, j],
                    m_plus
                )


    wall_x = [0.0]
    wall_y = [Y_THROAT]
    wall_theta = [theta_max]
    wall_nu = [0.0]
    wall_M = [1.0]
    wall_mu = [math.pi / 2.0]
    wall_Kplus = [0.0]

    for j in range(N):

        i = N - 1

        x_int = x[i, j]
        y_int = y[i, j]

        theta_int = theta[i, j]
        mu_int = mu[i, j]
        K_plus_int = K_plus[i, j]

        theta_w = theta_max - theta_initial[j]

        if j == N - 1:
            theta_w = 0.0

        nu_w = theta_w - K_plus_int

        M_w = mach_from_prandtl_meyer(nu_w, gamma)
        mu_w = mach_angle(M_w)

        m_char = math.tan(
            0.5 * (theta_int + theta_w)
            +
            0.5 * (mu_int + mu_w)
        )

        m_wall = math.tan(0.5 * (wall_theta[-1] + theta_w))

        x_w, y_w = intersect_lines(
            x_int,
            y_int,
            m_char,
            wall_x[-1],
            wall_y[-1],
            m_wall
        )

        if x_w <= wall_x[-1] or y_w <= 0.0:
            print(f"Warning: rejected non-forward wall point for Me={Me}, j={j}")
            continue

        wall_x.append(x_w)
        wall_y.append(y_w)
        wall_theta.append(theta_w)
        wall_nu.append(nu_w)
        wall_M.append(M_w)
        wall_mu.append(mu_w)
        wall_Kplus.append(K_plus_int)

    wall_x = np.array(wall_x)
    wall_y = np.array(wall_y)
    wall_theta = np.array(wall_theta)
    wall_nu = np.array(wall_nu)
    wall_M = np.array(wall_M)
    wall_mu = np.array(wall_mu)
    wall_Kplus = np.array(wall_Kplus)


    moc_records = []

    for i in range(N):
        for j in range(i + 1):

            T_T0, p_p0, rho_rho0, u_a0 = isentropic_ratios(M[i, j], gamma)

            moc_records.append({
                "i": i,
                "j": j,
                "x": x[i, j],
                "y": y[i, j],
                "theta_rad": theta[i, j],
                "theta_deg": math.degrees(theta[i, j]),
                "nu_rad": nu[i, j],
                "nu_deg": math.degrees(nu[i, j]),
                "M": M[i, j],
                "mu_rad": mu[i, j],
                "mu_deg": math.degrees(mu[i, j]),
                "K_minus_rad": K_minus[i, j],
                "K_plus_rad": K_plus[i, j],
                "K_minus_deg": math.degrees(K_minus[i, j]),
                "K_plus_deg": math.degrees(K_plus[i, j]),
                "T_T0": T_T0,
                "p_p0": p_p0,
                "rho_rho0": rho_rho0,
                "u_a0": u_a0
            })

    moc_df = pd.DataFrame(moc_records)

    center_records = []

    for i in range(N):
        j = i

        T_T0, p_p0, rho_rho0, u_a0 = isentropic_ratios(M[i, j], gamma)

        center_records.append({
            "x": x[i, j],
            "y": y[i, j],
            "theta_deg": math.degrees(theta[i, j]),
            "nu_deg": math.degrees(nu[i, j]),
            "M": M[i, j],
            "mu_deg": math.degrees(mu[i, j]),
            "K_minus_deg": math.degrees(K_minus[i, j]),
            "K_plus_deg": math.degrees(K_plus[i, j]),
            "T_T0": T_T0,
            "p_p0": p_p0,
            "rho_rho0": rho_rho0,
            "u_a0": u_a0
        })

    center_df = pd.DataFrame(center_records)

    wall_records = []

    for k in range(len(wall_x)):

        T_T0, p_p0, rho_rho0, u_a0 = isentropic_ratios(wall_M[k], gamma)

        wall_records.append({
            "k": k,
            "x": wall_x[k],
            "y": wall_y[k],
            "theta_deg": math.degrees(wall_theta[k]),
            "nu_deg": math.degrees(wall_nu[k]),
            "M": wall_M[k],
            "mu_deg": math.degrees(wall_mu[k]),
            "K_plus_deg": math.degrees(wall_Kplus[k]),
            "T_T0": T_T0,
            "p_p0": p_p0,
            "rho_rho0": rho_rho0,
            "u_a0": u_a0,
            "inv_sT": Y_THROAT / wall_y[k]
        })

    wall_df = pd.DataFrame(wall_records)

    moc_area_ratio = wall_y[-1] / Y_THROAT
    quasi_area_ratio = area_mach_ratio(Me, gamma)

    exit_center_M = center_df["M"].iloc[-1]
    exit_wall_M = wall_df["M"].iloc[-1]
    exit_wall_theta = wall_df["theta_deg"].iloc[-1]

    validation = {
        "Me": Me,
        "nu_e_deg": math.degrees(nu_e),
        "theta_max_deg": math.degrees(theta_max),
        "moc_exit_area_ratio": moc_area_ratio,
        "quasi1d_area_ratio": quasi_area_ratio,
        "area_percent_difference": 100.0 * (moc_area_ratio - quasi_area_ratio) / quasi_area_ratio,
        "exit_center_M": exit_center_M,
        "exit_wall_M": exit_wall_M,
        "exit_wall_theta_deg": exit_wall_theta,
        "center_M_error_percent": 100.0 * (exit_center_M - Me) / Me,
        "wall_M_error_percent": 100.0 * (exit_wall_M - Me) / Me,
        "nozzle_length": wall_x[-1],
        "exit_height": wall_y[-1]
    }

    return {
        "Me": Me,
        "gamma": gamma,
        "N": N,
        "nu_e": nu_e,
        "theta_max": theta_max,
        "theta_initial": theta_initial,
        "x": x,
        "y": y,
        "theta": theta,
        "nu": nu,
        "M": M,
        "mu": mu,
        "K_minus": K_minus,
        "K_plus": K_plus,
        "wall_x": wall_x,
        "wall_y": wall_y,
        "wall_theta": wall_theta,
        "wall_nu": wall_nu,
        "wall_M": wall_M,
        "moc_df": moc_df,
        "center_df": center_df,
        "wall_df": wall_df,
        "validation": validation
    }



def savefig(filename):
    path = os.path.join(FIG_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_characteristic_net(result):
    Me = result["Me"]
    x = result["x"]
    y = result["y"]
    N = result["N"]

    plt.figure(figsize=(8.0, 5.0))

    for i in range(N):
        xs = [0.0]
        ys = [Y_THROAT]

        for j in range(i + 1):
            xs.append(x[i, j])
            ys.append(y[i, j])

        plt.plot(xs, ys, color=BLUE, linewidth=0.75, alpha=0.75)

    for j in range(N):
        xs = []
        ys = []

        for i in range(j, N):
            xs.append(x[i, j])
            ys.append(y[i, j])

        if len(xs) > 1:
            plt.plot(xs, ys, color=RED, linewidth=0.75, alpha=0.75)

    for k in range(1, len(result["wall_x"])):
        j = k - 1
        i = N - 1

        plt.plot(
            [x[i, j], result["wall_x"][k]],
            [y[i, j], result["wall_y"][k]],
            color=RED,
            linewidth=0.75,
            alpha=0.75
        )

    plt.plot(result["wall_x"], result["wall_y"], color=BLACK, linewidth=2.2, label="Wall")

    plt.plot(
        [0.0, result["wall_x"][-1]],
        [0.0, 0.0],
        color=BLACK,
        linestyle="--",
        linewidth=1.2,
        label="Centerline"
    )

    plt.xlabel(r"$x/y_t$")
    plt.ylabel(r"$y/y_t$")
    plt.title(rf"Characteristic Net, $M_e={Me}$")
    plt.axis("equal")
    plt.legend(frameon=False)

    savefig(f"characteristic_net_Me_{str(Me).replace('.', 'p')}.png")


def plot_mach_points(result):
    Me = result["Me"]

    plt.figure(figsize=(8.0, 5.0))

    df = result["moc_df"]
    wall_df = result["wall_df"]

    sc1 = plt.scatter(
        df["x"],
        df["y"],
        c=df["M"],
        cmap="turbo",
        s=20,
        edgecolors="none",
        label="MOC points"
    )

    plt.scatter(
        wall_df["x"],
        wall_df["y"],
        c=wall_df["M"],
        cmap="turbo",
        s=25,
        marker="s",
        edgecolors="none"
    )

    cbar = plt.colorbar(sc1)
    cbar.set_label(r"$M$")

    plt.plot(result["wall_x"], result["wall_y"], color=BLACK, linewidth=2.2)
    plt.plot([0, result["wall_x"][-1]], [0, 0], color=BLACK, linestyle="--", linewidth=1.2)

    plt.xlabel(r"$x/y_t$")
    plt.ylabel(r"$y/y_t$")
    plt.title(rf"Mach Number at MOC Points, $M_e={Me}$")
    plt.axis("equal")

    savefig(f"mach_points_Me_{str(Me).replace('.', 'p')}.png")


def plot_profile_markers(result, var, ylabel, prefix):
    Me = result["Me"]
    center_df = result["center_df"]
    wall_df = result["wall_df"]

    plt.figure(figsize=(7.0, 4.5))

    plt.plot(
        center_df["x"],
        center_df[var],
        linestyle="None",
        marker="o",
        markersize=5.0,
        markerfacecolor=BLUE,
        markeredgecolor=BLUE,
        label="Centerline"
    )

    plt.plot(
        wall_df["x"],
        wall_df[var],
        linestyle="None",
        marker="s",
        markersize=5.0,
        markerfacecolor=RED,
        markeredgecolor=RED,
        label="Wall"
    )

    plt.xlabel(r"$x/y_t$")
    plt.ylabel(ylabel)
    plt.title(rf"{ylabel} Along Centerline and Wall, $M_e={Me}$")
    plt.legend(frameon=False)

    savefig(f"{prefix}_centerline_wall_Me_{str(Me).replace('.', 'p')}.png")


def plot_profiles(result):
    plot_profile_markers(result, "M", r"$M$", "M")
    plot_profile_markers(result, "p_p0", r"$p/p_0$", "p_p0")
    plot_profile_markers(result, "rho_rho0", r"$\rho/\rho_0$", "rho_rho0")
    plot_profile_markers(result, "u_a0", r"$u/a_0$", "u_a0")


def plot_inverse_streamline_spacing(result):
    Me = result["Me"]
    wall_df = result["wall_df"]

    plt.figure(figsize=(7.0, 4.5))

    plt.plot(
        wall_df["x"],
        wall_df["inv_sT"],
        linestyle="None",
        marker="o",
        markersize=5.0,
        markerfacecolor=BLUE,
        markeredgecolor=BLUE
    )

    plt.xlabel(r"$x/y_t$")
    plt.ylabel(r"$1/s_T$")
    plt.title(rf"Inverse Streamline Spacing, $M_e={Me}$")

    savefig(f"inv_sT_Me_{str(Me).replace('.', 'p')}.png")


def plot_wall_contours(all_results):
    plt.figure(figsize=(8.0, 5.0))

    colors = [BLUE, RED, GREEN]
    markers = ["o", "s", "^"]

    for result, color, marker in zip(all_results, colors, markers):
        Me = result["Me"]

        plt.plot(
            result["wall_x"],
            result["wall_y"],
            linestyle="None",
            marker=marker,
            markersize=4.0,
            markerfacecolor=color,
            markeredgecolor=color,
            label=rf"$M_e={Me}$"
        )

    max_x = max(r["wall_x"][-1] for r in all_results)

    plt.plot([0, max_x], [0, 0], color=BLACK, linestyle="--", linewidth=1.2)

    plt.xlabel(r"$x/y_t$")
    plt.ylabel(r"$y/y_t$")
    plt.title("MOC Minimum-Length Nozzle Wall Contours")
    plt.axis("equal")
    plt.legend(frameon=False)

    savefig("wall_contours_all_Me.png")


def plot_quasi1d_comparison(all_results):
    rows = []

    for result in all_results:
        rows.append(result["validation"])

    df = pd.DataFrame(rows)

    plt.figure(figsize=(7.0, 4.5))

    plt.plot(
        df["Me"],
        df["moc_exit_area_ratio"],
        linestyle="None",
        marker="o",
        markersize=7,
        markerfacecolor=BLUE,
        markeredgecolor=BLUE,
        label="MOC exit height ratio"
    )

    plt.plot(
        df["Me"],
        df["quasi1d_area_ratio"],
        linestyle="None",
        marker="s",
        markersize=7,
        markerfacecolor=RED,
        markeredgecolor=RED,
        label="Quasi-1D area-Mach"
    )

    plt.xlabel(r"$M_e$")
    plt.ylabel(r"$A_e/A^*$")
    plt.title("Exit Area Ratio Comparison")
    plt.legend(frameon=False)

    savefig("quasi1d_area_ratio_comparison.png")

    df.to_csv(os.path.join(DATA_DIR, "validation_summary.csv"), index=False)



def main():

    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    all_results = []

    print("\nRunning real MOC nozzle solver...\n")

    for Me in ME_LIST:

        print(f"Solving Me = {Me}")

        result = solve_minimum_length_nozzle(Me, N=N_WAVES, gamma=GAMMA)

        all_results.append(result)

        tag = str(Me).replace(".", "p")

        result["moc_df"].to_csv(
            os.path.join(DATA_DIR, f"moc_points_Me_{tag}.csv"),
            index=False
        )

        result["center_df"].to_csv(
            os.path.join(DATA_DIR, f"centerline_Me_{tag}.csv"),
            index=False
        )

        result["wall_df"].to_csv(
            os.path.join(DATA_DIR, f"wall_Me_{tag}.csv"),
            index=False
        )

        plot_characteristic_net(result)
        plot_mach_points(result)
        plot_profiles(result)
        plot_inverse_streamline_spacing(result)

        v = result["validation"]

        print(f"  nu_e                 = {v['nu_e_deg']:.6f} deg")
        print(f"  theta_max            = {v['theta_max_deg']:.6f} deg")
        print(f"  nozzle length        = {v['nozzle_length']:.6f}")
        print(f"  exit height          = {v['exit_height']:.6f}")
        print(f"  MOC Ae/A*            = {v['moc_exit_area_ratio']:.6f}")
        print(f"  quasi-1D Ae/A*       = {v['quasi1d_area_ratio']:.6f}")
        print(f"  area difference      = {v['area_percent_difference']:.6f} %")
        print(f"  exit center M        = {v['exit_center_M']:.6f}")
        print(f"  exit wall M          = {v['exit_wall_M']:.6f}")
        print(f"  exit wall theta      = {v['exit_wall_theta_deg']:.6f} deg")
        print("  data and figures saved\n")

    plot_wall_contours(all_results)
    plot_quasi1d_comparison(all_results)

    print("Done.")
    print(f"Figures saved in: {FIG_DIR}/")
    print(f"Data saved in:    {DATA_DIR}/\n")


if __name__ == "__main__":
    main()