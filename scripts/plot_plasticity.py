#!/usr/bin/env python3
"""Plot stress-strain and tangent error for J2 and Drucker-Prager plasticity.

Reads CSV data produced by the plot_data test executable.
Usage: build the plot_data target, run it, then run this script.
"""
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})


def read_csv(path):
    data = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for col in reader.fieldnames:
            data[col] = []
        for row in reader:
            for col in reader.fieldnames:
                data[col].append(float(row[col]))
    for col in data:
        data[col] = np.array(data[col])
    return data


LOAD_CASE_LABELS = {
    "eps_11": (r"$\varepsilon_{11}$", "Uniaxial strain $\\varepsilon_{11}$"),
    "eps_22": (r"$\varepsilon_{22}$", "Uniaxial strain $\\varepsilon_{22}$"),
    "eps_12": (r"$\varepsilon_{12}$", "Pure shear $\\varepsilon_{12}$"),
}

# Which stress components to plot for each load case
STRESS_COMPONENTS = {
    "eps_11": [("sig_11", r"$\sigma_{11}$"), ("sig_22", r"$\sigma_{22}$")],
    "eps_22": [("sig_22", r"$\sigma_{22}$"), ("sig_11", r"$\sigma_{11}$")],
    "eps_12": [("sig_12", r"$\sigma_{12}$"), ("sig_11", r"$\sigma_{11}$")],
}


def plot_load_case(j2, dp, tag, base):
    eps_label, title = LOAD_CASE_LABELS.get(tag, (tag, tag))
    stress_cols = STRESS_COMPONENTS.get(tag, [("sig_11", r"$\sigma_{11}$")])

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # --- Top left: stress-strain (primary component) ---
    ax = axes[0, 0]
    primary_col, primary_label = stress_cols[0]
    ax.plot(j2["eps_load"], j2[primary_col], "o-", ms=3, label=f"J2 {primary_label}")
    ax.plot(dp["eps_load"], dp[primary_col], "s-", ms=3, label=f"DP {primary_label}")
    if len(stress_cols) > 1:
        sec_col, sec_label = stress_cols[1]
        ax.plot(j2["eps_load"], j2[sec_col], "o--", ms=3, alpha=0.6,
                label=f"J2 {sec_label}")
        ax.plot(dp["eps_load"], dp[sec_col], "s--", ms=3, alpha=0.6,
                label=f"DP {sec_label}")
    ax.set_xlabel(eps_label)
    ax.set_ylabel("Stress [MPa]")
    ax.set_title("Stress-strain response")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Top right: hydrostatic pressure ---
    ax = axes[0, 1]
    ax.plot(j2["eps_load"], j2["pressure"], "o-", ms=3, label="J2")
    ax.plot(dp["eps_load"], dp["pressure"], "s-", ms=3, label="Drucker-Prager")
    ax.set_xlabel(eps_label)
    ax.set_ylabel(r"$p = I_1/3$ [MPa]")
    ax.set_title("Mean stress")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Bottom left: tangent error ---
    ax = axes[1, 0]
    ax.semilogy(j2["step"], j2["tangent_rel_error"], "o-", ms=3, label="J2")
    ax.semilogy(dp["step"], dp["tangent_rel_error"], "s-", ms=3, label="Drucker-Prager")
    ax.axhline(1e-6, color="gray", ls="--", lw=0.8, label=r"$10^{-6}$")
    ax.set_xlabel("Load step")
    ax.set_ylabel("Relative tangent error")
    ax.set_title("Consistent tangent accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    # shade elastic region
    j2_plastic = j2["alpha"] > 1e-15
    first_plastic = np.argmax(j2_plastic) if j2_plastic.any() else len(j2["step"])
    ax.axvspan(-0.5, first_plastic - 0.5, alpha=0.08, color="green")

    # --- Bottom right: equivalent plastic strain ---
    ax = axes[1, 1]
    ax.plot(j2["eps_load"], j2["alpha"], "o-", ms=3, label="J2")
    ax.plot(dp["eps_load"], dp["alpha"], "s-", ms=3, label="Drucker-Prager")
    ax.set_xlabel(eps_label)
    ax.set_ylabel(r"$\alpha$ (equiv. plastic strain)")
    ax.set_title("Plastic strain accumulation")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"J2 vs Drucker-Prager: {title}", fontsize=15, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out = base / f"plasticity_{tag}.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


def main():
    base = Path(__file__).resolve().parent.parent / "build"

    tags = ["eps_11", "eps_22", "eps_12"]
    found = False

    for tag in tags:
        j2_file = base / f"j2_{tag}.csv"
        dp_file = base / f"dp_{tag}.csv"

        if not j2_file.exists() or not dp_file.exists():
            print(f"Skipping {tag}: CSV files not found")
            continue

        found = True
        j2 = read_csv(j2_file)
        dp = read_csv(dp_file)
        plot_load_case(j2, dp, tag, base)

    if not found:
        print(f"No CSV files found. Run the plot_data executable first:")
        print(f"  cd {base} && ./tests/plot_data")
        sys.exit(1)


if __name__ == "__main__":
    main()
