#!/usr/bin/env python3
"""
Supplementary materials -- generation of manuscript Figures 3 and 4
====================================================================

This standalone script regenerates Figures 3 and 4 of the manuscript
from the raw experimental data files distributed alongside it.

Required input files (must sit in the same directory as this script):
    Summary_30lb(ni=1,no=1).xlsx
    Summary_30lb(ni=2,no=1).xlsx
    Summary_30lb(ni=3,no=1).xlsx
    Summary_60lb(ni=1,no=1).xlsx
    Summary_60lb(ni=2,no=1).xlsx
    Summary_60lb(ni=3,no=1).xlsx

Each Summary file contains, in a sheet called 'Summary', the per-step
means and standard deviations of every measured quantity, averaged
across three independent tightening trials (T1, T2, T3, present as
additional sheets in the same workbook).  This script does NOT
recompute those means from the raw trials -- it reads the tabulated
ave / std pairs directly.

Output files written into the current working directory:
    FIG3_baseline_failure.{pdf,svg,png}
    FIG4_corrected_vs_uncorrected.{pdf,svg,png}

----------------------------------------------------------------------
Physical model
----------------------------------------------------------------------
The pulling force F required to keep a surgeon's knot stationary
against frictional sliding is the sum of two Coulomb friction
contributions from the outer and inner braids:

    F  =  mu * ( P_o + P_i )

with

    P_o  =  2 pi n_o B h k_o^3      k_o  =  ( sqrt(12) (h R_lo) )^(-1/2)
                                          =  12^(-1/4) / sqrt(h R_lo)
    P_i  =  2 pi n_i B h k_i^3 * [ 1 + 3 (K/k_i)^2 + (1/2) (K/k_i)^4 ]
                                    k_i  =  ( sqrt(12) (h R_bi) )^(-1/2)
                                          =  12^(-1/4) / sqrt(h R_bi)
                                    K    =  1 / R_bi

The wavenumber formula is Eq. (3) of Jawed, Dieleman, Audoly & Reis,
Phys. Rev. Lett. 115, 118302 (2015): k = (sqrt(12) h R)^(-1/2).

where:
    mu   - Coulomb friction coefficient (single free parameter)
    B    - cord bending stiffness E*I
    h    - cord radius
    n_o  - number of outer-braid turns (always 1 in this study)
    n_i  - number of inner-braid turns (1, 2, or 3)
    R_lo - outer-loop radius (from geometric closure)
    R_bi - inner-braid loop radius (measured directly from each
           tightening step; xlsx column R_bi)
    L_bo - measured outer-braid length (sets P_o through L_bo -> R_lo
           via the closure of Section 3.1)

The helical wavenumbers k_o and k_i are fixed by the Jawed-2015
energy-selection law, which minimises the elastic energy of a strand
contacting a curved backbone of radius R.  Applying this law on the
inner braid -- rather than the topological identity k_i = 2 pi n_i /
L_bi often used in n-foil derivations -- makes P_i scale as n_i^1
instead of n_i^4 and is what brings the six experimental datasets onto
a single physical friction coefficient (see Section 3.3).  The curved-
braid enhancement bracket on P_i comes from the contact-pressure
analysis of Section 3.2.1.

The "uncorrected" baseline drops the inner-braid term and retains only
P_o (the n-foil prediction).  FIG3 illustrates the failure of this
baseline in the configuration where the inner contribution matters
most (0.74 mm cord, n_i = 2).  FIG4 shows the corrected model
alongside the uncorrected baseline for all six configurations.

----------------------------------------------------------------------
Friction-coefficient fit
----------------------------------------------------------------------
The unified mu shared by all six datasets is obtained by a log-space
(geometric-mean) least-squares fit:

    minimise   sum_i  ( log F_i - log mu - log g_i )^2
    where      g_i  =  P_o,i + P_i,i

The closed-form solution is

    mu  =  exp( mean_i  log( F_i / g_i ) )

i.e. the geometric mean of the per-point ratios F_i / g_i.  This
choice gives every data point equal weight on a multiplicative scale,
so neither the largest-F nor the smallest-F datasets dominate.

Requirements: Python >= 3.8, numpy, matplotlib, openpyxl.
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import openpyxl


# =============================================================================
# 1. Material and geometric parameters of the two suture sizes
# =============================================================================

# 30 lb USP-rated braid: effective bending stiffness B30 and half-
# diameter h30 (radius of an equivalent solid rod that matches the
# measured macroscopic bending stiffness).
B30 = 2.711e-6                          # N m^2
h30 = 0.524e-3 / 2                      # m   ( = 0.262 mm  ->  d = 0.53 mm )

# 60 lb USP-rated braid: composed of multiple sub-filaments of radius
# h60 = 0.127 mm.  Here B60 is the single-filament bending stiffness
# (E*I with E = 67.5 GPa for the polymer).  The macroscopic braid
# diameter is approximately 0.74 mm.
h60 = 0.127e-3                          # m
B60 = 67.5e9 * np.pi * h60**4 / 4       # N m^2


# =============================================================================
# 2. Plot style -- Computer Modern fonts at 10 pt, in-pointing ticks
# =============================================================================

mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["cmr10", "CMU Serif",
                              "Computer Modern Roman", "DejaVu Serif"]
mpl.rcParams["mathtext.fontset"] = "cm"
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["axes.formatter.use_mathtext"] = True
mpl.rcParams["font.size"] = 10
mpl.rcParams["axes.linewidth"] = 0.8
mpl.rcParams["xtick.major.width"] = 0.8
mpl.rcParams["ytick.major.width"] = 0.8
mpl.rcParams["xtick.direction"] = "in"
mpl.rcParams["ytick.direction"] = "in"

# Lab color palette
COL_BLUE  = np.array([  0, 174, 239]) / 255.0
COL_RED   = np.array([237,  28,  36]) / 255.0
COL_BLACK = np.array([  0,   0,   0]) / 255.0


# =============================================================================
# 3. Dataset metadata: short key -> diameter, ni, no, B, h, label, xlsx
# =============================================================================

DATA_META = {
    "60_ni1": dict(label=r"diameter = 0.74 mm,  $n_i = 1$",
                   B=B60, h=h60, ni=1, no=1,
                   xlsx="Summary_60lb(ni=1,no=1).xlsx"),
    "60_ni2": dict(label=r"diameter = 0.74 mm,  $n_i = 2$",
                   B=B60, h=h60, ni=2, no=1,
                   xlsx="Summary_60lb(ni=2,no=1).xlsx"),
    "60_ni3": dict(label=r"diameter = 0.74 mm,  $n_i = 3$",
                   B=B60, h=h60, ni=3, no=1,
                   xlsx="Summary_60lb(ni=3,no=1).xlsx"),
    "30_ni1": dict(label=r"diameter = 0.53 mm,  $n_i = 1$",
                   B=B30, h=h30, ni=1, no=1,
                   xlsx="Summary_30lb(ni=1,no=1).xlsx"),
    "30_ni2": dict(label=r"diameter = 0.53 mm,  $n_i = 2$",
                   B=B30, h=h30, ni=2, no=1,
                   xlsx="Summary_30lb(ni=2,no=1).xlsx"),
    "30_ni3": dict(label=r"diameter = 0.53 mm,  $n_i = 3$",
                   B=B30, h=h30, ni=3, no=1,
                   xlsx="Summary_30lb(ni=3,no=1).xlsx"),
}


# =============================================================================
# 4. Loader: read a Summary_*.xlsx into (mean, std) arrays
# =============================================================================
#
# The Summary sheet layout (after two header rows) is one row per
# tightening step, with columns:
#
#     1     2      3     4         5         6         7
#     e     F_ave  F_std L_bi_ave  L_bi_std  L_li_ave  L_li_std
#
#     8         9         10        11
#     L_bo_ave  L_bo_std  R_bi_ave  R_bi_std
#
# We unpack into two arrays of shape (N_steps, 6) packed as
#   [stroke_e, F, L_bi, L_li, L_bo, R_bi]   (xlsx ordering)
# with the means in `ave` and the per-trial standard deviations in
# `std`.  The std for column 0 (stroke) is set to zero because the
# stroke is a controlled testing-machine input, not a measurement.
# =============================================================================

def load_summary(xlsx_name):
    """Read Summary_*.xlsx -> (ave[N,6], std[N,6]) arrays."""
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, xlsx_name)
    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb["Summary"]
    ave_rows, std_rows = [], []
    for r in ws.iter_rows(min_row=3, values_only=True):
        if r[0] is None or not isinstance(r[0], (int, float)):
            continue
        e, Fa, Fs, Lbia, Lbis, Llia, Llis, Lboa, Lbos, Rbia, Rbis = r
        ave_rows.append([e,   Fa, Lbia, Llia, Lboa, Rbia])
        std_rows.append([0.0, Fs, Lbis, Llis, Lbos, Rbis])
    return (np.asarray(ave_rows, dtype=float),
            np.asarray(std_rows, dtype=float))


# Load all four datasets once at import.
DATA = {}
for _k, _meta in DATA_META.items():
    _ave, _std = load_summary(_meta["xlsx"])
    DATA[_k] = dict(_meta, ave=_ave, std=_std)


# =============================================================================
# 5. Model evaluation and unified-mu fit
# =============================================================================
#
# ---------------------------------------------------------------------------
# Inner-braid contact pressure.  k_i comes from the Jawed-2015 energy
# selection law applied to a strand contacting the inner-loop backbone of
# radius R_bi (read directly from the experiments, xlsx column R_bi).
# The curved-braid enhancement 1 + 3 (K/k_i)^2 + (1/2) (K/k_i)^4 from
# Section 3.2.1 multiplies the straight-backbone contact pressure.
# ---------------------------------------------------------------------------

def _inner_P(d):
    """Inner-braid contact pressure P_i (per tightening step)."""
    B, h, ni = d["B"], d["h"], d["ni"]
    Rbi = d["ave"][:, 5] * 1e-3
    k_i = 12.0**(-0.25) / np.sqrt(h * Rbi)
    K   = 1.0 / Rbi
    P_i_straight = 2.0 * np.pi * ni * B * h * k_i**3
    enhancement  = 1.0 + 3.0 * (K / k_i)**2 + 0.5 * (K / k_i)**4
    return P_i_straight * enhancement


def _factor_g(d):
    """Return g_i = P_o + P_i (per row)."""
    B, h, no = d["B"], d["h"], d["no"]
    Lbo = d["ave"][:, 2] * 1e-3
    P_o = 16.0 * np.pi**4 * no**4 * B * h / Lbo**3
    return P_o + _inner_P(d)


def evaluate_model(key, mu):
    """Evaluate the corrected model at every tightening step.

    Returns a dict with:
        F_exp   - measured force (N)
        F_th    - corrected model F = mu * (P_o + P_i)
        F_outer - uncorrected baseline F_o = mu * P_o
        e_full  - reconstructed end-to-end shortening (m)
    """
    d = DATA[key]
    B, h, no = d["B"], d["h"], d["no"]
    F_exp = d["ave"][:, 1]
    Lbo = d["ave"][:, 2] * 1e-3
    Llo = d["ave"][:, 3] * 1e-3
    Lbi = d["ave"][:, 4] * 1e-3
    Lli = d["ave"][:, 5] * 1e-3
    P_o = 16.0 * np.pi**4 * no**4 * B * h / Lbo**3
    P_i = _inner_P(d)
    F_outer = mu * P_o
    F_inner = mu * P_i
    F_th    = F_outer + F_inner
    e_full  = Lbo + Llo + 2.0 * Lbi + Lli
    return dict(F_exp=F_exp, F_th=F_th, F_outer=F_outer, e_full=e_full)


def fit_unified_mu_log(keys):
    """
    Geometric-mean (log-space) least-squares fit for the single mu
    shared across all datasets in ``keys``.  Closed-form, see header
    docstring.  Rows with F <= 0 are dropped (log undefined).
    """
    log_mu_sum = 0.0
    n_pts = 0
    for key in keys:
        d = DATA[key]
        F = d["ave"][:, 1]
        g = _factor_g(d)
        ok = ~np.isnan(F) & ~np.isnan(g) & (g > 0) & (F > 0)
        if not ok.any():
            continue
        log_mu_sum += float(np.sum(np.log(F[ok] / g[ok])))
        n_pts      += int(ok.sum())
    return float(np.exp(log_mu_sum / n_pts)) if n_pts > 0 else np.nan


# =============================================================================
# 6. Common plotting helpers
# =============================================================================

def _normalised_arrays(key, mu):
    """Pull and pre-process everything needed for one panel."""
    d = DATA[key]
    B, h = d["B"], d["h"]
    res = evaluate_model(key, mu)
    F_exp  = res["F_exp"]
    F_th   = res["F_th"]
    F_base = res["F_outer"]
    e_full = res["e_full"]
    F_std  = d["std"][:, 1]

    # Drop F = 0 rows (cannot be shown on log axis)
    ok = ~np.isnan(F_exp) & (F_exp > 0)
    e_mm = e_full[ok] * 1e3
    Fe   = F_exp[ok]
    Fes  = F_std[ok]
    Ft   = F_th[ok]
    Fb   = F_base[ok]
    order = np.argsort(e_mm)

    # Non-dimensionalise by h^2 / EI
    scale = h**2 / B
    return dict(
        e_mm  = e_mm[order],
        Fe_n  = (Fe  * scale)[order],
        Fes_n = (Fes * scale)[order],
        Ft_n  = (Ft  * scale)[order],
        Fb_n  = (Fb  * scale)[order],
    )


def _log_yerr(Fe_n, Fes_n):
    """Asymmetric yerr clipped to 99% of Fe so bars stay above zero."""
    return [np.minimum(Fes_n, 0.99 * Fe_n), Fes_n]


# =============================================================================
# 7. FIG 3 -- single-panel baseline failure (0.74 mm, n_i = 3)
# =============================================================================
#
# Shows the uncorrected n-foil baseline (blue) against the measured F
# (red, with 1-sigma vertical error bars) for the configuration in
# which the inner-braid contribution dominates most -- i.e. the case
# where the baseline fails most badly.  No horizontal error bars (the
# x uncertainty is dominated by trial-to-trial variability in the
# inner-loop length and is much wider than the y bars; including it
# clutters the plot without adding information).
# =============================================================================

def plot_FIG3(mu, out_stem="FIG3_baseline_failure", key="60_ni3"):
    a = _normalised_arrays(key, mu)
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    ax.plot(a["e_mm"], a["Fb_n"], "-", color=COL_BLUE, lw=1.4,
            label="uncorrected")
    ax.errorbar(a["e_mm"], a["Fe_n"], yerr=_log_yerr(a["Fe_n"], a["Fes_n"]),
                fmt="o", markerfacecolor="none",
                markeredgecolor=COL_RED, ecolor=COL_RED,
                elinewidth=0.8, capsize=2.0,
                markeredgewidth=1.0, markersize=5,
                label="experiments")
    ax.set_yscale("log")
    ytop = max(float(np.nanmax(a["Fe_n"] + a["Fes_n"])),
               float(np.nanmax(a["Fb_n"])))
    ybot = min(float(np.nanmin(a["Fe_n"])), float(np.nanmin(a["Fb_n"])))
    ax.set_ylim(ybot * 0.4, ytop * 4.0)
    epad = 0.04 * (a["e_mm"].max() - a["e_mm"].min())
    ax.set_xlim(a["e_mm"].min() - epad, a["e_mm"].max() + epad)
    ax.set_xlabel(r"end-to-end shortening, $e$ (mm)", fontsize=9)
    ax.set_ylabel(r"normalized pulling force,  $F\,h^{2}/EI$", fontsize=9)
    ax.legend(loc="upper right", fontsize=7, frameon=False)
    fig.tight_layout()
    for ext in ("pdf", "svg", "png"):
        fig.savefig(f"{out_stem}.{ext}",
                    dpi=300 if ext == "png" else None,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_stem}.{{pdf, svg, png}}")


# =============================================================================
# 8. FIG 4 -- six-panel corrected vs uncorrected, all configurations
# =============================================================================
#
# Layout: one column per cord diameter (left: 0.74 mm, right: 0.53 mm)
# and one row per inner-throw winding number n_i (top: n_i = 1,
# middle: n_i = 2, bottom: n_i = 3).
#     panel (a)  0.74 mm,  n_i = 1     panel (b)  0.53 mm,  n_i = 1
#     panel (c)  0.74 mm,  n_i = 2     panel (d)  0.53 mm,  n_i = 2
#     panel (e)  0.74 mm,  n_i = 3     panel (f)  0.53 mm,  n_i = 3
#
# Each panel shows the experimental data (red circles, 1-sigma
# vertical error bars), the uncorrected n-foil baseline (dashed blue)
# and the corrected outer+inner model (solid black), with both model
# curves evaluated at the SAME unified mu.  No suptitle.
# =============================================================================

def plot_FIG4(mu, out_stem="FIG4_corrected_vs_uncorrected"):
    keys = ["60_ni1", "30_ni1",     # row 1 (n_i = 1)
            "60_ni2", "30_ni2",     # row 2 (n_i = 2)
            "60_ni3", "30_ni3"]     # row 3 (n_i = 3)
    fig, axes = plt.subplots(3, 2, figsize=(7.0, 9.0))
    axes = axes.ravel()
    for ax, key in zip(axes, keys):
        d = DATA[key]
        a = _normalised_arrays(key, mu)
        ax.plot(a["e_mm"], a["Fb_n"], "--", color=COL_BLUE, lw=1.1,
                label="uncorrected")
        ax.plot(a["e_mm"], a["Ft_n"], "-", color=COL_BLACK, lw=1.3,
                label="corrected")
        ax.errorbar(a["e_mm"], a["Fe_n"],
                    yerr=_log_yerr(a["Fe_n"], a["Fes_n"]),
                    fmt="o", markerfacecolor="none",
                    markeredgecolor=COL_RED, ecolor=COL_RED,
                    elinewidth=0.8, capsize=2.0,
                    markeredgewidth=1.0, markersize=5,
                    label="experiments")
        ax.set_yscale("log")
        ytop = max(float(np.nanmax(a["Fe_n"] + a["Fes_n"])),
                   float(np.nanmax(a["Ft_n"])), float(np.nanmax(a["Fb_n"])))
        ybot = min(float(np.nanmin(a["Fe_n"])),
                   float(np.nanmin(a["Ft_n"])), float(np.nanmin(a["Fb_n"])))
        ax.set_ylim(ybot * 0.5, ytop * 3.0)
        # Panel label (a)/(b)/(c)/(d) in the upper-left of each panel.
        panel = "(" + chr(ord("a") + keys.index(key)) + ")"
        ax.text(0.04, 0.95, panel, transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="top", ha="left")
        ax.set_xlabel(r"end-to-end shortening, $e$ (mm)", fontsize=9)
        ax.set_ylabel(r"normalized pulling force,  $F\,h^{2}/EI$",
                      fontsize=9)
        ax.set_title(d["label"], fontsize=9, fontweight="normal")
        # Legend only on the first panel to keep the others uncluttered.
        if key == keys[0]:
            ax.legend(loc="upper right", fontsize=7, frameon=False)
    fig.tight_layout()
    for ext in ("pdf", "svg", "png"):
        fig.savefig(f"{out_stem}.{ext}",
                    dpi=300 if ext == "png" else None,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_stem}.{{pdf, svg, png}}")


# =============================================================================
# 9. Entry point
# =============================================================================

if __name__ == "__main__":
    keys = ["60_ni1", "60_ni2", "60_ni3",
            "30_ni1", "30_ni2", "30_ni3"]
    mu_uni = fit_unified_mu_log(keys)
    print(f"unified mu (log-space fit, 6 datasets) = {mu_uni:.4f}")
    plot_FIG3(mu_uni, out_stem="FIG3_baseline_failure", key="60_ni3")
    plot_FIG4(mu_uni, out_stem="FIG4_corrected_vs_uncorrected")
