#!/usr/bin/env python3
"""
Figure plotter for the two-rod difference-problem validation.

Reads `two_rod_results.txt` (the output of ``sim_two_rod_braid.py``)
and writes the manuscript figure:

    two_rod_combined.{pdf, svg, png}    (two side-by-side panels, page width)

    (a) 3D rendering of the simulated two-strand geometry on a curved
        backbone, including the clamped end nodes and the dashed
        backbone curve.
    (b) Contact-pressure enhancement P / P_straight vs the dimensionless
        ratio K/k between backbone curvature and helical wavenumber;
        simulation points (open red circles) overlaid on the analytical
        prediction 1 + 3 (K/k)^2 + (K/k)^4/2.

    The outlier point at R_curve = 10 mm (K/k ~ 0.38) lies outside the
    constant-separation difference-problem ansatz and is plotted with
    a cross to flag it.

The script depends only on numpy and matplotlib (no dismech).  The
small inline helpers reproduce just enough of the simulation geometry
(`_two_strand_helix_curved`, `_attach_loop_arcs_xyz`,
`_tube_surface`) to render panel (a) without re-running the
simulation; the numbers in panel (b) come straight from the results
file.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

# Font: Computer Modern (CMR10 for text, CMMI10 for math) as recommended in
# the group's style guide.  Matplotlib ships the actual CM fonts -- cmr10.ttf,
# cmmi10.ttf, cmsy10.ttf, cmex10.ttf -- in its bundled font directory, so we
# can use them directly without requiring a LaTeX installation.
#
# - General text (labels, ticks, legends): cmr10 via font.family + font.serif.
# - Math (everything inside $...$):        CM via mathtext.fontset = "cm",
#   which dispatches to cmmi10 (italic letters), cmr10 (upright digits),
#   cmsy10 (symbols), and cmex10 (large math).
#
# If you have a LaTeX install on PATH and prefer LaTeX-rendered text, set
# USE_TEX=True; matplotlib will then call pdflatex for every label, which
# embeds the same glyphs but via a slower path.

USE_TEX = False

if USE_TEX:
    mpl.rcParams["text.usetex"] = True
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.serif"] = ["Computer Modern Roman"]
    mpl.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}\usepackage{amssymb}"
else:
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.serif"] = ["cmr10", "CMU Serif",
                                  "Computer Modern Roman", "DejaVu Serif"]
    mpl.rcParams["mathtext.fontset"] = "cm"
    # When using cmr10 (which lacks a hyphen-minus glyph at the same code
    # point), matplotlib falls back to a Unicode minus that some versions
    # warn about.  Suppress by using ASCII minus.
    mpl.rcParams["axes.unicode_minus"] = False

mpl.rcParams["font.size"] = 10
# Use mathtext for tick formatting so CMR10 digits render through the same
# pipeline as the math labels (resolves the cmr10 hyphen-minus warning).
mpl.rcParams["axes.formatter.use_mathtext"] = True
mpl.rcParams["axes.linewidth"] = 0.8
mpl.rcParams["xtick.major.width"] = 0.8
mpl.rcParams["ytick.major.width"] = 0.8
mpl.rcParams["xtick.direction"] = "in"
mpl.rcParams["ytick.direction"] = "in"

# Group color palette from SamplePlot.m
COLPOS = np.array([
    [247, 148,  30],     # orange
    [  0, 166,  81],     # green
    [237,  28,  36],     # red
    [  0, 174, 239],     # blue
    [  0,   0,   0],     # black
], dtype=float) / 255.0

PWIDTH, PHEIGHT = 3.5, 3.0   # single-column figure size (inches)
PWIDTH_2COL = 7.0            # full-page width (two-panel figure)

# Axis labels (kept short here; will be re-typeset in Inkscape).
XLABEL = r"backbone curvature ratio  $K\,/\,k$"
YLABEL_KERNEL = r"bending kernel  $\langle |\delta''|^{2} \rangle$  (mm$^{-2}$)"
YLABEL_PRATIO = r"contact pressure ratio  $P\,/\,P_{\rm straight}$"


# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------

def load_results(path="two_rod_results.txt"):
    """
    Parse the two-rod sweep results into structured arrays.

    Returns
    -------
    dict with keys:
        R_curve    -- (N,) float, NaN for the straight baseline (R_curve = inf)
        Kk         -- (N,) float, K/k_i
        dd_sim     -- (N,) float, <|delta''|^2> from simulation
        dd_paper   -- (N,) float, h^2 (k^4 + 3 k^2 K^2 + K^4 / 2)
        enh_sim    -- (N,) float, dd_sim / dd_sim[straight]
        enh_paper  -- (N,) float, dd_paper / dd_paper[straight]
        valid      -- (N,) bool, True except at the geometric-breakdown point
    """
    R_curve, Kk, sim, paper, enh_sim, enh_paper = [], [], [], [], [], []
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            # The data rows start with the R_curve column.  Skip headers /
            # dashes / blank lines.
            if not s or s.startswith("Two-rod") or s.startswith("h=") \
               or s.startswith("baseline") or s.startswith("R_curve") \
               or s.startswith("-"):
                continue
            # Split on whitespace and pipes
            parts = [p for p in s.replace("|", " ").split() if p]
            # Expected: R_curve, K/k, dd_sim, dd_paper, ratio, enh_sim, enh_paper
            if len(parts) != 7:
                continue
            try:
                rc = float("inf") if parts[0] == "inf" else float(parts[0])
                R_curve.append(rc)
                Kk.append(float(parts[1]))
                sim.append(float(parts[2]))
                paper.append(float(parts[3]))
                enh_sim.append(float(parts[5]))
                enh_paper.append(float(parts[6]))
            except ValueError:
                continue

    R_curve = np.array(R_curve)
    Kk = np.array(Kk)
    sim = np.array(sim)
    paper = np.array(paper)
    enh_sim = np.array(enh_sim)
    enh_paper = np.array(enh_paper)

    # Mark the geometric-breakdown point (R_curve = 10 mm in this sweep)
    valid = R_curve > 10.0 + 1e-6  # everything except R_curve = 10
    # Convention: the straight baseline has R_curve = inf, which is also valid
    valid |= np.isinf(R_curve)

    return dict(R_curve=R_curve, Kk=Kk, dd_sim=sim, dd_paper=paper,
                enh_sim=enh_sim, enh_paper=enh_paper, valid=valid)


# ---------------------------------------------------------------------------
# Helpers for analytical curves
# ---------------------------------------------------------------------------

def paper_enhancement(Kk):
    """Analytical enhancement ratio: 1 + 3 (K/k)^2 + (K/k)^4 / 2."""
    return 1.0 + 3.0 * Kk**2 + 0.5 * Kk**4


def paper_dd_squared(Kk, dd_straight):
    """Analytical <|delta''|^2> as a function of K/k, with the straight value."""
    return dd_straight * paper_enhancement(Kk)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _draw_panel(ax, data, mode, theory_label):
    """
    Draw a single panel: theory curve + simulation symbols (+ outlier).

    Parameters
    ----------
    ax : matplotlib Axes
    data : dict from load_results
    mode : "kernel"      -> y = <|delta''|^2>
           "enhancement" -> y = <|delta''|^2> / <|delta''|^2>_str = P / P_straight
    theory_label : legend string for the analytical curve
    """
    valid = data["valid"]
    broken = ~valid

    Kk_dense = np.linspace(0, max(data["Kk"]) * 1.05, 300)
    if mode == "kernel":
        dd_str = data["dd_paper"][data["Kk"] == 0][0]
        y_theory = paper_dd_squared(Kk_dense, dd_str)
        y_sim = data["dd_sim"]
        ylabel = YLABEL_KERNEL
    elif mode == "enhancement":
        y_theory = paper_enhancement(Kk_dense)
        y_sim = data["enh_sim"]
        ylabel = YLABEL_PRATIO
    else:
        raise ValueError(mode)

    ax.plot(Kk_dense, y_theory,
            linestyle="-", color=COLPOS[4], linewidth=1.0,
            label=theory_label)
    ax.plot(data["Kk"][valid], y_sim[valid],
            linestyle="None", marker="o",
            markerfacecolor="none", markeredgecolor=COLPOS[2],
            markeredgewidth=1.0, markersize=6,
            label="simulation")
    if np.any(broken):
        ax.plot(data["Kk"][broken], y_sim[broken],
                linestyle="None", marker="x", color=COLPOS[2],
                markersize=6, markeredgewidth=1.2,
                label="sim (outside validity)")

    ax.set_xlabel(XLABEL)
    ax.set_ylabel(ylabel)
    ax.set_xlim(left=-0.01)

    # Place the legend just below the top edge, leaving room for the
    # (a)/(b) panel labels that sit at y = 0.98 in the combined figure.
    leg = ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.92),
                    frameon=False, fontsize=8)
    for sp in ax.spines.values():
        sp.set_visible(True)


def _two_strand_helix_curved(h, R_loop, n_cross=1, pts_per_turn=80,
                              R_curve=None):
    """
    Return the two centerlines (N, 3) each for a curved-backbone helical
    pair. Same conventions as make_two_strand_helix in sim_two_rod_braid.py
    (kept inline here so this plotting script has no dismech dependency).
    """
    k = 12**0.25 / np.sqrt(h * R_loop)
    winding = (2 * n_cross + 1) * np.pi
    n_pts = max(int(winding / (2 * np.pi) * pts_per_turn), 80) + 1
    half_s = winding / (2 * k)
    s = np.linspace(-half_s, half_s, n_pts)
    phi = k * s

    straight = (R_curve is None) or (R_curve > 1e12)
    if straight:
        nodes_1 = np.column_stack([h * np.cos(phi),
                                    h * np.sin(phi), s])
        nodes_2 = np.column_stack([-h * np.cos(phi),
                                   -h * np.sin(phi), s])
        return nodes_1, nodes_2, s, np.zeros_like(s)
    # Curved backbone: circle of radius R_curve in xz-plane
    theta = s / R_curve
    cx = -R_curve + R_curve * np.cos(theta)
    cz = R_curve * np.sin(theta)
    neg_Nx = np.cos(theta)
    neg_Nz = np.sin(theta)
    nodes_1 = np.column_stack([
        cx + h * np.cos(phi) * neg_Nx,
        h * np.sin(phi),
        cz + h * np.cos(phi) * neg_Nz,
    ])
    nodes_2 = np.column_stack([
        cx - h * np.cos(phi) * neg_Nx,
        -h * np.sin(phi),
        cz - h * np.cos(phi) * neg_Nz,
    ])
    return nodes_1, nodes_2, cx, cz


def _tube_surface(centerline, radius, n_theta=24,
                  ref_dir=np.array([0.0, 0.0, 1.0])):
    """
    Build a tube surface (X, Y, Z arrays for plot_surface) around a
    centerline polyline, using a rotation-minimising frame (parallel
    transport) for the normal/binormal so the tube doesn't twist
    spuriously along the path.
    """
    N = len(centerline)
    # Tangents via central differences
    tangents = np.gradient(centerline, axis=0)
    tlen = np.linalg.norm(tangents, axis=1, keepdims=True)
    tlen = np.maximum(tlen, 1e-15)
    tangents = tangents / tlen

    # Initial normal: project ref_dir into plane perpendicular to t0
    t0 = tangents[0]
    rd = ref_dir.copy()
    if abs(np.dot(t0, rd)) > 0.95:
        rd = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(t0, rd)) > 0.95:
            rd = np.array([0.0, 1.0, 0.0])
    N0 = rd - np.dot(rd, t0) * t0
    N0 = N0 / max(np.linalg.norm(N0), 1e-15)

    Ns = np.zeros_like(centerline)
    Ns[0] = N0
    for i in range(1, N):
        t_prev = tangents[i - 1]
        t_curr = tangents[i]
        axis = np.cross(t_prev, t_curr)
        an = np.linalg.norm(axis)
        N_prev = Ns[i - 1]
        if an < 1e-12:
            N_curr = N_prev
        else:
            axis = axis / an
            cos_th = float(np.clip(np.dot(t_prev, t_curr), -1.0, 1.0))
            sin_th = an
            # Rodrigues rotation
            N_curr = (N_prev * cos_th
                      + np.cross(axis, N_prev) * sin_th
                      + axis * np.dot(axis, N_prev) * (1.0 - cos_th))
        # Re-orthogonalise against current tangent
        N_curr = N_curr - np.dot(N_curr, t_curr) * t_curr
        N_curr = N_curr / max(np.linalg.norm(N_curr), 1e-15)
        Ns[i] = N_curr
    Bs = np.cross(tangents, Ns)

    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=True)
    cos_t = np.cos(theta)[None, :]
    sin_t = np.sin(theta)[None, :]
    X = (centerline[:, 0:1] + radius * (Ns[:, 0:1] * cos_t + Bs[:, 0:1] * sin_t))
    Y = (centerline[:, 1:2] + radius * (Ns[:, 1:2] * cos_t + Bs[:, 1:2] * sin_t))
    Z = (centerline[:, 2:3] + radius * (Ns[:, 2:3] * cos_t + Bs[:, 2:3] * sin_t))
    return X, Y, Z


def _tube_polys(X, Y, Z, base_color,
                light=np.array([0.3, 0.2, 1.0]), ambient=0.45):
    """
    Convert a tube surface grid (X, Y, Z, each shape (N_pts, n_theta))
    into a list of quad face vertex arrays plus per-face RGBA colors,
    with a simple Lambertian shading applied to ``base_color``.

    Used in place of ``plot_surface`` so that two (or more) strands can
    be combined into a single :class:`Poly3DCollection`.  Putting all
    faces in one collection lets matplotlib depth-sort them per-face,
    so a blue strand segment that is in front of the camera correctly
    occludes the orange strand behind it (and vice versa) wherever the
    two helices cross.  Two separate ``plot_surface`` calls cannot do
    this because each surface is its own collection and gets painted
    whole, in call order.
    """
    light = np.asarray(light, dtype=float)
    light = light / np.linalg.norm(light)
    base = np.asarray(base_color, dtype=float)[:3]

    # Grid of vertices and the four corner positions of each quad face.
    P = np.stack([X, Y, Z], axis=-1)
    p00 = P[:-1, :-1]
    p10 = P[1: , :-1]
    p11 = P[1: , 1: ]
    p01 = P[:-1, 1: ]
    quads = np.stack([p00, p10, p11, p01], axis=2).reshape(-1, 4, 3)

    # Face normals via the two diagonals (more stable on twisted quads
    # than a single edge cross-product).
    nvec = np.cross(p11 - p00, p01 - p10).reshape(-1, 3)
    nlen = np.linalg.norm(nvec, axis=1, keepdims=True)
    nlen = np.where(nlen > 1e-12, nlen, 1.0)
    nunit = nvec / nlen

    # Lambertian intensity with an ambient floor so the dark side of
    # each tube isn't pitch black.  abs(.) means we don't care which
    # way the face normal points -- the tube is opaque on both sides.
    intensity = ambient + (1.0 - ambient) * np.abs(nunit @ light)
    rgb = np.clip(base[None, :] * intensity[:, None], 0.0, 1.0)
    rgba = np.concatenate([rgb, np.ones((rgb.shape[0], 1))], axis=1)

    return list(quads), rgba


def _attach_loop_arcs_xyz(nodes, R_loop, n_arc=8):
    """
    Append a 3D circular arc of radius R_loop, tangent to the strand at
    each end.  Mirrors the construction in sim_two_rod_braid.attach_loop_arcs
    but resolved at higher density (n_arc nodes per end) for smooth tube
    rendering.  Both ends bend toward the global +x direction so the
    arcs lie in a single planar loop closing the braid.
    """
    bend_axis = np.array([1.0, 0.0, 0.0])

    def _project_perp(v, t):
        out = v - np.dot(v, t) * t
        n = np.linalg.norm(out)
        if n < 1e-12:
            alt = np.array([0.0, 1.0, 0.0]) - np.dot([0, 1, 0], t) * t
            out = alt
            n = np.linalg.norm(out)
        return out / n

    edge_len = float(np.linalg.norm(nodes[-1] - nodes[-2]))
    phi_step = edge_len / R_loop

    # Top end: outgoing tangent from the helix
    t_top = nodes[-1] - nodes[-2]
    t_top /= np.linalg.norm(t_top)
    n_top = _project_perp(bend_axis, t_top)
    c_top = nodes[-1] + R_loop * n_top
    top_arc = []
    for i in range(1, n_arc + 1):
        th = i * phi_step
        p = c_top - R_loop * np.cos(th) * n_top + R_loop * np.sin(th) * t_top
        top_arc.append(p)
    top_arc = np.array(top_arc)

    # Bottom end
    t_bot = nodes[0] - nodes[1]
    t_bot /= np.linalg.norm(t_bot)
    n_bot = _project_perp(bend_axis, t_bot)
    c_bot = nodes[0] + R_loop * n_bot
    bot_arc = []
    for i in range(1, n_arc + 1):
        th = i * phi_step
        p = c_bot - R_loop * np.cos(th) * n_bot + R_loop * np.sin(th) * t_bot
        bot_arc.append(p)
    bot_arc = np.array(bot_arc)[::-1]   # reverse so order goes outward -> helix

    full = np.vstack([bot_arc, nodes, top_arc])
    # Return indices of (a) the clamped nodes -- 3 at each end of the
    # *whole* strand (i.e. of `full`) -- so the caller can highlight them.
    n_total = len(full)
    clamped_idx = np.array([0, 1, 2,
                            n_total - 3, n_total - 2, n_total - 1])
    return full, clamped_idx


def _draw_geometry_panel(ax, h=1.0, R_loop=50.0, n_cross=1, R_curve=60.0):
    """
    3D visualisation of the two-rod simulation geometry, including
    boundary conditions:
      - two intertwined helical strands rendered as solid tubes;
      - the loop-arc extension at each end of each strand that imposes
        the boundary curvature 1/R_loop;
      - the clamped nodes (three per end per strand) highlighted with
        markers;
      - the dashed curved backbone of the inner braid.
    """
    # Helix region only.  Finer centerline sampling (pts_per_turn) makes
    # each tube quad shorter along the strand axis, which helps the
    # per-face depth sort inside the combined Poly3DCollection produce
    # cleaner interleaving where the two helices cross.
    nodes_1, nodes_2, _cx, _cz = _two_strand_helix_curved(
        h, R_loop, n_cross=n_cross, pts_per_turn=120, R_curve=R_curve
    )
    # Attach loop arcs at each end (matches sim_two_rod_braid.py)
    full_1, clamp_1 = _attach_loop_arcs_xyz(nodes_1, R_loop, n_arc=8)
    full_2, clamp_2 = _attach_loop_arcs_xyz(nodes_2, R_loop, n_arc=8)

    # Rotate (x, y, z) -> (z, x, y) so the helix axis (originally z)
    # spans the panel's x direction.
    def _rot(arr):
        return np.column_stack([arr[:, 2], arr[:, 0], arr[:, 1]])
    full_1 = _rot(full_1)
    full_2 = _rot(full_2)

    # Tube rendering -- both strands in one Poly3DCollection so that
    # matplotlib's per-face depth sort interleaves blue and orange
    # quads correctly where the two helices cross.  Two separate
    # plot_surface calls would each draw as one opaque blob in call
    # order, making whichever strand is drawn last (orange) always
    # appear in front, even on segments where the blue strand is
    # actually closer to the camera.
    # Finer azimuthal resolution -> smaller faces -> the centroid-based
    # depth sort matplotlib uses inside Poly3DCollection produces fewer
    # wrong-order artifacts where the two helices interpenetrate.
    X1, Y1, Z1 = _tube_surface(full_1, radius=h, n_theta=32)
    X2, Y2, Z2 = _tube_surface(full_2, radius=h, n_theta=32)
    faces_1, colors_1 = _tube_polys(X1, Y1, Z1, COLPOS[3])
    faces_2, colors_2 = _tube_polys(X2, Y2, Z2, COLPOS[0])
    strand_polys = Poly3DCollection(
        faces_1 + faces_2,
        facecolors=np.vstack([colors_1, colors_2]),
        edgecolor="none", linewidth=0,
        # antialiased=False removes the partial-transparency at quad
        # edges that, combined with per-face depth-sort artifacts, was
        # creating the stripey look on the tube surfaces.
        antialiased=False, zorder=0,
    )
    ax.add_collection3d(strand_polys)

    # Dashed backbone (rotated coords)
    if R_curve is not None and R_curve < 1e10:
        s_extent = max(abs(full_1[:, 0]).max(), 1.0)
        s_back = np.linspace(-s_extent, s_extent, 400)
        theta = s_back / R_curve
        cx_o = -R_curve + R_curve * np.cos(theta)
        cz_o = R_curve * np.sin(theta)
        ax.plot(cz_o, cx_o, np.zeros_like(s_back),
                "--", color=COLPOS[4], lw=1.2, alpha=0.75, zorder=10)

    # Clamped-node markers: black dots that sit on top of the strand
    # tubes (computed_zorder=False on the 3D axes lets the explicit
    # zorder=20 below win against the surfaces' default zorder=0).
    for arr, idx in [(full_1, clamp_1), (full_2, clamp_2)]:
        ax.scatter(arr[idx, 0], arr[idx, 1], arr[idx, 2],
                   color="k", s=30, depthshade=False, zorder=20,
                   edgecolors="none")

    # Equal data aspect, but make the rendered 3D box match the actual
    # data extents instead of forcing a cube.  The geometry is long and
    # thin (x ~ helix axis ~ 40 units, y,z ~ a few h), so a (1,1,1) box
    # leaves most of the panel empty above and below the strands.  Using
    # box_aspect=(dx,dy,dz) makes the screen-space box hug the geometry
    # while still keeping each axis at the same units-per-pixel, so
    # circles stay circular.
    all_pts = np.vstack([full_1, full_2])
    pad = h * 1.3
    xr = (all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad)
    yr = (all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad)
    zr = (all_pts[:, 2].min() - pad, all_pts[:, 2].max() + pad)
    dx = xr[1] - xr[0]; dy = yr[1] - yr[0]; dz = zr[1] - zr[0]
    ax.set_xlim(*xr)
    ax.set_ylim(*yr)
    ax.set_zlim(*zr)
    ax.set_box_aspect((dx, dy, dz))

    ax.set_axis_off()
    ax.view_init(elev=16, azim=-68)
    try:
        ax.dist = 4.5
    except Exception:
        pass
    # Grow the 3D axes within its subplot cell to use most of the
    # available area
    ax.set_position(ax.get_position().expanded(1.30, 1.30))

    # In-axes legend close to the geometry; placed lower so it aligns
    # vertically with the (b) panel label on the right plot (the 3D axes
    # are expanded above their nominal cell, which pushes y=1 above the
    # visible top edge).
    handles = [
        plt.Line2D([], [], color="k", marker="o", linestyle="None",
                   markersize=4, label="clamped nodes"),
        plt.Line2D([], [], color=COLPOS[4], linestyle="--", lw=1.2,
                   label="curved backbone"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=8,
              frameon=False, bbox_to_anchor=(0.0, 0.82))


def plot_combined(data, out_stem="two_rod_combined"):
    """
    Two-panel figure (page width):
      Left panel:  3D visualisation of the two-rod intertwined geometry
                   on a curved backbone.
      Right panel: pressure ratio P/P_straight from the simulation
                   compared with the analytical prediction.
    """
    import matplotlib.gridspec as gs
    fig = plt.figure(figsize=(PWIDTH_2COL, PHEIGHT))
    grid = gs.GridSpec(1, 2, width_ratios=[1.0, 1.0],
                       wspace=0.10, left=0.01, right=0.98,
                       top=0.97, bottom=0.16)
    # computed_zorder=False forces matplotlib to use the explicit zorder
    # we set on each 3D artist instead of its own depth-sort heuristic,
    # so the clamped-node scatter (zorder=20) draws on top of the tube
    # surfaces (default zorder=0) instead of being occluded by them.
    ax_geom = fig.add_subplot(grid[0, 0], projection="3d",
                              computed_zorder=False)
    _draw_geometry_panel(ax_geom, h=1.0, R_loop=50.0, n_cross=1,
                         R_curve=40.0)

    ax_enh = fig.add_subplot(grid[0, 1])
    _draw_panel(ax_enh, data, mode="enhancement",
                theory_label=r"theory  $1 + 3(K/k)^{2} + (K/k)^{4}/2$")

    # Panel labels (a), (b).
    # The 3D axes get expanded above their nominal cell (see set_position
    # call in _draw_geometry_panel), so transAxes y=1 sits above the
    # visible top edge. Lower the (a) label so it aligns vertically with
    # (b) on the right panel, which uses standard transAxes coords.
    ax_geom.text2D(0.02, 0.87, "(a)", transform=ax_geom.transAxes,
                   fontsize=10, fontweight="bold", va="top", ha="left")
    ax_enh.text(0.02, 0.98, "(b)", transform=ax_enh.transAxes,
                fontsize=10, fontweight="bold", va="top", ha="left")

    fig.savefig(f"{out_stem}.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(f"{out_stem}.svg", format="svg", bbox_inches="tight")
    fig.savefig(f"{out_stem}.png", format="png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  saved  {out_stem}.pdf, .svg, .png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    txt_path = sys.argv[1] if len(sys.argv) > 1 else "two_rod_results.txt"
    if not os.path.exists(txt_path):
        # Try relative to the script's directory
        here = os.path.dirname(os.path.abspath(__file__))
        alt = os.path.join(here, txt_path)
        if os.path.exists(alt):
            txt_path = alt
        else:
            raise FileNotFoundError(f"Cannot find {txt_path}")

    print(f"reading  {txt_path}")
    data = load_results(txt_path)
    print(f"  {len(data['Kk'])} data points, "
          f"K/k range = [{data['Kk'].min():.3f}, {data['Kk'].max():.3f}]")

    plot_combined(data)
