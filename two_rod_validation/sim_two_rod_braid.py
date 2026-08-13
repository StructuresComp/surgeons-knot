#!/usr/bin/env python3
"""
Two-rod braid: direct validation of the difference-problem contact pressure.

Two elastic rods wind helically about a common (straight or curved) backbone,
180 degrees out of phase, so their centerlines are at constant distance 2h.
Each rod has cross-section radius h, so they are in continuous tangential
contact along the braid.  The stress-free reference of each rod is straight;
bending pre-stress drives them against each other.

This complements sim_jawed_fig2b.py, which uses a single rod on a rigid
cylinder.  The two-rod sim directly measures the relative position
delta(s) = (r_1 - r_2)/2 and its second derivative, allowing a clean
comparison with the paper's prediction (Sec. 3.2):

    <|delta''|^2> = h^2 * (k^4 + 3 k^2 K^2 + K^4 / 2)

where K = 1/R_curve is the backbone curvature and k is the helical wavenumber.

Units: mm, N, s throughout.
"""

import numpy as np
import dismech


# ---------------------------------------------------------------------------
# Geometry: two intertwined helices on a (possibly curved) backbone
# ---------------------------------------------------------------------------

def make_two_strand_helix(h, R_loop, n_cross=1, pts_per_turn=30, R_curve=None):
    """
    Build the central helical portion of two strands, 180 degrees out of phase.

    Each strand has cross-section radius h.  Its centerline winds at distance h
    from the backbone, so the two centerlines are at distance 2h (touching).

    Parameters
    ----------
    h : strand cross-section radius (also = half-separation from backbone)
    R_loop : loop radius setting the wavenumber k = 12^(-1/4) / sqrt(h * R_loop)
             = (sqrt(12) h R_loop)^(-1/2), Eq. (3) of Jawed et al. 2015
    n_cross : crossing number (winding = (2n+1)*pi per strand)
    pts_per_turn : node density
    R_curve : backbone curvature radius (None => straight backbone along z)

    Returns
    -------
    nodes_1, nodes_2 : (N, 3) arrays
    k : helical wavenumber
    s : (N,) backbone arc-length coordinate at each node (shared by both strands)
    """
    k = 12**-0.25 / np.sqrt(h * R_loop)
    winding = (2 * n_cross + 1) * np.pi
    n_pts = max(int(winding / (2 * np.pi) * pts_per_turn), 40) + 1

    half_s = winding / (2 * k)
    s = np.linspace(-half_s, half_s, n_pts)
    phi = k * s

    straight = (R_curve is None) or (R_curve > 1e12)

    if straight:
        # Backbone = z-axis.  Frame: -N = (1, 0, 0), B = (0, 1, 0).
        nodes_1 = np.column_stack([
            h * np.cos(phi),
            h * np.sin(phi),
            s,
        ])
        nodes_2 = np.column_stack([
            -h * np.cos(phi),
            -h * np.sin(phi),
            s,
        ])
    else:
        # Curved backbone: circle of radius R_curve in xz-plane, centered at
        # (-R_curve, 0, 0), tangent to z-axis at origin.
        theta = s / R_curve
        cx = -R_curve + R_curve * np.cos(theta)
        cz = R_curve * np.sin(theta)
        # Local frame at each point:  -N = (cos theta, 0, sin theta),  B = ey
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

    return nodes_1, nodes_2, k, s


def attach_loop_arcs(strand_nodes, R_loop):
    """
    Append a 3-node circular arc at each end of a strand, imposing the
    discrete loop curvature 1/R_loop at the strand exit.

    Both ends bend toward global +x (same convention as sim_jawed_fig2b.py:
    'attach_loop_arcs'), so the two end arcs lie on the same side of the
    helix, consistent with a single planar loop closing the braid.
    """
    bend_axis = np.array([1.0, 0.0, 0.0])
    edge_len = float(np.linalg.norm(strand_nodes[-1] - strand_nodes[-2]))
    phi_arc = edge_len / R_loop

    def _arc(p0, t):
        n_dir = bend_axis - np.dot(bend_axis, t) * t
        n_norm = np.linalg.norm(n_dir)
        if n_norm < 1e-12:
            # Tangent is along bend_axis; fall back to +y as bend direction
            n_dir = np.array([0.0, 1.0, 0.0]) - np.dot([0, 1, 0], t) * t
            n_norm = np.linalg.norm(n_dir)
        n_dir = n_dir / n_norm
        c = p0 + R_loop * n_dir
        extras = []
        for i in (1, 2):
            th = i * phi_arc
            p = c - R_loop * np.cos(th) * n_dir + R_loop * np.sin(th) * t
            extras.append(p)
        return np.array(extras)

    t_top = strand_nodes[-1] - strand_nodes[-2]
    t_top /= np.linalg.norm(t_top)
    top_extra = _arc(strand_nodes[-1], t_top)

    t_bot = strand_nodes[0] - strand_nodes[1]
    t_bot /= np.linalg.norm(t_bot)
    bot_extra = _arc(strand_nodes[0], t_bot)
    bot_extra = bot_extra[::-1]

    return np.vstack([bot_extra, strand_nodes, top_extra])


def build_two_rod_topology(nodes_1, nodes_2):
    """
    Combine two strand node arrays into Dismech-compatible (nodes, edges)
    with TWO disconnected rod components.

    Returns
    -------
    nodes : (N1 + N2, 3)
    edges : ((N1-1) + (N2-1), 2)  rod-rod edges only, no cross-rod edges
    N1, N2 : int  number of nodes in each strand
    fixed : (12,) clamped node indices: 3 at each end of each strand
    """
    N1 = len(nodes_1)
    N2 = len(nodes_2)
    nodes = np.vstack([nodes_1, nodes_2])

    edges_1 = np.array([[i, i + 1] for i in range(N1 - 1)], dtype=int)
    edges_2 = np.array([[N1 + i, N1 + i + 1] for i in range(N2 - 1)], dtype=int)
    edges = np.vstack([edges_1, edges_2])

    fixed = np.array([
        0, 1, 2, N1 - 3, N1 - 2, N1 - 1,                 # strand 1 ends
        N1, N1 + 1, N1 + 2, N1 + N2 - 3, N1 + N2 - 2, N1 + N2 - 1,  # strand 2 ends
    ])

    return nodes, edges, N1, N2, fixed


def make_straight_reference(nodes, N1):
    """
    Stress-free reference for both strands: each laid out along +z with
    identical edge lengths to the deformed configuration.  The two strands
    are placed in different (x, y) so their reference shapes don't coincide.
    """
    nodes_1 = nodes[:N1]
    nodes_2 = nodes[N1:]

    def _ref_along_z(strand, x_offset, y_offset):
        dl = np.linalg.norm(np.diff(strand, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(dl)])
        ref = np.zeros_like(strand)
        ref[:, 0] = x_offset
        ref[:, 1] = y_offset
        ref[:, 2] = strand[0, 2] + s
        return ref

    # Offset references in different planes so they're geometrically distinct
    ref_1 = _ref_along_z(nodes_1, x_offset=0.0, y_offset=0.0)
    ref_2 = _ref_along_z(nodes_2, x_offset=10.0, y_offset=0.0)
    return np.vstack([ref_1, ref_2])


# ---------------------------------------------------------------------------
# Visualisation: 3D plot of initial configuration
# ---------------------------------------------------------------------------

def plot_initial_geometry(nodes, N1, fixed_idx, R_curve=None, title="Initial"):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    nodes_1 = nodes[:N1]
    nodes_2 = nodes[N1:]
    ax.plot(nodes_1[:, 0], nodes_1[:, 1], nodes_1[:, 2],
            "b.-", lw=1.5, ms=3, label="strand 1")
    ax.plot(nodes_2[:, 0], nodes_2[:, 1], nodes_2[:, 2],
            "g.-", lw=1.5, ms=3, label="strand 2")
    ax.scatter(nodes[fixed_idx, 0], nodes[fixed_idx, 1], nodes[fixed_idx, 2],
               c="red", s=50, zorder=5, label="clamped")

    if R_curve is not None and R_curve < 1e12:
        # Draw the backbone arc for reference
        s_all = np.linspace(nodes[:, 2].min(), nodes[:, 2].max(), 80)
        theta = s_all / R_curve
        cx = -R_curve + R_curve * np.cos(theta)
        cz = R_curve * np.sin(theta)
        ax.plot(cx, np.zeros_like(s_all), cz, "k--", lw=1.0, alpha=0.6,
                label="backbone")

    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title(title)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# |delta''|^2 measurement from converged node positions
# ---------------------------------------------------------------------------

def measure_delta_double_prime(final_nodes, N1, N_helix_pts,
                               N_arc=2, R_curve=None):
    """
    Compute delta(s) = (r_1(s) - r_2(s))/2 and its second derivative w.r.t.
    backbone arc-length s, then average |delta''|^2 over the helical region.

    The pairing is done by node index: node i of strand 1 corresponds to node
    i of strand 2, since they were initialized at the same backbone position.
    Node indexing within the strand: [N_arc bottom-arc nodes | N_helix_pts
    helical nodes | N_arc top-arc nodes].  We measure on the helical nodes
    only, skipping a margin to avoid boundary effects.

    Parameters
    ----------
    final_nodes : (N1+N2, 3)
    N1 : number of nodes in strand 1 (must equal N2 for paired comparison)
    N_helix_pts : number of helical nodes per strand (the original n_pts)
    N_arc : number of arc nodes at each end (default 2)
    R_curve : backbone curvature (None => straight)

    Returns
    -------
    dict with:
        delta : (N_helix_pts, 3) array of delta values
        delta_sq_avg : <|delta|^2>
        ddelta_sq_avg : <|delta''|^2>  (the key quantity)
        s_node : backbone arc-length at each helical node
        margin : number of edge nodes skipped from each end
    """
    strand_1 = final_nodes[:N1]
    strand_2 = final_nodes[N1:]

    # Slice out the helical region (excluding the 2 arc nodes at each end)
    helix_1 = strand_1[N_arc:N_arc + N_helix_pts]
    helix_2 = strand_2[N_arc:N_arc + N_helix_pts]

    # Relative position
    delta = 0.5 * (helix_1 - helix_2)

    # Project onto backbone to get s coordinate
    if R_curve is None or R_curve > 1e12:
        # Straight backbone (z-axis):  s = z of midpoint
        midpts = 0.5 * (helix_1 + helix_2)
        s_node = midpts[:, 2]
    else:
        midpts = 0.5 * (helix_1 + helix_2)
        # s = R_curve * arctan2(z, x + R_curve)
        s_node = R_curve * np.arctan2(midpts[:, 2], midpts[:, 0] + R_curve)

    # Compute delta'(s) and delta''(s) by central finite differences
    # Use np.gradient with the actual (possibly non-uniform) s spacing
    delta_prime = np.gradient(delta, s_node, axis=0, edge_order=2)
    delta_double_prime = np.gradient(delta_prime, s_node, axis=0, edge_order=2)

    delta_sq = np.sum(delta * delta, axis=1)
    ddelta_sq = np.sum(delta_double_prime * delta_double_prime, axis=1)

    # Drop edge nodes where finite differences are less accurate
    margin = max(3, N_helix_pts // 8)
    delta_sq_avg = float(np.mean(delta_sq[margin:-margin]))
    ddelta_sq_avg = float(np.mean(ddelta_sq[margin:-margin]))

    return {
        "delta": delta,
        "delta_prime": delta_prime,
        "delta_double_prime": delta_double_prime,
        "s_node": s_node,
        "delta_sq_avg": delta_sq_avg,
        "ddelta_sq_avg": ddelta_sq_avg,
        "delta_sq_array": delta_sq,
        "ddelta_sq_array": ddelta_sq,
        "margin": margin,
        "n_measured": N_helix_pts - 2 * margin,
    }


# ---------------------------------------------------------------------------
# Bending energy by strand (uses Dismech's discrete curvature)
# ---------------------------------------------------------------------------

def strand_bending_energy(strand_nodes, EI):
    """
    Sum of (B/2) kappa_i^2 V_i over internal nodes of a single strand.
    kappa_i is the turning-angle curvature from three consecutive nodes.
    """
    n = len(strand_nodes)
    if n < 3:
        return 0.0, 0.0
    edges = strand_nodes[1:] - strand_nodes[:-1]
    edge_lens = np.linalg.norm(edges, axis=1)

    E_bend = 0.0
    L_tot = 0.0
    for i in range(1, n - 1):
        l1 = edge_lens[i - 1]
        l2 = edge_lens[i]
        if l1 < 1e-14 or l2 < 1e-14:
            continue
        e1 = edges[i - 1]
        e2 = edges[i]
        ang = np.arctan2(np.linalg.norm(np.cross(e1, e2)), np.dot(e1, e2))
        V = 0.5 * (l1 + l2)
        kappa = ang / V
        E_bend += 0.5 * EI * kappa**2 * V
        L_tot += V
    return E_bend, L_tot


# ---------------------------------------------------------------------------
# Single simulation run
# ---------------------------------------------------------------------------

def run(h=1.0, R_loop=50.0, n_cross=1, E=60.0,
        dt=1e-3, total_time=5.0, kc_scale=1.0, pts_per_turn=30,
        delta_imc=None, show_initial=False, R_curve=None,
        max_steps=500):
    """
    Set up and relax a two-rod intertwined helix to equilibrium, then
    measure <|delta''|^2>.

    Parameters
    ----------
    h : strand cross-section radius (= half-separation from backbone)
    R_loop : loop radius (sets wavenumber k via Audoly)
    n_cross : crossing number (winding angle (2n+1)*pi per strand)
    E : Young's modulus
    dt : initial timestep
    total_time : overdamped relaxation duration
    pts_per_turn : node density
    R_curve : backbone curvature radius (None = straight)
    max_steps : hard cap on successful Newton steps
    """
    nodes_1_helix, nodes_2_helix, k_theory, s_helix = make_two_strand_helix(
        h, R_loop, n_cross=n_cross, pts_per_turn=pts_per_turn, R_curve=R_curve
    )
    n_helix = len(nodes_1_helix)

    nodes_1 = attach_loop_arcs(nodes_1_helix, R_loop)
    nodes_2 = attach_loop_arcs(nodes_2_helix, R_loop)

    nodes, edges, N1, N2, fixed = build_two_rod_topology(nodes_1, nodes_2)
    ref = make_straight_reference(nodes, N1)

    if show_initial:
        plot_initial_geometry(nodes, N1, fixed, R_curve=R_curve,
                              title=f"Two-rod init  (R_curve={R_curve})")

    faces = np.empty((0, 3), dtype=int)
    geo_ref = dismech.Geometry(ref, edges, faces, plot_from_txt=False)

    I_sec = np.pi * h**4 / 4.0
    gp = dismech.GeomParams(
        rod_r0=h, shell_h=0.0,
        axs=1e6 * np.pi * h**2,
        ixs1=I_sec, ixs2=I_sec, jxs=2.0 * I_sec,
    )
    mat = dismech.Material(
        density=1e-9, youngs_rod=E, youngs_shell=0.0,
        poisson_rod=0.3, poisson_shell=0.0,
    )
    sim = dismech.SimParams(
        static_sim=False, two_d_sim=False, use_mid_edge=False,
        use_line_search=True, line_search_iters=10, show_floor=False,
        log_data=False, log_step=1, dt=dt, max_iter=500,
        total_time=total_time, plot_step=0,
        tol=0.1, ftol=1e-4, dtol=1e-8,
    )
    env = dismech.Environment()
    env.add_force("damping", eta=500.0)

    robot = dismech.SoftRobot(gp, mat, geo_ref, sim, env)
    robot = robot.fix_nodes(fixed)

    # Place the rods at their intertwined initial state
    q0 = robot.state.q.copy()
    q0[:3 * len(nodes)] = nodes.ravel()
    robot = robot.update(q=q0, u=np.zeros_like(q0))

    stepper = dismech.ImplicitEulerTimeStepper(robot)

    # ---- Overdamped relaxation ----
    print(f"Two-rod sim:  h={h}  R_loop={R_loop}  n_cross={n_cross}  "
          f"R_curve={R_curve}  ({len(nodes)} nodes)", flush=True)
    dt_max = 50.0 * dt
    t, step_i = 0.0, 0
    while t < total_time and step_i < max_steps:
        dt_step = float(sim.dt)
        try:
            robot, f_norm = stepper.step(robot)
        except Exception:
            sim.dt = max(1e-6, sim.dt * 0.5)
            if sim.dt < 1e-6:
                print(f"  Stopping at t={t:.4f}", flush=True)
                break
            continue
        if step_i % 10 == 0:
            print(f"  step {step_i:4d}  t={t:.4f}  |f|={f_norm:.2e}  "
                  f"dt={dt_step:.2e}", flush=True)
        if sim.dt < dt_max:
            sim.dt = min(dt_max, sim.dt * 2.0)
        step_i += 1
        t += dt_step
    print(f"  Done: {step_i} steps, t_final={t:.4f}s", flush=True)

    # ---- Extract results ----
    final = robot.state.q[:3 * len(nodes)].reshape(len(nodes), 3)
    EI = E * I_sec

    # Bending energies per strand
    E_bend_1, L_1 = strand_bending_energy(final[:N1], EI)
    E_bend_2, L_2 = strand_bending_energy(final[N1:], EI)

    # |delta''|^2
    dd = measure_delta_double_prime(final, N1=N1, N_helix_pts=n_helix,
                                    N_arc=2, R_curve=R_curve)

    # Strand-strand separation (sanity check: should be ~ 2h along helix)
    helix_1 = final[2:2 + n_helix]
    helix_2 = final[N1 + 2:N1 + 2 + n_helix]
    sep = np.linalg.norm(helix_1 - helix_2, axis=1)

    # Backbone curvature
    K_backbone = 1.0 / R_curve if (R_curve is not None and R_curve < 1e12) else 0.0
    k_used = k_theory  # since we constrain the topology, k_theory is enforced

    # Paper prediction
    ddelta_sq_paper = h**2 * (k_used**4 + 3 * k_used**2 * K_backbone**2
                              + 0.5 * K_backbone**4)

    print(f"\n  ---- Results ----")
    print(f"  E_bend strand 1   = {E_bend_1:.6f} N*mm  (L={L_1:.3f})")
    print(f"  E_bend strand 2   = {E_bend_2:.6f} N*mm  (L={L_2:.3f})")
    print(f"  separation min/max = {sep.min():.4f} / {sep.max():.4f} mm "
          f"(expected ~ {2 * h:.4f})")
    print(f"  <|delta''|^2>_sim  = {dd['ddelta_sq_avg']:.6e}")
    print(f"  <|delta''|^2>_th   = {ddelta_sq_paper:.6e}  "
          f"(h^2 (k^4 + 3k^2 K^2 + K^4/2))")
    print(f"  ratio sim/th       = "
          f"{dd['ddelta_sq_avg']/ddelta_sq_paper:.4f}")

    return {
        "h": h, "R_loop": R_loop, "n_cross": n_cross,
        "R_curve": R_curve if R_curve is not None else float('inf'),
        "K_backbone": K_backbone, "k_theory": k_theory,
        "E_bend_1": E_bend_1, "E_bend_2": E_bend_2,
        "L_1": L_1, "L_2": L_2,
        "ddelta_sq_sim": dd["ddelta_sq_avg"],
        "ddelta_sq_paper": ddelta_sq_paper,
        "ratio": dd["ddelta_sq_avg"] / ddelta_sq_paper if ddelta_sq_paper > 0 else float('nan'),
        "sep_min": float(sep.min()), "sep_max": float(sep.max()),
        "delta_sq_avg": dd["delta_sq_avg"],
        "n_measured": dd["n_measured"],
        "final": final,
        "N1": N1, "n_helix": n_helix,
    }


# ---------------------------------------------------------------------------
# Validation sweep: straight + curved
# ---------------------------------------------------------------------------

def validate_two_rod(R_loop=50.0, h=1.0, n_cross=1, pts_per_turn=30,
                     out_file="two_rod_results.txt"):
    """
    Sweep R_curve and compare measured <|delta''|^2> with the paper's
    h^2 (k^4 + 3 k^2 K^2 + K^4/2) prediction.
    """
    print("\n" + "=" * 70)
    print("  TWO-ROD DIFFERENCE-PROBLEM VALIDATION")
    print("=" * 70)

    run_configs = [
        # (R_curve, total_time, max_steps)
        (None,  3.0,  500),
        (200.0, 3.0,  500),
        (100.0, 3.0,  500),
        (70.0,  5.0,  600),
        (50.0,  5.0,  600),
        (35.0,  8.0,  800),
        (30.0,  8.0,  800),
        (20.0, 15.0, 1500),
        (15.0, 20.0, 2000),
        (10.0, 25.0, 2500),
    ]

    results = []
    for rc, tt, msteps in run_configs:
        rc_label = "inf" if rc is None else f"{rc}"
        print(f"\n{'='*60}\n  R_curve = {rc_label}\n{'='*60}")
        try:
            res = run(h=h, R_loop=R_loop, n_cross=n_cross,
                      total_time=tt, pts_per_turn=pts_per_turn,
                      show_initial=False, R_curve=rc, max_steps=msteps)
            res["status"] = "ok"
        except Exception as exc:
            res = {"R_curve": rc if rc is not None else float('inf'),
                   "status": f"FAIL: {exc}"}
            print(f"  FAIL: {exc}")
        results.append(res)

    ok = [r for r in results if r.get("status") == "ok"]
    if not ok:
        print("No successful runs.")
        return results

    baseline = next((r for r in ok if r["K_backbone"] < 1e-12), None)
    dd_str = baseline["ddelta_sq_sim"] if baseline else None
    k_base = baseline["k_theory"] if baseline else None

    # Print summary
    print("\n\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    header = (f"{'R_curve':>8} {'K/k':>7} | "
              f"{'<dd2>_sim':>11} {'<dd2>_paper':>11} {'ratio':>7} | "
              f"{'enh_sim':>8} {'enh_paper':>9}")
    print(header)
    print("-" * len(header))

    for r in ok:
        K = r["K_backbone"]
        k = r["k_theory"]
        Kk = K / k if k > 0 else 0.0
        sim_val = r["ddelta_sq_sim"]
        paper_val = r["ddelta_sq_paper"]
        ratio = sim_val / paper_val if paper_val > 0 else float('nan')
        enh_sim = sim_val / dd_str if dd_str else float('nan')
        enh_paper = 1.0 + 3.0 * Kk**2 + 0.5 * Kk**4
        rc = r.get("R_curve", float('inf'))
        rc_str = f"{rc:>8.1f}" if rc < 1e10 else f"{'inf':>8}"
        print(f"{rc_str} {Kk:>7.4f} | "
              f"{sim_val:>11.4e} {paper_val:>11.4e} {ratio:>7.4f} | "
              f"{enh_sim:>8.4f} {enh_paper:>9.4f}")

    with open(out_file, "w") as fh:
        fh.write("Two-rod difference-problem validation\n")
        fh.write(f"h={h}  R_loop={R_loop}  n_cross={n_cross}\n")
        fh.write(f"baseline <|delta''|^2>_str = {dd_str}\n\n")
        fh.write(header + "\n" + "-" * len(header) + "\n")
        for r in ok:
            K = r["K_backbone"]
            k = r["k_theory"]
            Kk = K / k if k > 0 else 0.0
            sim_val = r["ddelta_sq_sim"]
            paper_val = r["ddelta_sq_paper"]
            ratio = sim_val / paper_val if paper_val > 0 else float('nan')
            enh_sim = sim_val / dd_str if dd_str else float('nan')
            enh_paper = 1.0 + 3.0 * Kk**2 + 0.5 * Kk**4
            rc = r.get("R_curve", float('inf'))
            rc_str = f"{rc:.1f}" if rc < 1e10 else "inf"
            fh.write(f"{rc_str:>8} {Kk:>7.4f} | "
                     f"{sim_val:>11.4e} {paper_val:>11.4e} {ratio:>7.4f} | "
                     f"{enh_sim:>8.4f} {enh_paper:>9.4f}\n")
    print(f"\nResults written to {out_file}")

    plot_two_rod_validation(ok, baseline)
    return results


def plot_two_rod_validation(results, baseline):
    import matplotlib.pyplot as plt
    if baseline is None:
        return
    dd_str = baseline["ddelta_sq_sim"]
    k_base = baseline["k_theory"]
    h = baseline["h"]

    K_arr = np.array([r["K_backbone"] for r in results])
    k_arr = np.array([r["k_theory"] for r in results])
    dd_sim = np.array([r["ddelta_sq_sim"] for r in results])
    dd_paper = np.array([r["ddelta_sq_paper"] for r in results])

    Kk = K_arr / k_arr
    Kk_dense = np.linspace(0, max(Kk) * 1.2 if max(Kk) > 0 else 0.5, 200)
    paper_curve = 1.0 + 3.0 * Kk_dense**2 + 0.5 * Kk_dense**4

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: <|delta''|^2> vs (K/k)^2
    ax = axes[0]
    ax.plot(Kk_dense, paper_curve * dd_str, "k-", lw=2,
            label=r"Paper: $h^2(k^4 + 3k^2 K^2 + K^4/2)$")
    ax.plot(Kk, dd_sim, "ro", ms=8, label="Simulation")
    for i, r in enumerate(results):
        rc = r.get("R_curve", float('inf'))
        lbl = r"$\infty$" if rc > 1e10 else f"{rc:.0f}"
        ax.annotate(f"$R_c$={lbl}", (Kk[i], dd_sim[i]),
                    fontsize=7, xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("$K/k$")
    ax.set_ylabel(r"$\langle|\delta''|^2\rangle$ (mm$^{-2}$)")
    ax.set_title("Two-rod sim vs paper formula")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: enhancement ratio
    ax = axes[1]
    ax.plot(Kk_dense, paper_curve, "k-", lw=2, label="Paper formula")
    ax.plot(Kk, dd_sim / dd_str, "ro", ms=8, label="Simulation")
    for i, r in enumerate(results):
        rc = r.get("R_curve", float('inf'))
        lbl = r"$\infty$" if rc > 1e10 else f"{rc:.0f}"
        ax.annotate(f"$R_c$={lbl}", (Kk[i], dd_sim[i] / dd_str),
                    fontsize=7, xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("$K/k$")
    ax.set_ylabel(r"$\langle|\delta''|^2\rangle / \langle|\delta''|^2\rangle_{\rm str}$")
    ax.set_title("Curvature enhancement of contact pressure")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.suptitle(f"Two-rod difference problem  (h={h}, R_loop={baseline['R_loop']})",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig("two_rod_validation.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Saved: two_rod_validation.png")


# ---------------------------------------------------------------------------
# Entry points: smoke test + full sweep
# ---------------------------------------------------------------------------

def smoke_test():
    """Single straight run for fast iteration on geometry/contact setup."""
    print("\n" + "=" * 70)
    print("  SMOKE TEST: straight backbone, short relaxation")
    print("=" * 70)
    res = run(h=1.0, R_loop=50.0, n_cross=1, total_time=2.0,
              pts_per_turn=25, show_initial=True, R_curve=None,
              max_steps=300)
    return res


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"

    if mode == "smoke":
        smoke_test()
    elif mode == "sweep":
        validate_two_rod(R_loop=50.0, h=1.0, n_cross=1, pts_per_turn=30)
    else:
        print(f"Unknown mode: {mode}. Use 'smoke' or 'sweep'.")
