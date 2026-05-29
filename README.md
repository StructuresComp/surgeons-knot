# Supplementary materials

## Layout

- `two_rod_validation/` — direct numerical validation of the curved-braid
  contact-pressure formula (manuscript Section 3.2.2).
  - `sim_two_rod_braid.py` — discrete elastic-rod simulation of two
    intertwined helical strands on a straight or curved backbone; sweeps
    the backbone curvature K and writes the measured bending kernel
    ⟨|δ″|²⟩ to `two_rod_results.txt`.
  - `plot_two_rod_validation.py` — reads `two_rod_results.txt` and
    produces the manuscript figure `two_rod_combined.{pdf,svg,png}`.
  - `two_rod_results.txt` — simulation output used by the plotter.

- `surgeon_knot/` — manuscript Figures 3 and 4 from the experimental
  surgeon's-knot data (Section 3.4).
  - `supplementary_figures_3_4.py` — reads the six `Summary_*.xlsx`
    files, fits a single Coulomb friction coefficient μ across all six
    experimental configurations, and writes
    `FIG3_baseline_failure.{pdf,svg,png}` (single-panel illustration of
    n-foil-baseline failure) and
    `FIG4_corrected_vs_uncorrected.{pdf,svg,png}` (six-panel corrected
    vs uncorrected model overlay).
  - `Summary_<diameter>lb(ni=N,no=1).xlsx` — per-step means and
    standard deviations of the experimental measurements, averaged
    across three independent tightening trials per configuration (also
    stored as additional sheets in the same workbook).

## Reproducing the figures

From the repository root:

```bash
pip install -r requirements.txt

# Manuscript Figure 2 (two-rod validation):
cd two_rod_validation && python plot_two_rod_validation.py && cd ..

# Manuscript Figures 3 + 4 (surgeon's-knot data):
cd surgeon_knot && python supplementary_figures_3_4.py && cd ..
```

`plot_two_rod_validation.py` reads the bundled `two_rod_results.txt`
and does not call the simulation.  To regenerate that file from
scratch, run `python sim_two_rod_braid.py` from the same directory;
the sweep takes a few minutes on a modern laptop and depends on
`dismech` (see Dependencies below).

## Dependencies

- Python ≥ 3.8
- `numpy`, `matplotlib`, `scipy`, `openpyxl` — all pip-installable; see
  `requirements.txt`.
- `dismech` (<https://github.com/StructuresComp/dismech-rods>) — only
  required to *rerun* the simulation in `sim_two_rod_braid.py`.  The
  rest of the pipeline runs from the bundled `two_rod_results.txt`
  without it.
