# `tpms2cfd`: Dual-Track Analytic B-Rep & Watertight Sheet-TPMS CAD-to-CFD Toolkit

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](pyproject.toml)

`tpms2cfd` (`v1.0.0`) is an open-source Python toolkit for synthesizing lightweight, CFD-ready Periodic Open Cellular Structures (POCS) and Triply Periodic Minimal Surface (TPMS) monoliths with automated inverse fluid-domain extraction, plenum buffering, deterministic boundary-zone tagging, and topological/DfAM/mesh-quality auditing.

## Key Capabilities

1. **Track A — Analytic OpenCASCADE B-Rep Engine (`brep_engine.py`)**:
   - Synthesizes pure analytic cylindrical/spherical/planar cellular solids (`sq_channel`, `gyroid_srs`, `diamond_pocs`, `helix_pocs`, `kelvin_pocs`) and walled multi-cell monoliths (`MONO_GYROID_SRS_30x30x60`) without facet approximation.
   - Uses balanced pairwise Boolean fusion (`fuzzy = 1e-4 mm`), automatic inverse fluid extraction with upstream (`15 mm`) and downstream (`20 mm`) plenums, and strict split readback (`ocp_parse_transfer_s` + `ocp_heal_audit_s`) verification.
2. **Track B — Watertight Implicit Sheet-TPMS Engine (`sheet_engine.py`)**:
   - Evaluates five uniform and axially graded Sheet-TPMS kernels (`gyroid_sheet`, `diamond_sheet`, `schwarz_p_sheet`, `iwp_sheet`, `graded_gyroid_sheet`) via oversized Marching Cubes, volume-preserving Taubin smoothing (`+0.09%` to `+0.41%` volume drift), and exact `manifold3d` Boolean clipping.
   - Exports both `:.8e`-precision multi-zone ASCII STL (`*_FLUID_NAMED.stl`) and a **single closed vertex-welded Fluent ASCII surface mesh (`*_FLUID_SURFACE.msh`)** with explicit zone tables (`bc_id = 20` `mass-flow-inlet`, `5` `pressure-outlet`, `7` `symmetry`, `3` `wall`), eliminating open-baffle failures in commercial watertight workflows.
3. **Automated Topological, Optical Bypass, DfAM & Mesh QA (`qa_audit.py`)**:
   - Verifies zero boundary/non-manifold/degenerate edges (`merge_digits=8`), zero odd-parity ray errors, `100%` through-flow connectivity, `0%` trapped voids, axial line-of-sight (`LOS`) optical bypass, post-smoothing `0.15 mm` 3D Euclidean Distance Transform (`EDT`) ridge thickness (`min / p05 / median`), and deterministic single-threaded Gmsh 3D tetrahedral quality (`median SICN`, `p01 SICN`, `% SICN < 0.10`, zero negative Jacobians).

## Verified Benchmark Highlights

- **B-Rep Compression & Speedup**: Up to **`65.1x`** smaller STEP size (`0.88 MiB` vs `57.41 MiB`) and **`210.7x`** fewer faces against a faceted Sheet-Gyroid STEP control; in identical-geometry ablations on `UC_GYROID_SRS_RVE40`, analytic B-Rep achieves **`20.7–21.3x`** STEP size reduction, **`19.4–20.4x`** OpenCASCADE parse speedup, and avoids downstream Gmsh `OCCMakeSolids` kernel failure.
- **Format Ablation on Identical `57,308` Sheet Triangles**: Vertex-welded Fluent `.msh` (`2.32 MiB`) eliminates **`60.4x`** file bloat (`140.41 MiB`, `122.4 s` OCCT readback, `0` solids) incurred by sewing surface facets into planar STEP.
- **Multi-Backend Verification**: Tested across `11` positive configurations in deterministic Gmsh 3D meshing (`0` negative Jacobians), `14` headless ANSYS SpaceClaim 2026 R1 conversions (`100%` pass), and `12` ANSYS Fluent 2026 R1 watertight polyhedral CFD benchmarks (including developing square-duct verification within `3.5%` of Shah–London laminar entry-flow theory).

## Installation & Quick Start

```bash
pip install -e .
python -m unittest discover -s tests -v
tpms2cfd build --track all --out-dir ./build_out
tpms2cfd audit --manifest ./build_out/manifest.json
```

To reproduce the local geometry, DfAM, Gmsh, and controlled Pair A/B representation ablations cleanly on CPU:

```bash
python benchmarks/run_local_benchmark.py --out-dir ./benchmarks/geometries --results-dir ./benchmarks/results
```

Commercial ANSYS SpaceClaim and Fluent verification runners (`benchmarks/run_kmou_spaceclaim_bench.py`, `benchmarks/run_kmou_fluent_bench.py`) are configured via environment variables (`TPMS2CFD_REMOTE_HOST`, `TPMS2CFD_REMOTE_BASE`, `TPMS2CFD_REMOTE_PYTHON`, `AWP_ROOT261`).

## Citation & Authors

If you use `tpms2cfd` in academic work, please cite [`CITATION.cff`](CITATION.cff):

- **Tae-Hyung Kim** (`ttthkim@gmail.com`, Korea Institute of Limestone and Advanced Materials, ORCID `0009-0003-5047-7501`) — First author
- **Bora Jeong** (`bora1106@kitech.re.kr`, Korea Institute of Industrial Technology)
- **Min-Woo Lee**
- **Young-Jin Kim** (Korea Institute of Limestone and Advanced Materials) — Corresponding author
- **Hong-Dae Kim** (`hdkim@kitech.re.kr`, Korea Institute of Industrial Technology) — Corresponding author
