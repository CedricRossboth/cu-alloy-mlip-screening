# Data

## Included

- `adsorbates/` — XYZ structures of adsorbate molecules used in the coupling reactions
- `cu_alloy.extxyz` — Screened copper alloy slab structures
- `select_bulks.csv` — Selected bulk material IDs from the Materials Project
- `select_bulks.py`, `exp_bulks.py` — Scripts to query and filter bulk structures

## Not included (too large for GitHub)

The following files are needed to rerun the full screening but exceed GitHub's size limits:

- `pkls/bulks.pkl`, `pkls/stable_mater_from_mp.pkl`, etc. — Bulk structures retrieved from the Materials Project via `data_retrieval.ipynb`
- `alloy/*_jobs/*.pkl` — Pre-split bulk structure batches for parallel SLURM jobs

To regenerate them, run `notebooks/data_retrieval.ipynb` with a valid [Materials Project API key](https://materialsproject.org/api).
