# Third-party notices

## MAPE-PPI

The implementation under `MAPE-PPI/` is adapted from:

- Project: MAPE-PPI
- Upstream: https://github.com/LirongWu/MAPE-PPI
- Paper: MAPE-PPI: Towards Effective and Efficient Protein-Protein Interaction
  Prediction via Microenvironment-Aware Protein Embedding (ICLR 2024)
- Copyright: Copyright (c) 2024 Lirong Wu
- License: MIT; the complete upstream notice is retained in `MAPE-PPI/LICENSE`.
- Upstream main revision inspected for license provenance:
  `c11708b4b36ade89448606a4a5a42e6e6e25d59a`.
  This is a provenance reference, not a claim that every adapted file is an
  unmodified copy of that revision.

Adaptations include portable data/configuration paths, shared backbone imports,
DGL batch/device compatibility, singleton pair indexing, removal of unused
imports and historical experiment branches, and use of the baseline without
NNI. The model equations and learned parameter names are retained.

## Dependencies and data

PyTorch, DGL, NumPy, SciPy, tqdm, and Foldseek are external dependencies and
retain their own licenses. Their binaries are not redistributed here. Dataset
and AlphaFold links refer to external providers; those assets are not covered
by this repository's MIT license or included in its source distribution.
