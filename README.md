# HiSCL-PPI

Code for **Structure-aware Contrastive Learning for Protein Network Inference**.
HiSCL-PPI adds residue-, segment-, and protein-level hierarchical contrastive
learning (HCL) and soft TM-score-guided contrastive learning (TCL) to MAPE-PPI.
It predicts seven interaction types on SHS27k, SHS148k, and STRING using fixed
Random/DFS/BFS partitions. Protein pretraining is followed by a GIN classifier
trained on fixed protein embeddings.

## Implementation status

This repository is named **HiSCL-PPI** and contains the implementation used for
our hierarchical InfoNCE and soft TM-score experiments.

**The ACCL surrogate described in Section 3.2.2 of the manuscript is not
implemented in this version.** Residue-, segment-, and protein-level HCL use
symmetric cross-view InfoNCE, as implemented in `src/losses.py`. Pooling uses
ordinary means; there is no ACCL cross-scale surrogate, OT pooling, or adversarial
optimization path. Adding that implementation is deferred. The `full` runner
variant means the complete **implemented HCL + soft TCL** pipeline; it does not
activate ACCL. Existing experiment results must not be attributed to an ACCL
run merely because ACCL is described in the manuscript.

| Component | Status | Source |
|---|---|---|
| MAPE-PPI backbone and downstream GIN | Included | `MAPE-PPI/src/` |
| Three-scale symmetric InfoNCE HCL | Included | `src/losses.py` |
| Soft TM-score-guided TCL | Included | `src/losses.py` |
| ACCL cross-scale surrogate | Not implemented | Manuscript Section 3.2.2 |

## Repository layout

```text
MAPE-PPI/src/          Single shared MAPE-PPI backbone, loader, and trainer
MAPE-PPI/preprocessing/ Original target-benchmark preprocessing reference
src/                  InfoNCE HCL, soft TCL, pretraining, runner, and aggregation
src/tm_matrix/        Directed query-normalized TM matrix construction and lookup
configs/backbone.json Shared dataset/split parameters used by every method
configs/method.json   One HCL/TCL configuration; ablations switch loss weights
scripts/              TM construction and test entry points
tests/                Synthetic numerical and complete-pipeline tests
environment.yml       CUDA environment matching the release validation stack
```

The backbone source is kept once and imported by the extension. Its model and
loss equations are retained; changes remove unused dependencies and add portable
input/configuration paths and CPU/GPU graph handling. There are no datasets,
checkpoints, figures, historical experiment branches, or run outputs in this
repository. Commands below run from the **HiSCL-PPI** repository root.

## Environment

Linux, Python 3.9, PyTorch 2.1.2, DGL 1.1.3, NumPy 1.24.4, and SciPy 1.13.1:

```bash
conda env create -f environment.yml
conda activate hiscl_ppi
python -c "import torch,dgl,numpy; print(torch.__version__,dgl.__version__,numpy.__version__)"
```

The CUDA environment requires an NVIDIA driver compatible with CUDA 12.1.
For CPU verification, install the corresponding CPU wheels in a Python 3.9
virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu
pip install dgl==1.1.3 numpy==1.24.4 scipy==1.13.1 tqdm==4.70.0
bash scripts/test.sh
```

The test command uses CPU synthetic graphs, temporary TM matrices, and short
training runs for all four paper variants; it does not require benchmark data.
GPU kernels can remain nondeterministic with a fixed seed. The environment used
for release validation differs from some historical paper runs, so exact
floating-point reproduction also depends on the original software and inputs.

## Data

Obtain the upstream MAPE-PPI datasets:

- [Processed protein/PPI data](https://drive.google.com/file/d/16JkdWJ92jy_kVGc92ngXlLKbcSLAPu0P/view)
- [Raw sequences, interactions, and structures](https://drive.google.com/file/d/1hJVrQXddB9JK68z7jlIcLfd9AmTWwgJr/view)

Extract the processed files into `data/processed_data/`. Alternatively, pass
`--data-root /path/to/data` to the commands below; this directory must contain
`processed_data/` and, for TCL experiments, `tm_matrix/`.
For each `DS` in `SHS27k`, `SHS148k`, and `STRING`, the required prepared inputs are:

```text
protein.nodes.DS.pt
protein.knn.edges.DS.npy
protein.rball.edges.DS.npy
protein.DS.sequences.dictionary.csv
DS_ppi.pkl
DS_ppi_label.pkl
DS_random.json, DS_dfs.json, DS_bfs.json
```

Retain the original fixed split files and processed protein order. A missing
split is rejected by the paper runner; it is never silently regenerated.
If PPI caches are absent, `prepare_graphs.py` creates them through the upstream
loader using `protein.actions.DS.txt`, the dictionary, and the fixed Random split.
DGL graph pickles are version-sensitive. Build graph caches in the installed
environment from the node and edge files, before any parallel run:

```bash
python src/prepare_graphs.py --data-root data
```

The resulting `DS_protein_graphs.pkl` must match the dictionary row order. For a
cache made with another DGL version, move it outside the data directory before
rebuilding. Edge-array elements must be lists of `(source, target)` tuples.
All submitted commands leave the fixed partitions intact.

### Target structural supervision

Only `full` and `without_hcl` require TM matrices. Provide the exact paper matrices
when available. Otherwise, the included public-service downloader can obtain
AlphaFold structures named by dictionary protein IDs:

```bash
python src/fetch_structures.py \
  --dictionary data/processed_data/protein.SHS27k.sequences.dictionary.csv \
  --output structures/shs27k
python src/fetch_structures.py \
  --dictionary data/processed_data/protein.STRING.sequences.dictionary.csv \
  --output structures/string
```

This step requires `curl` and internet access. It maps Ensembl protein IDs through
UniProt/Ensembl, downloads AlphaFold models, and records unavailable IDs and
sequence mismatches. Existing downloads are reused; mismatched structures are
reported and retained, matching the construction policy of the experiments.
Live databases change: these downloads cannot guarantee the historical structure
snapshot. Exact numerical reproduction requires the same structure snapshot,
protein-ID mapping, and fixed splits as the experiments.

Install the external [Foldseek](https://github.com/steineggerlab/foldseek) binary
(the experiments used version `463739e0014a1549a527de589102cde98f802f37`) and set
`FOLDSEEK` to its path. Then build SHS27k and STRING matrices and extract SHS148k:

```bash
export FOLDSEEK=/path/to/foldseek
export THREADS=8
bash scripts/build_tm.sh structures/shs27k/pdb \
  data/processed_data/protein.SHS27k.sequences.dictionary.csv \
  data/tm_matrix/shs27k_af2_qtm
bash scripts/build_tm.sh structures/string/pdb \
  data/processed_data/protein.STRING.sequences.dictionary.csv \
  data/tm_matrix/string_af2_qtm
python src/tm_matrix/subset_tm_matrix.py \
  --source data/tm_matrix/string_af2_qtm \
  --dictionary data/processed_data/protein.SHS148k.sequences.dictionary.csv \
  --output data/tm_matrix/shs148k_af2_qtm
```

Foldseek uses `query,target,qtmscore`, E-value 0.1, and its prefilter. The builder
keeps directed non-self hits with score >=0.5, takes the maximum duplicate score,
and retains unavailable proteins as empty rows. Each matrix directory contains
`mapping.json` and `tm_matrix.npz`. Missing entries are unavailable evidence.

## Run the implemented experiments

The implemented full variant uses **InfoNCE HCL and soft TCL**, HCL weight 1.0, TCL weight 0.3, protein
batch size 128, 50 pretraining epochs, and a 12,000-vector contrastive cap.
The downstream model runs for 500 epochs. Validation is performed at epochs
1, 11, ..., 491; the test score at the highest validation micro-F1 is reported,
with later checkpoints preferred for tied validation scores.

| Runner variant | HCL/soft-TCL experiment | HCL weight | Soft TCL weight |
|---|---|---:|---:|
| `baseline` | MAPE-PPI reproduction | 0 | 0 |
| `full` | HiSCL-PPI (implemented InfoNCE HCL + soft TCL) | 1.0 | 0.3 |
| `without_hcl` | HiSCL-PPI (w/o HCL) | 0 | 0.3 |
| `without_tcl` | HiSCL-PPI (w/o TCL) | 1.0 | 0 |

Run one experiment:

```bash
python src/run.py --variant full --dataset STRING --split bfs --seed 0 --gpu 0
```

Run the four implemented variants, three datasets, three partitions, and seeds 0-4 (180 runs):

```bash
python src/sweep.py --gpu-ids 0,1,2,3
```

Run the full implementation and its two component ablations:

```bash
python src/sweep.py --gpu-ids 0,1,2,3 --variants full without_hcl without_tcl
```

Use `--data-root` and `--output-root` for external inputs/outputs, or `--gpu cpu`
for a single CPU run. The sweep assigns one experiment to each specified GPU.
A successful run has `DONE=0`; rerunning identical completed settings skips it.
An incomplete run or a changed configuration requires a fresh output root.

The archived baseline used batch 512 for STRING-Random seeds 0-2 and batch 128
elsewhere; the runner preserves that documented exception, including the
embedding extraction batch. The three proposed variants always use batch 128.
Published reference results from external papers are not locally retrained.

Summarize validation-selected test scores over seeds:

```bash
python src/aggregate.py --root runs
```

Each run writes an isolated configuration snapshot, pretraining checkpoint,
downstream checkpoint, log, and `PerformMetrics_Metrics.csv`. The aggregator
uses the second-to-last metric (`test_val`), never the best test epoch.
All generated files stay under the selected output root.

## Attribution

The shared backbone is adapted from **MAPE-PPI: Towards Effective and Efficient
Protein-Protein Interaction Prediction via Microenvironment-Aware Protein
Embedding**, ICLR 2024 ([paper](https://openreview.net/forum?id=itGkF993gz),
[upstream code](https://github.com/LirongWu/MAPE-PPI)).
HCL and soft TCL introduce no additional learned modules. See `src/losses.py`
for the objectives and `src/pretrain.py` for their integration with the baseline.

## Release checks

The release passes 15 numerical, configuration, matrix, and pipeline tests.
All four runner variants also completed synthetic 50-epoch pretraining and
500-epoch downstream runs. A fixed-seed CPU comparison verifies identical loss
values, parameter updates, and RNG state for the three proposed variants before
and after removing the historical branches. TM construction was exercised with
the Foldseek binary on six external structures. No benchmark retraining or live
structure-download validation is claimed by these checks. The pinned libraries
were verified in a validation virtual environment reusing the installed stack.

## License

HiSCL-PPI is distributed under the [MIT License](LICENSE). The bundled MAPE-PPI
code retains its [upstream MIT notice](MAPE-PPI/LICENSE). See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for provenance and adaptations.
Dataset downloads and external dependencies retain their own terms.

Author names, author citation metadata, and personal repository links are omitted
from this anonymous review version. Third-party attribution and license notices
identify upstream dependencies, not the submitting authors.

## Development and submission checks

Run the source checks and synthetic tests before committing:

```bash
python -B scripts/check_release.py
bash scripts/test.sh
```

The GitHub Actions workflow runs these checks on CPU for pushes and pull
requests. Local passing tests do not imply that the hosted workflow has run.
The `.gitignore` excludes datasets, structure downloads, checkpoints, run logs,
Python environments, and credentials. Use a fresh `main` history containing
only this source directory, not the parent research workspace. For anonymous
review, do not share a repository URL or commit history identifying the authors.
Use a venue-permitted anonymous artifact service or an anonymous source archive.
