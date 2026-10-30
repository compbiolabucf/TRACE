# TRACE

<img src="figure/TRACE.png">

TRACE is a graph-based variational autoencoder for spatial transcriptomics cell-type deconvolution. It builds a sparse graph over measured spatial spots and simulated pseudo-spots, encodes their expression profiles with graph convolutions, and predicts a cell-type proportion for each spot using single-cell reference profiles. The included SMA-ME3 dataset provides an end-to-end example.

## Contents

```text
TRACE/
├── data/SMA-ME3/
│   ├── spot_expression.parquet      # Spot × gene raw counts
│   ├── spot_coordinates.parquet     # Spot × x/y coordinates
│   ├── cell_expression.parquet      # Reference cell × gene raw counts
│   ├── celltype_labels.parquet      # Reference cell × cell-type label
│   └── spot_ground_truth.parquet    # Optional evaluation truth
├── config.yaml                      # SMA-ME3 run configuration
├── train.py                         # Training and evaluation entry point
├── preprocess.py                    # Model preprocessing
├── io_utils.py                      # Strict Parquet input validation
├── adjacency.py                     # Sparse graph construction
├── models.py                        # TRACE graph VAE
├── losses.py                        # Training objective
├── metrics.py                       # Proportion evaluation metrics
├── plotting.py                      # Spatial proportion plot
├── requirements.txt
└── outputs/                         # Default run output directory
```

## Install and run

Python 3.10 or newer is recommended. Install a PyTorch build appropriate for your CPU/CUDA environment, then install the dependencies and run the included config:

```bash
cd ./TRACE
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python train.py --config config.yaml
```

TRACE automatically uses CUDA when it is available and otherwise runs on CPU. A short end-to-end check can be run with:

```bash
python train.py --config config.yaml \
  --set n_epochs=2 \
  --set pseudo_spot_simulation_paras.spot_num=100 \
  --set progress.enabled=false
```

Run outputs are written under `outputs/SMA-ME3/seed-6077/`. They include predicted spot proportions, evaluation metrics when the optional truth file is configured, the trained model checkpoint, selected marker genes, a loss plot, and a spatial pie plot.

## Input files

TRACE reads Parquet files only. The four required files remain separate:

- `spot_expression_file`: spot-by-gene raw integer counts.
- `spot_coordinates_file`: the same spot IDs in the same row order, with columns named `x` and `y`.
- `cell_expression_file`: reference cell-by-gene raw integer counts, with the same ordered genes as the spot matrix.
- `celltype_labels_file`: one `cell_type` column, indexed in the same order as the reference cell matrix.

The optional `spot_ground_truth_file` contains spot-by-cell-type proportions and is used only for post-training evaluation. It is not used to fit TRACE. All five paths are shown in `config.yaml`; change them to use a different dataset. Keep expression files as raw counts because TRACE performs library-size normalization and log transformation during preprocessing.

All tables use their observation IDs as the Parquet index, preserving spot/cell alignment and gene names without additional sidecar files.

## SMA-ME3 example data

The included processed SMA-ME3 files contain 521 spatial spots, 11,169 reference single cells, 351 shared genes, and 23 cell types.