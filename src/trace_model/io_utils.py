"""Strict Parquet input loader for TRACE's separated dataset files."""

from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


REQUIRED_PATH_KEYS = (
    "spot_expression_file",
    "spot_coordinates_file",
    "cell_expression_file",
    "celltype_labels_file",
)


def _read_parquet(path, key, required=True):
    if not path and not required:
        return None
    if not path:
        raise KeyError(f"Set paths.{key} to a Parquet input file")
    path = Path(path)
    if path.suffix.lower() != ".parquet":
        raise ValueError(f"TRACE accepts Parquet input files only; paths.{key} is {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Input file does not exist: {path}")
    return pd.read_parquet(path)


def _validate_expression(frame, name):
    if frame.empty or frame.index.has_duplicates or frame.columns.has_duplicates:
        raise ValueError(f"{name} must be nonempty with unique observation and gene names")
    if not all(is_numeric_dtype(dtype) for dtype in frame.dtypes):
        try:
            frame = frame.apply(pd.to_numeric, errors="raise")
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must contain numeric raw counts") from error
    values = frame.to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"{name} must contain finite, nonnegative raw counts")
    if not np.allclose(values, np.rint(values), atol=1e-6):
        raise ValueError(f"{name} must contain raw integer counts")
    frame.index = frame.index.astype(str)
    frame.columns = frame.columns.astype(str)
    return frame


def load_trace_inputs(paths):
    """Load four aligned Parquet inputs and optional spot truth.

    Required files are spot-by-gene counts, spot coordinates, cell-by-gene
    counts, and one cell-type label per reference cell. An optional fifth file
    contains spot-by-cell-type truth for post-training evaluation only.
    """
    missing = [key for key in REQUIRED_PATH_KEYS if not paths.get(key)]
    if missing:
        raise KeyError(f"Missing required TRACE input paths: {', '.join('paths.' + k for k in missing)}")

    spots = _validate_expression(
        _read_parquet(paths["spot_expression_file"], "spot_expression_file"),
        "Spot expression matrix",
    )
    cells = _validate_expression(
        _read_parquet(paths["cell_expression_file"], "cell_expression_file"),
        "Reference cell expression matrix",
    )
    if not spots.columns.equals(cells.columns):
        raise ValueError("Spot and reference expression files must have identical ordered gene columns")

    coordinates = _read_parquet(paths["spot_coordinates_file"], "spot_coordinates_file")
    if coordinates.index.has_duplicates or not coordinates.index.astype(str).equals(spots.index):
        raise ValueError("Spot coordinate row IDs must exactly match spot expression row IDs and order")
    if list(coordinates.columns) != ["x", "y"]:
        raise ValueError("Spot coordinates must contain exactly two columns named 'x' and 'y'")
    coordinates.index = coordinates.index.astype(str)
    coordinates = coordinates.astype(np.float64)
    if not np.isfinite(coordinates.to_numpy()).all():
        raise ValueError("Spot coordinates must be finite")

    labels = _read_parquet(paths["celltype_labels_file"], "celltype_labels_file")
    if labels.index.has_duplicates or not labels.index.astype(str).equals(cells.index):
        raise ValueError("Cell-type label row IDs must exactly match reference expression row IDs and order")
    if list(labels.columns) != ["cell_type"]:
        raise ValueError("Cell-type label file must contain exactly one column named 'cell_type'")
    labels.index = labels.index.astype(str)
    labels = labels["cell_type"].astype(str).str.strip()
    if labels.eq("").any() or labels.str.lower().eq("nan").any():
        raise ValueError("Every reference cell must have a nonempty cell-type label")

    truth = _read_parquet(paths.get("spot_ground_truth_file"), "spot_ground_truth_file", required=False)
    cell_types = sorted(labels.unique())
    if truth is not None:
        if truth.index.has_duplicates or not truth.index.astype(str).equals(spots.index):
            raise ValueError("Spot truth row IDs must exactly match spot expression row IDs and order")
        truth.index = truth.index.astype(str)
        truth.columns = truth.columns.astype(str)
        if set(truth.columns) != set(cell_types):
            raise ValueError("Spot truth cell types must match the reference cell-type labels")
        truth = truth.loc[:, cell_types].astype(np.float64)
        values = truth.to_numpy()
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Spot truth must contain finite, nonnegative proportions")
        if not np.allclose(values.sum(axis=1), 1.0, atol=1e-5):
            raise ValueError("Each spot truth row must sum to one")

    return {
        "spot_expression": spots,
        "spot_coordinates": coordinates,
        "cell_expression": cells,
        "cell_types": labels,
        "ground_truth": truth,
        "cell_type_names": cell_types,
    }
