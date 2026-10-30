"""TRACE preprocessing for its separated Parquet input files."""

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

from io_utils import load_trace_inputs


def _transform_expression(adata, normalize=True, log=True):
    """Library-size normalize and log-transform a raw-count AnnData matrix."""
    result = adata.copy()
    if normalize:
        sc.pp.normalize_total(result, target_sum=1e4)
    if log:
        sc.pp.log1p(result)
    return result


def _dense(matrix):
    return matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)


def _select_marker_genes(reference, marker_config, seed):
    """Rank reference markers with Scanpy's logistic-regression ranking."""
    cell_types = sorted(reference.obs["cell_type"].astype(str).unique())
    marker_data = reference.copy()
    marker_data.obs["cell_type"] = pd.Categorical(
        marker_data.obs["cell_type"].astype(str), categories=cell_types, ordered=True
    )
    if marker_config.get("preprocess", True):
        marker_data = _transform_expression(
            marker_data,
            normalize=bool(marker_config.get("normalize", True)),
            log=bool(marker_config.get("log", True)),
        )
    marker_data.layers["trace_marker_input"] = marker_data.X.copy()
    sc.tl.rank_genes_groups(
        marker_data,
        groupby="cell_type",
        method="logreg",
        layer="trace_marker_input",
        use_raw=False,
        n_genes=marker_data.n_vars,
        key_added="trace_logreg_markers",
        random_state=int(seed),
    )
    ranked = marker_data.uns["trace_logreg_markers"]["names"]
    top_n = min(int(marker_config.get("top_gene_per_type", 100)), marker_data.n_vars)
    selected = sorted(
        {
            str(ranked[group][rank])
            for group in cell_types
            for rank in range(top_n)
            if str(ranked[group][rank]) in marker_data.var_names
        }
    )
    if not selected:
        raise ValueError("Marker ranking returned no genes")
    return selected, cell_types


def _generate_pseudo_spots(reference, cell_types, config, seed):
    """Generate reproducible count-sum pseudo-spots and cell fractions."""
    n_spots = int(config["spot_num"])
    min_cells = int(config["min_cell_num_in_spot"])
    max_cells = int(config["max_cell_num_in_spot"])
    max_types = int(config["max_cell_types_in_spot"])
    if n_spots <= 0 or min_cells <= 0 or max_cells < min_cells or max_types <= 0:
        raise ValueError("Invalid pseudo-spot generation settings")

    labels = reference.obs["cell_type"].astype(str).to_numpy()
    cells_by_type = {
        cell_type: np.flatnonzero(labels == cell_type) for cell_type in cell_types
    }
    rng = np.random.default_rng(int(seed))
    count_matrix = reference.X.tocsr() if sparse.issparse(reference.X) else sparse.csr_matrix(reference.X)
    pseudo_counts = np.zeros((n_spots, reference.n_vars), dtype=np.float32)
    proportions = np.zeros((n_spots, len(cell_types)), dtype=np.float32)
    type_index = {cell_type: index for index, cell_type in enumerate(cell_types)}

    for spot in range(n_spots):
        cell_count = int(rng.integers(min_cells, max_cells + 1))
        type_count = int(rng.integers(1, min(max_types, cell_count, len(cell_types)) + 1))
        chosen_types = rng.choice(cell_types, size=type_count, replace=False)
        sampled_types = list(chosen_types)
        if cell_count > type_count:
            sampled_types.extend(
                rng.choice(chosen_types, size=cell_count - type_count, replace=True).tolist()
            )
        rng.shuffle(sampled_types)
        sampled_cells = [
            int(rng.choice(cells_by_type[cell_type])) for cell_type in sampled_types
        ]
        pseudo_counts[spot] = np.asarray(count_matrix[sampled_cells].sum(axis=0)).ravel()
        for cell_type in sampled_types:
            proportions[spot, type_index[cell_type]] += 1.0 / cell_count

    names = [f"pseudo_{index:05d}" for index in range(n_spots)]
    pseudo = ad.AnnData(
        X=sparse.csr_matrix(pseudo_counts),
        obs=pd.DataFrame(index=pd.Index(names, name="observation_id")),
        var=reference.var.copy(),
    )
    fraction_table = pd.DataFrame(proportions, index=names, columns=cell_types)
    return pseudo, fraction_table


def preprocess_trace(input_paths, output_path, marker_config, pseudo_config,
                     normalization_config, feature_config, seed=0):
    """Prepare real spots, pseudo-spots, reference profiles and optional truth."""
    loaded = load_trace_inputs(input_paths)
    spot_expression = loaded["spot_expression"]
    cell_expression = loaded["cell_expression"]
    labels = loaded["cell_types"]
    cell_types = loaded["cell_type_names"]
    genes = pd.Index(spot_expression.columns.astype(str), name="gene")

    real = ad.AnnData(
        X=sparse.csr_matrix(spot_expression.to_numpy(dtype=np.int32)),
        obs=pd.DataFrame(index=pd.Index(spot_expression.index.astype(str), name="spot_id")),
        var=pd.DataFrame(index=genes),
    )
    reference = ad.AnnData(
        X=sparse.csr_matrix(cell_expression.to_numpy(dtype=np.int32)),
        obs=pd.DataFrame(
            {"cell_type": labels.to_numpy()},
            index=pd.Index(cell_expression.index.astype(str), name="cell_id"),
        ),
        var=pd.DataFrame(index=genes),
    )
    reference.obs["cell_type"] = reference.obs["cell_type"].astype(str).str.strip()
    real_coordinates = pd.DataFrame(
        loaded["spot_coordinates"].to_numpy(dtype=np.float32),
        index=real.obs_names,
        columns=["x", "y"],
    )
    ground_truth = loaded["ground_truth"]
    if ground_truth is not None:
        ground_truth = ground_truth.astype(np.float32)

    selected_genes, ranked_cell_types = _select_marker_genes(reference, marker_config, seed)
    if ranked_cell_types != sorted(cell_types):
        raise ValueError("Reference labels and ground-truth cell-type ordering are inconsistent")
    common_genes = [gene for gene in selected_genes if gene in real.var_names]
    if not common_genes:
        raise ValueError("No selected marker genes overlap the spatial expression matrix")

    reference.obs["cell_type"] = pd.Categorical(
        reference.obs["cell_type"].astype(str), categories=cell_types, ordered=True
    )
    pseudo, pseudo_fractions = _generate_pseudo_spots(
        reference, cell_types, pseudo_config, seed
    )

    normalize = bool(normalization_config.get("normalize", True))
    log = bool(normalization_config.get("log", True))
    real_transformed = _transform_expression(real, normalize=normalize, log=log)
    pseudo_transformed = _transform_expression(pseudo, normalize=normalize, log=log)
    real_selected = real_transformed[:, common_genes].copy()
    pseudo_selected = pseudo_transformed[:, common_genes].copy()

    combined = ad.AnnData(
        X=sparse.vstack([real_selected.X, pseudo_selected.X], format="csr"),
        var=real_selected.var.copy(),
    )
    if bool(feature_config.get("scale", True)):
        sc.pp.scale(combined, zero_center=False)
    features = _dense(combined.X).astype(np.float32, copy=False)
    n_real = real.n_obs
    feature_columns = list(common_genes)
    real_features = pd.DataFrame(features[:n_real], index=real.obs_names, columns=feature_columns)
    pseudo_features = pd.DataFrame(
        features[n_real:], index=pseudo.obs_names, columns=feature_columns
    )
    all_features = pd.concat([real_features, pseudo_features], axis=0)

    reference_selected = _transform_expression(
        reference[:, common_genes].copy(), normalize=True, log=True
    )
    reference_expression = pd.DataFrame(
        _dense(reference_selected.X),
        index=reference_selected.obs_names,
        columns=feature_columns,
    )
    reference_means = reference_expression.assign(
        cell_type=reference_selected.obs["cell_type"].astype(str).to_numpy()
    ).groupby("cell_type", observed=False).mean().reindex(cell_types)
    if reference_means.isna().any().any():
        raise ValueError("Could not calculate a reference expression profile for every cell type")

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    marker_path = output_path / "marker_genes.txt"
    marker_path.write_text("".join(f"{gene}\n" for gene in common_genes))
    print(
        f"TRACE preprocessing: {real.n_obs:,} spatial spots, {reference.n_obs:,} "
        f"reference cells, {len(common_genes):,} marker genes, {len(cell_types)} cell types, "
        f"{pseudo.n_obs:,} pseudo-spots"
    )

    return {
        "real_spots": real_features,
        "real_coordinates": real_coordinates,
        "pseudo_spots": pseudo_features,
        "all_spots": all_features,
        "pseudo_dist": pseudo_fractions,
        "cell_types": np.asarray(cell_types, dtype=object),
        "sc_ref": reference_expression.assign(
            cell_type=reference_selected.obs["cell_type"].astype(str).to_numpy()
        ),
        "ground_truth": ground_truth,
    }
