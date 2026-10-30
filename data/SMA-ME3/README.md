# SMA-ME3 input files

The four required Parquet files are separate model inputs:

- `spot_expression.parquet`: 521 spots × 351 genes, raw counts.
- `spot_coordinates.parquet`: x/y coordinates for the same 521 spots.
- `cell_expression.parquet`: 11,169 reference cells × the same 351 genes, raw counts.
- `celltype_labels.parquet`: one cell-type label for each reference cell.

`spot_ground_truth.parquet` is a separate optional evaluation file with 521 spots × 23 cell-type proportions. TRACE does not use it during fitting.

Every Parquet index contains the spot or cell ID. The cell and spot expression tables use identical ordered gene columns.
